import os
import sys
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import project
import argparse
from train.train_util import init_model


from torch import nn

from architectures.config import RainConfig
from components.attention import MoTAttention, MoTGroupQueryAttention
from components.mlp import MoTFFN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniRain Pretraining")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument('--hidden_size', default=1024, type=int, help="隐藏层维度")
    parser.add_argument('--vision_ffn_scale', default=2, type=int, help="视觉 FFN 缩放比例")
    parser.add_argument('--model_type', default='v', type=str, help="模型类别")
    parser.add_argument('--num_hidden_layers', default=20, type=int, help="隐藏层数量")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--use_mot', default=1, type=int, choices=[0, 1], help="是否使用LOOP架构（0=否，1=是）")
    parser.add_argument('--use_bounce', default=0, type=int, choices=[0, 1], help="是否使用Bounce架构（0=否，1=是）")
    parser.add_argument('--loop_start', type=int, default=0, help="Loop起始层")
    parser.add_argument('--loop_end', type=int, default=0, help="Loop结束层")
    parser.add_argument('--max_loop_iter', type=int, default=1, help="Loop最大迭代次数")
    parser.add_argument("--use_block_attn_res", action="store_true", help="是否使用块注意力残差")
    parser.add_argument("--use_full_attn_res", action="store_true", help="是否使用全注意力残差")
    parser.add_argument("--block_size", default=4, type=int, help="块大小")

    args = parser.parse_args()

    config = RainConfig(hidden_size=args.hidden_size, n_hidden_layers=args.num_hidden_layers,
                        use_moe=bool(args.use_moe), use_mot=bool(args.use_mot),
                        use_bounce=bool(args.use_bounce), loop_start=args.loop_start,
                        loop_end=args.loop_end, max_loop_iter=args.max_loop_iter,
                        vision_ffn_scale=args.vision_ffn_scale,
                        block_size=args.block_size,
                        model_type=args.model_type,
                        block_attn_res=bool(args.use_block_attn_res),
                        full_attn_res=bool(args.use_full_attn_res))

    print(config)
    model, tokenizer = init_model(config, from_weight=args.from_weight, tokenizer_path=project.TOKENIZER_PATH)
    print(f"Model Type: {model.name}")

    # 模态编号约定：0=文本 1=图像，与 MoTFFN 的 mlp_t/mlp_v、modality_mask 语义一致
    VISION_EXPERT = 1

    def count_params(module, trainable_only: bool = False):
        """trainable_only=True 只统计 requires_grad 的参数；默认统计全部（冻结的预训练骨干也算体积）"""
        if module is None:
            return 0
        return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)

    def pair(module):
        """返回 (总参数, 可训练参数)"""
        return count_params(module), count_params(module, trainable_only=True)

    def fmt(total: int, trainable: int) -> str:
        text = f"{total/1e6:.2f}M"
        if total != trainable:
            text += f" (可训练 {trainable/1e6:.2f}M)"
        return text

    def vision_attn(module):
        """
        取注意力里属于视觉专家的参数，返回 (总参数, 可训练参数)。
        MoTGroupQueryAttention: wq/wk/wv/wo 均为 ModuleList[n_modalities]
        MoTAttention:          local_experts_wq/wk/wv/wo + 逐模态 norm
        普通 GQA 各模态共享同一套投影，没有模态专属参数 → (0, 0)
        """
        if isinstance(module, MoTGroupQueryAttention):
            names = ("wq", "wk", "wv", "wo")
        elif isinstance(module, MoTAttention):
            names = ("local_experts_wq", "local_experts_wk", "local_experts_wv", "local_experts_wo",
                     "local_experts_q_normalization", "local_experts_k_normalization",
                     "local_experts_attention_norm")
        else:
            return 0, 0
        total = trainable = 0
        for name in names:
            experts = getattr(module, name, None)  # q/k norm 只在 qk_normalization=True 时存在
            if not isinstance(experts, nn.ModuleList) or len(experts) <= VISION_EXPERT:
                continue
            t, r = pair(experts[VISION_EXPERT])
            total += t
            trainable += r
        return total, trainable

    backbone = model.model
    # 视觉 stem：兼容新旧命名 patch_embed / patch_embd，也兼容 ConvStem / ViTStem
    patch_embed = getattr(backbone, "patch_embed", None) or getattr(backbone, "patch_embd", None)
    layers = getattr(backbone, "layers", [])

    vision_stem = pair(patch_embed)
    vision_attn_pair = [0, 0]
    # MoT 架构下视觉 FFN 为每层 MoTFFN.mlp_v；非 MoT/非视觉模型没有独立视觉 FFN
    vision_ffn = [0, 0]
    for layer in layers:
        t, r = vision_attn(getattr(layer, "attn", None))
        vision_attn_pair[0] += t
        vision_attn_pair[1] += r
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, MoTFFN):
            t, r = pair(mlp.mlp_v)
            vision_ffn[0] += t
            vision_ffn[1] += r

    vision_total = (vision_stem[0] + vision_attn_pair[0] + vision_ffn[0],
                    vision_stem[1] + vision_attn_pair[1] + vision_ffn[1])
    model_total = pair(model)
    print(f'[ScalingLaw] Vision Stem Params: {fmt(*vision_stem)}')
    print(f'[ScalingLaw] Vision Attn Params: {fmt(*vision_attn_pair)}')
    print(f'[ScalingLaw] Vision FFN  Params: {fmt(*vision_ffn)}')
    print(f'[ScalingLaw] Vision Total Params: {fmt(*vision_total)}  '
          f'占比 {vision_total[0] / max(model_total[0], 1) * 100:.1f}%')
    print(f'[ScalingLaw] Model Params: {fmt(*model_total)}')
