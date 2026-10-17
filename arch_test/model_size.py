import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from train.train_util import init_model


from architectures.config import RainConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniRain Pretraining")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument('--hidden_size', default=1024, type=int, help="隐藏层维度")
    parser.add_argument('--model_type', default='v', type=str, help="模型类别")
    parser.add_argument('--num_hidden_layers', default=12, type=int, help="隐藏层数量")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--use_mot', default=1, type=int, choices=[0, 1], help="是否使用LOOP架构（0=否，1=是）")
    parser.add_argument('--use_bounce', default=0, type=int, choices=[0, 1], help="是否使用Bounce架构（0=否，1=是）")
    parser.add_argument('--loop_start', type=int, default=0, help="Loop起始层")
    parser.add_argument('--loop_end', type=int, default=0, help="Loop结束层")
    parser.add_argument('--max_loop_iter', type=int, default=1, help="Loop最大迭代次数")
    parser.add_argument("--use_block_attn_res", action="store_true", help="是否使用块注意力残差")
    parser.add_argument("--use_full_attn_res", action="store_true", help="是否使用全注意力残差")
    parser.add_argument("--block_size", default=2, type=int, help="块大小")

    args = parser.parse_args()

    config = RainConfig(hidden_size=args.hidden_size, n_hidden_layers=args.num_hidden_layers,
                        use_moe=bool(args.use_moe), use_mot=bool(args.use_mot),
                        use_bounce=bool(args.use_bounce), loop_start=args.loop_start,
                        loop_end=args.loop_end, max_loop_iter=args.max_loop_iter,
                        block_size=args.block_size,
                        model_type=args.model_type,
                        block_attn_res=bool(args.use_block_attn_res),
                        full_attn_res=bool(args.use_full_attn_res))

    print(config)
    model, tokenizer = init_model(config, from_weight=args.from_weight)
    print(f"Model Type: {model.name}")
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    vision_params = sum(p.numel() for p in model.model.patch_embd.parameters() if p.requires_grad)
    print(f'[ScalingLaw] Vision Embed Params: {vision_params/1e6:.2f}M')
    print(f'[ScalingLaw] Model Params: {model_params/1e6:.2f}M')
