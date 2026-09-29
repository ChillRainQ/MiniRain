import os

import torch
import argparse
from architectures.config import RainConfig
from architectures.rain import MiniRainVForCausalLM

# ---- 特殊 token id（按你的 tokenizer_config）----
IMG_START = 9    # <|vision_start|>
IMG_END   = 10   # <|vision_end|>
IMG_PAD   = 12   # <|image_pad|>，patch 占位
IMG_ROW   = 27   # <|buffer1|> 当作换行符（定了别的就改这里）
PAD, BOS, EOS = 0, 1, 2
TOK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tokenizers')

class VLMCollator:
    """
    输入: [{"image": Tensor[3,H,W] 或 None, "text": str}, ...]
    输出: input_ids / positions / labels / attention_mask / pixel_values
    序列格式: BOS [图像段] 文本 EOS，loss 只落在文本+EOS 上
    """
    def __init__(self, tokenizer, patch_size=16, max_len=2048, img_row_id=IMG_ROW):
        self.tokenizer = tokenizer
        self.patch_size = patch_size
        self.max_len = max_len
        self.img_row_id = img_row_id   # 设为 None 则不用换行符

    # ------------------------------------------------------------------
    def _text_span(self, text, cursor, learn):
        """文本段：行=列=cursor 递增。learn=True 时计入 loss。"""
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        pos = [(cursor + k, cursor + k) for k in range(len(ids))]
        labels = ids[:] if learn else [-100] * len(ids)
        return ids, pos, labels, cursor + len(ids)

    def _image_span(self, img, cursor):
        """图像段：R×C 个占位坑 + 每行换行 + 边界。全部不算 loss。返回 (ids, pos, labels, 新cursor, 坑数)"""
        C = img.shape[-1] // self.patch_size
        R = img.shape[-2] // self.patch_size
        ids, pos, labels = [IMG_START], [(cursor, cursor)], [-100]
        base = cursor + 1                        # patch 坐标起点
        for i in range(R):
            for j in range(C):
                ids.append(IMG_PAD)
                pos.append((base + i, base + j)) # patch(i,j) → 行+ i, 列+ j
                labels.append(-100)
            if self.img_row_id is not None:
                ids.append(self.img_row_id)
                pos.append((base + i, base + C)) # 换行：行不变，列出界一格
                labels.append(-100)
        ids.append(IMG_END)
        pos.append((base + R, base + R))
        labels.append(-100)
        return ids, pos, labels, base + R + 1, R * C

    # ------------------------------------------------------------------
    def encode_sample(self, image, text):
        """拼一条样本：BOS [图像段] caption EOS"""
        ids, pos, labels = [BOS], [(0, 0)], [-100]
        cursor, pixel_values = 1, []
        if image is not None:
            i_ids, i_pos, i_labels, cursor, _ = self._image_span(image, cursor)
            ids += i_ids; pos += i_pos; labels += i_labels
            pixel_values.append(image)
        t_ids, t_pos, t_labels, cursor = self._text_span(text, cursor, learn=True)
        ids += t_ids; pos += t_pos; labels += t_labels
        ids.append(EOS); pos.append((cursor, cursor)); labels.append(EOS)

        ids, labels = ids[:self.max_len], labels[:self.max_len]
        pos = pos[:self.max_len]
        return ids, pos, labels, pixel_values

    # ------------------------------------------------------------------
    def __call__(self, features):
        batch = [self.encode_sample(f.get("image"), f["text"]) for f in features]
        max_len = max(len(b[0]) for b in batch)

        input_ids, positions, labels, attn_mask = [], [], [], []
        pixel_values = []
        for ids, pos, lab, imgs in batch:
            pad_n = max_len - len(ids)
            input_ids.append(ids + [PAD] * pad_n)
            positions.append(pos + [(0, 0)] * pad_n)      # padding 位置无所谓，被 mask 挡掉
            labels.append(lab + [-100] * pad_n)
            attn_mask.append([1] * len(ids) + [0] * pad_n)
            pixel_values.extend(imgs)                      # 按样本顺序平铺，与坑位顺序一致

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "positions": torch.tensor(positions, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "pixel_values": pixel_values or None,
        }


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_PATH)
    col = VLMCollator(tok, patch_size=16)

    img = torch.rand(3, 64, 64)          # 4×4=16 patches
    out = col([{"image": img, "text": "一只狗在草地上跑"},
               {"image": None, "text": "今天天气很好"}])   # 纯文本也走同一入口

    ids, pos, lab = out["input_ids"][0], out["positions"][0], out["labels"][0]
    print("坑位数:", (ids == IMG_PAD).sum().item())        # 应=16
    print("换行数:", (ids == IMG_ROW).sum().item())        # 应=4
    for t, p, l in zip(ids.tolist(), pos.tolist(), lab.tolist()):
        print(f"id={t:>5}  pos=({p[0]:>2},{p[1]:>2})  label={l}")
    parser = argparse.ArgumentParser(description="MiniRain Pretraining")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument('--hidden_size', default=1024, type=int, help="隐藏层维度")
    parser.add_argument('--model_type', default='v', type=str, help="模型类别")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--use_loop', default=0, type=int, choices=[0, 1], help="是否使用LOOP架构（0=否，1=是）")
    parser.add_argument('--use_bounce', default=0, type=int, choices=[0, 1], help="是否使用Bounce架构（0=否，1=是）")
    parser.add_argument('--loop_start', type=int, default=0, help="Loop起始层")
    parser.add_argument('--loop_end', type=int, default=0, help="Loop结束层")
    parser.add_argument('--max_loop_iter', type=int, default=1, help="Loop最大迭代次数")
    parser.add_argument("--use_block_attn_res", action="store_true", help="是否使用块注意力残差")
    parser.add_argument("--use_full_attn_res", action="store_true", help="是否使用全注意力残差")
    parser.add_argument("--block_size", default=2, type=int, help="块大小")

    args = parser.parse_args()
    
    config = RainConfig(hidden_size=args.hidden_size, n_hidden_layers=args.num_hidden_layers,
                        use_moe=bool(args.use_moe), use_loop=bool(args.use_loop),
                        use_bounce=bool(args.use_bounce), loop_start=args.loop_start,
                        loop_end=args.loop_end, max_loop_iter=args.max_loop_iter,
                        block_size=args.block_size,
                        model_type=args.model_type,
                        block_attn_res=bool(args.use_block_attn_res),
                        full_attn_res=bool(args.use_full_attn_res))
    model = MiniRainVForCausalLM(config).cuda().bfloat16()
    batch = col([{"image": torch.rand(3, 64, 64), "text": "一只狗在草地上跑"},
                 {"image": torch.rand(3, 80, 48), "text": "测试"}])  # 不同尺寸，顺便测广播
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    batch["pixel_values"] = [x.cuda() for x in batch["pixel_values"]]
    out = model(**batch)
    out.loss.backward()
    print(out.loss.item())