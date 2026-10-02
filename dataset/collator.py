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

    位置编号规则（2D RoPE）:
      - 文本:      行=列=cursor，逐个递增
      - patch(i,j): 行=base+i, 列=base+j（base=图像起始行）
      - 换行:      行=base+i, 列=base+C（该行末尾，列出界一格）
      - 图后文本:  从 base+max(R,C)+1 继续（max 防止宽图列坐标与文本碰撞）
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
        """图像段：R×C 个占位坑 + 每行换行 + 边界。全部不算 loss。
        返回 (ids, pos, labels, 新cursor, 坑数)。
        新cursor按 max(R,C) 前进：宽图(C>R)时列坐标占到了 base+C，
        若只按 R 前进，后续文本的 (t,t) 会撞进图像的列坐标区间。"""
        C = img.shape[-1] // self.patch_size
        R = img.shape[-2] // self.patch_size
        ids, pos, labels = [IMG_START], [(cursor, cursor)], [-100]
        base = cursor + 1                        # patch 坐标起点
        for i in range(R):
            for j in range(C):
                ids.append(IMG_PAD)
                pos.append((base + i, base + j)) # patch(i,j) → 行+i, 列+j
                labels.append(-100)
            if self.img_row_id is not None:
                ids.append(self.img_row_id)
                pos.append((base + i, base + C)) # 换行：行不变，列出界一格
                labels.append(-100)
        ids.append(IMG_END)
        pos.append((base + R, base + R))
        labels.append(-100)
        return ids, pos, labels, base + max(R, C) + 1, R * C

    # ------------------------------------------------------------------
    def encode_sample(self, image, text):
        """拼一条样本：BOS [图像段] 文本 EOS"""
        ids, pos, labels = [BOS], [(0, 0)], [-100]
        cursor, pixel_values = 1, []
        img_len = 0
        if image is not None:
            i_ids, i_pos, i_labels, cursor, _ = self._image_span(image, cursor)
            ids += i_ids; pos += i_pos; labels += i_labels
            pixel_values.append(image)
            img_len = len(i_ids)
        t_ids, t_pos, t_labels, cursor = self._text_span(text, cursor, learn=True)
        ids += t_ids; pos += t_pos; labels += t_labels
        ids.append(EOS); pos.append((cursor, cursor)); labels.append(EOS)

        # 截断保护：图像段完整性优先，超长只砍文本尾部，绝不在图像段中间下刀
        if len(ids) > self.max_len:
            head_len = 1 + img_len                          # BOS + 图像段
            assert head_len + 1 < self.max_len, \
                f"图像段({head_len})+EOS 已超出 max_len={self.max_len}，图太大或 max_len 太小"
            keep_text = self.max_len - head_len - 1         # 留给文本的额度
            ids    = ids[:head_len] + ids[head_len:head_len + keep_text] + [EOS]
            pos    = pos[:head_len] + pos[head_len:head_len + keep_text] + [pos[-1]]
            labels = labels[:head_len] + labels[head_len:head_len + keep_text] + [EOS]
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

    # 方图 + 宽图 + 纯文本，三种形态一次测完
    out = col([
        {"image": torch.rand(3, 64, 64), "text": "一只狗在草地上跑"},  # R=4, C=4
        {"image": torch.rand(3, 48, 96), "text": "宽图测试"},  # R=3, C=6 验证max(R,C)
        {"image": None, "text": "今天天气很好"},  # 纯文本
    ])
    for b in range(3):
        print(f"--- 样本{b} ---")
        ids, pos, lab = out["input_ids"][b], out["positions"][b], out["labels"][b]
        print("坑位数:", (ids == IMG_PAD).sum().item())
        for t, p, l in zip(ids.tolist(), pos.tolist(), lab.tolist()):
            print(f"id={t:>5}  pos=({p[0]:>2},{p[1]:>2})  label={l}")
        print()
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