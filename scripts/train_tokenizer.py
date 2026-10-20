# train_tokenizer.py
"""
训练 MiniRain BPE tokenizer，vocab 16384
特殊 token 0~35 与原 tokenizer 完全对齐，BPE 词表从 36 开始填充

前置：pip install tokenizers
运行：python train_tokenizer.py
产物：minirain_tokenizer.json
"""
import json
import random
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

VOCAB_SIZE = 16384

# ===================== 特殊 token（顺序即 id，勿改） =====================
SPECIALS = [
    "<|endoftext|>",          # 0  PAD
    "<|im_start|>",            # 1  BOS
    "<|im_end|>",              # 2  EOS
    "<|object_ref_start|>",    # 3
    "<|object_ref_end|>",      # 4
    "<|box_start|>",           # 5
    "<|box_end|>",             # 6
    "<|quad_start|>",          # 7
    "<|quad_end|>",            # 8
    "<|vision_start|>",        # 9  img_start
    "<|vision_end|>",          # 10 img_end
    "<|vision_pad|>",          # 11
    "<|image_pad|>",           # 12 img_patch
    "<|video_pad|>",           # 13
    "<|audio_start|>",         # 14
    "<|audio_end|>",           # 15
    "<|audio_pad|>",           # 16
    "<tts_pad>",               # 17
    "<tts_text_bos>",          # 18
    "<tts_text_eod>",          # 19
    "<tts_text_bos_single>",   # 20
    "<tool_call>",             # 21
    "</tool_call>",            # 22
    "<tool_response>",         # 23
    "</tool_response>",        # 24
    "<think>",                 # 25
    "</think>",                # 26
    "<|buffer1|>",             # 27 img_row
    "<|buffer2|>",             # 28
    "<|buffer3|>",             # 29
    "<|buffer4|>",             # 30
    "<|buffer5|>",             # 31
    "<|buffer6|>",             # 32
    "<|buffer7|>",             # 33
    "<|buffer8|>",             # 34
    "<|buffer9|>",             # 35
]
assert len(SPECIALS) == 36
assert SPECIALS[0] == "<|endoftext|>"
assert SPECIALS[1] == "<|im_start|>" and SPECIALS[2] == "<|im_end|>"
assert SPECIALS[9] == "<|vision_start|>" and SPECIALS[10] == "<|vision_end|>"
assert SPECIALS[12] == "<|image_pad|>" and SPECIALS[27] == "<|buffer1|>"

# ===================== 采样配比（tokenizer 训练用 ~3GB 即可） =====================
SAMPLING = {
    # glob 模式: 目标采样 GB
    "texts/zh_CCI4.0*.jsonl":               1.2,
    "texts/zh_wikipedia*.jsonl":            0.3,
    "texts/zh_*Fineweb-Edu-Chinese*.jsonl": 0.1,
    "texts/en_fineweb-edu*.jsonl":          0.5,
    "texts/en_fineweb_sample*.jsonl":       0.1,
    "texts/code_python*.jsonl":             0.6,
    "texts/math_NuminaMath*.jsonl":         0.2,
    # 图文 caption 收集完后取消注释：
    # "muge.jsonl":    0.10,
    # "cococn.jsonl":  0.05,
    # "flickr.jsonl":  0.05,
}


def iter_sampled():
    """按配比从各 jsonl 随机采样文本行。"""
    for pattern, target_gb in SAMPLING.items():
        files = sorted(Path(".").glob(pattern))
        if not files:
            print(f"[跳过] 无文件匹配 {pattern}")
            continue
        target = target_gb * 1e9
        got = 0
        for fp in files:
            if got >= target:
                break
            # 按 目标/文件大小 估算采样概率（文件通常远大于目标）
            p = min(1.0, target * 1.3 / max(fp.stat().st_size, 1))
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    if got >= target:
                        break
                    if random.random() > p:
                        continue
                    try:
                        text = json.loads(line)["text"]
                    except (json.JSONDecodeError, KeyError):
                        continue
                    if len(text) < 20:
                        continue
                    got += len(text.encode("utf-8"))
                    yield text
        print(f"  {pattern}: 采样 {got/1e9:.2f} GB")


def main():
    random.seed(42)

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIALS,   # 占据 id 0~35，BPE merge 从 36 开始填
        show_progress=True,
    )

    print("开始采样 + 训练 BPE（约 20~40 分钟）...")
    tok.train_from_iterator(iter_sampled(), trainer)

    out = "minirain_tokenizer.json"
    tok.save(out)
    print(f"\n已保存: {out}")
    print(f"词表大小: {tok.get_vocab_size()}")

    # ---- 关键 id 校验（必须与 config 一致）----
    print("\n关键 id:")
    checks = {
        "<|endoftext|>": 0,
        "<|im_start|>": 1,
        "<|im_end|>": 2,
        "<|vision_start|>": 9,
        "<|vision_end|>": 10,
        "<|image_pad|>": 12,
        "<|buffer1|>": 27,
    }
    ok = True
    for t, expect in checks.items():
        actual = tok.token_to_id(t)
        mark = "OK" if actual == expect else "MISMATCH"
        if actual != expect:
            ok = False
        print(f"  {t:25s} id={actual} (期望 {expect}) {mark}")
    if not ok:
        raise RuntimeError("特殊 token id 未对齐，检查 SPECIALS 顺序")

    # ---- 压缩率快测 ----
    print("\n压缩率测试:")
    cases = [
        ("中文", "深度学习是机器学习的一个重要分支，它使用多层神经网络从数据中学习表示。"),
        ("英文", "Deep learning is a subset of machine learning that uses neural networks."),
        ("代码", "def forward(self, x):\n    return self.linear(x)  # comment"),
        ("数学", "设 $f(x) = x^2 + 2x + 1$，求 $f'(x)$ 在 $x=0$ 处的值。"),
        ("混合", "使用 transformer 模型处理 NLP 任务，learning rate 设为 2e-4。"),
    ]
    for label, s in cases:
        ids = tok.encode(s).ids
        print(f"  [{label}] {len(ids)/len(s):.2f} token/字符 "
              f"({len(s)} 字符 -> {len(ids)} token)")

    # ---- round-trip 测试 ----
    s = ("<|im_start|>图像序列 <|vision_start|><|image_pad|>"
         "<|buffer1|><|vision_end|> 测试123<|im_end|>")
    enc = tok.encode(s)
    dec = tok.decode(enc.ids)
    print(f"\nround-trip: {'OK' if dec == s else 'FAIL'}")
    print(f"  原: {s}")
    print(f"  回: {dec}")


if __name__ == "__main__":
    main()