# -*- coding: utf-8 -*-
"""
图像分辨率/长宽比分布统计 + 训练桶设计建议
============================================
扫描图文 jsonl，读每张图的真实 (W, H)，输出：
  1. 尺寸分桶统计（短边/长边分布）
  2. 长宽比分布（决定长方形桶怎么设）
  3. 给定候选桶集合后：每张图落入哪个桶、平均 patch 数、视觉 token 占比估算

用法：
  python bucket_stats.py --root /data/xu/xin/MiniRain/data/image_text \
      --jsonl pairs_all.jsonl --per-file 20000
"""
import argparse
import json
import math
import os
from collections import Counter, defaultdict

from PIL import Image


def ar_bucket(w, h):
    """长宽比归类：横/方/竖 + 比值档。"""
    r = max(w, h) / min(w, h)
    orient = "landscape" if w > h else ("portrait" if h > w else "square")
    return orient, round(min(r, 4.0), 1)


def nearest_bucket(w, h, buckets):
    """把 (w,h) 分配到面积不小于原图、长宽比最接近的桶。"""
    area = w * h
    best, best_cost = None, float("inf")
    for (bw, bh) in buckets:
        if bw * bh < area * 0.55:      # 缩得太狠的桶不考虑
            continue
        # 代价：长宽比失配 + 放大倍数惩罚（轻微）
        cost = abs(math.log((bw / bh) / (w / h))) + 0.15 * max(0, math.log((bw * bh) / area))
        if cost < best_cost:
            best_cost, best = cost, (bw, bh)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="图片根目录")
    ap.add_argument("--jsonl", nargs="+", required=True)
    ap.add_argument("--per-file", type=int, default=20000, help="每个 jsonl 抽样条数")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--imgs-per-seq", type=int, default=2)
    ap.add_argument("--out", default="bucket_full_stats.txt", help="统计结果落盘路径")
    args = ap.parse_args()

    import sys
    from datetime import datetime
    log_fp = open(args.out, "w", encoding="utf-8")
    class _Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            log_fp.write(s)
            log_fp.flush()
        def flush(self):
            sys.__stdout__.flush()
            log_fp.flush()
    sys.stdout = _Tee()
    print(f"# 统计时间 {datetime.now():%Y-%m-%d %H:%M:%S}  root={args.root}")
    print(f"# jsonl={args.jsonl} per-file={args.per_file}")

    stats = []  # (w, h, source)
    for fp in args.jsonl:
        path = fp if os.path.exists(fp) else os.path.join(args.root, fp)
        src = os.path.basename(path)
        n = 0
        with open(path) as f:
            for line in f:
                if n >= args.per_file:
                    break
                try:
                    d = json.loads(line)
                    p = os.path.join(args.root, d["image"])
                    if not os.path.exists(p):
                        continue
                    with Image.open(p) as im:
                        w, h = im.size
                    stats.append((w, h, src))
                    n += 1
                except Exception:
                    continue
        print(f"[scan] {src}: {n} 张")

    N = len(stats)
    print(f"\n===== 总量 {N} 张 =====")

    # ---- 短边/长边分布 ----
    short = Counter(min(w, h) // 64 * 64 for w, h, _ in stats)
    long_ = Counter(max(w, h) // 64 * 64 for w, h, _ in stats)
    print("\n--- 短边分布(像素, 按64取整) ---")
    for k in sorted(short):
        print(f"  {k:5d}px  {short[k]:6d}  {short[k]/N*100:5.1f}%")
    print("--- 长边分布(像素, 按64取整) ---")
    for k in sorted(long_):
        print(f"  {k:5d}px  {long_[k]:6d}  {long_[k]/N*100:5.1f}%")

    # ---- 长宽比分布 ----
    ars = Counter()
    for w, h, _ in stats:
        orient, r = ar_bucket(w, h)
        ars[(orient, r)] += 1
    print("\n--- 长宽比分布 (横/方/竖 : 比值) ---")
    for orient in ("square", "landscape", "portrait"):
        sub = {r: c for (o, r), c in ars.items() if o == orient}
        for r in sorted(sub):
            print(f"  {orient:9s} {r:3.1f}  {sub[r]:6d}  {sub[r]/N*100:5.1f}%")

    # ---- 候选桶评估 ----
    CANDIDATES = [
        # (名字, [(w,h), ...]) — 面积大致分级 + 覆盖横竖
        ("A_纯正方3档", [(256, 256), (384, 384), (512, 512)]),
        ("B_正方+横竖4档", [(256, 256), (384, 384), (512, 384), (384, 512)]),
        ("C_全组合6档", [(256, 256), (384, 384), (512, 512),
                        (512, 384), (384, 512), (512, 256), (256, 512)]),
    ]
    print("\n===== 候选桶评估 =====")
    for name, buckets in CANDIDATES:
        buckets = sorted(buckets, key=lambda b: b[0] * b[1])
        use = Counter()
        patch_total = 0
        upscale_area = []   # 放大倍数（面积比）
        for w, h, _ in stats:
            b = nearest_bucket(w, h, buckets)
            if b is None:
                continue
            use[b] += 1
            patch_total += (b[0] // 16) * (b[1] // 16)
            upscale_area.append((b[0] * b[1]) / (w * h))
        n_used = sum(use.values())
        avg_patch = patch_total / max(1, n_used)
        tok_per_seq = avg_patch * args.imgs_per_seq + 34 * args.imgs_per_seq  # patch + 行符估算
        vision_frac = tok_per_seq / args.seq_len
        # 放大损失：>2倍面积放大的图占比
        big_up = sum(1 for u in upscale_area if u > 2.0) / max(1, n_used)
        print(f"\n[{name}] 桶={buckets}")
        print(f"  平均每图 patch={avg_patch:.0f}  每序列(2图)视觉token≈{tok_per_seq:.0f}"
              f"  视觉占比≈{vision_frac*100:.1f}%")
        print(f"  面积放大>2倍的图占 {big_up*100:.1f}%")
        for b in sorted(use, key=lambda x: -use[x]):
            print(f"    桶 {b[0]}x{b[1]}: {use[b]:6d} ({use[b]/n_used*100:4.1f}%)")


if __name__ == "__main__":
    main()