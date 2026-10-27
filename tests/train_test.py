
import argparse
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _device import add_device_args, list_gpus, resolve_device
from architectures.config import RainConfig
from architectures.rain_v import MiniRainVForCausalLM
from perf_test import _build_muon_optimizer, bench, image_batch, peak_mb, text_batch

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
# 可被 --sweep 扫描的参数及其类型
SWEEPABLE = {
    "lr": float,
    "batch": int,
    "seq": int,
    "text_len": int,
    "img_size": int,
    "block_size": int,
    "weight_decay": float,
    "muon_lr": float,
    "dropout": float,
}


def build_model(args, device):
    cfg = RainConfig(
        hidden_size=args.hidden,
        n_hidden_layers=args.layers,
        use_mot=True,
        use_moe=False,
        vision_ffn_scale=args.vision_ffn_scale,
        block_size=args.block_size,
        block_attn_res=args.attn_res == "block",
        full_attn_res=args.attn_res == "full",
        dropout=args.dropout,
        model_type="v",
    )
    model = MiniRainVForCausalLM(cfg).to(device).to(DTYPES[args.dtype])
    return model, cfg


# ---------------------------------------------------------------- 训练测试
def run_once(args, device):
    """perf_test 训练段的同一套测试：文本 / 图像 ms·峰值，加优化器显存。"""
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    total_mb = (torch.cuda.get_device_properties(0).total_memory / 2 ** 20
                if device.startswith("cuda") else 0)
    model, _ = build_model(args, device)

    def train_step(ids, am, labels, positions=None, pixel_values=None):
        model.zero_grad(set_to_none=True)
        out = model(ids, attention_mask=am, labels=labels,
                    positions=positions, pixel_values=pixel_values)
        out.loss.backward()
        return out.loss

    model.train()
    ids, am = text_batch(args, args.batch, args.seq)
    labels = ids.clone()
    labels[am == 0] = -100
    run = lambda: train_step(ids, am, labels)
    text_ms = bench(run, max(3, args.iters // 2), args.warmup, device)
    text_mem = peak_mb(run, device)
    print(f"  {'文本 step':<18}{text_ms:>9.2f} ms  (B={args.batch} S={args.seq})"
          f"   峰值 +{text_mem:>7.1f} MB")

    model.train()
    i_ids, i_pos, i_am, i_lab, pix = image_batch(args, 1, args.text_len)
    run = lambda: train_step(i_ids, i_am, i_lab, positions=i_pos, pixel_values=pix)
    img_ms = bench(run, max(3, args.iters // 2), args.warmup, device)
    img_mem = peak_mb(run, device)
    print(f"  {'图像+文本 step':<18}{img_ms:>9.2f} ms  (S={i_ids.shape[1]})"
          f"   峰值 +{img_mem:>7.1f} MB")
    for name, mem in (("文本", text_mem), ("图像", img_mem)):
        if total_mb and mem > 0.85 * total_mb:
            print(f"      ! {name}峰值已超显存({total_mb:.0f}MB)的 85%，"
                  f"可能触发系统内存回退导致耗时失真")

    # 优化器状态：上面的峰值只含 权重+激活+梯度，训练真实占用还要加优化器状态
    model.train()
    opt = _build_muon_optimizer(model, None)
    for group in opt.param_groups:
        group["lr"] = args.muon_lr if group["use_muon"] else args.lr
    bench(run, 2, 1, device)              # 热身：梯度管道打通
    free_cuda()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    run()                                  # fwd+bwd（文本梯度）
    out = model(i_ids, attention_mask=i_am, labels=i_lab,
                positions=i_pos, pixel_values=pix)
    out.loss.backward()                    # 图像步补齐 vision 参数梯度
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    print(f"[muon] 无梯度参数 {len(missing)} 个: {missing[:5]}")
    opt.step()
    torch.cuda.synchronize()
    opt_mem = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
    print(f"  {'优化器 step':<18}{'':>9}      优化器状态峰值 +{opt_mem:>7.1f} MB")

    result = {
        "text_ms": text_ms, "text_mem": text_mem,
        "img_ms": img_ms, "img_mem": img_mem,
        "opt_mem": opt_mem,
        "params": sum(p.numel() for p in model.parameters()) / 1e6,
    }
    del model, opt
    free_cuda()
    return result


def free_cuda():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- 扫描
def expand_sweeps(base, sweeps):
    """把 --sweep k=v1,v2 展开成参数组合列表（笛卡尔积）。"""
    if not sweeps:
        return [dict(base)]
    keys = sorted(sweeps)
    for k in keys:
        if k not in SWEEPABLE:
            raise SystemExit(f"--sweep 不支持 {k}，可选: {', '.join(SWEEPABLE)}")
        if not sweeps[k]:
            raise SystemExit(f"--sweep {k} 没有给值")
    values = [[SWEEPABLE[k](v) for v in sweeps[k]] for k in keys]
    combos = []
    for combo in itertools.product(*values):
        over = dict(base)
        over.update(dict(zip(keys, combo)))
        combos.append(over)
    return combos


def fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def main():
    p = argparse.ArgumentParser(description="MiniRainVLM 训练超参数探索")
    add_device_args(p)
    # 模型
    p.add_argument("--hidden", type=int, default=1024, help="hidden_size")
    p.add_argument("--layers", type=int, default=20, help="层数")
    p.add_argument("--vision-ffn-scale", type=float, default=4)
    p.add_argument("--block-size", type=int, default=4, help="块注意力残差的块大小（子层数，需为偶数）")
    p.add_argument("--attn-res", choices=["none", "block", "full"], default="none",
                   help="残差聚合方式，实际是否启用取决于配置")
    p.add_argument("--dropout", type=float, default=0.0)
    # 训练
    p.add_argument("--batch", type=int, default=1, help="文本 step 的 batch")
    p.add_argument("--seq", type=int, default=1024, help="文本 step 的序列长度")
    p.add_argument("--text-len", type=int, default=256, help="图像样本里的文本段长度")
    p.add_argument("--img-size", type=int, default=64, help="图像边长，需被 16 整除")
    p.add_argument("--iters", type=int, default=10, help="每项测几次")
    p.add_argument("--lr", type=float, default=2e-3, help="AdamW 组学习率（embed / lm_head / norm）")
    p.add_argument("--muon-lr", type=float, default=0.02, help="Muon 组学习率（隐藏矩阵）")
    p.add_argument("--weight-decay", type=float, default=0.01, help="Muon 组 weight decay（AdamW 组为 0）")
    p.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=3, help="计时前热身步数")
    p.add_argument("--sweep", action="append", default=[], metavar="K=V1,V2",
                   help=f"扫描参数（可重复），可选: {', '.join(SWEEPABLE)}")
    args = p.parse_args()

    if args.list_gpus:
        list_gpus()
        raise SystemExit(0)
    device, _ = resolve_device(args.device)
    if args.attn_res == "full" and args.block_size:
        raise SystemExit("--attn-res full 与 --block-size 不能同时启用")
    if args.attn_res == "block" and (args.block_size < 2 or args.block_size % 2):
        raise SystemExit(f"--block-size 必须为偶数且 ≥2（当前 {args.block_size}）")

    sweeps = {}
    for item in args.sweep:
        if "=" not in item:
            raise SystemExit(f"--sweep 需要 K=V1,V2 形式，收到 {item!r}")
        k, v = item.split("=", 1)
        sweeps[k.strip()] = [x.strip() for x in v.split(",") if x.strip()]

    base = vars(args).copy()
    combos = expand_sweeps(base, sweeps)
    keys = sorted(sweeps)
    print(f"设备 {device} | dtype {args.dtype} | 模型 {args.hidden}x{args.layers} | "
          f"文本 {args.batch}x{args.seq} | 图像 {args.img_size}px + 文本 {args.text_len}")
    print(f"共 {len(combos)} 组配置" + (f"，扫描 {keys}" if keys else ""))

    results = []
    for i, over in enumerate(combos, 1):
        cfg = argparse.Namespace(**over)
        tag = "  ".join(f"{k}={fmt(over[k])}" for k in keys) if keys else "基准配置"
        print("\n" + "=" * 78)
        print(f"[{i}/{len(combos)}] {tag}")
        print("=" * 78)
        res = run_once(cfg, device)
        results.append((tag, over, res))
        print(f"  结果 | 文本 {res['text_ms']:.1f} ms | 图像 {res['img_ms']:.1f} ms | "
              f"优化器 {res['opt_mem']:.0f} MB | 参数 {res['params']:.1f}M")

    # ---------------------------------------------------------------- 汇总
    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    if keys:
        head = "".join(f"{k:>14}" for k in keys)
        print(f"{'#':>3}{head}{'文本ms':>9}{'图像ms':>9}{'优化器MB':>10}{'参数M':>8}")
        for i, (_, over, r) in enumerate(results, 1):
            vals = "".join(f"{fmt(over[k]):>14}" for k in keys)
            print(f"{i:>3}{vals}{r['text_ms']:>9.0f}{r['img_ms']:>9.0f}"
                  f"{r['opt_mem']:>10.0f}{r['params']:>8.1f}")
    else:
        r = results[0][2]
        print(f"  文本 {r['text_ms']:.1f} ms | 图像 {r['img_ms']:.1f} ms | "
              f"优化器 {r['opt_mem']:.0f} MB | 参数 {r['params']:.1f}M")
    print("\n完成。")


if __name__ == "__main__":
    main()
