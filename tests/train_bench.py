# -*- coding: utf-8 -*-
"""MiniRainVLM 训练步基准（从 perf_test.py 抽出的独立版）。

只测训练相关项：文本 fwd+bwd、图像+文本 fwd+bwd、MuonWithAuxAdam 显存探针。
用途：训练前锁 ms/step / 峰值，或对比编译模式（默认 vs max-autotune）。

用法:
    python tests/train_bench.py                              # 默认 B=5 S=4096 compile
    python tests/train_bench.py --no-compile                 # eager 对照
    python tests/train_bench.py --mode max-autotune          # autotune 对拍
    python tests/train_bench.py --train-batch 6 --train-seq 4096
"""
import argparse
import gc
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architectures.config import RainConfig
from architectures.rain_v import MiniRainVForCausalLM

try:
    # 纯 PyPI 包 muon-optimizer（Keller Jordan Muon）。单卡用 SingleDevice 变体，
    # 分布式 MuonWithAuxAdam 会调 dist.get_world_size()，未初始化 distributed 会崩。
    from muon import SingleDeviceMuonWithAuxAdam as _MuonCls
except ImportError:
    _MuonCls = None

IMG_START, IMG_END, IMG_PATCH, IMG_ROW, BOS = 9, 10, 12, 27, 1
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# 与 arch_test/model_size.py 的默认参数完全一致
BASE_CFG = dict(
    hidden_size=1024,
    n_hidden_layers=20,
    use_moe=False,
    use_mot=True,
    use_bounce=False,
    loop_start=0,
    loop_end=0,
    max_loop_iter=1,
    vision_ffn_scale=4,
    block_size=4,
    model_type="v",
    block_attn_res=False,
    full_attn_res=False,
)


# ---------------------------------------------------------------- 基础设施
class Tee:
    """stdout 同时写控制台和 txt 文件（行缓冲，崩了也不丢已写内容）。"""

    def __init__(self, path, echo=True):
        self.file = open(path, "w", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout
        self.echo = echo  # False = 只写文件，控制台完全静默

    def write(self, s):
        if self.echo:
            self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        if self.echo:
            self.stdout.flush()
        self.file.flush()


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def bench(fn, iters, warmup, device):
    """返回平均耗时（ms）。compile 模式下 warmup 会触发编译，耐心等待。"""
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / iters * 1000.0


def peak_mb(fn, device):
    """执行 fn 时的峰值显存增量（MB），CPU 上返回 nan。"""
    if not device.startswith("cuda"):
        fn()
        return float("nan")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2 ** 20


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_model(args):
    kw = dict(BASE_CFG)
    kw.update(hidden_size=args.hidden, n_hidden_layers=args.layers,
              block_size=args.block_size, vision_ffn_scale=args.vision_ffn_scale)
    cfg = RainConfig(**kw)
    model = MiniRainVForCausalLM(cfg).to(args.device).to(DTYPES[args.dtype])
    if args.compile:
        mode = None if args.mode == "default" else args.mode
        print(f"  [compile] torch.compile(mode={mode}) 中（首次调用会花几分钟编译）...")
        model = torch.compile(model, mode=mode)
    return model, cfg


# ---------------------------------------------------------------- 造数据
def text_batch(args, batch, seq_len, pad_frac=0.0):
    ids = torch.randint(100, 16000, (batch, seq_len), device=args.device)
    am = torch.ones(batch, seq_len, dtype=torch.long, device=args.device)
    if pad_frac > 0:
        n = max(1, int(seq_len * pad_frac))
        am[: max(1, batch // 2), seq_len - n:] = 0  # 右 padding
    return ids, am


def image_batch(args, batch, text_len):
    """按 collator 的位置规划造一条 [BOS 图像段 文本] 样本。"""
    patch = 16  # ConvStem 固定 16× 降采样
    r = c = args.img_size // patch
    ids_all, pos_all, pixel_values = [], [], []
    for _ in range(batch):
        row_ids, row_pos = [BOS], [(0, 0)]
        cur = 1
        row_ids.append(IMG_START)
        row_pos.append((cur, cur))
        base = cur + 1
        for i in range(r):
            for j in range(c):
                row_ids.append(IMG_PATCH)
                row_pos.append((base + i, base + j))
            row_ids.append(IMG_ROW)
            row_pos.append((base + i, base + c))
        row_ids.append(IMG_END)
        row_pos.append((base + r, base + r))
        cur = base + max(r, c) + 1
        txt = torch.randint(100, 16000, (text_len,), device=args.device).tolist()
        row_ids += txt
        row_pos += [(cur + k, cur + k) for k in range(text_len)]
        ids_all.append(row_ids)
        pos_all.append(row_pos)
        pixel_values.append(torch.randn(3, args.img_size, args.img_size, device=args.device))

    ids = torch.tensor(ids_all, device=args.device)
    pos = torch.tensor(pos_all, device=args.device)
    am = torch.ones_like(ids)
    labels = ids.clone()
    img_ids = torch.tensor([IMG_START, IMG_END, IMG_PATCH, IMG_ROW], device=args.device)
    labels[torch.isin(ids, img_ids)] = -100
    return ids, pos, am, labels, pixel_values


# ---------------------------------------------------------------- 优化器
def _build_muon_optimizer(model, cfg):
    """按 muon-optimizer 包的要求手动分 param_groups（use_muon 标志）。"""
    hidden_matrix_params = [p for n, p in model.named_parameters()
                            if p.ndim >= 2 and "embed" not in n and "lm_head" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for n, p in model.named_parameters()
                     if p.ndim < 2 and "embed" not in n and "lm_head" not in n]
    adam_groups = [
        dict(params=embed_params, lr=2e-3),
        dict(params=scalar_params, lr=2e-3),
    ]
    adam_groups = [dict(**g, betas=(0.9, 0.95), eps=1e-10,
                        weight_decay=0.0, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=0.02, momentum=0.95,
                      weight_decay=0.01, use_muon=True)
    groups = [*adam_groups, muon_group]
    print(f"[muon] muon组={len(hidden_matrix_params)} embed组={len(embed_params)} "
          f"scalar组={len(scalar_params)} 总参数张量={len(list(model.parameters()))}")

    # 校验：组间无重复、且覆盖全部参数
    seen, all_ids = set(), {id(p) for p in model.parameters()}
    for g in groups:
        for p in g["params"]:
            if id(p) in seen:
                names = [n for n, q in model.named_parameters() if q is p]
                raise ValueError(f"重复参数: {names}")
            seen.add(id(p))
    missing = all_ids - seen
    if missing:
        names = [n for n, p in model.named_parameters() if id(p) in missing]
        print(f"[muon] 注意: {len(missing)} 个参数未分组: {names}")

    return _MuonCls(groups)


# ---------------------------------------------------------------- 训练段
def section_train(args, model, total_mb):
    print("\n" + "=" * 78)
    print(f"训练一步（前向+反向，MoT）  B={args.train_batch}  S={args.train_seq}  "
          f"dtype={args.dtype}  compile={args.compile} mode={args.mode}")
    print("=" * 78)

    def train_step(ids, am, labels, positions=None, pixel_values=None):
        model.zero_grad(set_to_none=True)
        out = model(ids, attention_mask=am, labels=labels,
                    positions=positions, pixel_values=pixel_values)
        out.loss.backward()
        return out.loss

    model.train()
    ids, am = text_batch(args, args.train_batch, args.train_seq, pad_frac=0.0)
    labels = ids.clone()
    labels[am == 0] = -100
    run = lambda: train_step(ids, am, labels)
    ms = bench(run, max(3, args.iters // 2), args.warmup, args.device)
    mem = peak_mb(run, args.device)
    print(f"  {'文本(0% pad)':<20}{ms:>9.2f} ms/step   峰值 +{mem:>7.1f} MB")
    if total_mb and mem > 0.85 * total_mb:
        print(f"      ! 峰值已超显存({total_mb:.0f}MB)的 85%，耗时可能失真")

    model.train()
    i_ids, i_pos, i_am, i_lab, pix = image_batch(args, 1, args.text_len)
    run = lambda: train_step(i_ids, i_am, i_lab, positions=i_pos, pixel_values=pix)
    ms_img = bench(run, max(3, args.iters // 2), args.warmup, args.device)
    mem_img = peak_mb(run, args.device)
    print(f"  {'图像+文本':<20}{ms_img:>9.2f} ms/step   峰值 +{mem_img:>7.1f} MB")

    # 优化器显存：上面的峰值只含 权重+激活+梯度，训练真实占用还要加优化器状态
    if _MuonCls is None:
        print("  [muon] 未安装 muon-optimizer 包，跳过优化器显存测试"
              "（pip install muon-optimizer）")
    else:
        model.train()
        ids, am = text_batch(args, args.train_batch, args.train_seq, pad_frac=0.0)
        labels = ids.clone()
        labels[am == 0] = -100
        opt = _build_muon_optimizer(model, None)
        run = lambda: train_step(ids, am, labels)
        bench(run, 2, 1, args.device)          # 热身：权重/编译就绪，梯度管道打通
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        run()                                   # fwd+bwd（文本梯度在）
        mem_opt = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
        # 图像步：只前向+反向，不清零（train_step 会 zero_grad），让 vision 参数也有梯度
        out = model(i_ids, attention_mask=i_am, labels=i_lab,
                    positions=i_pos, pixel_values=pix)
        out.loss.backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        print(f"[muon] 无梯度参数 {len(missing)} 个: {missing[:5]}")
        opt.step()                              # 优化器状态 + step 临时缓冲
        torch.cuda.synchronize()
        mem_total = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
        resident = torch.cuda.memory_allocated() / 2 ** 20
        print(f"  MuonWithAuxAdam  训练步峰值 +{mem_opt:>7.1f} MB"
              f"   +optimizer.step 峰值 +{mem_total:>7.1f} MB"
              f"   全进程常驻 {resident:>7.0f} MB")
        if total_mb and resident > 0.80 * total_mb:
            print(f"      ! 常驻已超显存({total_mb:.0f}MB)的 80%，长跑有 OOM 风险")

    # 训练口径汇总（ETA 估算用）
    tok_per_micro = args.train_batch * args.train_seq
    print(f"\n  [汇总] {tok_per_micro} tok/micro  吞吐 {tok_per_micro / ms * 1000:>6.0f} tok/s  "
          f"峰值 {mem:>7.0f} MB")


# ---------------------------------------------------------------- 入口
def main():
    p = argparse.ArgumentParser(description="MiniRainVLM 训练步基准（独立版）")
    p.add_argument("--device", default="cuda:5" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--train-batch", type=int, default=5)
    p.add_argument("--train-seq", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=BASE_CFG["hidden_size"])
    p.add_argument("--layers", type=int, default=BASE_CFG["n_hidden_layers"])
    p.add_argument("--block-size", type=int, default=BASE_CFG["block_size"])
    p.add_argument("--vision-ffn-scale", type=int, default=BASE_CFG["vision_ffn_scale"])
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--text-len", type=int, default=256)
    p.add_argument("--compile", action="store_true", default=True,
                   help="torch.compile（默认开启，--no-compile 关闭）")
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--mode", choices=["default", "max-autotune"], default="default",
                   help="compile 模式：max-autotune 会实测选优 GEMM，编译时间长 5-10 倍")
    p.add_argument("--quiet", action="store_true", help="控制台静默，输出只写 results/*.txt")
    args = p.parse_args()
    torch._dynamo.config.recompile_limit = 10000

    # stdout 同时写入 results/train_bench_<时间戳>.txt
    os.makedirs("results", exist_ok=True)
    log_path = os.path.join("results", f"train_bench_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    sys.stdout = Tee(log_path, echo=not args.quiet)
    print(f"[log] 输出已写入 {log_path}")

    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    if args.dtype != "fp32" and not args.device.startswith("cuda"):
        print("[warn] CPU 上 bf16/fp16 很慢，建议 --dtype fp32")
    if args.compile and args.warmup < 3:
        args.warmup = 3
    torch.manual_seed(0)

    dev_idx = torch.device(args.device).index or 0
    dev_name = torch.cuda.get_device_name(dev_idx) if args.device.startswith("cuda") else "CPU"
    print("=" * 78)
    print(f"MiniRainVLM 训练步基准 | {dev_name} | torch {torch.__version__} | {args.dtype}")
    print(f"训练={args.train_batch}x{args.train_seq}  compile={args.compile} "
          f"mode={args.mode}  iters={args.iters}")
    print("=" * 78)

    model, cfg = build_model(args)
    total_mb = (torch.cuda.get_device_properties(dev_idx).total_memory / 2 ** 20
                if args.device.startswith("cuda") else 0.0)

    section_train(args, model, total_mb)
    free()
    print("\n完成。")


if __name__ == "__main__":
    main()