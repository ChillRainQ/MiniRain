# -*- coding: utf-8 -*-
"""MiniRainVLM 性能基准（只测 arch_test/model_size.py 中的 MoT 配置）

配置（model_size.py 默认值）：
    model_type='v'  hidden_size=1024  n_hidden_layers=20
    use_mot=True  use_moe=False  vision_ffn_scale=4  block_size=4
    block_attn_res=False  full_attn_res=False

整个脚本只构建/编译一次模型，所有 section 共用（含参数、compile cache）。

用法:
    python tests/perf_test.py
    python tests/perf_test.py --iters 20 --warmup 5
    python tests/perf_test.py --train-batch 9 --train-seq 4096 --grad-ckpt --no-gen
    python tests/perf_test.py --compile
    python tests/perf_test.py --dtype fp32 --device cpu --no-gen
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
from components.mlp import MoTFFN
from losses.loss import ChunkedCrossEntropy

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
    if args.heads is not None:
        kw["n_attn_heads"] = args.heads
    if args.kv_heads is not None:
        kw["n_key_value_heads"] = args.kv_heads
    if args.vocab is not None:
        kw["vocab_size"] = args.vocab
    if args.inter is not None:
        kw["intermediate_size"] = args.inter
    cfg = RainConfig(**kw)
    model = MiniRainVForCausalLM(cfg).to(args.device).to(DTYPES[args.dtype])
    if getattr(args, "grad_ckpt", False):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            print("  [ckpt] 梯度检查点已启用 (use_reentrant=False)")
        except (AttributeError, ValueError) as e:
            print(f"  [ckpt] 模型未实现 gradient_checkpointing，已跳过: {e}")
    if getattr(args, "compile", False):
        print("  [compile] torch.compile 中（首次调用会花几分钟编译，之后形状复用缓存）...")
        model = torch.compile(model)
    return model, cfg


def param_report(model, cfg):
    """与 model_size.py 相同的参数拆分。"""
    def count(module):
        return 0 if module is None else sum(p.numel() for p in module.parameters() if p.requires_grad)

    backbone = model.model
    patch_embed = getattr(backbone, "patch_embed", None) or getattr(backbone, "patch_embd", None)
    vision_embed = count(patch_embed)
    vision_ffn = sum(count(layer.mlp.mlp_v) for layer in getattr(backbone, "layers", [])
                     if isinstance(layer.mlp, MoTFFN))
    print(f"  配置 | type={cfg.type} block_size={cfg.block_size} "
          f"vision_ffn_scale={cfg.vision_ffn_scale} intermediate={cfg.intermediate_size} "
          f"heads={cfg.n_attn_heads}/{cfg.n_key_value_heads} vocab={cfg.vocab_size}")
    print(f"  参数 | Vision Embed {vision_embed / 1e6:.2f}M | Vision FFN {vision_ffn / 1e6:.2f}M | "
          f"Vision Total {(vision_embed + vision_ffn) / 1e6:.2f}M | Model {count(model) / 1e6:.2f}M")


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


# ---------------------------------------------------------------- 测试段
def section_forward(args, model):
    print("\n" + "=" * 78)
    print(f"1. 前向吞吐（MoT）  B={args.batch}  S=1024  dtype={args.dtype}")
    print("=" * 78)
    model.eval()
    with torch.no_grad():
        ids, _ = text_batch(args, args.batch, 1024)
        ms = bench(lambda: model(ids), args.iters, args.warmup, args.device)
        print(f"  {'S=1024':<10}{ms:>9.2f} ms{args.batch * 1024 / ms * 1000:>9.0f} tok/s")

        print("  -- 序列长度扫描（compile 下每个新形状会重新编译）--")
        for s in (512, 1024, 2048, 4096):
            ids, _ = text_batch(args, args.batch, s)
            ms = bench(lambda: model(ids), max(3, args.iters // 2), args.warmup, args.device)
            print(f"  {'S=' + str(s):<10}{ms:>9.2f} ms{args.batch * s / ms * 1000:>9.0f} tok/s")


def section_mask(args, model):
    print("\n" + "=" * 78)
    print(f"2. padding / 图像路径（MoT）  B={args.batch}  S=1024  dtype={args.dtype}")
    print("=" * 78)
    model.eval()
    ids, am_ones = text_batch(args, args.batch, 1024)
    _, am_pad = text_batch(args, args.batch, 1024, pad_frac=0.25)
    with torch.no_grad():
        ms = bench(lambda: model(ids), args.iters, args.warmup, args.device)
        print(f"  {'文本 无 mask':<24}{ms:>9.2f} ms")
        ms = bench(lambda: model(ids, attention_mask=am_ones), args.iters, args.warmup, args.device)
        print(f"  {'文本 mask 全 1':<24}{ms:>9.2f} ms")
        ms = bench(lambda: model(ids, attention_mask=am_pad), args.iters, args.warmup, args.device)
        print(f"  {'文本 25% padding':<24}{ms:>9.2f} ms")

        i_ids, i_pos, i_am, _, pix = image_batch(args, 1, args.text_len)
        ms = bench(lambda: model(i_ids, pixel_values=pix, positions=i_pos, attention_mask=i_am),
                   max(3, args.iters // 2), args.warmup, args.device)
        print(f"  {f'图像 {args.img_size}px + 文本{args.text_len}':<24}{ms:>9.2f} ms"
              f"   (S={i_ids.shape[1]})")


def section_train(args, model, total_mb):
    print("\n" + "=" * 78)
    print(f"3. 训练一步（前向+反向，MoT）  B={args.train_batch}  S={args.train_seq}  "
          f"dtype={args.dtype}")
    print("=" * 78)

    def train_step(ids, am, labels, positions=None, pixel_values=None):
        model.zero_grad(set_to_none=True)
        out = model(ids, attention_mask=am, labels=labels,
                    positions=positions, pixel_values=pixel_values)
        out.loss.backward()
        return out.loss

    model.train()
    ids, am = text_batch(args, args.train_batch, args.train_seq, pad_frac=0.25)
    labels = ids.clone()
    labels[am == 0] = -100
    run = lambda: train_step(ids, am, labels)
    ms = bench(run, max(3, args.iters // 2), args.warmup, args.device)
    mem = peak_mb(run, args.device)
    print(f"  {'文本(25% pad)':<20}{ms:>9.2f} ms/step   峰值 +{mem:>7.1f} MB")
    if total_mb and mem > 0.85 * total_mb:
        print(f"      ! 峰值已超显存({total_mb:.0f}MB)的 85%，可能触发系统内存回退导致耗时失真；"
              f"请调小 --train-batch/--train-seq")

    model.train()
    i_ids, i_pos, i_am, i_lab, pix = image_batch(args, 1, args.text_len)
    run = lambda: train_step(i_ids, i_am, i_lab, positions=i_pos, pixel_values=pix)
    ms = bench(run, max(3, args.iters // 2), args.warmup, args.device)
    mem = peak_mb(run, args.device)
    print(f"  {'图像+文本':<20}{ms:>9.2f} ms/step   峰值 +{mem:>7.1f} MB")
    if total_mb and mem > 0.85 * total_mb:
        print(f"      ! 峰值已超显存({total_mb:.0f}MB)的 85%，耗时可能失真")


def section_generate(args, model):
    print("\n" + "=" * 78)
    print(f"4. 生成（MoT）  prompt={args.gen_prompt}  new={args.gen_new}  dtype={args.dtype}")
    print("=" * 78)
    if getattr(args, "compile", False):
        print("  [warn] compile 模式下 decode 每个新长度都可能重编译，本段数字仅供参考")
    model.eval()
    ids = torch.randint(100, 16000, (1, args.gen_prompt), device=args.device)
    am = torch.ones_like(ids)
    for use_cache in (True, False):
        run = lambda: model.generate(input_ids=ids, attention_mask=am, max_new_tokens=args.gen_new,
                                     do_sample=False, use_cache=use_cache, eos_token_id=None)
        ms = bench(run, 2, 1, args.device)
        print(f"  use_cache={str(use_cache):<5}{ms:>9.1f} ms  "
              f"{ms / args.gen_new:>7.2f} ms/token  {args.gen_new / ms * 1000:>8.1f} tok/s")


def section_memory(args, model, cfg):
    print("\n" + "=" * 78)
    print("5. 显存与 CE（MoT）")
    print("=" * 78)
    model.eval()
    with torch.no_grad():
        for s in (1024, 2048):
            ids, _ = text_batch(args, args.batch, s)
            mem = peak_mb(lambda: model(ids), args.device)
            print(f"  {'前向峰值 S=' + str(s):<24}+{mem:>8.1f} MB")

        for s in (2048, 4096):
            fake = torch.randint(100, 16000, (1, s), device=args.device)
            mem = peak_mb(lambda: model.model._build_image_bidirectional_mask(fake, None, torch.bfloat16),
                          args.device)
            resident = s * s * 2 / 2 ** 20  # 单份 [B,1,S,S] bf16
            print(f"  {'图像 mask S=' + str(s):<24}常驻 {resident:>7.1f} MB  构建峰值 +{mem:>7.1f} MB")

    if args.device.startswith("cuda"):
        dtype = DTYPES[args.dtype]
        weight = torch.randn(cfg.vocab_size, cfg.hidden_size, device=args.device, dtype=dtype,
                             requires_grad=True)
        hidden = torch.randn(2048, cfg.hidden_size, device=args.device, dtype=dtype, requires_grad=True)
        target = torch.randint(0, cfg.vocab_size, (2048,), device=args.device)
        target[:200] = -100
        chunked = ChunkedCrossEntropy(4)

        def full_ce():
            weight.grad = None
            hidden.grad = None
            torch.nn.functional.cross_entropy(
                torch.nn.functional.linear(hidden, weight).float(), target, ignore_index=-100).backward()

        def chunk_ce():
            weight.grad = None
            hidden.grad = None
            chunked(weight, hidden, target).backward()

        for name, fn in (("全量 CE", full_ce), ("分块 CE(4)", chunk_ce)):
            ms = bench(fn, max(3, args.iters // 2), 2, args.device)
            mem = peak_mb(fn, args.device)
            print(f"  {name:<24}{ms:>9.2f} ms/step   峰值 +{mem:>7.1f} MB")
        del weight, hidden, target
        free()


# ---------------------------------------------------------------- 入口
def main():
    p = argparse.ArgumentParser(description="MiniRainVLM 性能基准（model_size.py 的 MoT 配置）")
    p.add_argument("--device", default="cuda:5" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--train-batch", type=int, default=1,
                   help="训练段 batch（block_size=4 训练显存约为前向的 6~7 倍，8GB 卡建议 1）")
    p.add_argument("--train-seq", type=int, default=1024, help="训练段文本序列长度")
    p.add_argument("--hidden", type=int, default=BASE_CFG["hidden_size"])
    p.add_argument("--layers", type=int, default=BASE_CFG["n_hidden_layers"])
    p.add_argument("--block-size", type=int, default=BASE_CFG["block_size"])
    p.add_argument("--vision-ffn-scale", type=int, default=BASE_CFG["vision_ffn_scale"])
    p.add_argument("--heads", type=int, default=None, help="默认沿用 RainConfig(8)")
    p.add_argument("--kv-heads", type=int, default=None, help="默认沿用 RainConfig(4)")
    p.add_argument("--vocab", type=int, default=None, help="默认沿用 RainConfig(16384)")
    p.add_argument("--inter", type=int, default=None, help="默认沿用 RainConfig 推导值")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--text-len", type=int, default=256)
    p.add_argument("--gen-prompt", type=int, default=512)
    p.add_argument("--gen-new", type=int, default=64)
    p.add_argument("--no-gen", action="store_true", help="跳过生成测试")
    p.add_argument("--compile", action="store_true", help="torch.compile 包模型（全脚本共享）")
    p.add_argument("--grad-ckpt", action="store_true", help="启用梯度检查点（测大 batch 用）")
    args = p.parse_args()

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
    print(f"MiniRainVLM 性能基准 | {dev_name} | torch {torch.__version__} | {args.dtype}")
    print(f"配置(model_size.py): type=v  hidden={args.hidden}  layers={args.layers}  "
          f"use_mot=1  use_moe=0  vision_ffn_scale={args.vision_ffn_scale}  "
          f"block_size={args.block_size}")
    print(f"负载: batch={args.batch}  训练={args.train_batch}x{args.train_seq}  "
          f"图像={args.img_size}px  文本={args.text_len}  iters={args.iters}  "
          f"compile={args.compile}  ckpt={args.grad_ckpt}")
    print("=" * 78)

    # 整个脚本只构建（并至多编译）一次模型
    model, cfg = build_model(args)
    param_report(model, cfg)
    total_mb = (torch.cuda.get_device_properties(dev_idx).total_memory / 2 ** 20
                if args.device.startswith("cuda") else 0.0)

    section_forward(args, model)
    free()
    section_mask(args, model)
    free()
    section_train(args, model, total_mb)
    free()
    if not args.no_gen:
        section_generate(args, model)
        free()
    section_memory(args, model, cfg)
    free()
    print("\n完成。")


if __name__ == "__main__":
    main()