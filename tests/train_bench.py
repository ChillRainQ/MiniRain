"""MiniRain 纯文本训练基准，使用合成 token，不测数据加载或保存开销。

uv run tests/train_bench.py --device cuda:1 --train-batch 1 --train-seq 4096
uv run tests/train_bench.py --no-compile --optimizer adamw
"""
import argparse
import atexit
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from architectures.config import RainConfig
from architectures.rain import MiniRainForCausalLM
from losses.loss import get_loss_fn
from train.train_util import get_muon_params, get_adamw_params

DTYPES = dict(bf16=torch.bfloat16, fp16=torch.float16, fp32=torch.float32, fp8=torch.bfloat16)


class Tee:
    def __init__(self, stream, logfile, echo):
        self.stream, self.logfile, self.echo = stream, logfile, echo

    def write(self, text):
        self.logfile.write(text)
        if self.echo:
            self.stream.write(text)
        return len(text)

    def flush(self):
        self.logfile.flush()
        if self.echo:
            self.stream.flush()


class TrainingLoss(torch.nn.Module):
    def __init__(self, model, backend, chunks):
        super().__init__()
        self.model = model
        self.loss_fn = None if backend == 'native' else get_loss_fn(chunks, backend)

    def forward(self, ids, mask, labels):
        if self.loss_fn is None:
            return self.model(ids, attention_mask=mask, labels=labels, use_cache=False).loss
        hidden, _, _ = self.model.model(ids, attention_mask=mask, use_cache=False)
        return self.external_loss(hidden, labels)

    @torch.compiler.disable
    def external_loss(self, hidden, labels):
        # Fused loss/custom backward already owns its kernels; avoid tracing
        # backend chunk loops and scalar extraction through Dynamo.
        # Liger expects [tokens, hidden] and [tokens], not batched tensors.
        shifted_hidden = hidden[:, :-1, :].reshape(-1, hidden.size(-1)).contiguous()
        shifted_labels = labels[:, 1:].reshape(-1).contiguous()
        return self.loss_fn(self.model.lm_head.weight, shifted_hidden, shifted_labels)


def build_optimizer(model, args):
    if args.optimizer == 'none':
        return None
    if args.optimizer == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                 weight_decay=0.01, fused=True)
    try:
        from muon import SingleDeviceMuonWithAuxAdam
    except ImportError as exc:
        raise RuntimeError('需要 muon-optimizer；也可指定 --optimizer adamw/none') from exc
    muon = get_muon_params(model, matrix_only=True, exclude_norm=True)
    adam = get_adamw_params(model, matrix_only=True, exclude_norm=True)
    grouped = muon + adam
    if len({id(p) for p in grouped}) != len(grouped) or {id(p) for p in grouped} != {id(p) for p in model.parameters()}:
        raise RuntimeError('Optimizer groups must cover each parameter exactly once')
    print(f'[optimizer] Muon={sum(p.numel() for p in muon):,}, AdamW={sum(p.numel() for p in adam):,}')
    return SingleDeviceMuonWithAuxAdam([
        dict(params=muon, use_muon=True, lr=args.lr * args.muon_lr_factor, momentum=0.95, weight_decay=0.01),
        dict(params=adam, use_muon=False, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)])


def check_gradients(model):
    bad = [name for name, p in model.named_parameters()
           if p.grad is None or not torch.isfinite(p.grad).all()]
    if bad:
        raise RuntimeError(f'Missing/nonfinite gradients: {bad[:10]}')


def accumulated_step(model, training, batches, optimizer, scaler, grad_clip,
                     autocast_context, stage=None):
    """One update: average equal-sized microbatch losses, clip/update once."""
    if stage is None:
        stage = lambda name, operation: operation()
    stage('zero_grad', lambda: model.zero_grad(set_to_none=True))
    mean_loss = None
    for ids, mask, labels in batches:
        def forward():
            with autocast_context():
                return training(ids, mask, labels)
        loss = stage('forward_loss', forward)
        stage('backward', lambda: scaler.scale(loss / len(batches)).backward())
        detached = loss.detach() / len(batches)
        mean_loss = detached if mean_loss is None else mean_loss + detached
    if optimizer is not None:
        def clip():
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        stage('unscale_clip', clip)
        def update():
            scaler.step(optimizer)
            scaler.update()
        stage('optimizer', update)
    return mean_loss


def main():
    p = argparse.ArgumentParser(description='MiniRain 纯文本训练性能基准（3 GDN : 1 GQA）')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', choices=list(DTYPES), default='bf16')
    p.add_argument('--iters', type=int, default=10, help='计时的完整累积更新步数')
    p.add_argument('--warmup', type=int, default=3, help='预热的完整累积更新步数')
    p.add_argument('--train-batch', type=int, default=1)
    p.add_argument('--grad-accum', type=int, default=1, help='每次更新累积的微批次数')
    p.add_argument('--train-seq', type=int, default=4096)
    p.add_argument('--hidden', type=int, default=1024)
    p.add_argument('--layers', type=int, default=24)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--kv-heads', type=int, default=4)
    p.add_argument('--intermediate-size', type=int, default=None)
    p.add_argument('--block-size', type=int, default=6)
    p.add_argument('--optimizer', choices=['muon_adamw', 'adamw', 'none'], default='muon_adamw')
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--muon-lr-factor', type=float, default=30.)
    p.add_argument('--grad-clip', type=float, default=1.)
    p.add_argument('--loss-backend', choices=['native', 'auto', 'liger', 'chunked'], default='auto',
                   help='默认 auto：优先 Liger，未安装则用分块 CE；native 为模型原生 CE')
    p.add_argument('--loss-chunks', type=int, default=32)
    p.add_argument('--attention-mask', choices=['none', 'ones'], default='none',
                   help='无 padding 合成数据默认不传 mask；ones 用于对照旧版变长路径')
    p.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--compile-dynamic', choices=['false', 'true'], default=None,
                   help='显式选择 torch.compile dynamic=False/True；不传则保持 PyTorch 默认')
    p.add_argument('--fp8-scale-interval', type=int, default=32)
    p.add_argument('--fp8-check-every', type=int, default=100)
    p.add_argument('--mode', choices=['default', 'max-autotune'], default='default')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--profile-iters', type=int, default=0,
                   help='另跑同步分阶段诊断，不计入正式吞吐；0 关闭')
    p.add_argument('--quiet', action='store_true', help='仅输出结果文件路径')
    args = p.parse_args()
    requested_compile = args.compile
    compile_dynamic = None if args.compile_dynamic is None else args.compile_dynamic == 'true'
    if min(args.fp8_scale_interval, args.fp8_check_every) < 1:
        p.error('FP8 scale/check intervals must be positive')
    if args.profile_iters < 0:
        p.error('--profile-iters 须为非负数')
    if min(args.iters, args.warmup, args.train_batch, args.grad_accum, args.layers, args.loss_chunks) < 1 or args.train_seq < 2:
        p.error('iters/warmup/batch/grad-accum/layers/chunks 须为正，seq 至少为 2')
    if args.heads < 1 or args.kv_heads < 1 or args.heads % args.kv_heads or args.hidden <= 0 or args.hidden % (2 * args.heads):
        p.error('hidden 须能被 2*heads 整除，heads 须能被 kv-heads 整除，均须为正')
    if args.block_size < 0 or args.block_size % 2 or (args.intermediate_size is not None and args.intermediate_size <= 0):
        p.error('block-size 须为 0 或正偶数，FFN 维度须为正')
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        p.error('真实 FLA GDN 训练基准需要 CUDA')
    directory = Path(__file__).resolve().parent / 'txt'
    directory.mkdir(parents=True, exist_ok=True)
    stem = f'train_bench_{time.strftime("%Y%m%d_%H%M%S")}_{time.time_ns() % 1000000}'
    path = directory / (stem + '.json')
    txt_path = directory / (stem + '.txt')
    original_stdout, original_stderr = sys.stdout, sys.stderr
    logfile = txt_path.open('w', encoding='utf-8', buffering=1)
    sys.stdout = Tee(original_stdout, logfile, not args.quiet)
    sys.stderr = Tee(original_stderr, logfile, not args.quiet)
    def restore_streams():
        sys.stdout, sys.stderr = original_stdout, original_stderr
        logfile.close()
    atexit.register(restore_streams)
    print(f'[log] {txt_path}', flush=True)
    if args.dtype == 'fp8':
        print(f'[fp8] GPU delayed scales interval={args.fp8_scale_interval}; '
              f'periodic finite check={args.fp8_check_every}; compile={args.compile}, dynamic={compile_dynamic}')
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    extra = {} if args.intermediate_size is None else dict(intermediate_size=args.intermediate_size)
    config = RainConfig(hidden_size=args.hidden, n_hidden_layers=args.layers,
                        n_attn_heads=args.heads, n_key_value_heads=args.kv_heads,
                        vocab_size=16384, tie_word_embeddings=True, block_size=args.block_size,
                        max_position_embeddings=args.train_seq, **extra)
    # FP32 主权重 + autocast，与混合精度训练一致。
    model = MiniRainForCausalLM(config).to(device).train()
    if args.dtype == 'fp8':
        from losses.loss import _HAS_LIGER
    if args.dtype == 'fp8' and (args.loss_backend == 'chunked' or
                                (args.loss_backend == 'auto' and not _HAS_LIGER)):
        # The fallback chunked loss explicitly computes its logits in FP32.
        # Keep the MVP LM head in BF16 autocast instead of quantizing it.
        # Installed Liger remains selected by auto as usual.
        print(f'[fp8] loss backend {args.loss_backend} -> native (BF16 logits, FP32 CE reduction)')
        args.loss_backend = 'native'
    training = TrainingLoss(model, args.loss_backend, args.loss_chunks).train()
    batches = []
    for _ in range(args.grad_accum):
        ids = torch.randint(100, config.vocab_size, (args.train_batch, args.train_seq), device=device)
        ids[:, 0] = config.bos_token_id
        mask = torch.ones_like(ids) if args.attention_mask == 'ones' else None
        batches.append((ids, mask, ids.clone()))
    fp8_info = None
    fp8_controller = None
    if args.dtype == 'fp8':
        from fp8_linear import enable_fp8, probe_backend
        backend = probe_backend(device)
        @torch.no_grad()
        def initial_losses():
            values = []
            for ids, mask, labels in batches:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    values.append(training(ids, mask, labels).float())
            return torch.stack(values)
        reference = initial_losses()
        fp8_info = enable_fp8(model, args.fp8_scale_interval, args.fp8_check_every)
        fp8_controller = fp8_info.pop('_controller')
        # Untimed BF16 calibration includes backward, so gradient scales are
        # initialized correctly (including grad_accum normalization).
        accumulated_step(model, training, batches, None,
                         torch.amp.GradScaler('cuda', enabled=False), 0.,
                         lambda: torch.autocast('cuda', dtype=torch.bfloat16))
        fp8_controller.finish_calibration()
        fp8_controller.check_now(model=model)
        model.zero_grad(set_to_none=True)
        actual = initial_losses()
        if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
            raise FloatingPointError('Nonfinite loss during BF16/FP8 comparison')
        fp8_info['backend'] = backend
        fp8_info['precision'] = dict(master_parameters='float32', linear_gemm='float8_e4m3fn',
                                     gemm_outputs='bfloat16', attention_norm_embedding='unchanged',
                                     lm_head='bf16_autocast; not quantized', loss_reduction='float32')
        fp8_info['initial_loss_comparison'] = dict(
            bf16_loss=reference.mean().item(), fp8_loss=actual.mean().item(),
            absolute_difference=(actual.mean()-reference.mean()).abs().item(),
            relative_difference=((actual.mean()-reference.mean()).abs()/reference.mean().abs().clamp_min(1e-12)).item(),
            max_microbatch_absolute_difference=(actual-reference).abs().max().item(),
            compared_microbatches=len(batches), same_weights_and_inputs=True)
        print(f'[fp8 vs bf16] {fp8_info["initial_loss_comparison"]}', flush=True)
        print(f'[fp8] converted={len(fp8_info["converted"])}, BF16 exceptions={fp8_info["bf16_linears"]}')
    optimizer = build_optimizer(model, args)
    if args.compile:
        torch._dynamo.config.recompile_limit = max(64, torch._dynamo.config.recompile_limit)
        if args.dtype == 'fp8':
            torch._dynamo.utils.counters.clear()
        training = torch.compile(training, mode=args.mode, dynamic=compile_dynamic)
    scaler = torch.amp.GradScaler('cuda', enabled=args.dtype == 'fp16')
    if not args.quiet:
        print(f'[bench] MiniRain {args.train_batch}x{args.train_seq}, optimizer={args.optimizer}, '
              f'accum={args.grad_accum}, effective_batch={args.train_batch * args.grad_accum}, '
              f'loss={args.loss_backend}, mask={args.attention_mask}, compile={args.compile}; warmup={args.warmup}', flush=True)

    def step(profile=None):
        def stage(name, operation):
            if profile is None:
                return operation()
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            value = operation()
            torch.cuda.synchronize(device)
            profile[name] = profile.get(name, 0.) + (time.perf_counter() - start) * 1000
            return value
        def autocast_context():
            return torch.autocast('cuda', dtype=DTYPES[args.dtype]) if args.dtype != 'fp32' else nullcontext()
        loss = accumulated_step(model, training, batches, optimizer, scaler,
                                args.grad_clip, autocast_context, stage)
        if fp8_controller is not None:
            fp8_controller.after_step(loss, model)
        return loss

    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(args.warmup):
        loss = step()
    torch.cuda.synchronize(device)
    warmup_seconds = time.perf_counter() - start
    if fp8_controller is not None:
        fp8_controller.check_now(loss, model)
    else:
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite warmup loss')
        check_gradients(model)
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(args.iters):
        loss = step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    # 在计时之后检查，不给性能数字加入逐参数同步开销。
    peak_allocated = torch.cuda.max_memory_allocated(device) / 2**20
    peak_reserved = torch.cuda.max_memory_reserved(device) / 2**20
    if fp8_controller is not None:
        fp8_controller.check_now(loss, model)
    else:
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite measured loss')
        check_gradients(model)
    ms = elapsed / args.iters * 1000
    result = dict(args=vars(args), model_class=type(model).__name__,
                  requested_compile=requested_compile, compile_dynamic=compile_dynamic, fp8=fp8_info,
                  parameters=sum(p.numel() for p in model.parameters()), config=config.to_dict(),
                  optimizer_implementation=('torch.optim.AdamW(fused=True)' if args.optimizer == 'adamw'
                                            else args.optimizer),
                  device=torch.cuda.get_device_name(device), torch_version=torch.__version__,
                  warmup_seconds=warmup_seconds, milliseconds_per_step=ms,
                  milliseconds_per_microbatch_amortized=ms / args.grad_accum,
                  effective_batch_size=args.train_batch * args.grad_accum,
                  input_tokens_per_update=args.train_batch * args.train_seq * args.grad_accum,
                  measured_microbatches=args.iters * args.grad_accum,
                  measured_optimizer_updates=args.iters if optimizer is not None else 0,
                  tokens_per_second=args.train_batch * args.train_seq * args.grad_accum * 1000 / ms,
                  predicted_tokens_per_second=args.train_batch * (args.train_seq - 1) * args.grad_accum * 1000 / ms,
                  peak_allocated_mib=peak_allocated, peak_reserved_mib=peak_reserved,
                  resident_allocated_mib=torch.cuda.memory_allocated(device) / 2**20,
                  final_loss=loss.item(), gradients_finite=True)
    if args.profile_iters:
        profile = {}
        for _ in range(args.profile_iters):
            diagnostic_loss = step(profile)
        check_gradients(model)
        if not torch.isfinite(diagnostic_loss):
            raise RuntimeError('Nonfinite diagnostic loss')
        result['synchronized_phase_ms'] = {name: value / args.profile_iters for name, value in profile.items()}
        result['profile_unit'] = 'accumulated_update_step'
        print(f'[profile: separate synchronized diagnostic] {result["synchronized_phase_ms"]}')
    if fp8_controller is not None:
        fp8_info['stats'] = fp8_controller.report()
        if args.compile:
            result['fp8_dynamo'] = dict(
                stats=dict(torch._dynamo.utils.counters['stats']),
                graph_breaks=dict(torch._dynamo.utils.counters['graph_break']))
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if not args.quiet:
        print(f'MiniRain ({result["parameters"]:,} params), {result["device"]}, {args.dtype}, compile={args.compile}')
        print(f'Warmup/compile: {warmup_seconds:.2f}s (excluded from timing)')
        print(f'{ms:.2f} ms/update | {ms / args.grad_accum:.2f} ms/micro (amortized) | '
              f'{result["tokens_per_second"]:,.0f} input tok/s | '
              f'peak allocated={peak_allocated:.1f} MiB | peak reserved={peak_reserved:.1f} MiB | loss={loss.item():.4f}')
    print(f'[result] {path}')
    print(f'[log] {txt_path}')
    if args.quiet:
        original_stdout.write(f'[result] {path}\n[log] {txt_path}\n')


if __name__ == '__main__':
    main()
