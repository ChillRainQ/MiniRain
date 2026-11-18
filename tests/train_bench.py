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

DTYPES = dict(bf16=torch.bfloat16, fp16=torch.float16, fp32=torch.float32)


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
        return self.loss_fn(self.model.lm_head.weight, hidden[:, :-1].contiguous(),
                            labels[:, 1:].contiguous())


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


def main():
    p = argparse.ArgumentParser(description='MiniRain 纯文本训练性能基准（3 GDN : 1 GQA）')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', choices=list(DTYPES), default='bf16')
    p.add_argument('--iters', type=int, default=10)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--train-batch', type=int, default=1)
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
    p.add_argument('--loss-backend', choices=['native', 'auto', 'liger', 'chunked'], default='native',
                   help='默认模型原生 CE；其余是显式替代路径，比较吞吐时须保持一致')
    p.add_argument('--loss-chunks', type=int, default=32)
    p.add_argument('--attention-mask', choices=['none', 'ones'], default='none',
                   help='无 padding 合成数据默认不传 mask；ones 用于对照旧版变长路径')
    p.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--mode', choices=['default', 'max-autotune'], default='default')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--profile-iters', type=int, default=0,
                   help='另跑同步分阶段诊断，不计入正式吞吐；0 关闭')
    p.add_argument('--quiet', action='store_true', help='仅输出结果文件路径')
    args = p.parse_args()
    if args.profile_iters < 0:
        p.error('--profile-iters 须为非负数')
    if min(args.iters, args.warmup, args.train_batch, args.layers, args.loss_chunks) < 1 or args.train_seq < 2:
        p.error('iters/warmup/batch/layers/chunks 须为正，seq 至少为 2')
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
    optimizer = build_optimizer(model, args)
    training = TrainingLoss(model, args.loss_backend, args.loss_chunks).train()
    if args.compile:
        torch._dynamo.config.recompile_limit = max(64, torch._dynamo.config.recompile_limit)
        training = torch.compile(training, mode=args.mode)
    scaler = torch.amp.GradScaler('cuda', enabled=args.dtype == 'fp16')
    ids = torch.randint(100, config.vocab_size, (args.train_batch, args.train_seq), device=device)
    ids[:, 0] = config.bos_token_id
    mask = torch.ones_like(ids) if args.attention_mask == 'ones' else None
    labels = ids.clone()
    if not args.quiet:
        print(f'[bench] MiniRain {args.train_batch}x{args.train_seq}, optimizer={args.optimizer}, '
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
        stage('zero_grad', lambda: model.zero_grad(set_to_none=True))
        def forward():
            context = torch.autocast('cuda', dtype=DTYPES[args.dtype]) if args.dtype != 'fp32' else nullcontext()
            with context:
                return training(ids, mask, labels)
        loss = stage('forward_loss', forward)
        stage('backward', lambda: scaler.scale(loss).backward())
        if optimizer is not None:
            def clip():
                scaler.unscale_(optimizer)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            stage('unscale_clip', clip)
            def update():
                scaler.step(optimizer)
                scaler.update()
            stage('optimizer', update)
        return loss.detach()

    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(args.warmup):
        loss = step()
    torch.cuda.synchronize(device)
    warmup_seconds = time.perf_counter() - start
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
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite measured loss')
    # 在计时之后检查，不给性能数字加入逐参数同步开销。
    peak_allocated = torch.cuda.max_memory_allocated(device) / 2**20
    peak_reserved = torch.cuda.max_memory_reserved(device) / 2**20
    check_gradients(model)
    ms = elapsed / args.iters * 1000
    result = dict(args=vars(args), model_class=type(model).__name__,
                  parameters=sum(p.numel() for p in model.parameters()), config=config.to_dict(),
                  optimizer_implementation=('torch.optim.AdamW(fused=True)' if args.optimizer == 'adamw'
                                            else args.optimizer),
                  device=torch.cuda.get_device_name(device), torch_version=torch.__version__,
                  warmup_seconds=warmup_seconds, milliseconds_per_step=ms,
                  tokens_per_second=args.train_batch * args.train_seq * 1000 / ms,
                  predicted_tokens_per_second=args.train_batch * (args.train_seq - 1) * 1000 / ms,
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
        print(f'[profile: separate synchronized diagnostic] {result["synchronized_phase_ms"]}')
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if not args.quiet:
        print(f'MiniRain ({result["parameters"]:,} params), {result["device"]}, {args.dtype}, compile={args.compile}')
        print(f'Warmup/compile: {warmup_seconds:.2f}s (excluded from timing)')
        print(f'{ms:.2f} ms/step | {result["tokens_per_second"]:,.0f} input tok/s | '
              f'peak allocated={peak_allocated:.1f} MiB | peak reserved={peak_reserved:.1f} MiB | loss={loss.item():.4f}')
    print(f'[result] {path}')
    print(f'[log] {txt_path}')
    if args.quiet:
        original_stdout.write(f'[result] {path}\n[log] {txt_path}\n')


if __name__ == '__main__':
    main()
