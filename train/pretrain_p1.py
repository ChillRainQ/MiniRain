import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path


def configure_gpus():
    """Use direct CUDA indices, without visibility remapping."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--gpu', '--gpus', dest='gpus', default=None)
    selected, _ = parser.parse_known_args()
    if selected.gpus is None:
        return
    value = selected.gpus.strip().lower()
    if value == 'cpu':
        return
    ids = value.split(',')
    if (not ids or any(not item.isdecimal() for item in ids) or
            len(set(map(int, ids))) != len(ids)):
        parser.error('--gpus must be GPU indices such as 0 or 0,1, or cpu')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    if len(ids) > 1 and int(os.environ.get('WORLD_SIZE', '1')) == 1:
        if sys.platform == 'win32':
            parser.error('Multi-GPU training uses NCCL and requires Linux/WSL')
        import subprocess
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   '--nnodes=1', f'--nproc_per_node={len(ids)}',
                   str(Path(__file__).resolve()), *sys.argv[1:]]
        raise SystemExit(subprocess.call(command))


if __name__ == '__main__':
    # --help must remain available without starting distributed workers.
    if '-h' not in sys.argv and '--help' not in sys.argv:
        configure_gpus()

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, BatchSampler, DistributedSampler, RandomSampler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from architectures.config import RainConfig
from architectures.rain_v import MiniRainVForCausalLM
from components.attention import MoTGroupQueryAttention, MoTAttention
from components.mlp import MoTFFN
from dataset.datasets import PretrainBinDataset
import project
from train.ckpt import load_ckpt_path, save_rolling_ckpt, save_exp, save_state_dict
from train.metrics import TrainingMetrics
from train.numerics import diagnose_loss
from train.train_util import (setup_seed, init_distributed_mode, Logger, get_lr,
                              get_muon_params, get_adamw_params, SkipBatchSampler,
                              format_training_progress)


def parse_args():
    p = argparse.ArgumentParser(description='MiniRain-Vision P1 pretraining')
    p.add_argument('--gpu', '--gpus', dest='gpus', type=str, default=None,
                   help='Direct CUDA index, e.g. 1 selects cuda:1; 0,1 launches DDP; cpu forces CPU')
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data/pretrain_bins')
    p.add_argument('--convstem-ckpt', type=Path, default=ROOT / 'init/ConvStem.pt',
                   help='Distilled ConvStem checkpoint loaded before freezing vision')
    p.add_argument('--exp-name', type=str, default=None,
                   help='Experiment name under ROOT/exp; default: pretrain_<hidden-size>_<layers>')
    p.add_argument('--output-dir', type=Path, default=None,
                   help='Override experiment root; P1 outputs go into its p1/ subdirectory')
    p.add_argument('--hidden-size', type=int, default=1024)
    p.add_argument('--layers', type=int, default=20)
    p.add_argument('--vision-ffn-scale', type=int, default=2)
    p.add_argument('--block-size', type=int, default=4)
    p.add_argument('--max-seq-len', type=int, default=4096)
    p.add_argument('--batch-size', type=int, default=2, help='Micro batch per GPU')
    p.add_argument('--grad-accum', type=int, default=8)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--max-steps', type=int, default=0, help='Optimizer steps; 0 = full epochs')
    p.add_argument('--lr', type=float, default=5e-4, help='AdamW learning rate')
    p.add_argument('--muon-lr-factor', type=float, default=30.0)
    p.add_argument('--muon-momentum', type=float, default=0.95)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--warmup-ratio', type=float, default=0.02)
    p.add_argument('--min-lr-ratio', type=float, default=0.1)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--optimizer', choices=['muon_adamw', 'adamw'], default='muon_adamw')
    p.add_argument('--dtype', choices=['bf16', 'fp16', 'fp32'], default='bf16')
    p.add_argument('--loss-backend', choices=['auto', 'liger', 'chunked'], default='auto')
    p.add_argument('--loss-chunks', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--pad-multiple', type=int, default=64)
    p.add_argument('--padding', choices=['fixed', 'dynamic'], default='fixed',
                   help='fixed: pad all batches to max-seq-len; dynamic: pad to batch maximum')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--log-every', type=int, default=8,
                   help='Print progress every N optimizer steps; metrics are saved every step')
    p.add_argument('--save-every', type=int, default=1000,
                   help='Checkpoint interval in optimizer steps; keep 5 numbered checkpoints plus latest.pt')
    p.add_argument('--resume', type=Path, default=None, help='Full training checkpoint')
    p.add_argument('--auto-resume', action='store_true',
                   help='Resume from <output-dir>/latest.pt when it exists')
    p.add_argument('--compile', action='store_true', help='Enable torch.compile (optional)')
    p.add_argument('--full-attn-res', action='store_true')
    p.add_argument('--block-attn-res', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Check dataset + model without training')
    p.add_argument('--ddp-find-unused', action=argparse.BooleanOptionalAction, default=True,
                   help='Safer for custom MoT branches; disable after all trainable params are verified')
    a = p.parse_args()
    if a.batch_size <= 0 or a.grad_accum <= 0 or a.epochs <= 0 or a.max_seq_len < 3:
        p.error('invalid batch/accum/epochs/max-seq-len')
    if a.pad_multiple < 1 or a.log_every < 1 or a.save_every < 1 or a.loss_chunks < 1:
        p.error('invalid padding/log/save interval')
    if not 0.0 <= a.muon_momentum < 1.0:
        p.error('muon-momentum must be in [0, 1)')
    if (a.lr <= 0 or a.muon_lr_factor <= 0 or a.weight_decay < 0 or
            not 0 <= a.warmup_ratio < 1 or not 0 <= a.min_lr_ratio <= 1 or
            a.grad_clip <= 0 or a.max_steps < 0 or a.num_workers < 0):
        p.error('invalid learning-rate/decay/warmup/clipping/steps/workers configuration')
    if a.exp_name is None:
        a.exp_name = f'pretrain_{a.hidden_size}_{a.layers}'
    if a.exp_name in ('.', '..') or Path(a.exp_name).name != a.exp_name:
        p.error('exp-name must be one directory name')
    return a


class Collate:
    def __init__(self, pad_id, max_length, multiple, padding='fixed'):
        self.pad_id, self.max_length, self.multiple = pad_id, max_length, multiple
        self.padding = padding

    def __call__(self, items):
        longest = max(len(x) for x in items)
        width = (self.max_length if self.padding == 'fixed' else
                 min(self.max_length, math.ceil(longest / self.multiple) * self.multiple))
        # Keep at least 2 positions for shifted LM loss.
        x = torch.full((len(items), width), self.pad_id, dtype=torch.long)
        y = torch.full((len(items), width), -100, dtype=torch.long)
        mask = torch.zeros((len(items), width), dtype=torch.long)
        for i, ids in enumerate(items):
            n = len(ids)
            x[i, :n] = ids
            y[i, :n] = ids
            mask[i, :n] = 1
        return x, y, mask


def load_distilled_convstem(model, path):
    """Load only the distilled student ConvStem, before P1 freezes vision."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Distilled ConvStem checkpoint not found: {path}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f'Invalid ConvStem checkpoint type: {type(checkpoint).__name__}')
    state = checkpoint.get('student')
    if not isinstance(state, dict) or not state:
        raise ValueError('ConvStem checkpoint is missing a non-empty "student" state_dict')
    backbone = getattr(model, 'model', None)
    stem = getattr(backbone, 'patch_embed', None)
    if not isinstance(stem, torch.nn.Module):
        raise RuntimeError('Model has no model.patch_embed ConvStem')
    try:
        stem.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            'Distilled ConvStem is incompatible with this model; '
            'the bundled checkpoint expects hidden_size=1024'
        ) from exc
    return int(checkpoint.get('step', 0))


def freeze_vision(model):
    """Freeze only modality=1 parameters, not the entire MoT Attention."""
    count = 0
    for module in model.modules():
        targets = []
        if isinstance(module, MoTGroupQueryAttention):
            targets = ['wq', 'wk', 'wv', 'wo']
        elif isinstance(module, MoTAttention):
            targets = ['local_experts_wq', 'local_experts_wk', 'local_experts_wv',
                       'local_experts_wo', 'local_experts_q_normalization',
                       'local_experts_k_normalization', 'local_experts_attention_norm']
        for name in targets:
            group = getattr(module, name, None)
            if isinstance(group, torch.nn.ModuleList) and len(group) > 1:
                for p in group[1].parameters():
                    if p.requires_grad:
                        count += p.numel()
                        p.requires_grad_(False)
        if isinstance(module, MoTFFN):
            for sub in (module.mlp_v, module.norm_v):
                for p in sub.parameters():
                    if p.requires_grad:
                        count += p.numel()
                        p.requires_grad_(False)
    backbone = getattr(model, 'model', None)
    if backbone is not None:
        for name in ('patch_embed', 'patch_embd', 'vision_stem', 'conv_stem', 'img_norm'):
            sub = getattr(backbone, name, None)
            if isinstance(sub, torch.nn.Module):
                for p in sub.parameters():
                    if p.requires_grad:
                        count += p.numel()
                        p.requires_grad_(False)
    # Fail closed on unexpected vision-only module names.
    leftover = [n for n,p in model.named_parameters() if p.requires_grad and
                ('.mlp_v.' in n or '.norm_v.' in n or
                 '.patch_embed.' in n or '.patch_embd.' in n or
                  '.vision_stem.' in n or '.conv_stem.' in n or '.img_norm.' in n)]
    if leftover:
        raise RuntimeError(f'Unfrozen vision params: {leftover[:8]}')
    return count


def optimizers(model, args):
    muon_params = (get_muon_params(model, matrix_only=True, exclude_norm=True)
                   if args.optimizer == 'muon_adamw' else [])
    adam_params = (get_adamw_params(model, matrix_only=True, exclude_norm=True)
                   if args.optimizer == 'muon_adamw' else
                   [p for p in model.parameters() if p.requires_grad])
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if sum(p.numel() for p in (*muon_params, *adam_params)) != total:
        raise RuntimeError('Optimizer partition missing parameters')
    if args.optimizer == 'muon_adamw':
        if not muon_params:
            raise RuntimeError('No hidden-layer matrices for Muon')
        try:
            from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
        except ImportError as exc:
            raise RuntimeError(
                'Muon training requires the "muon-optimizer" dependency declared in '
                'pyproject.toml; install the project dependencies first'
            ) from exc
        groups = [
            dict(params=muon_params, lr=args.lr * args.muon_lr_factor,
                 momentum=args.muon_momentum, weight_decay=args.weight_decay,
                 use_muon=True),
            dict(params=adam_params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=args.weight_decay, use_muon=False),
        ]
        cls = (MuonWithAuxAdam if dist.is_available() and dist.is_initialized()
               else SingleDeviceMuonWithAuxAdam)
        if dist.is_available() and dist.is_initialized():
            remainder = len(muon_params) % dist.get_world_size()
            if remainder and 2 * remainder < dist.get_world_size():
                raise ValueError(
                    'Installed distributed Muon padding cannot cover this parameter count '
                    'and world size; use a world size dividing the Muon parameter count'
                )
        optimizer = cls(groups)
    else:
        optimizer = torch.optim.AdamW(adam_params, lr=args.lr, betas=(0.9, 0.95),
                                      weight_decay=args.weight_decay)
    return optimizer, sum(p.numel() for p in muon_params), sum(p.numel() for p in adam_params)


def serialized_args(args):
    return {key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()}


def local_rng():
    return {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(),
            'python': random.getstate(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['cuda'])


def cpu_state(value):
    """Collect optimizer shards without serializing CUDA tensors across ranks."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(v) for v in value)
    return value


def select_device(gpus):
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if gpus is not None and gpus.strip().lower() == 'cpu':
        if world > 1:
            raise ValueError('CPU mode does not support the NCCL distributed path')
        return torch.device('cpu')
    if gpus is not None:
        ids = [int(item) for item in gpus.split(',')]
        if len(set(ids)) != len(ids) or any(index < 0 for index in ids):
            raise ValueError('GPU indices must be distinct nonnegative integers')
        if not torch.cuda.is_available():
            raise RuntimeError(f'--gpu {gpus} requested CUDA, but CUDA is unavailable')
        if max(ids) >= torch.cuda.device_count():
            raise RuntimeError(f'Requested GPU {max(ids)}, but only {torch.cuda.device_count()} devices are available')
        if world > len(ids) or local_rank >= len(ids):
            raise RuntimeError('Distributed worker count exceeds the number of selected GPUs')
        return torch.device(f'cuda:{ids[local_rank]}')
    return torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')


def main():
    args = parse_args()
    device = select_device(args.gpus)
    if device.type == 'cuda':
        torch.cuda.set_device(device.index)
    local_rank = init_distributed_mode(device_index=device.index)
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    experiment_dir = (ROOT / project.TRAIN_OUTPUT_DIR / args.exp_name
                      if args.output_dir is None else args.output_dir)
    if not experiment_dir.is_absolute():
        experiment_dir = ROOT / experiment_dir
    experiment_dir = experiment_dir.resolve()
    args.output_dir = experiment_dir / 'p1'
    if rank == 0:
        Logger(f'[Device] selected={device}; current='
               f'{torch.cuda.current_device() if device.type == "cuda" else "cpu"}; '
               f'name={torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"}; '
               f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "unset")}', flush=True)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    setup_seed(args.seed + rank, deterministic=False)
    if rank == 0:
        save_exp(args.exp_name, output_dir=experiment_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        Logger(f'[Experiment] {experiment_dir}; stage directory: {args.output_dir}', flush=True)
    if distributed:
        dist.barrier()
    data = PretrainBinDataset(
        bin_path=args.data_dir / 'p1_text.bin',
        idx_path=args.data_dir / 'p1_text.idx',
        max_length=args.max_seq_len,
        pad_to_max=False,
        stage='p1',
    )
    collate = Collate(data.pad_id, args.max_seq_len, args.pad_multiple, args.padding)
    # Fixed model initialization seed across all ranks.
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    config = RainConfig(hidden_size=args.hidden_size, n_hidden_layers=args.layers,
                        use_mot=True, use_moe=False, model_type='v',
                        vocab_size=data.vocab_size, pad_token_id=data.pad_id,
                        bos_token_id=data.bos_id, eos_token_id=data.eos_id,
                        vision_ffn_scale=args.vision_ffn_scale, block_size=args.block_size,
                        full_attn_res=args.full_attn_res,
                        block_attn_res=args.block_attn_res,
                        loss_backend=args.loss_backend, loss_chunks=args.loss_chunks,
                        use_bounce=False)
    model = MiniRainVForCausalLM(config).to(device)
    distilled_step = load_distilled_convstem(model, args.convstem_ckpt)
    frozen = freeze_vision(model)
    if args.dry_run:
        if rank == 0:
            x, y, mask = collate([data[i] for i in range(min(2, len(data)))])
            Logger(f'[DRY RUN] batch={tuple(x.shape)}, supervised={int((y[:,1:] != -100).sum())}')
        data.close()
        if distributed:
            dist.destroy_process_group()
        return
    optimizer, nmuon, nadam = optimizers(model, args)
    if rank == 0:
        Logger(f'[P1] records={len(data):,} stored_tokens={len(data.tokens):,}', flush=True)
        Logger(f'[ConvStem] loaded={args.convstem_ckpt} distilled_step={distilled_step:,}', flush=True)
        Logger(f'[MoT] vision parameters frozen={frozen/1e6:.2f}M; '
              f'muon={nmuon/1e6:.2f}M adamw={nadam/1e6:.2f}M', flush=True)
    # Different dropout/RNG streams per rank after identical model initialization.
    torch.manual_seed(args.seed + rank)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed + rank)

    if args.compile:
        model = torch.compile(model)
    # DDP only sees trainable params; frozen vision branch is intentionally absent.
    if distributed:
        model = DDP(model, device_ids=[device.index], find_unused_parameters=args.ddp_find_unused)

    raw = model.module if distributed else model
    raw = getattr(raw, '_orig_mod', raw)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'fp16' and device.type == 'cuda'))
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[args.dtype]
    sampler = (DistributedSampler(data, num_replicas=world, rank=rank, shuffle=True,
                                  seed=args.seed, drop_last=True) if distributed
               else RandomSampler(data))
    base_batches = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batches_per_epoch = len(base_batches)
    steps_per_epoch = math.ceil(batches_per_epoch / args.grad_accum)
    if batches_per_epoch == 0:
        raise ValueError('Dataset is smaller than one micro batch; reduce --batch-size')
    available_steps = steps_per_epoch * args.epochs
    if args.max_steps > available_steps:
        raise ValueError(
            f'--max-steps={args.max_steps} exceeds {available_steps} steps available '
            f'in {args.epochs} epoch(s)'
        )
    max_steps = args.max_steps or steps_per_epoch * args.epochs
    warmup = round(max_steps * args.warmup_ratio)
    start_epoch, next_batch, global_step, consumed_tokens = 0, 0, 0, 0
    resume_rng = None
    resume_path = args.resume
    if resume_path is None and args.auto_resume:
        candidate = args.output_dir / 'latest.pt'
        resume_path = candidate if candidate.is_file() else None
    if resume_path is not None:
        if not resume_path.is_absolute():
            resume_path = ROOT / resume_path
        ck = load_ckpt_path(resume_path, adjust_step_for_world_size=False)
        if ck is None:
            raise FileNotFoundError(resume_path)
        if int(ck.get('world_size', 1)) != world:
            raise ValueError('Exact resume requires the same world size')
        if (ck['config']['hidden_size'] != config.hidden_size or
                ck['config']['n_hidden_layers'] != config.n_hidden_layers):
            raise ValueError('Checkpoint architecture mismatch')
        geometry = {'records': len(data), 'stored_tokens': len(data.tokens),
                    'batches_per_epoch': batches_per_epoch, 'max_steps': max_steps}
        if ck.get('data_geometry') != geometry:
            raise ValueError('Exact resume requires unchanged dataset and training length')
        old_args = ck.get('args', {})
        if old_args.get('loss_backend', 'auto') != args.loss_backend:
            raise ValueError('Cannot exact-resume with changed loss backend; use a new experiment')
        if old_args.get('padding', 'dynamic') != args.padding:
            raise ValueError('Cannot exact-resume with changed padding; use the original mode or a new experiment')
        exact_keys = (
            'hidden_size', 'layers', 'vision_ffn_scale', 'block_size',
            'full_attn_res', 'block_attn_res', 'batch_size', 'grad_accum',
            'seed', 'max_seq_len', 'pad_multiple', 'optimizer', 'dtype', 'epochs',
            'lr', 'muon_lr_factor', 'muon_momentum', 'weight_decay', 'warmup_ratio',
            'min_lr_ratio', 'grad_clip',
        )
        for key in exact_keys:
            if key in old_args and old_args[key] != getattr(args, key):
                raise ValueError(f'Cannot exact-resume with changed {key}')
        if old_args.get('max_steps', args.max_steps) != args.max_steps:
            raise ValueError('Cannot exact-resume with changed max_steps')
        raw.load_state_dict(ck['model'], strict=True)
        if 'optimizer' not in ck:
            raise ValueError(
                'Checkpoint predates the unified optimizer and cannot be '
                'exact-resumed; start a new P1 run'
            )
        shards = ck.get('optimizer_states')
        if distributed and (shards is None or len(shards) != world):
            raise ValueError('Exact DDP resume requires optimizer states for every rank')
        optimizer.load_state_dict(shards[rank] if shards is not None else ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
        saved_step = ck.get('step', ck.get('global_step'))
        if saved_step is None:
            raise ValueError('Checkpoint is missing step/global_step')
        start_epoch, next_batch, global_step = ck['epoch'], ck['next_batch'], saved_step
        consumed_tokens = ck.get('consumed_tokens', 0)
        states = ck.get('rng_states')
        if states is None or len(states) != world:
            raise ValueError('Exact resume requires RNG states for every rank')
        resume_rng = states[rank]
    if resume_rng is not None:
        restore_rng(resume_rng)

    if rank == 0:
        Logger(f'[Run] ranks={world}, micro_B={args.batch_size}, accum={args.grad_accum}, '
              f'steps/epoch={steps_per_epoch}, total_steps={max_steps}', flush=True)
        log = TrainingMetrics(
            args.output_dir, resume_step=global_step if resume_path is not None else None,
            config_dir=experiment_dir,
            metadata=dict(stage='p1', args=serialized_args(args), model_config=config.to_dict(),
                          world_size=world, device=str(device), torch_version=str(torch.__version__),
                          trainable_params=sum(p.numel() for p in raw.parameters() if p.requires_grad),
                          frozen_params=frozen, total_steps=max_steps),
        )
    else:
        log = None

    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    elapsed_before = log.elapsed_before if log is not None else 0.
    measured_step_seconds = 0.
    measured_steps = 0
    stop = False
    for epoch in range(start_epoch, args.epochs):
        if distributed:
            sampler.set_epoch(epoch)
        else:
            # Per-epoch deterministic order on single GPU too, including resume.
            generator = torch.Generator().manual_seed(args.seed + epoch)
            sampler = RandomSampler(data, generator=generator)
            base_batches = BatchSampler(sampler, args.batch_size, drop_last=True)
        offset = next_batch if epoch == start_epoch else 0
        loader = DataLoader(data, batch_sampler=SkipBatchSampler(
                                sampler, args.batch_size, offset, drop_last=True),
                            collate_fn=collate, num_workers=args.num_workers,
                            generator=torch.Generator().manual_seed(args.seed + epoch + rank),
                            pin_memory=(device.type == 'cuda'), persistent_workers=False)
        it = iter(loader)
        batch_pos = offset
        while batch_pos < batches_per_epoch and global_step < max_steps:
            step_started = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            group = min(args.grad_accum, batches_per_epoch - batch_pos)
            loss_normalizer = group * args.batch_size * args.max_seq_len
            local_loss_sum = 0.
            local_tokens = 0
            for j in range(group):
                ids, labels, mask = next(it)
                ids = ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                valid = int((labels[:, 1:] != -100).sum().item())
                local_tokens += valid
                sync = contextlib.nullcontext() if (not distributed or j == group-1) else model.no_sync()
                amp = (torch.autocast(device_type='cuda', dtype=dtype)
                       if device.type == 'cuda' and args.dtype != 'fp32'
                       else contextlib.nullcontext())
                with sync:
                    with amp:
                        out = model(ids, labels=labels, attention_mask=mask, use_cache=False)
                        aux = getattr(out, 'aux_loss', None)
                        invalid_loss = (out.loss is None or not torch.isfinite(out.loss).all() or
                                        (aux is not None and not torch.isfinite(aux).all()))
                        failed = torch.tensor(int(invalid_loss), device=device)
                        if distributed:
                            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
                        if failed.item():
                            from train.ckpt import save_ckpt
                            diagnostic = diagnose_loss(raw, out, ids, labels)
                            diagnostic.update(step=global_step, micro_batch=j, rank=rank,
                                              batch_pos=batch_pos, compiled=args.compile,
                                              loss_backend=args.loss_backend)
                            path = args.output_dir / f'nonfinite_step_{global_step:08d}_rank{rank}.pt'
                            save_ckpt(path, step=global_step, diagnostic=diagnostic,
                                      input_ids=ids.detach().cpu(), labels=labels.detach().cpu(),
                                      attention_mask=mask.detach().cpu())
                            print('[Numerics] ' + json.dumps(diagnostic, ensure_ascii=False), flush=True)
                            if log is not None:
                                log.export(status='failed_nonfinite')
                                log.close()
                            raise RuntimeError(f'Nonfinite/None loss at step {global_step}; diagnostic: {path}')
                        total = out.loss + (aux if aux is not None else 0.)
                        local_loss_sum += float(total.detach()) * valid
                        # Accumulate token sums, then normalize after DDP has
                        # averaged gradients. Dynamic micro batches differ in size.
                        loss = total * (valid / loss_normalizer)
                    scaler.scale(loss).backward()
                batch_pos += 1
                del ids, labels, mask, out, loss
            step_number = global_step + 1
            scheduled_lr = get_lr(step_number, max_steps, args.lr,
                                  warmup_steps=warmup, min_lr_ratio=args.min_lr_ratio)
            for pg in optimizer.param_groups:
                factor = args.muon_lr_factor if pg.get('use_muon', False) else 1.0
                pg['lr'] = scheduled_lr * factor
            scaler.unscale_(optimizer)
            token_count = torch.tensor(float(local_tokens), device=device, dtype=torch.float64)
            if distributed:
                dist.all_reduce(token_count)
            missing = [n for n, p in raw.named_parameters() if p.requires_grad and p.grad is None]
            bad = torch.tensor(int(bool(missing) or any(
                not torch.isfinite(p.grad).all() for p in raw.parameters()
                if p.requires_grad and p.grad is not None)), device=device)
            if distributed:
                dist.all_reduce(bad, op=dist.ReduceOp.MAX)
            if bad.item():
                raise RuntimeError(f'Missing or nonfinite gradients at step {global_step}: {missing[:8]}')
            for p in raw.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad.mul_(world * loss_normalizer / token_count.item())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (p for p in raw.parameters() if p.requires_grad), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            metrics = torch.tensor([local_loss_sum, float(local_tokens)],
                                   device=device, dtype=torch.float64)
            if distributed:
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            consumed_tokens += int(metrics[1].item())
            loss_value = metrics[0].item() / metrics[1].item()
            if rank == 0:
                step_seconds = max(1e-6, time.perf_counter() - step_started)
                measured_step_seconds += step_seconds
                measured_steps += 1
                elapsed = elapsed_before + time.perf_counter() - started
                item = {'step': global_step, 'epoch': epoch, 'next_batch': batch_pos,
                        'loss': loss_value,
                        'lr': next(pg['lr'] for pg in optimizer.param_groups
                                   if not pg.get('use_muon', False)),
                        'muon_lr': next((pg['lr'] for pg in optimizer.param_groups
                                         if pg.get('use_muon', False)), None),
                        'grad_norm': float(grad_norm), 'grad_scale': scaler.get_scale(),
                        'tokens_seen': consumed_tokens,
                        'step_tokens': int(metrics[1].item()),
                        'tokens_per_sec': metrics[1].item() / step_seconds,
                        'step_seconds': step_seconds, 'elapsed_seconds': elapsed,
                        'eta_seconds': (max_steps - global_step) * measured_step_seconds / measured_steps,
                        'cuda_allocated_mb_rank0': (torch.cuda.memory_allocated(device) / 2**20
                                                    if device.type == 'cuda' else None),
                        'cuda_peak_mb_rank0': (torch.cuda.max_memory_allocated(device) / 2**20
                                              if device.type == 'cuda' else None)}
                log.log(item)
                if global_step % args.log_every == 0 or global_step == 1 or global_step == max_steps:
                    Logger(format_training_progress(item, max_steps, args.epochs), flush=True)
            at_epoch_end = batch_pos == batches_per_epoch
            need_save = (global_step % args.save_every == 0 or
                         global_step == max_steps or
                         (at_epoch_end and epoch == args.epochs - 1))
            if need_save:
                rng = local_rng()
                if distributed:
                    states = [None] * world
                    dist.all_gather_object(states, rng)
                    optimizer_states = [None] * world if rank == 0 else None
                    dist.gather_object(cpu_state(optimizer.state_dict()), optimizer_states, dst=0)
                else:
                    states = [rng]
                    optimizer_states = None
                e_next, b_next = (epoch+1, 0) if at_epoch_end else (epoch, batch_pos)
                if rank == 0:
                    Logger(f'[Checkpoint] Saving step {global_step:,}...', flush=True)
                    save_rolling_ckpt(
                        args.output_dir, keep=5, model=raw,
                        epoch=e_next, step=global_step, scaler=scaler,
                        optimizer=optimizer, config=config.to_dict(),
                        args=serialized_args(args), next_batch=b_next,
                        consumed_tokens=consumed_tokens, rng_states=states,
                        optimizer_states=optimizer_states,
                        data_geometry=dict(records=len(data), stored_tokens=len(data.tokens),
                                           batches_per_epoch=batches_per_epoch, max_steps=max_steps),
                    )
                    log.export()
                    Logger(f'[Checkpoint] Saved: {args.output_dir / "latest.pt"}', flush=True)
                if distributed:
                    dist.barrier()
        next_batch = 0
        if global_step >= max_steps:
            stop = True
        if stop:
            break
    if rank == 0:
        save_state_dict(raw, args.output_dir / 'p1_final_weights.pth')
        Logger('[DONE] ' + str(args.output_dir / 'p1_final_weights.pth'), flush=True)
        log.export(status='completed')
        log.close()
    data.close()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
