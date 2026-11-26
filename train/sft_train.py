"""Fine-tune a dense MiniRain base checkpoint on compiled RainPocket SFT data."""

import argparse
from collections import deque
from contextlib import nullcontext
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, help="Base weights or pretraining experiment")
    parser.add_argument("--model-config", type=Path, help="Override the base model JSON")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/sft/compiled_16k")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "exp/rainpocket_v1_sft")
    parser.add_argument("--gpu", type=int, default=0, help="Physical CUDA device index")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--loss-backend", choices=("auto", "liger", "chunked", "native"), default="auto")
    parser.add_argument("--loss-chunks", type=int, default=32)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--keep-checkpoints", type=int, default=5)
    parser.add_argument("--val-samples", type=int, default=128, help="Fixed held-out rows; 0 disables validation")
    parser.add_argument("--val-every", type=int, default=1000, help="Validate every N optimizer updates")
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--stop-after", type=int, default=0, help="Stop after this many updates in this invocation")
    parser.add_argument("--resume", action="store_true", help="Resume from output-dir/latest.pt")
    parser.add_argument("--dry-run", action="store_true", help="Validate compiled data and model on CPU")
    args = parser.parse_args()
    if args.ckpt is None and not args.resume:
        parser.error("--ckpt is required for a new SFT run")
    if min(args.batch_size, args.grad_accum, args.epochs, args.loss_chunks,
           args.log_every, args.save_every, args.keep_checkpoints, args.val_every, args.val_batch_size) < 1:
        parser.error("batch, accumulation, epochs, loss chunks and intervals must be positive")
    if min(args.gpu, args.num_workers, args.seed, args.stop_after, args.val_samples) < 0:
        parser.error("gpu, workers, seed, stop-after and val-samples must be nonnegative")
    if not all(math.isfinite(x) for x in (args.lr, args.grad_clip, args.weight_decay,
                                          args.warmup_ratio, args.min_lr_ratio)):
        parser.error("optimizer settings must be finite")
    if (min(args.lr, args.grad_clip) <= 0 or args.weight_decay < 0
            or not 0 <= args.warmup_ratio < 1 or not 0 <= args.min_lr_ratio <= 1):
        parser.error("invalid optimizer settings or schedule ratios")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if not parsed.dry_run:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from architectures.config import RainConfig
from architectures.rain import MiniRainForCausalLM
from cli.eval import find_weight, read_checkpoint
from dataset.datasets import SftBinDataset
from train.ckpt import load_ckpt_path, save_exp, save_rolling_ckpt, save_state_dict
from train.metrics import TrainingMetrics
from train.pretrain import restore_rng, rng_state, validation_indices
from train.text_training import TextTrainingLoss, build_optimizer
from train.train_util import SkipBatchSampler, format_training_progress, get_lr, setup_seed


def move_batch(ids, labels, data, device):
    # Trim the common right-padding tail before taking the batch to CUDA.
    end = int((ids.ne(data.pad_token_id) | labels.ne(-100)).any(dim=0).nonzero()[-1, 0]) + 1
    end = min(data.max_length, math.ceil(end / 128) * 128)
    return ids[:, :end].to(device, non_blocking=True), labels[:, :end].to(device, non_blocking=True)


@torch.no_grad()
def evaluate_sft(model, loss_fn, loader, data, device, dtype):
    started = time.perf_counter()
    was_training = model.training
    state = rng_state()
    model.eval()
    weighted_loss = torch.zeros((), device=device, dtype=torch.float64)
    targets = 0
    try:
        for ids, labels in loader:
            count = int(labels[:, 1:].ne(-100).sum())
            if not count:
                continue
            ids, labels = move_batch(ids, labels, data, device)
            amp = torch.autocast("cuda", dtype=dtype) if dtype != torch.float32 else nullcontext()
            with amp:
                loss = loss_fn(ids, labels)
            weighted_loss += loss.double() * count
            targets += count
        if not targets:
            raise ValueError("SFT validation subset has no supervised tokens")
        loss = (weighted_loss / targets).item()
        if not math.isfinite(loss):
            raise FloatingPointError("Nonfinite SFT validation loss")
        return dict(val_loss=loss, val_perplexity=math.exp(loss) if loss < 700 else None,
                    val_tokens=targets, val_records=len(loader.dataset),
                    val_seconds=time.perf_counter() - started)
    finally:
        model.train(was_training)
        restore_rng(state)


def run(args):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("SFT supports one GPU")
    output = args.output_dir.resolve()
    if args.resume:
        checkpoint = load_ckpt_path(output / "latest.pt", adjust_step_for_world_size=False)
        if not checkpoint or checkpoint.get("contract", {}).get("stage") != "sft":
            raise ValueError("Resume requires an SFT checkpoint in output-dir/latest.pt")
        config_values, state = checkpoint["config"], checkpoint["model"]
    else:
        checkpoint = None
        config_values, state = read_checkpoint(find_weight(args.ckpt.resolve()), args.model_config)
    config = RainConfig.from_dict(config_values)
    if config.type_id != 1:
        raise ValueError("SFT requires a dense MiniRain checkpoint")

    with SftBinDataset(args.data_dir) as data:
        if data.vocab_size != config.vocab_size or data.max_length > config.max_position_embeddings:
            raise ValueError("SFT vocabulary or sequence length does not match the base model")
        pad_id = getattr(config, "pad_token_id", None)
        if pad_id is not None and pad_id != data.pad_token_id:
            raise ValueError("SFT PAD token does not match the base model")
        val_count = min(args.val_samples, len(data) - 1)
        held_out = validation_indices(len(data), 1, val_count, args.seed) if val_count else []
        train_data = Subset(data, np.delete(np.arange(len(data)), held_out)) if held_out else data
        val_loader = (DataLoader(Subset(data, held_out), batch_size=args.val_batch_size,
                                 shuffle=False, pin_memory=True,
                                 generator=torch.Generator().manual_seed(args.seed)) if held_out else None)
        batches_per_epoch = math.ceil(len(train_data) / args.batch_size)
        total_steps = args.epochs * math.ceil(batches_per_epoch / args.grad_accum)
        if args.dry_run:
            with torch.device("meta"):
                model = MiniRainForCausalLM(config)
            model.load_state_dict(state, strict=True, assign=True)
            print(f"[Dry run] train_rows={len(train_data):,} val_rows={val_count:,} length={data.max_length:,} "
                  f"compiled_supervised_tokens={data.supervised_tokens:,} updates={total_steps:,}", flush=True)
            return
        if checkpoint is None and output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Experiment is nonempty: {output}. Use --resume or a new output directory.")
        if not torch.cuda.is_available():
            raise ValueError("SFT training requires CUDA; use --dry-run for CPU validation")
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        setup_seed(args.seed, deterministic=False)
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
        model = MiniRainForCausalLM(config)
        model.load_state_dict(state, strict=True)
        del state
        model.to(device).train()
        optimizer = build_optimizer(model, dict(optimizer="adamw", lr=args.lr, weight_decay=args.weight_decay))
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
        loss_fn = TextTrainingLoss(model, args.loss_backend, args.loss_chunks)
        val_loss_fn = loss_fn
        if args.compile:
            loss_fn = torch.compile(loss_fn, dynamic=True)

        data_files = {}
        for field, relative in data.manifest["files"].items():
            stat = (data.data_dir / relative).stat()
            data_files[field] = [stat.st_size, stat.st_mtime_ns]
        training_config = {key: str(value) if isinstance(value, Path) else value
                           for key, value in vars(args).items()}
        contract = dict(
            stage="sft", total_steps=total_steps, model=config.to_dict(),
            data_files=data_files, validation_indices=held_out,
            manifest_sha256=hashlib.sha256((data.data_dir / "manifest.json").read_bytes()).hexdigest(),
            settings={key: getattr(args, key) for key in (
                "batch_size", "grad_accum", "epochs", "lr", "warmup_ratio", "min_lr_ratio",
                "weight_decay", "grad_clip", "dtype", "loss_backend", "loss_chunks", "seed",
            )},
        )
        epoch_start = batch_start = step = tokens_seen = 0
        if checkpoint is not None:
            saved_contract = dict(checkpoint["contract"])
            saved_contract.setdefault("validation_indices", [])
            if saved_contract != contract:
                raise ValueError("Resume requires the same SFT data, validation split, model and optimizer schedule. "
                                 "For checkpoints from before validation support, pass --val-samples 0.")
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            epoch_start, batch_start = checkpoint["epoch"], checkpoint["next_batch"]
            step, tokens_seen = checkpoint["step"], checkpoint["tokens_seen"]
            restore_rng(checkpoint["rng"])
            del checkpoint
        else:
            save_exp(output.name, output_dir=output)
            config.to_json_file(output / "model_config.json", use_diff=False)
        metrics = TrainingMetrics(output, resume_step=step if args.resume else None,
                                  metadata=dict(model_config=config.to_dict(), configuration=training_config,
                                                total_steps=total_steps, device=str(device)))
        shutdown = False

        def request_stop(signum, frame):
            nonlocal shutdown
            shutdown = True

        previous_handler = signal.signal(signal.SIGINT, request_stop)
        started = time.perf_counter()
        durations = deque(maxlen=32)
        updates = 0
        warmup = round(total_steps * args.warmup_ratio)
        print(f"[SFT] {device} {torch.cuda.get_device_name(device)} | "
              f"{len(train_data):,} train rows | {val_count:,} val rows | "
              f"{total_steps:,} updates | resume_step={step}", flush=True)
        try:
            for epoch in range(epoch_start, args.epochs):
                batch_position = batch_start if epoch == epoch_start else 0
                remaining = range(batch_position * args.batch_size, len(train_data))
                loader = DataLoader(train_data, batch_sampler=SkipBatchSampler(remaining, args.batch_size),
                                    pin_memory=True, num_workers=args.num_workers,
                                    generator=torch.Generator().manual_seed(args.seed + epoch))
                iterator = iter(loader)
                while batches := list(islice(iterator, args.grad_accum)):
                    update_start = time.perf_counter()
                    torch.cuda.reset_peak_memory_stats(device)
                    counts = [int(labels[:, 1:].ne(-100).sum()) for _, labels in batches]
                    targets = sum(counts)
                    if not targets:
                        raise ValueError("SFT accumulation group has no supervised tokens")
                    optimizer.zero_grad(set_to_none=True)
                    loss_sum = torch.zeros((), device=device)
                    for (ids, labels), count in zip(batches, counts, strict=True):
                        if not count:
                            continue
                        ids, labels = move_batch(ids, labels, data, device)
                        amp = torch.autocast("cuda", dtype=dtype) if dtype != torch.float32 else nullcontext()
                        with amp:
                            loss = loss_fn(ids, labels)
                        scaler.scale(loss * (count / targets)).backward()
                        loss_sum += loss.detach().float() * count
                    loss_value = (loss_sum / targets).item()
                    if not math.isfinite(loss_value):
                        raise FloatingPointError(f"Nonfinite SFT loss at step {step + 1}")
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
                    lr = get_lr(step + 1, total_steps, args.lr, warmup_steps=warmup, min_lr_ratio=args.min_lr_ratio)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    scaler.step(optimizer)
                    scaler.update()
                    torch.cuda.synchronize(device)
                    seconds = time.perf_counter() - update_start
                    if updates == 1:
                        durations.clear()
                    durations.append(seconds)
                    step += 1
                    updates += 1
                    tokens_seen += targets
                    batch_position += len(batches)
                    next_epoch, next_batch = ((epoch + 1, 0) if batch_position == batches_per_epoch
                                              else (epoch, batch_position))
                    record = dict(
                        step=step, epoch=epoch, next_batch=next_batch, loss=loss_value,
                        lr=lr, grad_norm=float(norm), grad_scale=scaler.get_scale(), tokens_seen=tokens_seen,
                        step_tokens=targets, tokens_per_sec=targets / seconds, step_seconds=seconds,
                        elapsed_seconds=metrics.elapsed_before + time.perf_counter() - started,
                        eta_seconds=(total_steps - step) * sum(durations) / len(durations),
                        cuda_allocated_mb_rank0=torch.cuda.memory_allocated(device) / 2**20,
                        cuda_peak_mb_rank0=torch.cuda.max_memory_allocated(device) / 2**20,
                    )
                    shutdown = shutdown or (args.stop_after > 0 and updates >= args.stop_after)
                    if val_loader is not None and (step % args.val_every == 0 or step == total_steps or shutdown):
                        record.update(evaluate_sft(model, val_loss_fn, val_loader, data, device, dtype))
                        print(f"[SFT Val] step={step:,} loss={record['val_loss']:.6f} "
                              f"ppl={record['val_perplexity']} tokens={record['val_tokens']:,} "
                              f"rows={record['val_records']:,}", flush=True)
                    metrics.log(record)
                    if step == 1 or step % args.log_every == 0 or step == total_steps or shutdown:
                        print(format_training_progress(record, total_steps, args.epochs).replace("[P1]", "[SFT]", 1),
                              flush=True)
                    if step % args.save_every == 0 or step == total_steps or shutdown:
                        save_rolling_ckpt(output, step=step, keep=args.keep_checkpoints,
                                         model=model, optimizer=optimizer, scaler=scaler, epoch=next_epoch,
                                         next_batch=next_batch, tokens_seen=tokens_seen, rng=rng_state(),
                                         contract=contract, config=config.to_dict())
                        metrics.export(status="running")
                    if shutdown:
                        break
                if shutdown:
                    break
            if step == total_steps:
                save_state_dict(model, output / "final_weights.pth", dtype=dtype)
            metrics.export(status="completed" if step == total_steps else "stopped")
        finally:
            signal.signal(signal.SIGINT, previous_handler)
            metrics.close()


if __name__ == "__main__":
    run(parsed)
