"""Single-GPU pure-text MiniRain pretraining configured by JSON."""
import argparse
from collections import deque
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULTS = dict(device='cuda:0', output_dir='exp/pretrain_text_001', batch_size=1,
    grad_accum=8, epochs=1, max_steps=0, optimizer='muon_adamw', lr=5e-4,
    muon_lr_factor=30., muon_momentum=0.95, weight_decay=0.1, warmup_ratio=0.02,
    min_lr_ratio=0.1, grad_clip=1., dtype='bf16', loss_backend='auto', loss_chunks=32,
    compile=True, compile_dynamic=False, compile_recompile_limit=64,
    num_workers=4, seed=42, log_every=8, save_every=1000, keep_checkpoints=5)


def load_configuration(path):
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding='utf-8'))
    required = {'model_config', 'data', 'training'}
    if not required <= set(document) or set(document) - required - {'validation'}:
        raise ValueError('JSON requires model_config, data, training; validation is optional')
    training = dict(document['training'])
    unknown = set(training) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown training settings: {sorted(unknown)}')
    training = DEFAULTS | training
    data = document['data']
    if set(data) != {'bin', 'idx', 'max_length'}:
        raise ValueError('data requires exactly bin, idx and max_length')
    integer_keys = ('batch_size', 'grad_accum', 'epochs', 'loss_chunks',
                    'compile_recompile_limit', 'log_every', 'save_every', 'keep_checkpoints')
    for key in integer_keys:
        if type(training[key]) is not int or training[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('max_steps', 'num_workers', 'seed'):
        if type(training[key]) is not int or training[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    for key in ('compile', 'compile_dynamic'):
        if type(training[key]) is not bool:
            raise ValueError(f'{key} must be a JSON boolean')
    for key in ('lr', 'muon_lr_factor', 'grad_clip', 'weight_decay', 'warmup_ratio', 'min_lr_ratio', 'muon_momentum'):
        if type(training[key]) not in (float, int) or not math.isfinite(training[key]):
            raise ValueError(f'{key} must be a finite number')
    if min(training[k] for k in ('lr', 'muon_lr_factor', 'grad_clip')) <= 0 or training['weight_decay'] < 0:
        raise ValueError('Invalid learning rate, clipping or weight decay')
    if not 0 <= training['warmup_ratio'] < 1 or not 0 <= training['muon_momentum'] < 1 or not 0 <= training['min_lr_ratio'] <= 1:
        raise ValueError('Invalid warmup/minimum LR/momentum ratio')
    if training['optimizer'] not in ('muon_adamw', 'adamw') or training['dtype'] not in ('bf16', 'fp16', 'fp32'):
        raise ValueError('Unsupported optimizer or dtype')
    if training['loss_backend'] not in ('auto', 'native', 'liger', 'chunked'):
        raise ValueError('Unsupported loss backend')
    device = training['device']
    if not isinstance(device, str) or not device.startswith('cuda:') or not device[5:].isdecimal():
        raise ValueError('training.device must be an explicit CUDA index, e.g. cuda:1')
    if not isinstance(training['output_dir'], str) or not training['output_dir'].strip():
        raise ValueError('training.output_dir must be a nonempty path')
    if type(data['max_length']) is not int or data['max_length'] < 3:
        raise ValueError('data.max_length must be an integer >= 3')
    resolve = lambda value: str((path.parent / value).resolve())
    validation = document.get('validation')
    if validation is not None:
        if not isinstance(validation, dict) or set(validation) != {'bin', 'idx', 'every', 'max_batches', 'batch_size'}:
            raise ValueError('validation requires bin, idx, every, max_batches and batch_size')
        for key in ('every', 'batch_size', 'max_batches'):
            if type(validation[key]) is not int or validation[key] < (0 if key == 'max_batches' else 1):
                raise ValueError(f'Invalid validation.{key}; max_batches=0 means the full validation set')
        validation = dict(validation, bin=resolve(validation['bin']), idx=resolve(validation['idx']))
        if validation['bin'] == resolve(data['bin']) or validation['idx'] == resolve(data['idx']):
            raise ValueError('Validation files must be separate from training files')
    model = document['model_config']
    if isinstance(model, str):
        model = json.loads(Path(resolve(model)).read_text(encoding='utf-8'))
    if not isinstance(model, dict) or model.get('use_persistent_memory', False):
        raise ValueError('model_config must describe the main text model without persistent memory')
    if model.get('architectures', ['MiniRainForCausalLM']) != ['MiniRainForCausalLM']:
        raise ValueError('Expected the pure-text MiniRainForCausalLM architecture')
    return dict(model_config=model, data=dict(data, bin=resolve(data['bin']), idx=resolve(data['idx'])),
                training=dict(training, output_dir=resolve(training['output_dir'])), validation=validation)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'pretrain.json')
    parser.add_argument('--gpu', type=int, help='Direct physical CUDA index, overrides JSON device')
    parser.add_argument('--resume', help='Full checkpoint path, or auto for the experiment latest.pt')
    parser.add_argument('--dry-run', action='store_true', help='Validate data and model shapes without training')
    parser.add_argument('--stop-after', type=int, default=0,
                        help='Stop cleanly after N updates this invocation, preserving the full LR schedule')
    args = parser.parse_args()
    if args.stop_after < 0 or (args.gpu is not None and args.gpu < 0):
        parser.error('stop-after and GPU index must be nonnegative')
    return args


# Select visibility before importing model libraries which may initialize CUDA.
if __name__ == '__main__':
    args = parse_args()
    resolved = load_configuration(args.config)
    if args.gpu is not None:
        resolved['training']['device'] = f'cuda:{args.gpu}'
    if resolved['training']['device'].startswith('cuda'):
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset
from architectures.config import RainConfig
from architectures.rain import MiniRainForCausalLM
from dataset.datasets import PretrainBinDataset
from train.ckpt import save_exp, save_rolling_ckpt, load_ckpt_path, save_state_dict
from train.metrics import TrainingMetrics
from train.train_util import setup_seed, get_lr, SkipBatchSampler, format_training_progress
from train.text_training import TextCollate, TextTrainingLoss, build_optimizer


def rng_state():
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])


def data_signature(data):
    signature = dict(records=len(data), stored_tokens=len(data.tokens))
    for key, path in (('bin', data.bin_path), ('idx', data.idx_path)):
        stat = path.stat()
        signature[key] = dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    manifest = data.bin_path.parent / 'manifest.json'
    signature['manifest_sha256'] = (hashlib.sha256(manifest.read_bytes()).hexdigest()
                                    if manifest.is_file() else None)
    return signature


def validation_indices(count, batch_size, max_batches, seed):
    """Fixed uniform sample WITHOUT replacement across the entire held-out set."""
    limit = count if max_batches == 0 else min(count, batch_size * max_batches)
    if limit == count:
        return range(count)
    return torch.randperm(count, generator=torch.Generator().manual_seed(seed))[:limit].tolist()


@torch.no_grad()
def evaluate(model, loss_function, data, validation, device, dtype, seed):
    state, was_training = rng_state(), model.training
    try:
        model.eval()
        sample = Subset(data, validation_indices(len(data), validation['batch_size'], validation['max_batches'], seed))
        loader = DataLoader(sample, batch_size=validation['batch_size'], shuffle=False,
            collate_fn=TextCollate(data.pad_id, data.max_length), num_workers=0,
            generator=torch.Generator().manual_seed(seed), pin_memory=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        weighted_loss = torch.zeros((), device=device, dtype=torch.float64)
        tokens = 0
        for ids, labels in loader:
            count = int((labels[:, 1:] != -100).sum())
            ids, labels = ids.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            amp = torch.autocast('cuda', dtype=dtype) if dtype != torch.float32 else contextlib.nullcontext()
            with amp:
                loss = loss_function(ids, labels)
            weighted_loss += loss.double() * count
            tokens += count
        value = (weighted_loss / max(tokens, 1)).item()
        if tokens <= 0 or not math.isfinite(value):
            raise FloatingPointError('Validation has no targets or nonfinite loss')
        return dict(val_loss=value, val_perplexity=math.exp(value) if value < 700 else None,
                    val_tokens=tokens, val_records=len(sample), val_seconds=time.perf_counter()-started)
    finally:
        model.train(was_training)
        restore_rng(state)


def run(args, resolved):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This entry point supports one GPU; do not launch with multi-rank torchrun')
    settings, data_settings = resolved['training'], resolved['data']
    config = RainConfig.from_dict(resolved['model_config'])
    if config.type_id != 1 or data_settings['max_length'] > config.max_position_embeddings:
        raise ValueError('Requires dense MiniRain and max_length <= model context length')
    with contextlib.ExitStack() as resources:
        data = resources.enter_context(PretrainBinDataset(data_settings['bin'], data_settings['idx'],
            max_length=data_settings['max_length'], pad_to_max=False, packed=None))
        if (data.vocab_size, data.bos_id, data.eos_id) != (config.vocab_size, config.bos_token_id, config.eos_token_id):
            raise ValueError('Dataset tokenizer vocabulary/BOS/EOS do not match the model JSON')
        validation = resolved.get('validation')
        val_data = None
        if validation is not None:
            val_data = resources.enter_context(PretrainBinDataset(validation['bin'], validation['idx'],
                max_length=data_settings['max_length'], pad_to_max=False, packed=None))
            if (val_data.vocab_size, val_data.bos_id, val_data.eos_id, val_data.pad_id) != (
                    data.vocab_size, data.bos_id, data.eos_id, data.pad_id):
                raise ValueError('Validation and training tokenizer metadata do not match')
        batches_per_epoch = math.ceil(len(data) / settings['batch_size'])
        steps_per_epoch = math.ceil(batches_per_epoch / settings['grad_accum'])
        total_steps = settings['max_steps'] or steps_per_epoch * settings['epochs']
        if not 0 < total_steps <= steps_per_epoch * settings['epochs']:
            raise ValueError('max_steps exceeds the steps available in configured epochs')
        if args.dry_run:
            with torch.device('meta'):
                model = MiniRainForCausalLM(config)
            print(f'[Dry run] {type(model).__name__}; parameters={sum(p.numel() for p in model.parameters()):,}; '
                  f'updates/epoch={steps_per_epoch:,}; total_updates={total_steps:,}; '
                  f'validation_records={len(val_data) if val_data is not None else 0:,}', flush=True)
            return
        device = torch.device(settings['device'])
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('Training requires CUDA; use --dry-run for CPU validation')
        torch.cuda.set_device(device)
        print(f'[Device] selected={device}; current={torch.cuda.current_device()}; '
              f'name={torch.cuda.get_device_name(device)}', flush=True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        setup_seed(settings['seed'], deterministic=False)
        model = MiniRainForCausalLM(config).to(device).train()
        optimizer = build_optimizer(model, settings)
        scaler = torch.amp.GradScaler('cuda', enabled=settings['dtype'] == 'fp16')
        training = TextTrainingLoss(model, settings['loss_backend'], settings['loss_chunks'])
        validation_loss = (TextTrainingLoss(model, settings['loss_backend'], settings['loss_chunks'])
                           if val_data is not None else None)
        if settings['compile']:
            import torch._dynamo.config as dynamo_config
            dynamo_config.recompile_limit = settings['compile_recompile_limit']
            training = torch.compile(training, dynamic=settings['compile_dynamic'])
            print(f'[Compile] enabled, dynamic={settings["compile_dynamic"]}; first update includes compilation', flush=True)
        experiment = Path(settings['output_dir'])
        directory = experiment / 'pretrain'
        resume_path = directory / 'latest.pt' if args.resume == 'auto' else Path(args.resume).resolve() if args.resume else None
        signature = data_signature(data)
        # Device/output placement may change; mathematical training settings may not.
        contract = dict(model=config.to_dict(), data=signature, data_order='sequential',
                        training={k: v for k, v in settings.items()
                                  if k not in ('device', 'output_dir', 'num_workers', 'log_every', 'save_every', 'keep_checkpoints')},
                        total_steps=total_steps,
                        validation=dict(settings=validation, data=data_signature(val_data)) if val_data is not None else None)
        epoch_start = next_batch = global_step = tokens_seen = 0
        if resume_path is not None:
            checkpoint = load_ckpt_path(resume_path, adjust_step_for_world_size=False)
            if checkpoint is None:
                raise FileNotFoundError(resume_path)
            old_contract = checkpoint.get('contract', {})
            old_contract.setdefault('validation', None)
            # Replay pre-shuffle-format runs in their original order.
            if old_contract.setdefault('data_order', 'legacy_random') == 'legacy_random':
                contract['data_order'] = 'legacy_random'
            if old_contract != contract:
                raise ValueError('Resume requires unchanged model/data/optimizer/batch/LR schedule settings')
            model.load_state_dict(checkpoint['model'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            epoch_start, next_batch, global_step = checkpoint['epoch'], checkpoint['next_batch'], checkpoint['step']
            tokens_seen = checkpoint['tokens_seen']
            restore_rng(checkpoint['rng'])
            del checkpoint
        elif experiment.exists() and any(experiment.iterdir()):
            raise FileExistsError(f'Experiment is nonempty; use --resume auto or a new output_dir: {experiment}')
        save_exp(experiment.name, output_dir=experiment)
        (experiment / 'training_config.json').write_text(json.dumps(resolved, ensure_ascii=False, indent=2), encoding='utf-8')
        config.to_json_file(experiment / 'model_config.json', use_diff=False)
        log = TrainingMetrics(directory, resume_step=global_step if resume_path else None,
            config_dir=experiment, metadata=dict(configuration=resolved, model_config=config.to_dict(),
                device=str(device), world_size=1, torch_version=str(torch.__version__), total_steps=total_steps,
                parameters=sum(p.numel() for p in model.parameters())))
        stop = [False]
        previous_handler = signal.getsignal(signal.SIGINT)
        def request_stop(signum, frame):
            stop[0] = True
            print('[Stop] Finishing this optimizer update before saving.', flush=True)
        signal.signal(signal.SIGINT, request_stop)
        started = time.perf_counter()
        elapsed_before = log.elapsed_before
        durations = deque(maxlen=32)
        updates_this_run = 0
        warmup = round(total_steps * settings['warmup_ratio'])
        dtype = dict(bf16=torch.bfloat16, fp16=torch.float16, fp32=torch.float32)[settings['dtype']]
        print(f'[Run] {experiment}; micro_batch={settings["batch_size"]}; accum={settings["grad_accum"]}; '
              f'updates/epoch={steps_per_epoch:,}; total_updates={total_steps:,}; resume_step={global_step:,}; '
              f'data_order={contract["data_order"]}', flush=True)
        try:
            for epoch in range(epoch_start, settings['epochs']):
                if global_step >= total_steps:
                    break
                offset = next_batch if epoch == epoch_start else 0
                if contract['data_order'] == 'legacy_random':
                    sampler = RandomSampler(data, generator=torch.Generator().manual_seed(settings['seed'] + epoch))
                    batch_sampler = SkipBatchSampler(sampler, settings['batch_size'], offset)
                else:
                    remaining = range(offset * settings['batch_size'], len(data))
                    batch_sampler = SkipBatchSampler(remaining, settings['batch_size'])
                loader = DataLoader(data, batch_sampler=batch_sampler,
                    collate_fn=TextCollate(data.pad_id, data.max_length), num_workers=settings['num_workers'],
                    pin_memory=True, generator=torch.Generator().manual_seed(settings['seed'] + epoch))
                iterator = iter(loader)
                batch_position = offset
                while batch_position < batches_per_epoch and global_step < total_steps:
                    torch.cuda.synchronize(device)
                    step_started = time.perf_counter()
                    torch.cuda.reset_peak_memory_stats(device)
                    group = min(settings['grad_accum'], batches_per_epoch - batch_position)
                    batches = [next(iterator) for _ in range(group)]
                    counts = [int((labels[:, 1:] != -100).sum()) for _, labels in batches]
                    valid_tokens = sum(counts)
                    if valid_tokens <= 0:
                        raise ValueError('Optimizer update contains no supervised tokens')
                    optimizer.zero_grad(set_to_none=True)
                    loss_sum = torch.zeros((), device=device)
                    for (ids, labels), count in zip(batches, counts):
                        ids, labels = ids.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                        amp = torch.autocast('cuda', dtype=dtype) if settings['dtype'] != 'fp32' else contextlib.nullcontext()
                        with amp:
                            loss = training(ids, labels)
                        scaler.scale(loss * (count / valid_tokens)).backward()
                        loss_sum += loss.detach().float() * count
                    loss_value = (loss_sum / valid_tokens).item()
                    if not math.isfinite(loss_value):
                        raise FloatingPointError(f'Nonfinite loss before update {global_step+1}')
                    scaler.unscale_(optimizer)
                    missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
                    if missing:
                        raise RuntimeError(f'Missing gradients: {missing[:8]}')
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['grad_clip'], error_if_nonfinite=True)
                    scheduled_lr = get_lr(global_step+1, total_steps, settings['lr'], warmup_steps=warmup,
                                          min_lr_ratio=settings['min_lr_ratio'])
                    for pg in optimizer.param_groups:
                        pg['lr'] = scheduled_lr * (settings['muon_lr_factor'] if pg.get('use_muon', False) else 1.)
                    scaler.step(optimizer)
                    scaler.update()
                    torch.cuda.synchronize(device)
                    seconds = time.perf_counter() - step_started
                    # Drop first-update compilation from ETA once a steady update exists.
                    if updates_this_run == 1:
                        durations.clear()
                    durations.append(seconds)
                    batch_position += group
                    global_step += 1
                    updates_this_run += 1
                    tokens_seen += valid_tokens
                    at_end = batch_position == batches_per_epoch
                    next_epoch, next_position = (epoch+1, 0) if at_end else (epoch, batch_position)
                    item = dict(step=global_step, epoch=epoch, next_batch=next_position,
                        loss=loss_value, lr=scheduled_lr,
                        muon_lr=scheduled_lr*settings['muon_lr_factor'] if settings['optimizer']=='muon_adamw' else None,
                        grad_norm=float(grad_norm), grad_scale=scaler.get_scale(), tokens_seen=tokens_seen,
                        step_tokens=valid_tokens, tokens_per_sec=valid_tokens/seconds,
                        step_seconds=seconds, elapsed_seconds=elapsed_before+time.perf_counter()-started,
                        eta_seconds=(total_steps-global_step)*sum(durations)/len(durations),
                        cuda_allocated_mb_rank0=torch.cuda.memory_allocated(device)/2**20,
                        cuda_peak_mb_rank0=torch.cuda.max_memory_allocated(device)/2**20)
                    stopping = stop[0] or (args.stop_after > 0 and updates_this_run >= args.stop_after)
                    if val_data is not None and (global_step % validation['every'] == 0 or global_step == total_steps or stopping):
                        item.update(evaluate(model, validation_loss, val_data, validation, device, dtype, settings['seed']))
                        print(f'[Val] step={global_step:,} loss={item["val_loss"]:.6f} '
                              f'ppl={item["val_perplexity"]} tokens={item["val_tokens"]:,} '
                              f'records={item["val_records"]:,}', flush=True)
                    log.log(item)
                    if global_step == 1 or global_step % settings['log_every'] == 0 or global_step == total_steps or stopping:
                        print(format_training_progress(item, total_steps, settings['epochs']).replace('[P1]', '[Pretrain]', 1), flush=True)
                    if global_step % settings['save_every'] == 0 or global_step == total_steps or stopping:
                        save_rolling_ckpt(directory, step=global_step, keep=settings['keep_checkpoints'],
                            model=model, optimizer=optimizer, scaler=scaler, epoch=next_epoch,
                            next_batch=next_position, tokens_seen=tokens_seen, rng=rng_state(),
                            contract=contract, config=config.to_dict(), training_config=resolved)
                        log.export(status='stopped' if stopping and global_step < total_steps else 'running')
                        print(f'[Checkpoint] {directory / "latest.pt"}', flush=True)
                    if stopping:
                        break
                next_batch = 0
                if stopping:
                    break
            completed = global_step >= total_steps
            if completed:
                save_state_dict(model, directory / 'final_weights.pth', dtype=dtype)
            log.export(status='completed' if completed else 'stopped')
        except BaseException:
            log.export(status='failed')
            raise
        finally:
            log.close()
            signal.signal(signal.SIGINT, previous_handler)
        print(f'[Done] step={global_step:,}/{total_steps:,}; status={"completed" if completed else "stopped"}', flush=True)


if __name__ == '__main__':
    run(args, resolved)
