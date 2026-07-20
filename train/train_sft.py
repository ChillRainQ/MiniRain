import sys
import os
# 将项目根目录（当前文件的父目录的父目录）添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from dataset.datasets import SFTDataset
from train.train_util import init_model, Logger
from train.train_util import is_main_process
from contextlib import nullcontext
import argparse
import time
import torch
import torch.distributed as dist
from torch import optim, nn
from architectures.config import RainConfig
from train.train_util import get_lr, init_distributed_mode, setup_seed, get_checkpoint, get_adam_params, get_muon_params, SkipBatchSampler, save



def train_epoch(model: nn.Module | DistributedDataParallel, scaler: GradScaler, muon, adam, epoch:int, epochs:int,
                learning_rate:float, device: str, loader, iters, start_step: int = 0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        last_step = step
        lr = get_lr(epoch * iters + step, epochs * iters, learning_rate)
        if muon is not None:
            for param_group in muon.param_groups:
                param_group['lr'] = lr * 30

        for param_group in adam.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            aux_loss = res.aux_loss if res.aux_loss is not None else 0.0
            loss = (res.loss + aux_loss) / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            if muon is not None:
                # 与预训练一致：裁剪前两个优化器都要 unscale，
                # 否则 adam 管的参数在 scaled 状态下被裁剪，阈值失真
                scaler.unscale_(muon)
                scaler.unscale_(adam)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                # 分别 step 两个优化器
                scaler.step(muon)
                scaler.step(adam)
                scaler.update()
                # 清空梯度
                muon.zero_grad(set_to_none=True)
                adam.zero_grad(set_to_none=True)
            else:
                scaler.unscale_(adam)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(adam)
                scaler.update()
                adam.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            muon_lr = muon.param_groups[-1]['lr'] if muon is not None else 0
            adam_lr = adam.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(
                f'Epoch:[{epoch + 1}/{epochs}]({step}/{iters}), '
                f'loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, '
                f'lr_muon: {muon_lr:.8f}, lr_adam: {adam_lr:.8f}, eta: {eta_min:.1f}min'
            )

        # ========== 断点保存：统一走 get_checkpoint ==========
        if step % args.save_interval == 0 or step == iters:
            if is_main_process():
                # 权重导出在每个 epoch 结束（done 命名规则）
                if step == iters:
                    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
                    raw_model = getattr(raw_model, '_orig_mod', raw_model)
                    moe_suffix = '_moe' if config.use_moe else ''
                    weight_path = f'{args.save_dir}/{args.save_weight}_{config.hidden_size}_{config.n_hidden_layers}{moe_suffix}_{epoch + 1}done.pth'
                    save(raw_model.state_dict(), weight_path)

                # 完整断点（模型 + 双优化器 + scaler）
                get_checkpoint(config, weight=args.save_weight, model=model,
                               save_dir='../checkpoints', epoch=epoch, step=step,
                               muon=muon, adam=adam, scaler=scaler)
            if dist.is_initialized():
                dist.barrier()
        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        # 与主循环一致：裁剪前两个优化器都 unscale
        if muon is not None:
            scaler.unscale_(muon)
        scaler.unscale_(adam)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if muon is not None:
            scaler.step(muon)
        scaler.step(adam)
        scaler.update()
        if muon is not None:
            muon.zero_grad(set_to_none=True)
        adam.zero_grad(set_to_none=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MiniRain SFT")
    parser.add_argument("--save_dir", default="../outputs", type=str, help="模型保存目录")
    parser.add_argument("--load_dir", default="../ready", type=str, help="模型加载目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", default=2, type=int, help="训练轮数")
    parser.add_argument("--batch_size", default=64, type=int, help="批次大小")
    # SFT 数据量大（≥预训练）时按"继续预训练"对待，lr 取预训练峰值的 1/10 左右，
    # 建议扫 1e-4 / 5e-5 / 2e-5；数据量回到常规 SFT 规模时调回 1e-5
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="学习率（AdamW组，Muon组为其30倍）")
    parser.add_argument("--device", default="cuda", type=str, help="训练设备")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="混合精度")
    parser.add_argument("--num_workers", default=4, type=int, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=768, type=int, help='训练的最大截断长度（中文1token≈1.5~1.7字符）')
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--use_loop', default=0, type=int, choices=[0, 1], help="是否使用LOOP架构（0=否，1=是）")
    parser.add_argument('--use_bounce', default=0, type=int, choices=[0, 1], help="是否使用Bounce架构（0=否，1=是）")
    parser.add_argument('--loop_start', type=int, default=0, help="Loop起始层")
    parser.add_argument('--loop_end', type=int, default=0, help="Loop结束层")
    parser.add_argument('--max_loop_iter', type=int, default=1, help="Loop最大迭代次数")
    parser.add_argument("--data_path", type=str, default="../data/sft_t2t.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_compile", default=1, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # 训练环境初始化
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # 训练环境检查
    os.makedirs(args.save_dir, exist_ok=True)
    config = RainConfig(hidden_size=args.hidden_size, n_hidden_layers=args.num_hidden_layers,
                        use_moe=bool(args.use_moe), use_loop=bool(args.use_loop),
                        use_bounce=bool(args.use_bounce), loop_start=args.loop_start,
                        loop_end=args.loop_end, max_loop_iter=args.max_loop_iter)
    print(config)
    # 尝试获取断点
    ckp_data = get_checkpoint(config, weight=args.save_weight,
                              save_dir='../checkpoints') if args.from_resume == 1 else None

    # 混合精度
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # 模型（load_dir 指向 ../ready，init_model 支持匹配 *_ready.pth）
    model, tokenizer = init_model(config, from_weight=args.from_weight,
                                  save_dir=args.load_dir, device=args.device)
    sft_dataset = SFTDataset(args.data_path, tokenizer, max_seq_len=args.max_seq_len)
    train_sampler = DistributedSampler(sft_dataset) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))

    # 参数分组后断言无遗漏、无重叠（embedding/lm_head/1D参数 归 AdamW，与预训练一致）
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    muon_params = get_muon_params(model)
    adam_params = get_adam_params(model)
    assert sum(p.numel() for p in muon_params) + sum(p.numel() for p in adam_params) == model_params, \
        '参数分组有遗漏或重叠，请检查 get_muon_params/get_adam_params'
    Logger(f'[Optimizer] Muon params: {sum(p.numel() for p in muon_params)/1e6:.2f}M, '
           f'AdamW params: {sum(p.numel() for p in adam_params)/1e6:.2f}M')
    muon = optim.Muon(muon_params, lr=args.learning_rate * 30)
    adam = optim.AdamW(adam_params, lr=args.learning_rate)

    # 断点恢复
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        if 'muon' in ckp_data:
            muon.load_state_dict(ckp_data['muon'])
        if 'adam' in ckp_data:
            adam.load_state_dict(ckp_data['adam'])
        if 'scaler' in ckp_data:
            scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        Logger(f'[Checkpoint] 从 epoch={start_epoch}, step={start_step} 续训')

    # 多卡支持与模型编译
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # run
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(sft_dataset)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(sft_dataset, batch_sampler=batch_sampler, num_workers=args.num_workers,
                            pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(model, scaler, muon, adam, epoch, args.epochs,
                        args.learning_rate, args.device, loader, len(loader) + skip, start_step)
        else:
            train_epoch(model, scaler, muon, adam, epoch, args.epochs,
                        args.learning_rate, args.device, loader, len(loader), 0)
    if is_main_process():
        print("MiniRain SFT done!")
        # 保存最终结果（ready 命名规则，save() 内部已确保目录存在）
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        moe_suffix = '_moe' if config.use_moe else ''
        weight_path = f'../ready/{args.save_weight}_{config.hidden_size}_{config.n_hidden_layers}{moe_suffix}_ready.pth'
        save(state_dict, weight_path)
    # clear
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()