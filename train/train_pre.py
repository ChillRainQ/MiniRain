import os
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from dataset.datasets import PretrainDataset
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
        labels = input_ids.to(device)
        last_step = step
        lr = get_lr(epoch * iters + step, epochs * iters, learning_rate)
        if muon is not None:
            for param_group in muon.param_groups:
                param_group['lr'] = lr * 30

        for param_group in adam.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            if muon is not None:
                # 梯度裁剪前 unscale（只需一个优化器）
                scaler.unscale_(muon)
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
            muon_lr = 0
            if muon is not None:
                muon_lr = muon.param_groups[-1]['lr']
            adam_lr = adam.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(
                f'Epoch:[{epoch + 1}/{epochs}]({step}/{iters}), '
                f'loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, '
                f'lr_muon: {muon_lr:.8f}, lr_adam: {adam_lr:.8f}, eta: {eta_min:.1f}min'
            )

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            moe_suffix = '_moe' if config.use_moe else ''
            done_suffix = f"_{epoch + 1}done" if step == iters else ''
            weight_path = f'{args.save_dir}/{args.save_weight}_{config.hidden_size}_{config.n_hidden_layers}{moe_suffix}{done_suffix}.pth'
            save(state_dict, weight_path)

            # 保存完整检查点（用于续训）
            checkpoint = {
                'model': raw_model.state_dict(),
                'muon': muon.state_dict(),
                'adam': adam.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'step': step,
                'config': config,
            }
            ckpt_dir = '../checkpoints'
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = f'{ckpt_dir}/checkpoint_{args.save_weight}_{epoch}_{step}.pt'
            torch.save(checkpoint, ckpt_path)
            Logger(f'Checkpoint saved to {ckpt_path}')
            # ---------- 清理旧检查点：最多保留 3 个 ----------
            import glob
            pattern = f'{ckpt_dir}/checkpoint_{args.save_weight}_*.pt'
            ckpt_files = glob.glob(pattern)
            if len(ckpt_files) > 3:
                # 按修改时间排序（最新的在前）
                ckpt_files.sort(key=os.path.getmtime, reverse=True)
                # 删除除最新的 3 个之外的所有文件
                for old_file in ckpt_files[3:]:
                    os.remove(old_file)
                    Logger(f'Removed old checkpoint: {old_file}')
            model.train()
            del state_dict
        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        if muon is not None:
            scaler.unscale_(muon)
        else:
            scaler.unscale_(adam)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if muon is not None:
            scaler.step(muon)
        scaler.step(adam)
        scaler.update()
        if muon is not None:
            muon.zero_grad(set_to_none=True)
        adam.zero_grad(set_to_none=True)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../outputs", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--use_loop', default=0, type=int, choices=[0, 1], help="是否使用LOOP架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../data/pretrain_t2t.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniRain-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=1, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()
    
    # 训练环境初始化
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # 训练环境检查
    os.makedirs(args.save_dir, exist_ok=True)
    config = RainConfig(hidden_size=args.hidden_size, n_hidden_layers=args.num_hidden_layers, 
                        use_moe=bool(args.use_moe), use_loop=bool(args.use_loop))
    print(config)

    ckp_data = get_checkpoint(config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # 混合精度 
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # 模型...abs
    model, tokenizer = init_model(config, 
                                  from_weight=args.from_weight, device=args.device)
    pretrain_dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(pretrain_dataset) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))
    muon = optim.Muon(get_muon_params(model), lr=args.learning_rate * 30)
    adam = optim.AdamW(get_adam_params(model), lr=args.learning_rate)
    
    # 断点恢复
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        muon.load_state_dict(ckp_data['muon'])
        adam.load_state_dict(ckp_data['adam'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        
    # 多卡支持与模型编译
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
        
    # run
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(pretrain_dataset)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(pretrain_dataset, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(model, scaler, muon, adam, epoch, args.epochs,
                        args.learning_rate, args.device, loader, len(loader) + skip, start_step)
        else:
            train_epoch(model, scaler, muon, adam, epoch, args.epochs,
                        args.learning_rate, args.device, loader, len(loader), 0)
    print("MiniRain pretrain done!")
    # 保存最终结果
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
