import sys
import os
# 将项目根目录（当前文件的父目录的父目录）添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from dataset.datasets import PretrainDataset
from train.train_util import init_model, Logger, ScalingLawLogger, swanlab_login
from train.train_util import is_main_process
from contextlib import nullcontext
import argparse
import time
import torch
import torch.distributed as dist
import swanlab
from torch import nn
from architectures.config import RainConfig
from train.train_util import get_lr, init_distributed_mode, setup_seed, get_adamw_params, get_muon_params, SkipBatchSampler, save, MuonWithAdamW
from train.ckpt import save_exp, save_ckpt, save_state_dict, load_ckpt


def train_epoch(model: nn.Module | DistributedDataParallel, scaler: GradScaler, optimizer: MuonWithAdamW,
                epoch: int, epochs: int, learning_rate: float, device: str, loader, iters, start_step: int = 0,
                wandb=None, logger=None):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        last_step = step
        lr = get_lr(epoch * iters + step, epochs * iters, learning_rate)
        # Muon 组用 30 倍学习率, AdamW 组用基础学习率
        for param_group in optimizer.muon.param_groups:
            param_group['lr'] = lr * 30
        for param_group in optimizer.adamw.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            # 修复：aux_loss 可能为 None（非 MoE 时），与原日志处的判空保持一致
            aux_loss = res.aux_loss if res.aux_loss is not None else 0.0
            loss = (res.loss + aux_loss) / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            # 统一优化器: 先 unscale 再裁剪, 最后 step + 清梯度
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            muon_lr = optimizer.muon.param_groups[-1]['lr']
            adam_lr = optimizer.adamw.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(
                f'Epoch:[{epoch + 1}/{epochs}]({step}/{iters}), '
                f'loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, '
                f'lr_muon: {muon_lr:.8f}, lr_adam: {adam_lr:.8f}, eta: {eta_min:.1f}min'
            )
            if wandb and is_main_process():
                swanlab.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss,
                                 "lr_muon": muon_lr, "lr_adam": adam_lr, "epoch_time": eta_min})

            # ========== 记录 Scaling Law 数据（仅主进程） ==========
            if logger is not None and is_main_process():
                # 修复：传全局步数，否则第二个 epoch 的 step 从头开始，token 曲线回退
                logger.log(epoch * iters + step, current_loss)

        # ========== 断点保存：统一走 train/ckpt.py ==========
        if step % args.save_interval == 0 or step == iters:
            model.eval()
            if is_main_process():
                if step == iters:
                    # 每个 epoch 结束另存一份纯权重, 便于直接部署
                    save_state_dict(model, exp_name, f'{prefix}_{epoch + 1}done.pth')
                # 保存完整断点（模型 + 优化器 + scaler）
                save_ckpt(exp_name, f'{prefix}_resume.pth', model=model, epoch=epoch, step=step,
                          optimizer=optimizer, scaler=scaler)
            # 等其他 rank 同步后再继续，避免写盘期间有 rank 提前进入下一步
            if dist.is_initialized():
                dist.barrier()
            model.train()
        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        # 与主循环一致: 收尾处理不足一个累积周期的残差梯度
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniRain Pretraining")
    parser.add_argument("--save_dir", type=str, default="../outputs", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument('--exp_name', type=str, default='', help="实验名, 对应输出目录 outputs/<exp_name>, 为空则自动生成")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Muon 权重衰减")
    parser.add_argument("--momentum", type=float, default=0.95, help="Muon 动量")
    parser.add_argument("--nesterov", type=int, default=1, choices=[0, 1], help="Muon 是否使用 Nesterov")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--use_loop', default=0, type=int, choices=[0, 1], help="是否使用LOOP架构（0=否，1=是）")
    parser.add_argument('--use_bounce', default=0, type=int, choices=[0, 1], help="是否使用Bounce架构（0=否，1=是）")
    # 修复：三个 loop 参数补默认值，否则 int(None) 启动即崩
    parser.add_argument('--loop_start', type=int, default=0, help="Loop起始层")
    parser.add_argument('--loop_end', type=int, default=0, help="Loop结束层")
    parser.add_argument('--max_loop_iter', type=int, default=1, help="Loop最大迭代次数")
    parser.add_argument("--data_path", type=str, default="../data/pretrain_t2t.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniRain-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=1, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--scaling_log_dir", type=str, default="../train_logs", help="Scaling Law 分析日志保存目录")
    parser.add_argument("--use_block_attn_res", action="store_true", help="是否使用块注意力残差")
    parser.add_argument("--use_full_attn_res", action="store_true", help="是否使用全注意力残差")
    parser.add_argument("--block_size", default=0, type=int, help="块大小")
    parser.add_argument("--dropout", default=0, type=float, help="dropout概率")
    args = parser.parse_args()

    # 训练环境初始化
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # 训练环境检查
    os.makedirs(args.save_dir, exist_ok=True)
    # 实验目录名(外界维护), 实验产物统一输出到 outputs/<exp_name>
    exp_name = args.exp_name or f"{args.save_weight}_{args.hidden_size}_{args.num_hidden_layers}" + ('_moe' if args.use_moe else '')
    if is_main_process():
        save_exp(exp_name)
        Logger(f'[Experiment] 实验目录: {exp_name}')
    config = RainConfig(hidden_size=int(args.hidden_size), n_hidden_layers=int(args.num_hidden_layers),
                        use_moe=bool(args.use_moe), use_loop=bool(args.use_loop),
                        use_bounce=bool(args.use_bounce), loop_start=int(args.loop_start), loop_end=int(args.loop_end),
                        max_loop_iter=int(args.max_loop_iter), block_attn_res=bool(args.use_block_attn_res),
                        full_attn_res=bool(args.use_full_attn_res), block_size=int(args.block_size),
                        dropout=float(args.dropout))
    Logger(config)
    if args.use_wandb and is_main_process():
        swanlab_login()
        swanlab.init(
            # 设置将记录此次实验的项目信息
            project=args.wandb_project,
            workspace="Chill_Rain",
            # 跟踪超参数和实验元数据
            config=config.to_dict()
        )
    moe_suffix = '_moe' if config.use_moe else ''
    prefix = f'{args.save_weight}_{config.hidden_size}_{config.n_hidden_layers}{moe_suffix}'
    ckp_data = load_ckpt(exp_name, f'{prefix}_resume.pth') if args.from_resume == 1 else None

    # 混合精度
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # 模型...
    model, tokenizer = init_model(config,
                                  from_weight=args.from_weight, device=args.device)

    # ========== 计算参数量并初始化 Scaling Law Logger ==========
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f'[ScalingLaw] 模型参数量: {model_params/1e6:.2f}M, '
           f'全局batch_size: {args.batch_size * world_size}, '
           f'每步tokens: {args.batch_size * world_size * args.max_seq_len}')

    scaling_logger = ScalingLawLogger(
        log_dir=args.scaling_log_dir,
        model_params=model_params,
        model_name=args.save_weight,
        batch_size=args.batch_size,
        seq_len=args.max_seq_len,
        world_size=world_size
    )

    pretrain_dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(pretrain_dataset) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))

    # 修复：参数分组后断言无遗漏、无重叠（embedding/lm_head 归 AdamW）
    muon_params = get_muon_params(model)
    adam_params = get_adamw_params(model)
    assert sum(p.numel() for p in muon_params) + sum(p.numel() for p in adam_params) == model_params, \
        '参数分组有遗漏或重叠，请检查 get_muon_params/get_adam_params'
    Logger(f'[Optimizer] Muon params: {sum(p.numel() for p in muon_params)/1e6:.2f}M, '
           f'AdamW params: {sum(p.numel() for p in adam_params)/1e6:.2f}M')
    # 统一优化器: Muon(隐藏层矩阵) + AdamW(embed/lm_head/bias/norm)
    optimizer = MuonWithAdamW(model, lr=args.learning_rate, weight_decay=args.weight_decay,
                              momentum=args.momentum, nesterov=bool(args.nesterov))

    # 断点恢复
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        if 'optimizer' in ckp_data:
            optimizer.load_state_dict(ckp_data['optimizer'])
        else:
            # 兼容旧格式检查点: muon / adam 分开保存
            if 'muon' in ckp_data:
                optimizer.muon.load_state_dict(ckp_data['muon'])
            if 'adam' in ckp_data:
                optimizer.adamw.load_state_dict(ckp_data['adam'])
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
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    # run
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(pretrain_dataset)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(pretrain_dataset, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(model, scaler, optimizer, epoch, args.epochs,
                        args.learning_rate, args.device, loader, len(loader) + skip, start_step,
                        logger=scaling_logger, wandb=args.use_wandb)
        else:
            train_epoch(model, scaler, optimizer, epoch, args.epochs,
                        args.learning_rate, args.device, loader, len(loader), 0,
                        logger=scaling_logger, wandb=args.use_wandb)
    if is_main_process():
        scaling_logger.save()
        print("MiniRain pretrain done!")
        # 保存最终结果（save() 内部已确保 ../ready 目录存在）
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        moe_suffix = '_moe' if config.use_moe else ''
        weight_path = f'../ready/{args.save_weight}_{config.hidden_size}_{config.n_hidden_layers}{moe_suffix}_ready.pth'
        save(state_dict, weight_path)
        swanlab.finish()
    # clear
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()