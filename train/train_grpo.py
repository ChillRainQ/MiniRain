import argparse
import math
import os
import re
import sys
# 将项目根目录（当前文件的父目录的父目录）添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DistributedSampler, DataLoader
import torch.nn.functional as F
from architectures.config import RainConfig
from dataset.datasets import RLAIFDataset
from train.rollout_engine import create_rollout_engine
from train.train_util import init_distributed_mode, setup_seed, get_checkpoint, init_model, LMForRewardModel, Logger, \
    is_main_process, SkipBatchSampler, save, LMForRewardModel1


def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    # 连续生成的n元组
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def old_calculate_rewards(prompts, responses, reward_model):
    """
    计算奖励
    :param prompts:
    :param responses:
    :param reward_model:
    :return:
    """
    rewards = torch.zeros(len(responses), device=args.device)
    with torch.no_grad():
        rewards_scores = []
        batch_size = len(prompts)
        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                # prompt与对应的response
                response = responses[response_idx]
                prompt = prompts[i]

                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]

                answer = response
                # role1 鼓励回答在 20-800 字符内
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5

                # role2 截取最终结果，不要思考过程
                if "</think>" in response:
                    think_content, answer_content = response.split("</think>", 1)
                    # role2.1 鼓励思考长度在 20-300 字符内
                    rewards[response_idx] += 0.5 if 20 <= len(think_content.strip()) <= 300 else -0.5
                    # role2.2 鼓励只出现一次 </think>
                    rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25

                    answer = answer_content.strip()
                # role3 重复惩罚
                rewards[response_idx] -= rep_penalty(answer)
                # 打分
                score = reward_model.get_score(messages, answer)
                rewards_scores.append(score)
        reward_model_scores = torch.tensor(rewards_scores, device=args.device)
        rewards += reward_model_scores
    return rewards

def calculate_rewards(prompts, responses, reward_model):
    """
    计算奖励 = 规则奖励（逐条计算） + 评委奖励（批量计算）
    :param prompts:   list[str]，长度 B（每个 prompt 的原始 chat 模板文本）
    :param responses: list[str]，长度 B*num_generations（模型采样出的回答）
    :param reward_model: LMForRewardModel 实例
    :return: Tensor[B*num_gen]，每条回答的总奖励
    """
    rewards = torch.zeros(len(responses), device=args.device)
    with torch.no_grad():
        rewards_scores = []      # 收集待评委打分的数据
        batch_size = len(prompts)

        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                # 从 chat 模板文本中解析出多轮对话历史
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]

                answer = response
                # rule1 鼓励回答在 20-800 字符内（含思考过程的整体长度）
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5

                # rule2 有思考过程时：截取 </think> 之后的最终答案交给评委
                if "</think>" in response:
                    think_content, answer_content = response.split("</think>", 1)
                    # rule2.1 鼓励思考长度在 20-300 字符内
                    rewards[response_idx] += 0.5 if 20 <= len(think_content.strip()) <= 300 else -0.5
                    # rule2.2 鼓励只出现一次 </think>
                    rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25
                    answer = answer_content.strip()

                # rule3 重复惩罚（只罚最终答案部分）
                rewards[response_idx] -= rep_penalty(answer)

                # 打分
                score = reward_model.get_score(messages, answer)
                rewards_scores.append(score)
        reward_model_scores = torch.tensor(rewards_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def grpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model, start_step=0, wandb=None, use_sglang=False):
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)

        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        outputs = rollout_result.output_ids
        completion_ids = rollout_result.completion_ids
        completions = rollout_result.completions
        old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()
        prompt_lens = rollout_result.prompt_lens.to(args.device)
        full_mask = (outputs != tokenizer.pad_token_id).long()
        logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(completion_ids.size(1), device=args.device).unsqueeze(0)

        rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)

        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model

        with autocast_ctx:
            res = model_unwrapped(outputs, attention_mask=full_mask)
            aux_loss = res.aux_loss if config.use_moe else torch.tensor(0.0, device=args.device)

            per_token_logprobs = F.log_softmax(res.logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        with (torch.no_grad()):
            ref_per_token_logps = F.log_softmax(ref_model(outputs, attention_mask=full_mask).logits[:, :-1, :],dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1,
                                                                                                                   logp_pos)
        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-' * 100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    Logger(f"{'=' * 28} [DEBUG] gen[{j}] RESPONSE_BEGIN {'=' * 28}")
                    Logger(completions[idx])
                    Logger(f"{'=' * 29} [DEBUG] gen[{j}] RESPONSE_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{j}] reward={rewards[idx].item():.4f}")
                Logger('=' * 100)

        grouped_rewards = rewards.view(-1, args.num_generations)
        # 求组内均值和标准差
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
        # 计算优势
        advantages = (rewards - mean_r) / (std_r + 1e-4)
        # completion掩码
        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        # 获取eos标记
        is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask
        # 记录eos索引的张量
        eos_idx = torch.full(
            (is_eos.size(0),),  # 形状 [batch_size]
            is_eos.size(1) - 1,  # 填充值 = seq_len - 1
            dtype=torch.long,
            device=args.device
        )
        mask = is_eos.any(dim=1)
        values = is_eos.int().argmax(dim=1)
        eos_idx[mask] = values[mask]

        completions_mask = torch.arange(is_eos.size(1),device=args.device).expand(is_eos.size(0), -1)
        completions_mask = (completions_mask <= eos_idx.unsqueeze(1)) & completion_pad_mask

        # kl计算
        kl_div = ref_per_token_logps - per_token_logprobs
        per_token_kl = torch.exp(kl_div) - kl_div - 1

        ratio = torch.exp(per_token_logprobs - old_per_token_logps)
        if args.loss_type == "cispo":
            """
            cispo只裁剪上界
            """
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logprobs
                               - args.beta * per_token_kl)
        else:
            raise ValueError("只能使用cispo")
        # 求每个序列的有效loss，然后取每个序列的均值，最后再做pingjun
        policy_loss = ((per_token_loss * completions_mask).sum(dim=1) / completions_mask.sum(dim=1).clamp(min=1)).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completions_mask.sum(dim=1).float().mean().item()
            kl_ref_val = ((ref_per_token_logps - per_token_logprobs) * completions_mask).sum().item() / max(
                completions_mask.sum().item(), 1)
            advantages_mean_val = advantages.mean().item()
            advantages_std_val = advantages.std().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, '
                   f'Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, '
                   f'Actor Loss: {policy_loss_val:.4f}, Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if config.use_moe else ''
            done_suffix = f"_{epoch + 1}done" if step == iters else ""
            ckp = f'{args.save_dir}/{args.save_weight}_{config.hidden_size}_{config.n_hidden_layers}{moe_suffix}{done_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            get_checkpoint(config, weight=args.save_weight, model=model, optimizer=optimizer,
                          epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(model)

        del prompt_inputs, outputs, completion_ids, per_token_logprobs, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completions_mask, completion_pad_mask, prompt_lens, logp_pos
        if is_main_process() and args.debug_mode:
            Logger(f"[MEM] step={step}, allocated={torch.cuda.memory_allocated() / 1024 ** 3:.2f}GiB, "
                   f"reserved={torch.cuda.memory_reserved() / 1024 ** 3:.2f}GiB, "
                   f"max_allocated={torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GiB")
    if step > start_step and step % args.accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniRain RLAIF")
    parser.add_argument("--save_dir", default="../outputs", type=str, help="模型保存目录")
    parser.add_argument("--load_dir", default="../ready", type=str, help="模型加载目录")
    parser.add_argument('--save_weight', default='rlaif', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=1152, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=12, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../data/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--num_generations", type=int, default=3, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.1, help="KL惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../reward/Skywork-Reward-V2-Qwen3-1.7B", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniRain-GRPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_grpo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # 训练环境初始化
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # 训练环境检查
    os.makedirs(args.save_dir, exist_ok=True)
    config = RainConfig(hidden_size=args.hidden_size, n_hidden_layers=args.num_hidden_layers,
                        max_seq_len=args.max_seq_len + args.max_gen_len,
                        use_moe=bool(args.use_moe))
    Logger(config)
    # 尝试获取断点
    ckp_data = get_checkpoint(config, weight=args.save_weight,
                              save_dir='../checkpoints') if args.from_resume == 1 else None

    # 混合精度
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # 模型初始化
    model, tokenizer = init_model(config, from_weight=args.from_weight,
                                  save_dir=args.load_dir, device=args.device)
    ref_model, _ = init_model(config, from_weight=args.from_weight,
                              save_dir=args.load_dir, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    wandb = None
    # 数据集初始化
    rlaif_dataset = RLAIFDataset(args.data_path, tokenizer, max_length=config.max_seq_len, thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(rlaif_dataset) if dist.is_initialized() else None

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # 数据迭代
    loader_for_count = DataLoader(rlaif_dataset, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    # 恢复断点
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    # 分布式训练与模型编译
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(model)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    rollout_engine.update_policy(model)

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch);
        indices = torch.randperm(len(rlaif_dataset)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(rlaif_dataset, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step,
                             wandb, use_sglang=(args.rollout_engine == "sglang"))
        else:
            grpo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb,
                             use_sglang=(args.rollout_engine == "sglang"))
    if is_main_process():
        print("MiniRain GRPO done!")
        # 保存最终结果（ready 命名规则，save() 内部已确保目录存在）
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        moe_suffix = '_moe' if config.use_moe else ''
        weight_path = f'../ready/{args.save_weight}_{config.hidden_size}_{config.n_hidden_layers}{moe_suffix}_ready.pth'
        save(state_dict, weight_path)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

