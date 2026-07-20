from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from architectures.rain import MiniRainForCausalLM
import math
import csv
import glob
import os
import numpy as np
import torch
import random
import torch.distributed as dist




def save(state_dict, weight_path):
    # 修复：保存前确保目录存在（否则 ../ready/ 等未创建目录会直接崩）
    dir_name = os.path.dirname(weight_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    torch.save({k: v.half().cpu() for k, v in state_dict.items()}, weight_path)

def get_lr(current_step, total_steps, lr):
    """
    余弦退火
    """
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))

def setup_seed(seed: int):
    """
    种子设置
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def Logger(content):
    if is_main_process():
        print(content)

def get_model_params(model, config) :
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')

def init_model(config, from_weight='pretrain', tokenizer_path='../tokenizers', save_dir='../outputs', device='cuda'):
    """
    初始化模型
    """
    device = device if torch.cuda.is_available() else 'cpu'
    model = MiniRainForCausalLM(config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if from_weight != "none":
        moe_suffix = '_moe' if config.use_moe else ''
        # 修复：训练脚本导出的权重文件名含层数和 done 后缀，
        # 如 pretrain_768_8_2done.pth，用 glob 匹配；找不到再回退旧命名
        pattern = f'{save_dir}/{from_weight}_{config.hidden_size}_{config.n_hidden_layers}{moe_suffix}*done.pth'
        # 按修改时间取最新的 done 文件（字典序在 epoch >= 10 时会排错，如 10done 排在 2done 前）
        candidates = sorted(glob.glob(pattern), key=os.path.getmtime)
        if candidates:
            weight_path = candidates[-1]
        else:
            weight_path = f'{save_dir}/{from_weight}_{config.hidden_size}{moe_suffix}.pth'
        Logger(f'[init_model] 加载权重: {weight_path}')
        weights = torch.load(weight_path, map_location=device)
        # 修复：打印 missing/unexpected，避免 strict=False 静默吞掉权重不匹配
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if missing or unexpected:
            Logger(f'[init_model] 权重不完全匹配: missing={missing}, unexpected={unexpected}')

    get_model_params(model, config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


def get_checkpoint(lm_config, weight='full_sft', model=None, muon=None, adam=None,
                   epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        if muon is not None:
            resume_data['muon'] = muon.state_dict()
        if adam is not None:
            resume_data['adam'] = adam.state_dict()
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def get_muon_params(model):
    """获取适用于 Muon 优化器的参数：隐藏层 ndim >= 2 的权重矩阵。
    修复：embedding 和 lm_head 虽然也是 2D，但必须交给 AdamW（Muon 的要求）。"""
    return [p for n, p in model.named_parameters()
            if p.requires_grad and p.ndim >= 2
            and not any(k in n for k in ('embed', 'lm_head', 'head'))]


def get_adam_params(model):
    """获取适用于 AdamW 优化器的参数：bias、norm scale 等 ndim < 2 的参数，
    以及 embedding / lm_head（修复：这两类从 Muon 划归 AdamW）。"""
    return [p for n, p in model.named_parameters()
            if p.requires_grad
            and (p.ndim < 2 or any(k in n for k in ('embed', 'lm_head', 'head')))]


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)


class ScalingLawLogger:
    """
    为 Chinchilla-style Scaling Law 分析保存训练日志。
    自动将多模型数据合并到标准格式 (loss.csv, tokens.csv)。
    列名格式: loss-P{参数量}M-{模型名}，与 Chinchilla 分析脚本兼容。
    """
    def __init__(self, log_dir: str, model_params: int, model_name: str,
                 batch_size: int, seq_len: int, world_size: int = 1):
        self.log_dir = log_dir
        self.model_params = model_params
        self.model_name = model_name
        # 每步处理的全局 token 数（数据并行下乘以 world_size）
        self.tokens_per_step = batch_size * seq_len * world_size
        self.records = []  # [(step, loss, tokens), ...]
        os.makedirs(log_dir, exist_ok=True)

        # 列名必须包含 -P{params}M- 以便 Chinchilla 脚本解析
        params_m = model_params / 1e6
        self.col_name = f"loss-P{params_m:.2f}M-{model_name}"

    def log(self, step: int, loss: float):
        """在训练循环中调用，记录当前全局 step 的 loss"""
        tokens = step * self.tokens_per_step
        self.records.append((step, loss, tokens))

    def save(self):
        """训练结束后调用，合并到标准 CSV 格式"""
        if not self.records or not is_main_process():
            return

        loss_path = os.path.join(self.log_dir, 'loss.csv')
        tokens_path = os.path.join(self.log_dir, 'tokens.csv')

        # 当前模型数据: step -> (loss, tokens)
        step_data = {step: (loss, tokens) for step, loss, tokens in self.records}

        def read_csv(path):
            if not os.path.exists(path):
                return ['step'], {}
            with open(path, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader)
                data = {}
                for row in reader:
                    if not row or not row[0].strip():
                        continue
                    try:
                        step = int(row[0])
                        data[step] = row[1:]
                    except ValueError:
                        continue
                return header, data

        def merge_csv(path, fmt):
            """合并当前模型数据到 CSV。
            修复：按列名定位索引写入，不再写死最后一列——
            否则重训同名模型且该列不在末尾时，会覆盖别的模型的数据。"""
            header, data = read_csv(path)
            if self.col_name not in header:
                header.append(self.col_name)
            col_idx = header.index(self.col_name)
            all_steps = sorted(set(data.keys()) | set(step_data.keys()))

            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for step in all_steps:
                    row = [''] * len(header)
                    row[0] = step
                    if step in data:
                        existing = data[step]
                        for i, v in enumerate(existing[:len(header) - 1]):
                            row[i + 1] = v
                    if step in step_data:
                        row[col_idx] = fmt(step_data[step])
                    writer.writerow(row)

        merge_csv(loss_path, lambda v: f"{v[0]:.6f}")
        merge_csv(tokens_path, lambda v: f"{v[1]:.0f}")

        Logger(f'[ScalingLaw] 已保存 {self.col_name} 到 {self.log_dir}')
        Logger(f'[ScalingLaw] 共 {len(self.records)} 条记录，'
               f'参数量: {self.model_params/1e6:.2f}M, '
               f'每步tokens: {self.tokens_per_step}')