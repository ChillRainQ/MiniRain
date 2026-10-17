from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from components.norm import RMSNorm, NormArgs


@dataclass
class MLPArgs:
    hidden_size: int
    intermediate_size: int
    bias: bool
    beta1: int = 4
    beta2: int = 25

class MyMLP(ABC, nn.Module):
    """MLP 抽象基类。"""

    def __init__(self):
        super().__init__()
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


class MLP(MyMLP):
    """标准两层 MLP（GELU 激活）。

    Linear(dim, hidden_dim) → GELU → Linear(hidden_dim, dim)
    """

    def __init__(self, dim: int, hidden_dim: int | None = None, bias: bool = True):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class SwiGLU(MyMLP):
    def __init__(self, args: MLPArgs):
        super().__init__()
        self.gate = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.up = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))

class LowRankMLP(MyMLP):
    ...

class SiTUGLU(MyMLP):
    def __init__(self, args: MLPArgs):
        super().__init__()
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.w1 = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.w2 = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.w3 = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(self.situ(self.w1(x), self.beta1) * self.softcap(self.w2(x), self.beta2))

    def situ(self, x: torch.Tensor, beta: int):
        return self.softcap(x, beta=beta) * F.sigmoid(x)

    def softcap(self, x: torch.Tensor, beta: int):
        return beta * torch.tanh(x / beta)

def build_norm_fn(norm_type, dim, norm_eps, norm_affine):
    pass


class MoTFFN(nn.Module):
    """
    MOT FFN
    """
    def __init__(self, n_modalities: int, args: MLPArgs, norm_args: NormArgs):
        super().__init__()
        self.n_modalities = n_modalities
        self.mlp_t = SwiGLU(args)
        self.mlp_v = SwiGLU(MLPArgs(hidden_size=args.hidden_size,
                                    intermediate_size=(args.intermediate_size // 2),
                                    bias=args.bias))
        self.norm_t = RMSNorm(norm_args)
        self.norm_v = RMSNorm(norm_args)

    def forward(self, x, img_idx=None):
        if img_idx is None:
            return self.mlp_t(self.norm_t(x))

        B, T, D = x.shape
        flat = x.reshape(-1, D)
        txt_mask = torch.ones(B * T, dtype=torch.bool, device=x.device)
        txt_mask[img_idx] = False
        txt_idx = txt_mask.nonzero(as_tuple=True)[0]

        out = None  # 延迟创建：dtype 跟随 expert 输出，兼容 autocast
        for idx, norm, mlp in ((txt_idx, self.norm_t, self.mlp_t),
                               (img_idx, self.norm_v, self.mlp_v)):
            if idx.numel() == 0:
                continue
            y = mlp(norm(flat.index_select(0, idx)))
            if out is None:
                out = y.new_empty(B * T, D)
            out.index_copy_(0, idx, y)
        return out.view(B, T, D)


class Router(nn.Module):
    """
    标准MoE路由器
    """
    def __init__(self, dim: int, num_experts: int, top_k: int, noise_scale: float = 0.0):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.noise_scale = noise_scale
        self.gate = nn.Linear(dim, self.num_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(self.num_experts))

    def forward(self, x: torch.Tensor):
        logits = self.gate(x)
        if self.training and self.noise_scale > 0.0:
            noise = torch.randn_like(logits) * self.noise_scale
            logits += noise
        logits += self.expert_bias
        # weights (token, experts)
        weights = F.softmax(logits, dim=-1)
        # topk_idx (num_token, top_k)
        topk_weights, topk_idx = torch.topk(weights, self.top_k, dim=-1)
        # weight 缩放，让和为1
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return logits, topk_weights, topk_idx

class AuxMoE(nn.Module):
    def __init__(self, dim: int, intermediate_size: int, num_experts: int, top_k: int = 2,
                 num_shared_experts: int = 1, noise_scale: float = 0.0, aux_scale: float = 0.0):
        assert top_k <= num_experts
        super().__init__()
        self.dim = dim
        self.top_k = top_k
        self.aux_scale = aux_scale
        mlp_args = MLPArgs(hidden_size=dim, intermediate_size=intermediate_size, bias=False)
        self.experts = nn.ModuleList([SwiGLU(mlp_args) for _ in range(num_experts)])
        self.num_experts = len(self.experts)
        self.shared_experts = nn.ModuleList([SwiGLU(mlp_args) for _ in range(num_shared_experts)]) if num_shared_experts > 0 else None
        self.router = Router(dim=self.dim, top_k=self.top_k, noise_scale=noise_scale, num_experts= num_experts)

    def forward(self, x: torch.Tensor):
        batch_shape = x.shape
        # 按token变形， token位置由 batch_idx与seq_idx两个坐标确定
        # (b, s, d) -> (b * s, d)
        x = x.view(-1, self.dim)
        num_tokens = x.shape[0]
        routed_out = torch.zeros_like(x)
        share_output = torch.zeros_like(x)
        router_logits, expert_weights, expert_idx = self.router(x)
        # 共享专家
        if self.shared_experts is not None:
            for expert in self.shared_experts:
                share_output += expert(x)
        # 按照 expert 遍历
        for expert_id in range(self.num_experts):
            # 获取当前 expert 的 mask
            mask = (expert_idx == expert_id)
            if not mask.any():
                continue
            # 当前 expert 要处理的 token nonzero就是找到True的位置
            token_indices = mask.nonzero(as_tuple=False)
            selected_tokens = token_indices[:, 0]
            # 获取top_k的位置 这个很重要，要拿gate权值的
            k_positions = token_indices[:, 1]
            expert_input = x[selected_tokens]

            gate = expert_weights[selected_tokens, k_positions].unsqueeze(-1)
            expert_output = self.experts[expert_id](expert_input) * gate
            # 填入已处理的token routed_out (tokens, dim) 得把 token 放到 0的位置
            routed_out.index_add_(0, selected_tokens, expert_output)
        output = routed_out + share_output
        output = output.view(batch_shape)
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        # 辅助损失
        if self.training:
            # p = 1/T * (sum(s_i))
            p = F.softmax(router_logits, dim=-1).mean(dim=0)
            # expert_mask = F.one_hot(expert_idx, self.num_experts).sum(dim=1).float()
            with torch.no_grad():
                expert_mask = torch.zeros_like(p)  # (T, N')
                expert_mask.scatter_(1, expert_idx, 1.0)  # (T, N')
            per_expert_count = expert_mask.sum(dim=0)
            f = per_expert_count * (self.num_experts / (self.top_k * num_tokens))
            aux_loss = self.aux_loss_alpha * (f * p).sum()
        return output, router_logits, aux_loss


class LosslessMoE(nn.Module):
    def __init__(self, dim: int, intermediate_size: int, num_experts: int, top_k: int = 2,
                 num_shared_experts: int = 1, noise_scale: float = 0.0, aux_scale: float = 0.0):
        assert top_k <= num_experts
        super().__init__()
        self.dim = dim
        self.top_k = top_k
        self.aux_scale = aux_scale
        mlp_args = MLPArgs(hidden_size=dim, intermediate_size=intermediate_size, bias=False)
        self.experts = nn.ModuleList([SwiGLU(mlp_args) for _ in range(num_experts)])
        self.num_experts = len(self.experts)
        self.shared_experts = nn.ModuleList(
            [SwiGLU(mlp_args) for _ in range(num_shared_experts)]) if num_shared_experts > 0 else None
        self.router = Router(dim=self.dim, top_k=self.top_k, noise_scale=noise_scale, num_experts=num_experts)

    def forward(self, x: torch.Tensor):
        batch_shape = x.shape
        x = x.view(-1, self.dim)
        num_tokens = x.shape[0]
        shared_output = torch.zeros_like(x)
        routed_output = torch.zeros_like(x)
        router_logits, expert_weights, expert_idx = self.router(x)
        if self.shared_experts is not None:
            for expert in self.shared_experts:
                shared_output += expert(x)
        for expert_id in range(self.num_experts):
            mask = (expert_idx == expert_id)
            if not mask.any():
                continue
            token_indices = mask.nonzero(as_tuple=False)
            selected_tokens = token_indices[:, 0]
            k_position = token_indices[:, 1]
            expert_input = x[selected_tokens]
            gate = expert_weights[selected_tokens, k_position].unsqueeze(-1)
            expert_output = self.experts[expert_id](expert_input) * gate
            routed_output.index_add_(0, selected_tokens, expert_output)
        output = routed_output + shared_output
        output = output.view(batch_shape)
        if self.training:
            with torch.no_grad():
                # c_i: 每个专家实际被分配了多少 token（含 top_k 重复）
                expert_mask = torch.zeros(num_tokens, self.num_experts, device=x.device)
                expert_mask.scatter_(1, expert_idx, 1.0)
                c_i = expert_mask.sum(dim=0)
            c = c_i.new_tensor((self.top_k * num_tokens) / self.num_experts)
            e_i = c - c_i
            self.router.expert_bias = self.router.expert_bias + self.aux_scale * torch.sign(e_i)
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return output, router_logits, aux_loss

