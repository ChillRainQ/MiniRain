from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class MLPArgs:
    hidden_size: int
    intermediate_size: int
    bias: bool

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
    """SwiGLU 门控 MLP。

    与标准 MLP 不同，SwiGLU 将输入投影到三个权重空间，其中两个经过
    SiLU 激活后逐元素相乘。hidden_dim 通常取 8/3 * dim（约 2.67x），
    而非标准 MLP 的 4x，以保证参数量相当。

    SwiGLU(x) = (xW_gate ⊙ SiLU(xW_up)) W_down
    """

    def __init__(self, args: MLPArgs):
        super().__init__()
        # hidden_dim = args.hidden_size or int(8 * args.hidden_size / 3)
        self.gate = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.up = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Router(nn.Module):
    """MoE 路由器，将 token 路由到 top-k 个专家。"""

    def __init__(self, dim: int, num_experts: int, top_k: int, noise_scale: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_scale = noise_scale
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        """返回 (router_logits, expert_weights, expert_indices)。"""
        logits = self.gate(x)  # (..., num_experts)

        if self.training and self.noise_scale > 0:
            noise = torch.randn_like(logits) * self.noise_scale
            logits = logits + noise

        weights = F.softmax(logits, dim=-1)
        topk_weights, topk_idx = torch.topk(weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_idx = cast(torch.LongTensor, topk_idx)

        return logits, topk_weights, topk_idx


class MoE(nn.Module):
    """Mixture of Experts。

    将输入通过路由器分发到 top-k 个专家，输出为各专家输出的加权和。
    可选地包含一个共享专家（shared expert），处理所有 token，再将结果与
    路由专家的输出相加（DeepSeek-MoE 风格）。

    支持负载均衡辅助损失（auxiliary loss）以鼓励 token 均匀分配到各专家。

    注意：MoE 返回 (output, aux_loss)，接口不同于 MyMLP（后者只返回 Tensor），
    因此 MoE 不能直接作为 LlamaDecoderBlock 的 mlp 参数。在 block 中包装即可。

    用法：
        moe = MoE(
            dim=512,
            expert_fn=lambda d: MLP(d, hidden_dim=1024),
            num_experts=8,
            top_k=2,
        )
        out, aux_loss = moe(x)
    """

    def __init__(
        self,
        dim: int,
        expert_fn: Callable[[int], nn.Module],
        num_experts: int,
        top_k: int = 2,
        shared_expert: Callable[[int], nn.Module] | None = None,
        noise_scale: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_experts, "top_k 不能超过专家数"
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = Router(dim, num_experts, top_k, noise_scale)
        self.experts = nn.ModuleList([expert_fn(dim) for _ in range(num_experts)])

        self.shared = None
        if shared_expert is not None:
            self.shared = shared_expert(dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        x_flat: torch.Tensor = x.view(-1, self.dim)  # (B*L, D)

        router_logits: torch.Tensor
        expert_weights: torch.Tensor
        expert_idx: torch.LongTensor
        router_logits, expert_weights, expert_idx = self.router(x_flat)

        out: torch.Tensor = torch.zeros_like(x_flat)
        for expert_id in range(self.num_experts):
            mask = expert_idx == expert_id  # (B*L, top_k)
            if not torch.any(mask).item():
                continue
            rows, cols = mask.nonzero(as_tuple=True)
            selected_input: torch.Tensor = x_flat[rows]
            selected_weights: torch.Tensor = expert_weights[rows, cols]
            expert_out: torch.Tensor = self.experts[expert_id](selected_input)
            out.index_add_(0, rows, expert_out * selected_weights.unsqueeze(-1))

        # 负载均衡损失
        router_probs: torch.Tensor = F.softmax(router_logits, dim=-1)

        expert_mask: torch.Tensor = torch.zeros_like(router_probs)
        expert_mask.scatter_(1, expert_idx, 1.0)  # (B*L, num_experts)

        frac_selected: torch.Tensor = expert_mask.float().mean(dim=0)
        avg_prob: torch.Tensor = router_probs.mean(dim=0)
        aux_loss: torch.Tensor = (frac_selected * avg_prob).sum() * self.num_experts

        if self.shared is not None:
            out = out + cast(torch.Tensor, self.shared(x_flat))

        out = out.view(orig_shape)
        return out, aux_loss
