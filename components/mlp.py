from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import PreTrainedConfig

from architectures.config import RainConfig
from components.norm import RMSNorm, NormArgs, TorchRMSNorm
try:
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    _HAS_LIGER_SWIGLU = True
except ImportError:
    _HAS_LIGER_SWIGLU = False

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


class FastWeightProgrammer(nn.Module):
    """学习 prompt 条件的低秩更新规则：ΔW = scale * Σᵣ αᵣ uᵣvᵣᵀ。

    u/v 是可学习的更新基，controller 只生成每个基的混合系数，避免直接回归完整权重。
    最后一层零初始化使初始 ΔW=0，启用 FWP 不会改变未训练模型的行为。
    """
    names = ("gate", "up", "down")

    def __init__(self, linears: dict[str, nn.Linear], prompt_size: int, rank: int,
                 hidden_size: int, update_scale: float):
        super().__init__()
        self.update_scale = update_scale
        self.controller = nn.Sequential(
            nn.Linear(prompt_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, 3 * rank))
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)
        self.u = nn.ParameterDict()
        self.v = nn.ParameterDict()
        for name, linear in linears.items():
            out_features, in_features = linear.weight.shape
            self.u[name] = nn.Parameter(torch.empty(rank, out_features))
            self.v[name] = nn.Parameter(torch.empty(rank, in_features))
            nn.init.normal_(self.u[name], std=0.02)
            nn.init.normal_(self.v[name], std=0.02)
        # 仅缓存每个样本的低秩系数，不进 state_dict；首轮创建，多轮累加，新会话清空
        self._alpha_cache = None

    def forward(self, prompt_context: torch.Tensor):
        return self.controller(prompt_context).chunk(3, dim=-1)

    def apply(self, name, x, linear, alpha, batch_idx=None):
        """y = Wx + scale·Σᵣαᵣuᵣ(vᵣᵀx)，不构造完整 ΔW 或逐 token 权重。"""
        if batch_idx is not None:
            alpha = alpha.index_select(0, batch_idx)
        if x.dim() == 2:
            projected = torch.einsum("ni,ri->nr", x, self.v[name])
            update = torch.einsum("nr,ro->no", projected * alpha, self.u[name])
        else:
            projected = torch.einsum("bti,ri->btr", x, self.v[name])
            update = torch.einsum("btr,ro->bto", projected * alpha.unsqueeze(1), self.u[name])
        return linear(x) + update * self.update_scale

    def cache(self, coefficients):
        self._alpha_cache = tuple(alpha.detach() for alpha in coefficients)

    def clear(self):
        self._alpha_cache = None

class SwiGLU(MyMLP):
    """纯 SwiGLU 快路径：不内置 FWP，self.fwp 恒为 None（FWP 默认关）。

    需要 FWP 用 SwiGLUWithFWP。这里保留 fwp 属性，是为了让“未接入 FWP”可被显式断言，
    而不是靠 hasattr 判断。
    """

    def __init__(self, args: MLPArgs, config: PreTrainedConfig | None = None, fwp: bool = False):
        super().__init__()
        self.gate = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.up = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.fwp = None  # 默认关：本类不构建 FWP

    def forward(self, x: torch.Tensor, fwp_context: torch.Tensor | None = None,
                batch_idx: torch.Tensor | None = None, fwp_accumulate: bool = False) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))

class SwiGLUWithFWP(SwiGLU):
    """opt-in 的 FWP 版本。

    继承 SwiGLU，故 isinstance(m, SwiGLU) 仍成立，block 里的分支无需区分两种类。
    两层开关：模块级 fwp=True 且 config.fwp_rank > 0，任一不满足即退化为普通 SwiGLU。
    """

    def __init__(self, args: MLPArgs, config: PreTrainedConfig | None = None, fwp: bool = False):
        super().__init__(args, config, fwp)
        # FWP 默认关闭：需显式传 fwp=True 才创建，开启后 rank 仍由 config.fwp_rank 决定
        fwp_rank = int(getattr(config, "fwp_rank", 0)) if fwp else 0
        self.fwp = FastWeightProgrammer(
            {"gate": self.gate, "up": self.up, "down": self.down},
            prompt_size=args.hidden_size, rank=fwp_rank,
            hidden_size=int(getattr(config, "fwp_hidden_size", 128)),
            update_scale=float(getattr(config, "fwp_update_scale", 1.0)),
        ) if fwp_rank > 0 else None

    def forward(self, x: torch.Tensor, fwp_context: torch.Tensor | None = None,
                batch_idx: torch.Tensor | None = None, fwp_accumulate: bool = False) -> torch.Tensor:
        coefficients = None
        if self.fwp is not None:
            if fwp_context is not None:
                coefficients = self.fwp(fwp_context)
                if fwp_accumulate and self.fwp._alpha_cache is not None:
                    coefficients = tuple(old + new for old, new in zip(self.fwp._alpha_cache, coefficients))
                if not self.training:
                    self.fwp.cache(coefficients)
            else:
                coefficients = self.fwp._alpha_cache
            if coefficients is None:
                raise RuntimeError("FWP 缺少 prompt 更新：请先执行 prefill 或传入 fwp_context")
        # Some Liger/Triton versions fail mutation analysis under torch.compile
        # (e.g. constexpr gate_multiplier.to()). Let Inductor fuse native ops.
        if (coefficients is None and _HAS_LIGER_SWIGLU and x.is_cuda
                and not torch.compiler.is_compiling()):
            return self.down(LigerSiLUMulFunction.apply(self.gate(x), self.up(x)))
        if coefficients is None:
            return self.down(F.silu(self.gate(x)) * self.up(x))
        gate_alpha, up_alpha, down_alpha = coefficients
        gate = self.fwp.apply("gate", x, self.gate, gate_alpha, batch_idx)
        up = self.fwp.apply("up", x, self.up, up_alpha, batch_idx)
        return self.fwp.apply("down", F.silu(gate) * up, self.down, down_alpha, batch_idx)

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
    def __init__(self, n_modalities: int, args: MLPArgs, norm_args: NormArgs, config: PreTrainedConfig):
        super().__init__()
        self.n_modalities = n_modalities
        # MoT 文本/视觉专家均使用普通 SwiGLU，不接入 FWP。
        self.mlp_t = SwiGLU(args)
        self.mlp_v = SwiGLU(MLPArgs(hidden_size=args.hidden_size,
                                    intermediate_size=int(args.intermediate_size // config.vision_ffn_scale),
                                    bias=args.bias))
        self.norm_t = TorchRMSNorm(norm_args)
        self.norm_v = TorchRMSNorm(norm_args)

    def forward(self, x, img_idx=None):
        # 纯文本快速路径（decode 常态）：无图像 token 时直接走文本专家，省掉 nonzero/分拣
        if img_idx is None or img_idx.numel() == 0:
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
                total_tokens = c_i.new_tensor(float(num_tokens))
                if dist.is_available() and dist.is_initialized():
                    # 所有 rank 必须依据全局 token 分布更新同一份 bias；局部更新会让路由漂移。
                    dist.all_reduce(c_i, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
                target = self.top_k * total_tokens / self.num_experts
                self.router.expert_bias.add_(self.aux_scale * torch.sign(target - c_i))
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return output, router_logits, aux_loss

