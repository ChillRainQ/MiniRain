import torch
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedConfig

from components.mlp import SwiGLU, FusedSwiGLU, LosslessMoE, MoTFFN, AdaptiveSwiGLU
from components.mlp import MLPArgs
from components.norm import NormArgs
from components.attention import AttentionArgs, MoTGroupQueryAttention
from components.attention import GroupQueryAttention
import torch.nn as nn

from components.norm import RMSNorm, TorchRMSNorm
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


# CUDA FP32 BlockAttnRes kernels; unsupported cases use the reference below.
if triton is not None:
    @triton.jit
    def _forward(V, W, Q, Y, R: tl.constexpr, D: tl.constexpr, N: tl.constexpr,
                 EPS: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr):
        row = tl.program_id(0)
        d = tl.arange(0, BD)
        n = tl.arange(0, BN)
        x = tl.load(V + (n[:, None] * R + row) * D + d[None, :],
                    (n[:, None] < N) & (d[None, :] < D), 0).to(tl.float32)
        w = tl.load(W + d, d < D, 0).to(tl.float32)
        q = tl.load(Q + d, d < D, 0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x*x, 1) / D + EPS)
        logits = tl.sum(x * w[None, :] * q[None, :], 1) * inv
        logits = tl.where(n < N, logits, -float('inf'))
        p = tl.exp(logits - tl.max(logits, 0))
        p = p / tl.sum(p, 0)
        y = tl.sum(p[:, None] * x, 0)
        tl.store(Y + row * D + d, y, d < D)


    @triton.jit
    def _backward(V, W, Q, DY, DX, DP, R: tl.constexpr, D: tl.constexpr,
                  N: tl.constexpr, EPS: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr):
        row = tl.program_id(0)
        d = tl.arange(0, BD)
        n = tl.arange(0, BN)
        mask = (n[:, None] < N) & (d[None, :] < D)
        x = tl.load(V + (n[:, None] * R + row) * D + d[None, :], mask, 0).to(tl.float32)
        w = tl.load(W + d, d < D, 0).to(tl.float32)
        q = tl.load(Q + d, d < D, 0).to(tl.float32)
        g = tl.load(DY + row * D + d, d < D, 0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x*x, 1) / D + EPS)
        dot = tl.sum(x * w[None, :] * q[None, :], 1)
        logits = tl.where(n < N, dot * inv, -float('inf'))
        p = tl.exp(logits - tl.max(logits, 0))
        p = p / tl.sum(p, 0)
        dp = tl.sum(x * g[None, :], 1)
        ds = p * (dp - tl.sum(p * dp, 0))
        dx = p[:, None] * g[None, :] + ds[:, None] * (
            inv[:, None] * w[None, :] * q[None, :] -
            (dot * inv * inv * inv / D)[:, None] * x)
        tl.store(DX + (n[:, None] * R + row) * D + d[None, :], dx, mask)
        dwq = tl.sum((ds * inv)[:, None] * x, 0)
        tl.store(DP + row * D + d, dwq, d < D)


    class _FusedAttnRes(torch.autograd.Function):
        @staticmethod
        def forward(ctx, values, weight, query, eps):
            n, rows, dim = values.shape
            output = torch.empty((rows, dim), device=values.device, dtype=values.dtype)
            _forward[(rows,)](values, weight, query, output, rows, dim, n, eps,
                              triton.next_power_of_2(dim), triton.next_power_of_2(n), num_warps=4)
            ctx.save_for_backward(values, weight, query)
            ctx.eps = eps
            return output

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_output):
            values, weight, query = ctx.saved_tensors
            n, rows, dim = values.shape
            dx = torch.empty_like(values)
            partial = torch.empty((rows, dim), device=values.device, dtype=torch.float32)
            _backward[(rows,)](values, weight, query, grad_output.contiguous(), dx, partial,
                               rows, dim, n, ctx.eps, triton.next_power_of_2(dim),
                               triton.next_power_of_2(n), num_warps=4)
            # Reduce per-token parameter gradients without atomic contention.
            combined = partial.sum(0)
            return dx, combined * query, combined * weight, None


    def fused_attn_res(values, weight, query, eps):
        shape = values.shape[1:]
        result = _FusedAttnRes.apply(values.reshape(values.shape[0], -1, values.shape[-1]),
                                    weight, query, eps)
        return result.reshape(shape)

else:
    fused_attn_res = None

class FullAttnRes(nn.Module):
    """
    全注意力残差
    """
    def __init__(self, config: PreTrainedConfig):
        super().__init__()
        self.attn_res_norm = TorchRMSNorm(NormArgs(dim=config.hidden_size, eps=config.rms_norm_eps))
        self.attn_res_query = nn.Parameter(torch.zeros(config.hidden_size))

    def forward(self, history_input: list[torch.Tensor]):
        # 堆叠历史输入 (N, B, T, D)
        V = torch.stack(history_input, dim=0)
        K = self.attn_res_norm(V)
        # KQ -> logits (N, B, T, D) * (1, 1, 1, D) -> (N, B, T)
        logits = (K * self.attn_res_query).sum(dim=-1)
        alpha = torch.softmax(logits, dim=0)
        # (N, B, T, 1) * (N, B, T, D) -> (B, T, D)
        output = (alpha.unsqueeze(-1) * V).sum(dim=0)
        return output

class BlockAttnRes(nn.Module):
    """
    块注意力残差
    """
    def __init__(self, config: PreTrainedConfig):
        super().__init__()
        self.attn_res_norm = TorchRMSNorm(NormArgs(dim=config.hidden_size, eps=config.rms_norm_eps))
        self.attn_res_query = nn.Parameter(torch.zeros(config.hidden_size))
        # 训练期可选梯度检查点（config.attn_res_checkpoint）：反向重算聚合，
        # 省掉 stack 缓冲的持有；推理/关闭时完全不生效
        self.use_checkpoint = bool(getattr(config, "attn_res_checkpoint", False))

    def _aggregate(self, *hs):
        """堆叠聚合：一次 norm + 一次归约。

        单候选也走同一路径，不早退——否则 q/norm 拿不到 0 梯度而变成 None，
        DDP 会判为未使用参数。
        """
        V = torch.stack(hs, dim=0)                                        # [N, B, T, D]
        if (fused_attn_res is not None and V.is_cuda and V.dtype == torch.float32
                and self.attn_res_norm.weight.dtype == torch.float32
                and self.attn_res_query.dtype == torch.float32
                and V.shape[0] <= 16 and V.shape[-1] <= 2048):
            return fused_attn_res(V, self.attn_res_norm.weight, self.attn_res_query,
                                  self.attn_res_norm.eps)
        logits = (self.attn_res_norm(V) * self.attn_res_query).sum(dim=-1)  # [N, B, T]
        alpha = torch.softmax(logits.float(), dim=0).to(V.dtype)
        return (alpha.unsqueeze(-1) * V).sum(dim=0)

    def forward(self, inter_block_history: list[torch.Tensor],
                partial_block: torch.Tensor):
        """
        :param inter_block_history: 块间历史表示（每一个块的最后一个output）
        :param partial_block: 块内历史表示
        :return:
        """
        hs = inter_block_history + [partial_block]
        # 检查点只省「为反向保留的中间量」：输入 hs 本身就是既有激活，不额外占用显存
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self._aggregate, *hs, use_reentrant=False)
        return self._aggregate(*hs)

    def forward_old(self, inter_block_history: list[torch.Tensor],
                partial_block: torch.Tensor):
        """
        :param inter_block_history: 块间历史表示（每一个块的最后一个output）
        :param partial_block: 块内历史表示
        :return:
        """
        # (N, B, T, D)
        V = torch.stack(inter_block_history + [partial_block], dim=0)
        K = self.attn_res_norm(V)
        # KQ -> logits (N, B, T, D) * (1, 1, 1, D) -> (N, B, T)
        logits = (K * self.attn_res_query).sum(dim=-1)
        alpha = torch.softmax(logits, dim=0)
        # (N, B, T, 1) * (N, B, T, D) -> (B, T, D)
        output = (alpha.unsqueeze(-1) * V).sum(dim=0)
        return output

    def forward_loop(self, inter_block_history: list[torch.Tensor],
                     partial_block: torch.Tensor):
        """优化前的逐候选循环实现，仅保留作对照/回退，勿用于生产路径。

        每个候选各做一次 norm + 一次 GEMV，decode 时 kernel launch 数随候选数线性增长。
        """
        hs = inter_block_history + [partial_block]
        logits = torch.stack([torch.matmul(self.attn_res_norm(h), self.attn_res_query)
                              for h in hs])                                      # [N, B, T]
        alpha = torch.softmax(logits.float(), dim=0).to(partial_block.dtype)
        output = torch.zeros_like(hs[0])
        for a, h in zip(alpha, hs):
            output = output + a.unsqueeze(-1) * h
        return output

class BlockAttnResMiniRainBlock(nn.Module):
    def __init__(self, layer_id: int, config: PreTrainedConfig):
        super().__init__()
        self.layer_id = layer_id
        self.block_size = config.block_size
        self.layer_number = config.n_hidden_layers
        self.attn = GroupQueryAttention(AttentionArgs(config))
        self.input_layernorm = TorchRMSNorm(NormArgs(dim=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attn_layernorm = TorchRMSNorm(NormArgs(dim=config.hidden_size, eps=config.rms_norm_eps))
        self.mlp = FusedSwiGLU(MLPArgs(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
                                         bias=False))
        self.attn_res = BlockAttnRes(config)
        self.mlp_res = BlockAttnRes(config)
    def _flush(self, past_hidden_states, partial_block):
        """普通层：历史不变，partial_block 原样往后传。"""
        return past_hidden_states, partial_block


    def forward(self, hidden_states, position_embeddings,
                past_key_value=None, use_cache=False, attention_mask=None,
                past_hidden_states=None, image_bidirectional_mask=None, modality_mask=None,
                img_idx=None):
        partial_block = hidden_states
        attn_input = self.attn_res(past_hidden_states, partial_block)
        # flush 决策在 trace 时按子类解析，不产生 layer_id guard
        past_hidden_states, partial_block = self._flush(past_hidden_states, partial_block)
        attn_out, past_key_value = self.attn(self.input_layernorm(attn_input),
                                             position_embeddings, past_key_value,
                                             use_cache, attention_mask, image_bidirectional_mask,
                                             modality_mask=modality_mask)
        partial_block = partial_block + attn_out if partial_block is not None else attn_out
        mlp_input = self.mlp_res(past_hidden_states, partial_block)
        mlp_out = self.mlp(self.post_attn_layernorm(mlp_input))
        if isinstance(mlp_out, tuple):
            mlp_out = mlp_out[0]  # MoE 返回 (out, logits, aux)
        output = partial_block + mlp_out
        return output, past_key_value, past_hidden_states

class FlushBlockAttnResMiniRainBlock(BlockAttnResMiniRainBlock):
    """flush 层（每 block_size//2 层一个）：把 partial_block 追加进历史后清空。"""

    is_flush = True

    def _flush(self, past_hidden_states, partial_block):
        past_hidden_states.append(partial_block)
        return past_hidden_states, None

class FullAttnResBlock(nn.Module):
    def __init__(self, layer_id: int, config: PreTrainedConfig):
        super().__init__()
        self.layer_id = layer_id
        self.attn = GroupQueryAttention(AttentionArgs(config))
        self.input_layernorm = TorchRMSNorm(NormArgs(dim=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attn_layernorm = TorchRMSNorm(NormArgs(dim=config.hidden_size, eps=config.rms_norm_eps))
        if config.type_id == 1:
            self.mlp = SwiGLU(MLPArgs(hidden_size=config.hidden_size,
                                             intermediate_size=config.intermediate_size, bias=False))
        elif config.type_id == 2:
            self.mlp = LosslessMoE(dim=config.hidden_size,
                                  intermediate_size=config.intermediate_size, num_experts=config.n_experts)
        self.mlp_res = FullAttnRes(config)
        self.attn_res = FullAttnRes(config)

    def forward(self, hidden_states, position_embeddings,
                past_key_value=None, use_cache=False, attention_mask=None,
                past_hidden_states=None, image_bidirectional_mask=None, modality_mask=None):
        attn_input = self.attn_res(past_hidden_states + [hidden_states])
        attn_out, past_key_value = self.attn(
            self.input_layernorm(attn_input), position_embeddings,
            past_key_value, use_cache, attention_mask, image_bidirectional_mask)
        # attn_out = hidden_states + attn_out
        past_hidden_states.append(hidden_states)
        mlp_input = self.mlp_res(past_hidden_states + [attn_out])
        mlp_out = self.mlp(self.post_attn_layernorm(mlp_input))
        if isinstance(mlp_out, tuple): mlp_out = mlp_out[0]  # MoE 返回 (out, logits, aux)
        # output = attn_out + mlp_out
        past_hidden_states.append(attn_out)
        return mlp_out, past_key_value, past_hidden_states

    def forward_old(self, hidden_states, position_embeddings,
                past_key_value=None, use_cache=False, attention_mask=None,
                past_hidden_states=None, image_bidirectional_mask=None, modality_mask=None):
        attn_input = self.attn_res(past_hidden_states + [hidden_states])
        attn_out, past_key_value = self.attn(
            self.input_layernorm(attn_input), position_embeddings,
            past_key_value, use_cache, attention_mask, image_bidirectional_mask, modality_mask=modality_mask)
        # attn_out = hidden_states + attn_out
        past_hidden_states.append(hidden_states)
        mlp_input = self.mlp_res(past_hidden_states + [attn_out])
        mlp_out = self.mlp(self.post_attn_layernorm(mlp_input))
        # output = attn_out + mlp_out
        past_hidden_states.append(attn_out)
        return mlp_out, past_key_value, past_hidden_states

class BlockAttnResBlock(nn.Module):
    is_flush = False
    def __init__(self, layer_id: int, config: PreTrainedConfig):
        super().__init__()
        self.layer_id = layer_id
        self.block_size = config.block_size
        self.layer_number = config.n_hidden_layers
        # 模态专属 QKVO 只在 MoT（type_id=3）时构建，否则复用单套投影，避免纯文本模型白付双份注意力
        self.attn = (MoTGroupQueryAttention if getattr(config, "type_id", 1) == 3
                     else GroupQueryAttention)(AttentionArgs(config))
        self.input_layernorm = TorchRMSNorm(NormArgs(dim=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attn_layernorm = TorchRMSNorm(NormArgs(dim=config.hidden_size, eps=config.rms_norm_eps))
        if config.type_id == 1:
            self.mlp = SwiGLU(MLPArgs(hidden_size=config.hidden_size,
                                             intermediate_size=config.intermediate_size, bias=False))
        elif config.type_id == 2:
            self.mlp = LosslessMoE(dim=config.hidden_size,
                                   intermediate_size=config.intermediate_size, num_experts=config.n_experts)
        elif config.type_id == 3:
            self.mlp = MoTFFN(n_modalities=2,
                              args=MLPArgs(hidden_size=config.hidden_size,
                                           intermediate_size=config.intermediate_size, bias=False),
                              norm_args=NormArgs(dim=config.hidden_size, eps=config.rms_norm_eps), config=config)
            self.post_attn_layernorm = None
        elif config.type_id == 4:
            self.mlp = AdaptiveSwiGLU(MLPArgs(hidden_size=config.hidden_size,
                                             intermediate_size=config.intermediate_size, bias=False), config=config)
        self.attn_res = BlockAttnRes(config)
        self.mlp_res = BlockAttnRes(config)

    # ---------------------------------------------------------------- flush 钩子
    def _flush(self, past_hidden_states, partial_block):
        """普通层：历史不变，partial_block 原样往后传。"""
        return past_hidden_states, partial_block

    def forward(self, hidden_states, position_embeddings,
                past_key_value=None, use_cache=False, attention_mask=None,
                past_hidden_states=None, image_bidirectional_mask=None, modality_mask=None,
                img_idx=None):
        partial_block = hidden_states
        attn_input = self.attn_res(past_hidden_states, partial_block)
        # flush 决策在 trace 时按子类解析，不产生 layer_id guard
        past_hidden_states, partial_block = self._flush(past_hidden_states, partial_block)
        attn_out, past_key_value = self.attn(self.input_layernorm(attn_input),
                                             position_embeddings, past_key_value,
                                             use_cache, attention_mask, image_bidirectional_mask,
                                             modality_mask=modality_mask)
        partial_block = partial_block + attn_out if partial_block is not None else attn_out
        mlp_input = self.mlp_res(past_hidden_states, partial_block)
        # 当前 1=图像统一走 mlp_v；2/3 为后续视觉子模态预留，届时在此按 id 分派
        if isinstance(self.mlp, MoTFFN):
            if img_idx is None and modality_mask is not None:
                img_idx = (modality_mask.reshape(-1) > 0).nonzero(as_tuple=True)[0]
            mlp_out = self.mlp(mlp_input, img_idx=img_idx)
        else:
            mlp_out = self.mlp(self.post_attn_layernorm(mlp_input))
        if isinstance(mlp_out, tuple):
            mlp_out = mlp_out[0]  # MoE 返回 (out, logits, aux)
        output = partial_block + mlp_out
        return output, past_key_value, past_hidden_states


class FlushBlockAttnResBlock(BlockAttnResBlock):
    """flush 层（每 block_size//2 层一个）：把 partial_block 追加进历史后清空。"""

    is_flush = True

    def _flush(self, past_hidden_states, partial_block):
        past_hidden_states.append(partial_block)
        return past_hidden_states, None
