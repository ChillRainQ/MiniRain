import torch
from transformers import PreTrainedConfig

from components.mlp import SwiGLU, LosslessMoE, MoTFFN
from components.mlp import MLPArgs
from components.norm import NormArgs
from components.attention import AttentionArgs
from components.attention import GroupQueryAttention
import torch.nn as nn

from components.norm import RMSNorm, TorchRMSNorm

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

    def forward(self, inter_block_history: list[torch.Tensor],
                partial_block: torch.Tensor):
        """
        :param inter_block_history: 块间历史表示（每一个块的最后一个output）
        :param partial_block: 块内历史表示
        :return:
        """
        hs = inter_block_history + [partial_block]
        # 一次性堆叠算 norm/logits，避免逐候选的 Python 循环（decode 时是 launch-bound 主因）
        # 注意：单候选时不早退，否则 q/norm 拿不到 0 梯度而变成 None（DDP/优化器会判为未使用参数）
        V = torch.stack(hs, dim=0)                                        # [N, B, T, D]
        logits = (self.attn_res_norm(V) * self.attn_res_query).sum(dim=-1)  # [N, B, T]
        alpha = torch.softmax(logits.float(), dim=0).to(partial_block.dtype)
        return (alpha.unsqueeze(-1) * V).sum(dim=0)

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

class MiniRainBlock(nn.Module):
    def __init__(self, layer_id: int, config: PreTrainedConfig):
        super().__init__()
        self.layer_id = layer_id
        self.full_attn_res = config.full_attn_res
        self.block_attn_res = config.block_attn_res
        self.block_size = config.block_size
        self.layer_number = config.n_hidden_layers
        self.attn = GroupQueryAttention(AttentionArgs(config))
        self.input_layernorm = TorchRMSNorm(NormArgs(dim=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attn_layernorm = TorchRMSNorm(NormArgs(dim=config.hidden_size, eps=config.rms_norm_eps))
        self.mlp = SwiGLU(MLPArgs(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size, bias=False))


    def forward(self, hidden_states, position_embeddings,
                past_key_value=None, use_cache=False, attention_mask=None,
                past_hidden_states=None):
        attn_input = hidden_states
        attn_out, past_key_value = self.attn(
            self.input_layernorm(attn_input),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        mlp_input = attn_input + attn_out
        mlp_out = self.mlp(
            self.post_attn_layernorm(mlp_input)
        )
        output = mlp_out + mlp_input
        return output, past_key_value, past_hidden_states

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
        self.full_attn_res = config.full_attn_res
        self.block_attn_res = config.block_attn_res
        self.block_size = config.block_size
        self.layer_number = config.n_hidden_layers
        self.attn = GroupQueryAttention(AttentionArgs(config))
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
                                             use_cache, attention_mask, image_bidirectional_mask)
        partial_block = partial_block + attn_out if partial_block is not None else attn_out
        mlp_input = self.mlp_res(past_hidden_states, partial_block)
        # MoTFFN 期望 img_idx（索引），由 VLM 入口统一算一次（避免每层 nonzero）；
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
# class BlockAttnResBlock(nn.Module):
#     def __init__(self, layer_id: int, config: PreTrainedConfig):
#         super().__init__()
#         self.layer_id = layer_id
#         self.full_attn_res = config.full_attn_res
#         self.block_attn_res = config.block_attn_res
#         self.block_size = config.block_size
#         self.layer_number = config.n_hidden_layers
#         self.attn = GroupQueryAttention(AttentionArgs(config))
#         self.input_layernorm = TorchRMSNorm(NormArgs(dim=config.hidden_size, eps=config.rms_norm_eps))
#         self.post_attn_layernorm = TorchRMSNorm(NormArgs(dim=config.hidden_size, eps=config.rms_norm_eps))
#         if config.type_id == 1:
#             self.mlp = SwiGLU(MLPArgs(hidden_size=config.hidden_size,
#                                       intermediate_size=config.intermediate_size, bias=False))
#         elif config.type_id == 2:
#             self.mlp = LosslessMoE(dim=config.hidden_size,
#                                    intermediate_size=config.intermediate_size, num_experts=config.n_experts)
#         elif config.type_id == 3:
#             self.mlp = MoTFFN(n_modalities=2,
#                               args=MLPArgs(hidden_size=config.hidden_size,
#                                            intermediate_size=config.intermediate_size, bias=False),
#                               norm_args=NormArgs(dim=config.hidden_size, eps=config.rms_norm_eps), config=config)
#             self.post_attn_layernorm = None
#         self.attn_res = BlockAttnRes(config)
#         self.mlp_res = BlockAttnRes(config)
#
#
#     def forward(self, hidden_states, position_embeddings,
#                 past_key_value=None, use_cache=False, attention_mask=None,
#                 past_hidden_states=None, image_bidirectional_mask=None, modality_mask=None):
#         partial_block = hidden_states
#         attn_input = self.attn_res(past_hidden_states, partial_block)
#         if self.block_size >= 2 and self.layer_id % (self.block_size // 2) == 0:
#             past_hidden_states.append(partial_block)
#             partial_block = None
#         attn_out, past_key_value = self.attn(self.input_layernorm(attn_input),
#                                              position_embeddings, past_key_value,
#                                              use_cache, attention_mask, image_bidirectional_mask)
#         partial_block = partial_block + attn_out if partial_block is not None else attn_out
#         mlp_input = self.mlp_res(past_hidden_states, partial_block)
#         # MoTFFN 期望 img_idx（索引），从 modality_mask 转换；其他 MLP 不接受此参数
#         if isinstance(self.mlp, MoTFFN):
#             img_idx = (modality_mask.reshape(-1) == 1).nonzero(as_tuple=True)[0] if modality_mask is not None else None
#             mlp_out = self.mlp(mlp_input, img_idx=img_idx)
#         else:
#             mlp_out = self.mlp(self.post_attn_layernorm(mlp_input))
#         if isinstance(mlp_out, tuple): mlp_out = mlp_out[0]  # MoE 返回 (out, logits, aux)
#         output = partial_block + mlp_out
#         return output, past_key_value, past_hidden_states