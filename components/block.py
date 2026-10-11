import torch
from transformers import PreTrainedConfig

from components.mlp import SwiGLU, LosslessMoE
from components.mlp import MLPArgs
from components.norm import NormArgs
from components.attention import AttentionArgs, MoTGroupQueryAttention
from components.attention import GroupQueryAttention
import torch.nn as nn

from components.norm import RMSNorm

class FullAttnRes(nn.Module):
    """
    全注意力残差
    """
    def __init__(self, config: PreTrainedConfig):
        super().__init__()
        self.attn_res_norm = RMSNorm(NormArgs(dim=config.hidden_size, eps=config.rms_norm_eps))
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
        self.attn_res_norm = RMSNorm(NormArgs(dim=config.hidden_size, eps=config.rms_norm_eps))
        self.attn_res_query = nn.Parameter(torch.zeros(config.hidden_size))

    def forward(self, inter_block_history: list[torch.Tensor],
                partial_block: torch.Tensor):
        """
        :param inter_block_history: 块间历史表示（每一个块的最后一个output）
        :param partial_block: 块内历史表示
        :return:
        """
        hs = inter_block_history + [partial_block]
        logits = torch.stack([
            torch.matmul(self.attn_res_norm(h), self.attn_res_query)
            for h in hs
        ])
        alpha = torch.softmax(logits.float(), dim=0).to(partial_block.dtype)
        output = torch.zeros_like(hs[0])
        for a, h in zip(alpha, hs):
            output = output + a.unsqueeze(-1) * h
        return output

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
        self.input_layernorm = RMSNorm(NormArgs(dim=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attn_layernorm = RMSNorm(NormArgs(dim=config.hidden_size, eps=config.rms_norm_eps))
        self.mlp = SwiGLU(MLPArgs(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size, bias=False))
        assert not (self.block_attn_res and self.full_attn_res)
        if self.full_attn_res:
            self.attn_res = FullAttnRes(config)
            self.mlp_res = FullAttnRes(config)
        elif self.block_attn_res:
            self.attn_res = BlockAttnRes(config)
            self.mlp_res = BlockAttnRes(config)


    def forward(self, hidden_states, position_embeddings,
                past_key_value=None, use_cache=False, attention_mask=None,
                past_hidden_states=None):
        partial_block = None
        if self.full_attn_res and self.attn_res:
            past_hidden_states.append(hidden_states)
            attn_input = self.attn_res(past_hidden_states)
        elif self.block_attn_res and self.attn_res:
            partial_block = hidden_states
            attn_input = self.attn_res(past_hidden_states, partial_block)
            if self.layer_id % (self.block_size // 2) == 0:
                past_hidden_states.append(partial_block)
                partial_block = None
        else:
            attn_input = hidden_states
        attn_out, past_key_value = self.attn(
            self.input_layernorm(attn_input),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        if self.full_attn_res and self.mlp_res:
            past_hidden_states.append(attn_out)
            mlp_input = self.mlp_res(past_hidden_states)
        elif self.block_attn_res and self.mlp_res:
            partial_block = partial_block + attn_out if partial_block is not None else attn_out
            mlp_input = self.mlp_res(past_hidden_states, partial_block)
        else:
            mlp_input = attn_input + attn_out
        mlp_out = self.mlp(
            self.post_attn_layernorm(mlp_input)
        )
        if self.full_attn_res:
            output = mlp_out
        elif self.block_attn_res:
           output = partial_block + mlp_out
        else:
            output = mlp_out + mlp_input
        return output, past_key_value, past_hidden_states

class FullAttnResBlock(nn.Module):
    def __init__(self, layer_id: int, config: PreTrainedConfig):
        super().__init__()
        self.layer_id = layer_id
        self.attn = GroupQueryAttention(AttentionArgs(config))
        self.input_layernorm = RMSNorm(NormArgs(dim=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attn_layernorm = RMSNorm(NormArgs(dim=config.hidden_size, eps=config.rms_norm_eps))
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
    def __init__(self, layer_id: int, config: PreTrainedConfig):
        super().__init__()
        self.layer_id = layer_id
        self.full_attn_res = config.full_attn_res
        self.block_attn_res = config.block_attn_res
        self.block_size = config.block_size
        self.layer_number = config.n_hidden_layers
        self.attn = MoTGroupQueryAttention(AttentionArgs(config))
        self.input_layernorm = RMSNorm(NormArgs(dim=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attn_layernorm = RMSNorm(NormArgs(dim=config.hidden_size, eps=config.rms_norm_eps))
        if config.type_id == 1:
            self.mlp = SwiGLU(MLPArgs(hidden_size=config.hidden_size,
                                      intermediate_size=config.intermediate_size, bias=False))
        elif config.type_id == 2:
            self.mlp = LosslessMoE(dim=config.hidden_size,
                                   intermediate_size=config.intermediate_size, num_experts=config.n_experts)
        self.attn_res = BlockAttnRes(config)
        self.mlp_res = BlockAttnRes(config)


    def forward(self, hidden_states, position_embeddings,
                past_key_value=None, use_cache=False, attention_mask=None,
                past_hidden_states=None, image_bidirectional_mask=None, modality_mask=None):
        partial_block = hidden_states
        attn_input = self.attn_res(past_hidden_states, partial_block)
        if self.block_size >= 2 and self.layer_id % (self.block_size // 2) == 0:
            past_hidden_states.append(partial_block)
            partial_block = None
        attn_out, past_key_value = self.attn(self.input_layernorm(attn_input),
                                             position_embeddings, past_key_value,
                                             use_cache, attention_mask, image_bidirectional_mask, modality_mask=modality_mask)
        partial_block = partial_block + attn_out if partial_block is not None else attn_out
        mlp_input = self.mlp_res(past_hidden_states, partial_block)
        mlp_out = self.mlp(self.post_attn_layernorm(mlp_input))
        output = partial_block + mlp_out
        return output, past_key_value, past_hidden_states