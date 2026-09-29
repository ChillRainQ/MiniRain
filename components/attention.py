from transformers import PreTrainedConfig

from components.norm import *
from dataclasses import dataclass
import math


import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    if cos.dim() == 3:
        unsqueeze_dim = 2
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

@dataclass
class AttentionArgs:
    """
    注意力参数。
    """
    config: PreTrainedConfig


class GroupQueryAttention(nn.Module):
    """分组查询注意力机制。"""
    def __init__(self, args: AttentionArgs):
        super().__init__()
        config = args.config
        
        self.n_key_value_heads = config.n_key_value_heads
        self.n_local_heads = config.n_attn_heads
        self.n_local_kv_heads = self.n_key_value_heads
        # 重复次数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        # 是否为因果注意力
        self.is_causal = True
        # 注意力矩阵
        self.wq = nn.Linear(config.hidden_size, config.n_attn_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.n_key_value_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.hidden_size, self.n_key_value_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_attn_heads * self.head_dim, config.hidden_size, bias=False)
        
        self.q_norm = RMSNorm(NormArgs(dim=self.head_dim, eps=config.rms_norm_eps))
        self.k_norm = RMSNorm(NormArgs(dim=self.head_dim, eps=config.rms_norm_eps))
        self.attn_dropout = nn.Dropout(config.n_dropout)
        self.resid_dropout = nn.Dropout(config.n_dropout)
        self.dropout = config.n_dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask = None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        # 位置编码
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        # kv_cahce
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
            
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(output))
        return output, past_kv
        
        
class MoTAttention(nn.Module):
    def __init__(self, dim: int,
                 head_dim: int,
                 n_heads: int,
                 dropout: float,
                 norm_eps: float = 1e-5,
                 qk_normalization: bool = False,
                 n_modalities: int = 2):
        super().__init__()

        self.n_modalities = n_modalities

        self.local_experts_wq = self._create_experts(dim, head_dim * n_heads)
        self.local_experts_wk = self._create_experts(dim, head_dim * n_heads)
        self.local_experts_wv = self._create_experts(dim, head_dim * n_heads)
        self.local_experts_wo = self._create_experts(head_dim * n_heads, dim)

        self.head_dim = head_dim

        if qk_normalization:
            # 对每种模态的 Q 和 K 分别做 LayerNorm
            self.local_experts_q_normalization = self._create_norms(
                head_dim, self.n_modalities, eps=norm_eps
            )
            self.local_experts_k_normalization = self._create_norms(
                head_dim, self.n_modalities, eps=norm_eps
            )

        self.local_experts_attention_norm = self._create_norms(
            dim, self.n_modalities
        )
        # 全局的自注意力，这里后面可以用自己实现的Attention来替换。
        self.attentions = torch.nn.MultiheadAttention(
            embed_dim=n_heads * head_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, modality_masks: torch.Tensor):
        # 提取所有的QKV
        expert_outputs_xq, expert_outputs_xk, expert_outputs_xv = self._process_qkv(
            x, modality_masks
        )
        # qkv合并
        xq = self._merge_qkv(expert_outputs_xq, modality_masks)
        xk = self._merge_qkv(expert_outputs_xk, modality_masks)
        xv = self._merge_qkv(expert_outputs_xv, modality_masks)

        # 全局自注意力计算
        attn_output, attn_output_weights = self.attention_comp(
            xq, xk, xv, attn_mask=attn_mask
        )

        attn_output = self._process_final_output(attn_output, modality_masks)

        return attn_output, attn_output_weights

    def _merge_qkv(self, expert_outputs: list, modality_masks:torch.Tensor):
        # 确保形状一致
        merged = torch.empty_like(expert_outputs[0])
        for i, expert_output in enumerate(expert_outputs):
            merged[modality_masks[i]] = expert_output
        return merged

    def _process_qkv(self, x: torch.Tensor, modality_mask: torch.Tensor):
        expert_outputs_xq, expert_outputs_xk, expert_outputs_xv = [], [], []
        """
        modality_mask: (1, n_modalities)
        """
        for i in range(self.n_modalities):
            # 获取对应模态的输入x
            expert_input = x[modality_mask[i]]
            # 对模态获取其qkv
            xq = self.local_experts_wq[i](expert_input)
            xk = self.local_experts_wk[i](expert_input)
            xv = self.local_experts_wv[i](expert_input)
            if hasattr(self, "local_experts_q_normalization"):
                xq = self.local_experts_q_normalization[i](xq)
                xk = self.local_experts_k_normalization[i](xk)
            # 合并所有的qkv
            expert_outputs_xq.append(xq)
            expert_outputs_xk.append(xk)
            expert_outputs_xv.append(xv)

        return expert_outputs_xq, expert_outputs_xk, expert_outputs_xv

    def _create_experts(self, input_dim: int, output_dim: int):
        """
        为各个模态创建 qkvo
        :param input_dim:
        :param output_dim:
        :return:
        """
        return torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    input_dim,
                    output_dim,
                    bias=False,  # 论文及常见 LLM 实现中通常不使用 bias
                )
                for _ in range(self.n_modalities)
            ]
        )

    def _create_norms(self, dim: int, n_modalities: int, eps: float=1e-5):
        """
        为各个模态创建norm
        """
        return torch.nn.ModuleList(
            [torch.nn.LayerNorm(dim, eps=eps) for _ in range(n_modalities)]
        )

    def _process_final_output(self, output, modality_masks):
        """
        处理注意力的最终输出
        """
        expert_outputs = []
        for i in range(self.n_modalities):
            # 取出属于第 i 种模态的注意力输出
            expert_input = output[modality_masks[i]]
            # 模态专属的输出投影 Wo
            expert_output = self.local_experts_wo[i](expert_input)

            # 模态专属的 LayerNorm
            expert_output = self.local_experts_attention_norm[i](expert_output)
            expert_outputs.append(expert_output)

        # 重新拼回完整序列顺序
        return self._merge_modalities(expert_outputs, modality_masks)
        