from components import norm
from components.norm import *
from architectures.config import RainConfig
from dataclasses import dataclass
import math


import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
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
    config: RainConfig


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
        
        
        
        
        