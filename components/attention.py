import inspect

from transformers import PreTrainedConfig

from components.norm import *
from dataclasses import dataclass
import math


import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    应用位置编码
    """
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
    """分组查询注意力机制。

    依赖外部定义：AttentionArgs、NormArgs、TorchRMSNorm 和
    apply_rotary_pos_emb。

    attention_mask 使用 [B, KV_len] 格式，1/True 表示有效 token，
    0/False 表示 padding；image_bidirectional_mask（别名 modality_mask）
    应为可广播到 [B, H, Q_len, KV_len] 的 mask。布尔或整型 0/1 mask
    按 1/True 可见、0/False 屏蔽处理；浮点 mask 按加性偏置处理。
    """

    @staticmethod
    def _repeat_kv(x, n_rep):
        """将 [B, Hkv, T, D] 展开为 [B, H, T, D]。"""
        if n_rep == 1:
            return x
        bsz, n_kv_heads, seq_len, head_dim = x.shape
        return (
            x[:, :, None, :, :]
            .expand(bsz, n_kv_heads, n_rep, seq_len, head_dim)
            .reshape(bsz, n_kv_heads * n_rep, seq_len, head_dim)
        )

    @staticmethod
    def _prepare_attn_mask(mask, query):
        """将 mask 转为 SDPA 支持的类型，并统一可见性语义。"""
        if mask.dtype == torch.bool:
            return mask.to(device=query.device)

        # 整型 0/1 mask 按可见性处理：1 可见，0 屏蔽。
        if not mask.dtype.is_floating_point:
            is_binary = bool(torch.all((mask == 0) | (mask == 1)).item())
            if is_binary:
                return mask.to(device=query.device, dtype=torch.bool)

        # 浮点 mask 或其他整型 mask 按加性注意力偏置处理。
        return mask.to(device=query.device, dtype=query.dtype)

    def __init__(self, args: AttentionArgs):
        super().__init__()
        config = args.config

        if config.n_attn_heads <= 0 or config.n_key_value_heads <= 0:
            raise ValueError("注意力头数必须为正整数")
        if config.n_attn_heads % config.n_key_value_heads != 0:
            raise ValueError(
                "n_attn_heads 必须能被 n_key_value_heads 整除，"
                f"当前为 {config.n_attn_heads} 和 {config.n_key_value_heads}"
            )

        self.n_key_value_heads = config.n_key_value_heads
        self.n_local_heads = config.n_attn_heads
        self.n_local_kv_heads = self.n_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True

        self.wq = nn.Linear(
            config.hidden_size,
            config.n_attn_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            config.hidden_size,
            self.n_key_value_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            config.hidden_size,
            self.n_key_value_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            config.n_attn_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )

        self.q_norm = TorchRMSNorm(
            NormArgs(dim=self.head_dim, eps=config.rms_norm_eps)
        )
        self.k_norm = TorchRMSNorm(
            NormArgs(dim=self.head_dim, eps=config.rms_norm_eps)
        )
        self.attn_dropout = nn.Dropout(config.n_dropout)
        self.resid_dropout = nn.Dropout(config.n_dropout)
        self.dropout = config.n_dropout
        self.flash = (
            hasattr(F, "scaled_dot_product_attention") and config.flash_attn
        )

    def forward(
        self,
        x,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
        image_bidirectional_mask=None,
        modality_mask=None,
    ):
        # 兼容多模态注意力中常用的 modality_mask 参数名。
        if modality_mask is not None:
            if image_bidirectional_mask is not None:
                raise ValueError(
                    "modality_mask 和 image_bidirectional_mask 只能传入一个"
                )
            image_bidirectional_mask = modality_mask

        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # 缓存保存 RoPE 后的 [B, T, Hkv, D]，与历史缓存直接拼接。
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq = xq.transpose(1, 2)  # [B, H, Q_len, D]
        xk = xk.transpose(1, 2)  # [B, Hkv, KV_len, D]
        xv = xv.transpose(1, 2)
        kv_len = xk.size(-2)
        dropout_p = self.dropout if self.training else 0.0

        # 始终显式展开 KV 头，不再依赖 SDPA 后端的原生 GQA 支持。
        # 缓存已在上方保存，因此这里不会增大 KV Cache。
        xk = self._repeat_kv(xk, self.n_rep)
        xv = self._repeat_kv(xv, self.n_rep)

        if image_bidirectional_mask is not None:
            # 显式 mask 已包含因果、图像段内双向和 padding 屏蔽。
            attn_mask = self._prepare_attn_mask(image_bidirectional_mask, xq)
            if self.flash:
                output = F.scaled_dot_product_attention(
                    xq,
                    xk,
                    xv,
                    dropout_p=dropout_p,
                    attn_mask=attn_mask,
                )
            else:
                scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
                if attn_mask.dtype == torch.bool:
                    scores = scores.masked_fill(
                        ~attn_mask,
                        torch.finfo(scores.dtype).min,
                    )
                else:
                    scores = scores + attn_mask
                output = self.attn_dropout(
                    F.softmax(scores.float(), dim=-1).type_as(xq)
                ) @ xv

        elif (
            self.flash
            and (
                attention_mask is None
                or bool(torch.all(attention_mask != 0).item())
            )
            # 多 token 带缓存时需要右下对齐的显式因果 mask，走手写分支。
            and (past_key_value is None or seq_len == 1)
        ):
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=dropout_p,
                # 单 token 解码可见全部历史 KV，不能启用因果遮挡。
                is_causal=self.is_causal and past_key_value is None,
            )

        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            min_score = torch.finfo(scores.dtype).min

            if self.is_causal:
                # 当前 Q 对应完整 KV 序列的最后 seq_len 个位置，掩码右下对齐。
                q_positions = torch.arange(
                    kv_len - seq_len, kv_len, device=scores.device
                )
                k_positions = torch.arange(kv_len, device=scores.device)
                causal_mask = (
                    k_positions.unsqueeze(0) > q_positions.unsqueeze(1)
                )
                scores = scores.masked_fill(causal_mask, min_score)

            if attention_mask is not None:
                if (
                    attention_mask.ndim != 2
                    or attention_mask.shape[0] != bsz
                    or attention_mask.shape[1] != kv_len
                ):
                    raise ValueError(
                        "attention_mask 应为 [B, KV_len]，当前形状为 "
                        f"{tuple(attention_mask.shape)}，期望为 {(bsz, kv_len)}"
                    )
                if attention_mask.dtype == torch.bool:
                    padding_mask = ~attention_mask
                else:
                    padding_mask = attention_mask == 0
                scores = scores.masked_fill(
                    padding_mask[:, None, None, :], min_score
                )

            output = self.attn_dropout(
                F.softmax(scores.float(), dim=-1).type_as(xq)
            ) @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(output))
        return output, past_kv

# class GroupQueryAttention(nn.Module):
#     """分组查询注意力机制。"""
#     def __init__(self, args: AttentionArgs):
#         super().__init__()
#         config = args.config
#
#         self.n_key_value_heads = config.n_key_value_heads
#         self.n_local_heads = config.n_attn_heads
#         self.n_local_kv_heads = self.n_key_value_heads
#         # 重复次数
#         self.n_rep = self.n_local_heads // self.n_local_kv_heads
#         self.head_dim = config.head_dim
#         # 是否为因果注意力
#         self.is_causal = True
#         # 注意力矩阵
#         self.wq = nn.Linear(config.hidden_size, config.n_attn_heads * self.head_dim, bias=False)
#         self.wk = nn.Linear(config.hidden_size, self.n_key_value_heads * self.head_dim, bias=False)
#         self.wv = nn.Linear(config.hidden_size, self.n_key_value_heads * self.head_dim, bias=False)
#         self.wo = nn.Linear(config.n_attn_heads * self.head_dim, config.hidden_size, bias=False)
#
#         self.q_norm = TorchRMSNorm(NormArgs(dim=self.head_dim, eps=config.rms_norm_eps))
#         self.k_norm = TorchRMSNorm(NormArgs(dim=self.head_dim, eps=config.rms_norm_eps))
#         self.attn_dropout = nn.Dropout(config.n_dropout)
#         self.resid_dropout = nn.Dropout(config.n_dropout)
#         self.dropout = config.n_dropout
#         self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn
#
#     def forward(self, x, position_embeddings, past_key_value=None, use_cache=False,
#                 attention_mask=None, image_bidirectional_mask=None, modality_mask=None):
#         bsz, seq_len, _ = x.shape
#         xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
#         xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
#         xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
#         xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
#         xq, xk = self.q_norm(xq), self.k_norm(xk)
#         # 位置编码
#         cos, sin = position_embeddings
#         xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
#         # kv_cahce
#         if past_key_value is not None:
#             xk = torch.cat([past_key_value[0], xk], dim=1)
#             xv = torch.cat([past_key_value[1], xv], dim=1)
#         past_kv = (xk, xv) if use_cache else None
#
#         xq, xk, xv = (
#             xq.transpose(1, 2),
#             repeat_kv(xk, self.n_rep).transpose(1, 2),
#             repeat_kv(xv, self.n_rep).transpose(1, 2)
#         )
#         use_flash = self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None)
#         if image_bidirectional_mask is not None:
#             # 显式 mask：因果 + 图像段内双向（已含 padding 屏蔽）
#             if use_flash:
#                 output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, attn_mask=image_bidirectional_mask)
#             else:
#                 scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
#                 scores = scores + image_bidirectional_mask
#                 output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
#         elif use_flash and (attention_mask is None or torch.all(attention_mask == 1)):
#             output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
#         else:
#             scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
#             if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
#             if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
#             output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
#
#         output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
#         output = self.resid_dropout(self.wo(output))
#         return output, past_kv



class MoTGroupQueryAttention(nn.Module):
    """MoT 版 GQA：QKVO 按模态解耦，注意力计算全局共享。
    modality_mask: (B, T) int，0=文本 1=图像；为 None 时全部走模态 0（纯文本退化为普通 GQA）。
    """
    def __init__(self, args: AttentionArgs):
        super().__init__()
        config = args.config
        self.n_key_value_heads = config.n_key_value_heads
        self.n_local_heads = config.n_attn_heads
        self.n_rep = self.n_local_heads // self.n_key_value_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.n_modalities = getattr(config, "n_modalities", 2)

        def experts(in_dim, out_dim) -> nn.ModuleList:
            return nn.ModuleList(
                [nn.Linear(in_dim, out_dim, bias=False) for _ in range(self.n_modalities)]
            )

        self.wq = experts(config.hidden_size, self.n_local_heads * self.head_dim)
        self.wk = experts(config.hidden_size, self.n_key_value_heads * self.head_dim)
        self.wv = experts(config.hidden_size, self.n_key_value_heads * self.head_dim)
        self.wo = experts(self.n_local_heads * self.head_dim, config.hidden_size)

        self.q_norm = TorchRMSNorm(NormArgs(dim=self.head_dim, eps=config.rms_norm_eps))
        self.k_norm = TorchRMSNorm(NormArgs(dim=self.head_dim, eps=config.rms_norm_eps))
        self.attn_dropout = nn.Dropout(config.n_dropout)
        self.resid_dropout = nn.Dropout(config.n_dropout)
        self.dropout = config.n_dropout
        self.flash = hasattr(F, 'scaled_dot_product_attention') and config.flash_attn

    def _modal_indices(self, modality_mask):
        """
        获取不同模态索引
        """
        flat = modality_mask.reshape(-1)
        return [(flat == i).nonzero(as_tuple=True)[0] for i in range(self.n_modalities)]

    def _project_modal(self, layers, x, idxs):
        """
        token分拣
        """
        # 为空时默认全为文本
        if idxs is None:
            return layers[0](x)
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        out = None
        # token分拣
        for layer, idx in zip(layers, idxs):
            if idx.numel() == 0:
                continue
            y = layer(flat.index_select(0, idx))
            if out is None:
                out = y.new_zeros(B * T, y.shape[-1])  # 顺带解决第3条：zeros 兜底
            out.index_copy_(0, idx, y)
        return out.view(B, T, -1)

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False,
                attention_mask=None, image_bidirectional_mask=None, modality_mask=None):
        bsz, seq_len, _ = x.shape
        idxs = self._modal_indices(modality_mask) if modality_mask is not None else None
        xq = self._project_modal(self.wq, x, idxs)
        xk = self._project_modal(self.wk, x, idxs)
        xv = self._project_modal(self.wv, x, idxs)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_key_value_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_key_value_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )
        # 带 cache 解码时，mask 切出当前 query 对应的行
        if image_bidirectional_mask is not None and image_bidirectional_mask.shape[-2] != seq_len:
            image_bidirectional_mask = image_bidirectional_mask[:, :, -seq_len:, :]
        use_flash = self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None)
        if image_bidirectional_mask is not None:
            if use_flash:
                output = F.scaled_dot_product_attention(
                    xq, xk, xv, dropout_p=self.dropout if self.training else 0.0,
                    attn_mask=image_bidirectional_mask)
            else:
                scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
                scores = scores + image_bidirectional_mask
                output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        elif use_flash and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(
                xq, xk, xv, dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal:
                scores[:, :, :, -seq_len:] += torch.full(
                    (seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self._project_modal(self.wo, output, idxs)   # wo 也按模态
        output = self.resid_dropout(output)
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
    def attention_comp(self, q, k, v, attn_mask=None):
        pass
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
        