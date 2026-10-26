import torch
import math
import torch.nn as nn
from transformers import PreTrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from architectures.config import RainConfig
from components.block import BlockAttnResBlock
from components.cv import ConvStem
from components.norm import NormArgs, RMSNorm, TorchRMSNorm
import torch.nn.functional as F
from losses.loss import get_loss_fn

def precompute_freqs_cis_2d(head_dim: int, end: int = 32*1024, rope_base: float = 1e6, rope_scaling: dict | None = None):
    # 每个轴分到 head_dim//2 个维度，对应 head_dim//4 个频率
    half = head_dim // 2
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, half, 2).float() / half)), 1.0
    if rope_scaling is not None:  # YaRN，按轴内维度 half 计算，与 1D 版同式
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (half * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), half // 2 - 1)
            ramp = torch.clamp((torch.arange(half // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end)
    ang = torch.outer(t, freqs)          # [P, half/2]
    return ang.cos() * attn_factor, ang.sin() * attn_factor   # 两张表：按行/列索引查

def build_2d_position_embeddings(positions, cos_tab, sin_tab):
    rows, cols = positions[..., 0], positions[..., 1]
    cos = torch.cat([cos_tab[rows], cos_tab[cols]], dim=-1)
    sin = torch.cat([sin_tab[rows], sin_tab[cols]], dim=-1)
    return torch.cat([cos, cos], dim=-1), torch.cat([sin, sin], dim=-1)


class MiniRainVLM(nn.Module):
    name = "MiniRain-Vision"
    def __init__(self, config: PreTrainedConfig):
        super().__init__()
        self.config = config
        self.model_type = self.config.type
        self.vocab_size, self.n_hidden_layers = self.config.vocab_size, self.config.n_hidden_layers
        # 文本
        self.embed = nn.Embedding(self.vocab_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.n_dropout)
        self.layers = nn.ModuleList([BlockAttnResBlock(i, self.config) for i in range(self.n_hidden_layers)])
        self.text_norm = TorchRMSNorm(NormArgs(dim=self.config.hidden_size, eps=self.config.rms_norm_eps))
        # 图像
        self.patch_size = config.patch_size
        assert self.patch_size == 16, f"ConvStem 固定 16× 降采样，config.patch_size={self.patch_size} 暂不支持"
        # 对于如此尺寸的小模型，可能需要更强的视觉先验，一层卷积可能不行，本身参数就不足。
        # self.patch_embd = nn.Conv2d(3, config.hidden_size,
        #                              kernel_size=self.patch_size, stride=self.patch_size)
        self.patch_embed = ConvStem(hidden=config.hidden_size)
        self.img_norm = TorchRMSNorm(NormArgs(dim=self.config.hidden_size, eps=self.config.rms_norm_eps))
        self.img_patch = self.config.img_patch_token
        self.img_start = self.config.img_start_token
        self.img_end = self.config.img_end_token
        self.img_row = self.config.img_row_token
        cos_tab, sin_tab = precompute_freqs_cis_2d(
            self.config.head_dim,
            end=self.config.max_position_embeddings,
            rope_base=self.config.rope_theta,
            rope_scaling=getattr(self.config, "rope_scaling", None))
        self.register_buffer("rope2d_cos", cos_tab, persistent=False)
        self.register_buffer("rope2d_sin", sin_tab, persistent=False)

    def prepare_inputs_embeds(self, input_ids, pixel_values=None):
        inputs_embeds = self.embed(input_ids)
        if pixel_values is None:
            return inputs_embeds
        patches = []
        for img in pixel_values:
            p = self.patch_embed(img.unsqueeze(0).to(next(self.patch_embed.parameters()).dtype))
            patches.append(self.img_norm(p.flatten(2).transpose(1, 2).squeeze(0)))
        patches = torch.cat(patches, dim=0)

        # patch 位置：conv 输出覆盖
        mask = input_ids == self.img_patch
        if not torch.compiler.is_compiling():
            assert mask.sum().item() == patches.shape[0], "patch 数量与 <|image_pad|> 位置数不一致"
        inputs_embeds[mask] = patches.to(inputs_embeds.dtype)

        # 边界 token 位置（start/end/row）：embedding 也过 img_norm，对齐分布
        boundary_ids = torch.tensor([self.img_start, self.img_end, self.img_row],
                                    dtype=torch.long, device=input_ids.device)
        boundary_mask = torch.isin(input_ids, boundary_ids)
        inputs_embeds[boundary_mask] = self.img_norm(inputs_embeds[boundary_mask])

        return inputs_embeds

    def _build_image_bidirectional_mask(self, input_ids, attention_mask, dtype):
        device = input_ids.device
        bsz, seq_len = input_ids.shape
        img_ids = torch.tensor([self.img_start, self.img_end, self.img_patch, self.img_row],
                               dtype=torch.long, device=device)
        is_img = torch.isin(input_ids, img_ids)  # [B, S]
        # 段号：每遇到一个 <|vision_start|> 递增，同一张图的 token 段号相同
        seg = torch.cumsum((input_ids == self.img_start).long(), dim=1)  # [B, S]
        same_img = is_img.unsqueeze(2) & is_img.unsqueeze(1) & \
                   (seg.unsqueeze(2) == seg.unsqueeze(1))  # 同段才双向
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        allow = causal.unsqueeze(0) | same_img
        if attention_mask is not None:
            allow = allow & attention_mask[:, None, :].bool()  # 只屏蔽 key 维（见下）
        mask = torch.where(allow, torch.zeros((), dtype=dtype, device=device),
                           torch.full((), float("-inf"), dtype=dtype, device=device))
        return mask.unsqueeze(1)

    def forward(self, input_ids, pixel_values=None, positions=None,
                attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * self.n_hidden_layers
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.prepare_inputs_embeds(input_ids, pixel_values))

        # padding 判断只做一次（避免注意力里每层 .item() 同步）；全有效时置 None 走因果快路径
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                has_pad = not bool(attention_mask.all())
            else:
                has_pad = bool((attention_mask == 0).any())
            if not has_pad:
                attention_mask = None

        image_bidirectional_mask = None
        modality_mask = None
        if pixel_values is not None:
            image_bidirectional_mask = self._build_image_bidirectional_mask(input_ids, attention_mask, hidden_states.dtype)
            img_ids = torch.tensor([self.img_start, self.img_end, self.img_patch, self.img_row],
                                   dtype=torch.long, device=input_ids.device)
            modality_mask = torch.isin(input_ids, img_ids).long()

        if positions is None:  # 纯文本路径：行=列=下标，退化为1D
            pos = torch.arange(start_pos, start_pos + seq_length, device=input_ids.device)
            pos = pos.unsqueeze(0).expand(batch_size, -1)
            positions = torch.stack([pos, pos], dim=-1)
        position_embeddings = build_2d_position_embeddings(positions, self.rope2d_cos, self.rope2d_sin)

        presents, past_hidden_states = [], []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present, past_hidden_states = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
                past_hidden_states=past_hidden_states,
                image_bidirectional_mask=image_bidirectional_mask,
                modality_mask=modality_mask
            )
            presents.append(present)
        hidden_states = self.text_norm(hidden_states)
        aux_loss = hidden_states.new_zeros(1).squeeze()
        return hidden_states, presents, aux_loss


class MiniRainVForCausalLM(PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed.weight"}
    def __init__(self, config: PreTrainedConfig):
        super().__init__(config)
        self.config = config if config is not None else RainConfig()
        self.model = MiniRainVLM(self.config)
        self.name = self.model.name
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed.weight = self.lm_head.weight
        self.loss_fn = get_loss_fn()
        self.post_init()

    def forward(self, input_ids, pixel_values=None, positions=None, attention_mask=None,
                past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids, pixel_values=pixel_values, positions=positions,
            attention_mask=attention_mask, past_key_values=past_key_values,
            use_cache=use_cache, **kwargs)

        loss = None
        if labels is None:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
        else:
            logits = None
            # shift 作用在 hidden/labels 上（不是 logits），对齐原版 CE 的语义
            h = hidden_states[:, :-1, :].reshape(-1, hidden_states.size(-1))  # [B*(T-1), H]
            y = labels[..., 1:].reshape(-1)  # [B*(T-1)]，图像段/pad 已是 -100
            loss = self.loss_fn(self.lm_head.weight, h, y)

        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits,
                                         past_key_values=past_key_values, hidden_states=hidden_states)
    # def forward(self, input_ids, pixel_values=None, positions=None, attention_mask=None,
    #             past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
    #     hidden_states, past_key_values, aux_loss = self.model(
    #         input_ids, pixel_values=pixel_values, positions=positions,
    #         attention_mask=attention_mask, past_key_values=past_key_values,
    #         use_cache=use_cache, **kwargs)
    #     slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    #     logits = self.lm_head(hidden_states[:, slice_indices, :])
    #     loss = None
    #     if labels is not None:
    #         x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
    #         loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
    #     return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits,
    #                                      past_key_values=past_key_values, hidden_states=hidden_states)

    @torch.inference_mode()
    def generate(self, inputs=None, pixel_values=None, positions=None, attention_mask=None,
                 max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2,
                 streamer=None, use_cache=True, num_return_sequences=1, do_sample=True,
                 repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        if positions is not None: positions = positions.repeat(num_return_sequences, 1, 1)
        if pixel_values is not None and num_return_sequences > 1:
            pixel_values = list(pixel_values) * num_return_sequences
        past_key_values = kwargs.pop("past_key_values", None)
        # 记录 prompt 末尾的位置编号：图像段占用了额外的行号，
        # decode 新 token 的位置必须从这里+1继续，不能用 past_len
        next_pos = int(positions[..., 0].max().item()) + 1 if positions is not None else None
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            step_positions = None
            if next_pos is not None and past_len > 0:
                # decode 阶段：行=列=连续编号，接在 prompt 末尾之后
                pos = torch.full((input_ids.shape[0], 1), next_pos, dtype=torch.long, device=input_ids.device)
                step_positions = torch.stack([pos, pos], dim=-1)
            # outputs = self.forward(
            #     input_ids[:, past_len:],
            #     pixel_values=pixel_values if past_len == 0 else None,  # 图像只在 prefill 进
            #     positions=step_positions if past_len > 0 else positions,
            #     attention_mask=attention_mask,
            #     past_key_values=past_key_values, use_cache=use_cache, **kwargs)
            outputs = self.forward(
                input_ids[:, past_len:],
                pixel_values=pixel_values if past_len == 0 else None,
                positions=step_positions if past_len > 0 else positions,
                attention_mask=attention_mask,
                past_key_values=past_key_values, use_cache=use_cache,
                logits_to_keep=1 if past_len == 0 else 0,  # prefill 只留最后一个
                **kwargs)
            if next_pos is not None: next_pos += 1
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            # pyrefly: ignore [unsupported-operation]
            logits = outputs.logits[:, -1, :] / temperature
            # 禁止生成视觉占位/边界 token（含换行 img_row），防止输出污染
            logits[:, [self.config.img_start_token, self.config.img_end_token, 11,
                       self.config.img_patch_token, 13, self.config.img_row_token]] = -float('inf')
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids