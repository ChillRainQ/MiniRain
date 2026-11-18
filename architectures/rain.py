from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers import GenerationMixin, PreTrainedConfig
from transformers import PreTrainedModel
from components.block import BlockAttnRes, FlushBlockAttnResMiniRainBlock, BlockAttnResMiniRainBlock
from components.norm import NormArgs, TorchRMSNorm
from architectures.config import RainConfig
import torch.nn.functional as F
import torch, math
import torch.nn as nn
import copy


def cache_length(cache):
    if cache is None:
        return 0
    if hasattr(cache, 'get_seq_length'):
        return int(cache.get_seq_length())
    if isinstance(cache, (tuple, list)):
        if not cache:
            return 0
        if len(cache) == 2 and isinstance(cache[0], torch.Tensor):
            return cache[0].shape[1]
        return cache_length(cache[0])
    raise TypeError(f'Unsupported cache type: {type(cache).__name__}')


def repeat_layer_cache(cache, repeats):
    if cache is None:
        return None
    if hasattr(cache, 'batch_repeat_interleave'):
        result = copy.deepcopy(cache)
        result.batch_repeat_interleave(repeats)
        return result
    return tuple(value.repeat_interleave(repeats, dim=0) for value in cache)


class LegacyHybridCache(list):
    is_compileable = False

    def get_seq_length(self):
        return cache_length(self[0]) if self else 0


def precompute_freqs_cis(dim: int, end: int = 32 * 1024, rope_base: float = 1e6, rope_scaling: dict | None = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None:  # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0,
                               1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


class MiniRainModel(nn.Module):
    name = "MiniRain"

    def __init__(self, config: PreTrainedConfig):
        super().__init__()
        if config.type_id != 1:
            raise ValueError('MiniRain is a dense text model; disable use_mot/use_moe/use_adaptive')
        if config.n_hidden_layers <= 0:
            raise ValueError('n_hidden_layers must be positive')
        if config.block_size < 0 or config.block_size % 2:
            raise ValueError('block_size must be zero or a positive even number of sublayers')
        if config.head_dim <= 0 or config.head_dim % 2:
            raise ValueError('1D RoPE requires a positive, even head_dim')
        self.config = config
        self.model_type = self.config.type
        self.vocab_size, self.n_hidden_layers = self.config.vocab_size, self.config.n_hidden_layers
        self.embed = nn.Embedding(self.vocab_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.n_dropout)
        self.layers = nn.ModuleList([
            (FlushBlockAttnResMiniRainBlock
             if (config.block_size >= 2 and i % (config.block_size // 2) == 0)
             else BlockAttnResMiniRainBlock)(i, config)
            for i in range(config.n_hidden_layers)
        ])
        self.final_attn_res = (BlockAttnRes(config)
                               if config.block_size >= 2
                               else None)
        self.norm = TorchRMSNorm(NormArgs(dim=self.config.hidden_size, eps=self.config.rms_norm_eps))
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings,
                                                    rope_base=self.config.rope_theta,
                                                    rope_scaling=self.config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False,
                **kwargs):
        batch_size, seq_length = input_ids.shape

        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * self.n_hidden_layers
        if len(past_key_values) != self.n_hidden_layers:
            raise ValueError('past_key_values must contain one entry per layer')
        start_pos = cache_length(past_key_values)
        if any(cache_length(value) != start_pos for value in past_key_values):
            raise ValueError('All layer caches must have the same sequence length')
        if start_pos + seq_length > self.config.max_position_embeddings:
            raise ValueError('Sequence exceeds max_position_embeddings')
        hidden_states = self.dropout(self.embed(input_ids))
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length],
                               self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        past_hidden_states = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present, past_hidden_states = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
                past_hidden_states=past_hidden_states,
            )
            presents.append(present)
        if self.final_attn_res is not None:
            hidden_states = self.final_attn_res(past_hidden_states, hidden_states)
        hidden_states = self.norm(hidden_states)
        aux_loss = hidden_states.new_zeros(1).squeeze()
        return hidden_states, presents, aux_loss


class MiniRainForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = RainConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed.weight"}

    def __init__(self, config: PreTrainedConfig):
        super().__init__(config)
        self.config = config if config is not None else RainConfig()
        self.model = MiniRainModel(self.config)
        self.name = self.model.name
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed.weight = self.lm_head.weight
        self.post_init()

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, MiniRainModel):
            # HF materializes non-persistent buffers as empty tensors when
            # loading from meta. Rebuild RoPE instead of using uninitialized data.
            with torch.device(module.freqs_cos.device):
                cos, sin = precompute_freqs_cis(
                    self.config.head_dim, self.config.max_position_embeddings,
                    self.config.rope_theta, self.config.rope_scaling)
            module.freqs_cos.copy_(cos)
            module.freqs_sin.copy_(sin)

    def forward(self, input_ids, attention_mask=None, past_key_values=None,
                use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache,
                                                              **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        loss = None
        if labels is not None:
            full_logits = self.lm_head(hidden_states)
            x, y = full_logits[..., :-1, :].float().contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
            logits = full_logits[:, slice_indices, :]
        else:
            logits = self.lm_head(hidden_states[:, slice_indices, :])
        # pyrefly: ignore [bad-argument-type]
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values,
                                         hidden_states=hidden_states)

    @torch.inference_mode()
    # pyrefly: ignore [bad-override]
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50,
                 eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True,
                 repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs)
        past_key_values = kwargs.pop("past_key_values", None)
        return_kv = kwargs.pop('return_kv', False)
        if past_key_values is None:
            input_ids, attention_mask = self._left_pad_for_generation(input_ids, attention_mask)
        input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
        attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0) if attention_mask is not None else None
        if past_key_values is not None and num_return_sequences > 1:
            past_key_values = [repeat_layer_cache(value, num_return_sequences) for value in past_key_values]
        do_sample = do_sample and temperature > 0
        kwargs['logits_to_keep'] = 1
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = cache_length(past_key_values)
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache,
                                   **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)],
                                       -1) if attention_mask is not None else None
            # pyrefly: ignore [unsupported-operation]
            logits = outputs.logits[:, -1, :].float()
            if do_sample:
                logits = logits / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]);
                    score = logits[i, seen];
                    logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if do_sample and top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if do_sample and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(
                logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1),
                                                                  next_token.new_full((next_token.shape[0], 1),
                                                                                      eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if return_kv: return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids

    @staticmethod
    def _left_pad_for_generation(input_ids, attention_mask):
        if attention_mask is None:
            return input_ids, attention_mask
        ids, mask = input_ids.clone(), attention_mask.clone()
        for row in range(input_ids.shape[0]):
            valid = attention_mask[row].bool()
            right_padded = bool(valid[0]) and not bool(valid[-1]) and bool(
                (valid[:-1].to(torch.int8) >= valid[1:].to(torch.int8)).all())
            if right_padded:
                order = torch.cat([(~valid).nonzero(as_tuple=True)[0], valid.nonzero(as_tuple=True)[0]])
                ids[row] = input_ids[row].index_select(0, order)
                mask[row] = attention_mask[row].index_select(0, order)
        return ids, mask

    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False

    @staticmethod
    def _expand_inputs_for_generation(expand_size=1, is_encoder_decoder=False, input_ids=None, **model_kwargs):
        input_ids, model_kwargs = GenerationMixin._expand_inputs_for_generation(
            expand_size=expand_size, is_encoder_decoder=is_encoder_decoder,
            input_ids=input_ids, **model_kwargs)
        cache = model_kwargs.get('past_key_values')
        if cache is not None and expand_size > 1:
            model_kwargs['past_key_values'] = LegacyHybridCache(
                [repeat_layer_cache(value, expand_size) for value in cache])
        return input_ids, model_kwargs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        kwargs.pop('next_sequence_length', None)
        kwargs.pop('is_first_iteration', None)
        past_len = cache_length(past_key_values)
        return dict(input_ids=input_ids[:, past_len:], past_key_values=past_key_values,
                    attention_mask=attention_mask, **{'logits_to_keep': 1, **kwargs})

    @torch.inference_mode()
    def generate_hf(self, inputs=None, attention_mask=None, max_new_tokens=8192,
                    temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2,
                    streamer=None, use_cache=True, num_return_sequences=1, do_sample=True,
                    repetition_penalty=1.0, **kwargs):
        """HF decoding over the same per-layer KV cache as generate()."""
        input_ids = kwargs.pop('input_ids', inputs)
        if kwargs.get('past_key_values') is None:
            input_ids, attention_mask = self._left_pad_for_generation(input_ids, attention_mask)
        elif type(kwargs['past_key_values']) is list:
            kwargs['past_key_values'] = LegacyHybridCache(kwargs['past_key_values'])
        do_sample = do_sample and temperature > 0
        sampling = dict(temperature=temperature, top_p=top_p, top_k=top_k) if do_sample else {}
        return super().generate(input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, eos_token_id=eos_token_id, streamer=streamer,
            use_cache=use_cache, num_return_sequences=num_return_sequences, do_sample=do_sample,
            repetition_penalty=repetition_penalty, disable_compile=True, **sampling, **kwargs)









