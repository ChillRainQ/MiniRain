from components.mlp import SwiGLU
from components.mlp import MLPArgs
from components.norm import NormArgs
from components.attention import AttentionArgs
from components.attention import GroupQueryAttention
from architectures.config import RainConfig
import torch.nn as nn

from components.norm import RMSNorm



class MiniRainBlock(nn.Module):
    def __init__(self, layer_id: int, config: RainConfig):
        super().__init__()
        self.layer_id = layer_id
        self.attn = GroupQueryAttention(AttentionArgs(config))
        self.input_layernorm = RMSNorm(NormArgs(dim=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attn_layernorm = RMSNorm(NormArgs(dim=config.hidden_size, eps=config.rms_norm_eps))
        self.mlp = SwiGLU(MLPArgs(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size, bias=False))
        
    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        res = hidden_states
        hidden_states, past_key_value = self.attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states += res
        output = hidden_states + self.mlp(
            self.post_attn_layernorm(hidden_states)
        )
        return output, past_key_value
    
