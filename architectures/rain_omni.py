from torch import nn
from torch.nn import functional as F

from architectures.config import RainOmniConfig
from architectures.rain import MiniRainForCausalLM


class MMAudioProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        return self.mlp(x)


class MMVisionProjector(nn.Module):
    def __init__(self, in_dim, out_dim, source_tokens=64, target_tokens=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        return self.mlp(x)


class MiniRainOmni(MiniRainForCausalLM):
    config_class = RainOmniConfig
    def __init__(self, config: RainOmniConfig, audio_encoder_path: str, vision_encoder_path: str):
        config = config or RainOmniConfig()
        super().__init__(config)
