from torch import nn
from torch.nn import functional as F

from architectures.config import RainOmniConfig
from architectures.rain import MiniRainForCausalLM

# 映射
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


# 映射
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

class TalkerHead(nn.Module):
    def __init__(self, in_features, out_features, num_layers=8, rank=256):
        ...

class TalkerModule(nn.Module):
    def __init__(self, config):
        ...

class MiniRainOmni(MiniRainForCausalLM):
    config_class = RainOmniConfig
    def __init__(self, config: RainOmniConfig, audio_encoder_path: str, vision_model_path: str):
        config = config or RainOmniConfig()
        super().__init__(config)
        object.__setattr__(self, 'thinker', self.model)
        object.__setattr__(self.model, 'lm_head', self.lm_head)

        # 投影器
        self.audio_proj = MMAudioProjector(config.audio_hidden_size, config.hidden_size)
        self.vision_proj = MMVisionProjector(config.image_hidden_size, config.hidden_size,
                                             target_tokens=config.image_token_len)
        self.talker = TalkerModule(config)
        # 编码器
        audio_encoder, audio_processor = self.load_sensevoice(audio_encoder_path)
        object.__setattr__(self, 'audio_encoder', audio_encoder)
        object.__setattr__(self, 'audio_processor', audio_processor)
        vision_encoder, vision_processor = self.load_vision(vision_model_path)
        object.__setattr__(self, 'vision_encoder', vision_encoder)
        object.__setattr__(self, 'vision_processor', vision_processor)