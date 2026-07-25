from transformers import PretrainedConfig
import math
model_type: dict[int, str] = {
    1 : "DENSE",
    2 : "MOE",
    3 : "DENSE-LOOP",
    4 : "MOE-LOOP",
    5 : "MOE-BOUNCE",
    6 : "DENSE-BOUNCE",
}

class RainConfig(PretrainedConfig):
    def __init__(self, hidden_size=768, n_hidden_layers=8,
                 use_moe=False, use_loop=False, use_bounce=False,  **kwargs):
        super().__init__(**kwargs)
        if use_moe:
            self.type_id = 2
            if use_loop:
                self.type_id = 4
            elif use_bounce:
                self.type_id = 5
        else:
            self.type_id = 1
            if use_loop:
                self.type_id = 3
            elif use_bounce:
                self.type_id = 6
        self.vocab_size: int = int(kwargs.get("vocab_size", 6400))
        self.hidden_size: int = hidden_size
        self.n_hidden_layers: int = n_hidden_layers
        self.type: str = model_type[self.type_id]
        self.bos_token_id: int = int(kwargs.get("bos_token_id", 1))
        self.eos_token_id: int = int(kwargs.get("eos_token_id", 2))
        self.n_dropout: float = float(kwargs.get("dropout", 0.0))
        self.flash_attn: bool = True
        self.n_attn_heads: int = int(kwargs.get("n_attn_heads", 8))
        self.n_key_value_heads: int = kwargs.get("n_key_value_heads", 4)
        self.head_dim: int = int(kwargs.get("head_dim", self.hidden_size // self.n_attn_heads))
        self.hidden_act: str = kwargs.get("hidden_act", 'silu')
        self.norm_type: str = kwargs.get("norm_type", "rms")
        self.max_position_embeddings: int = int(kwargs.get("max_position_embeddings", 32768))
        self.intermediate_size: int = int(kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64))
        self.rms_norm_eps: float = float(kwargs.get("rms_norm_eps", 1e-6))
        self.rope_theta: float = float(kwargs.get("rope_theta", 1e6))
        self.tie_word_embeddings: bool = bool(kwargs.get("tie_word_embeddings", True))
        self.inference_rope_scaling: bool = bool(kwargs.get("inference_rope_scaling", False))
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        # MOE config
        self.use_moe = use_moe
        self.norm_topk_prob: bool = bool(kwargs.get("norm_topk_prob", True))
        self.n_experts: int = 0 if not use_moe else kwargs.get("n_experts", 4)
        self.n_share_experts: int = 0 if not use_moe else kwargs.get("n_share_experts", 0)
        self.per_active_experts: int = 0 if not use_moe else kwargs.get("per_active_experts", 1)
        self.router_aux_loss_coef: float = float(kwargs.get("router_aux_loss_coef", 5e-4))
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        # Loop flag
        self.use_loop = use_loop
        # Bounce flag
        self.use_bounce = use_bounce
        # loop config
        self.loop_start: int = int(kwargs.get("loop_start", 0))
        self.loop_end: int = int(kwargs.get("loop_end", self.n_hidden_layers))
        self.max_loop_iter: int = int(kwargs.get("max_loop_iter", 3))
        # bounce config：区间 [bounce_start, bounce_end] 含端点，首层与末两层不进区间
        self.bounce_start: int = int(kwargs.get("bounce_start", min(2, self.n_hidden_layers)))
        self.bounce_end: int = int(kwargs.get("bounce_end", self.n_hidden_layers - 1 - 2))
        self.phase_init_std: float = float(kwargs.get("phase_init_std", 0.02))
        # attn_res
        self.full_attn_res = bool(kwargs.get("full_attn_res", False))
        self.block_attn_res = bool(kwargs.get("block_attn_res", False))
        self.block_number = int(kwargs.get("block_number", 0))


class RainOmniConfig(RainConfig):
    ...