"""统计 MiniRain 纯文本模型参数量，无需加载权重或 tokenizer。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from architectures.config import RainConfig
from architectures.rain import MiniRainForCausalLM
from components.attention import GroupQueryAttention


def estimate_flops(model, seq_len):
    """Leading logical FLOPs, MAC=2; training ~3x forward, without recomputation.

    Includes all Linear projections (including tied LM head) and causal GQA.
    Excludes norms, nonlinearities,
    AttnRes aggregation, CE, optimizer and chunk-kernel/padding overhead.
    This is an analytical estimate, not a profiler measurement.
    """
    linear = 2 * sum(m.weight.numel() for m in model.modules() if isinstance(m, torch.nn.Linear))
    attention = 0
    for layer in model.model.layers:
        if isinstance(layer.attn, GroupQueryAttention):
            attention += 2 * model.config.hidden_size * (seq_len + 1)
    forward = linear + attention
    return dict(linear=linear,
                causal_gqa=attention, forward_per_token=forward,
                training_per_token_estimate=3 * forward)


def main():
    parser = argparse.ArgumentParser(
        description="MiniRain 纯文本参数量统计（默认采用 20B token 推荐结构）")
    parser.add_argument('--hidden_size', default=1024, type=int, help="隐藏层维度")
    parser.add_argument('--n_attn_heads', default=8, type=int)
    parser.add_argument('--n_key_value_heads', default=4, type=int)
    parser.add_argument('--intermediate_size', default=None, type=int, help="默认按 hidden_size 的3倍向上取整到64的倍数")
    parser.add_argument('--seq_len', default=4096, type=int, help="FLOPs 估算的完整序列长度")
    parser.add_argument('--json_output', type=Path, default=None)
    parser.add_argument('--num_hidden_layers', default=24, type=int, help="Transformer 层数量")
    parser.add_argument('--block_size', default=6, type=int, help="残差块大小，以注意力/FFN 子层计数；0 表示不分块")
    args = parser.parse_args()
    if args.n_attn_heads <= 0 or args.n_key_value_heads <= 0 or args.n_attn_heads % args.n_key_value_heads:
        parser.error('注意力头数须为正，且 n_attn_heads 能被 n_key_value_heads 整除')
    if args.hidden_size <= 0 or args.hidden_size % (2 * args.n_attn_heads):
        parser.error('--hidden_size 须能被 2 * n_attn_heads 整除（RoPE head_dim 为偶数）')
    if args.seq_len <= 0 or (args.intermediate_size is not None and args.intermediate_size <= 0):
        parser.error('序列长度和 FFN 中间维度须为正数')
    if args.num_hidden_layers <= 0:
        parser.error('--num_hidden_layers 须为正整数')
    if args.block_size < 0 or args.block_size % 2:
        parser.error('--block_size 须为 0 或正偶数')

    extra = {} if args.intermediate_size is None else dict(intermediate_size=args.intermediate_size)
    config = RainConfig(
        hidden_size=args.hidden_size, n_hidden_layers=args.num_hidden_layers,
        n_attn_heads=args.n_attn_heads, n_key_value_heads=args.n_key_value_heads,
        vocab_size=16384, tie_word_embeddings=True,
        max_position_embeddings=args.seq_len, block_size=args.block_size, **extra)
    # 参数形状即可确定参数量；meta 避免分配完整权重或占用 GPU。
    with torch.device('meta'):
        model = MiniRainForCausalLM(config)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gqa_layers = sum(isinstance(layer.attn, GroupQueryAttention) for layer in model.model.layers)
    print(f'Model Type: {model.name} ({type(model).__name__})')
    print(f'Architecture: layers={config.n_hidden_layers}, hidden={config.hidden_size}, '
          f'FFN={config.intermediate_size}, heads={config.n_attn_heads}, '
          f'kv_heads={config.n_key_value_heads}, block_size={config.block_size}')
    print(f'Attention Layers: GQA={gqa_layers}')
    # parameters() 会去重，绑定的 embedding/lm_head 仅计一次。
    print(f'[ScalingLaw] Model Params: {total / 1e6:.2f}M ({total:,})')
    print(f'[ScalingLaw] Trainable Params: {trainable / 1e6:.2f}M ({trainable:,})')
    flops = estimate_flops(model, args.seq_len)
    print(f'[FLOPs estimate] T={args.seq_len}, forward={flops["forward_per_token"] / 1e9:.4f} GF/token, '
          f'training~{flops["training_per_token_estimate"] / 1e9:.4f} GF/token (logical, excludes chunk overhead/recompute)')
    if args.json_output is not None:
        result = dict(hidden_size=config.hidden_size, num_hidden_layers=config.n_hidden_layers,
                      n_attn_heads=config.n_attn_heads, n_key_value_heads=config.n_key_value_heads,
                      intermediate_size=config.intermediate_size, block_size=config.block_size,
                      seq_len=args.seq_len, parameters=total, trainable_parameters=trainable,
                      flops=flops)
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
