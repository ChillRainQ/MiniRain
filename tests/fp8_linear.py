"""Benchmark-only FP8 prototype: native scaled_mm, FP32 master parameters.

Strict per-GEMM finite checks synchronize the GPU. This intentionally favors
diagnosability over speed; no inference quantization or global monkeypatching.
"""
import types
import torch
import torch.nn.functional as F

DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(DTYPE).max


def _quantize(value, stats):
    value = value.detach().float()
    amax = value.abs().amax()
    maximum = amax.item()
    if not torch.isfinite(amax).item():
        raise FloatingPointError('FP8 operand contains inf/nan')
    scale = (amax / FP8_MAX).clamp_min(1e-12)
    quantized = (value / scale).clamp(-FP8_MAX, FP8_MAX).to(DTYPE)
    stats['amax_max'] = max(stats.get('amax_max', 0.), maximum)
    stats['quantizations'] = stats.get('quantizations', 0) + 1
    return quantized.contiguous(), scale


def _mm(a, b, scale_a, scale_b, stats):
    # cuBLASLt requires row-major A and column-major B for this FP8 recipe.
    a = a.contiguous()
    b = b.t().contiguous().t()
    with torch.autocast('cuda', enabled=False):
        result = torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b,
                                  out_dtype=torch.bfloat16, use_fast_accum=False)
    if not torch.isfinite(result).all().item():
        raise FloatingPointError('FP8 GEMM produced inf/nan')
    stats['scaled_mm_calls'] = stats.get('scaled_mm_calls', 0) + 1
    return result


class _FP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stats):
        if x.device.type != 'cuda' or weight.device != x.device:
            raise ValueError('FP8 Linear requires CUDA operands on the same device')
        shape = x.shape
        m, k, n = x.numel() // shape[-1], shape[-1], weight.shape[0]
        mp, kp, np = ((v + 15) // 16 * 16 for v in (m, k, n))
        # Match BF16 autocast inputs before FP8 quantization.
        a = F.pad(x.reshape(m, k).to(torch.bfloat16), (0, kp-k, 0, mp-m))
        w = F.pad(weight.to(torch.bfloat16), (0, kp-k, 0, np-n))
        aq, sa = _quantize(a, stats)
        wq, sw = _quantize(w, stats)
        out = _mm(aq, wq.t(), sa, sw, stats)[:m, :n]
        if bias is not None:
            out = out + bias.to(torch.bfloat16)
        ctx.save_for_backward(aq, wq, sa, sw)
        ctx.shape, ctx.sizes = shape, (m, k, n, mp, kp, np)
        ctx.input_dtype, ctx.weight_dtype = x.dtype, weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.stats = stats
        return out.reshape(*shape[:-1], n)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        aq, wq, sa, sw = ctx.saved_tensors
        m, k, n, mp, kp, np = ctx.sizes
        g = grad_output.reshape(m, n).to(torch.bfloat16)
        gq, sg = _quantize(F.pad(g, (0, np-n, 0, mp-m)), ctx.stats)
        dx = _mm(gq, wq, sg, sw, ctx.stats)[:m, :k]
        dw = _mm(gq.t(), aq, sg, sa, ctx.stats)[:n, :k]
        db = g.sum(0).to(ctx.bias_dtype) if ctx.bias_dtype is not None else None
        return dx.reshape(ctx.shape).to(ctx.input_dtype), dw.to(ctx.weight_dtype), db, None


class FP8Linear(torch.nn.Linear):
    @classmethod
    def from_linear(cls, source, stats):
        result = cls(source.in_features, source.out_features, source.bias is not None, device='meta')
        result.weight, result.bias = source.weight, source.bias
        result.stats = stats
        result.train(source.training)
        return result

    def forward(self, x):
        return _FP8Linear.apply(x, self.weight, self.bias, self.stats)


def _swiglu_forward(self, x):
    # The production FusedSwiGLU uses functional linear and bypasses child
    # modules. Override only this benchmark instance; retain separate weights.
    packed = torch.cat([self.gate.weight, self.up.weight], 0)
    gate, up = _FP8Linear.apply(x, packed, None, self._fp8_stats).chunk(2, -1)
    return self.down(F.silu(gate) * up)


def enable_fp8(model):
    from components.mlp import FusedSwiGLU
    stats = dict(amax_max=0., quantizations=0, scaled_mm_calls=0)
    converted, skipped = [], []
    def convert(parent, prefix=''):
        fused = isinstance(parent, FusedSwiGLU)
        if fused:
            parent._fp8_stats = stats
            parent.forward = types.MethodType(_swiglu_forward, parent)
            converted.append(prefix + '.gate_up(functional)')
        for name, child in list(parent.named_children()):
            full = prefix + '.' + name if prefix else name
            if fused and name in ('gate', 'up'):
                continue
            if isinstance(child, torch.nn.Linear):
                if full == 'lm_head' or min(child.in_features, child.out_features) < 16:
                    skipped.append(full)
                else:
                    setattr(parent, name, FP8Linear.from_linear(child, stats))
                    converted.append(full)
            else:
                convert(child, full)
    convert(model)
    return dict(stats=stats, converted=converted, bf16_linears=skipped,
                recipe='e4m3fn dynamic per-tensor; fwd/dgrad/wgrad',
                strict_finite_checks=True)


def probe_backend(device):
    stats = {}
    a = torch.eye(16, device=device)
    q, scale = _quantize(a, stats)
    try:
        out = _mm(q, q.t(), scale, scale, stats)
        torch.testing.assert_close(out.float(), a, atol=0.02, rtol=0.02)
    except RuntimeError as exc:
        raise RuntimeError(f'Native FP8 scaled_mm unavailable on {torch.cuda.get_device_name(device)} '
                           f'(SM{torch.cuda.get_device_capability(device)}, torch={torch.__version__})') from exc
    return dict(torch_version=torch.__version__, capability=torch.cuda.get_device_capability(device),
                scaled_mm_probe_passed=True)
