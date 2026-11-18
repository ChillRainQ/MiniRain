"""Benchmark-only delayed FP8, compile-capturable native GEMM/custom quant op.

Forward/backward contain no scalar reads or Python counters. GPU scales stay
fixed between controller updates; amax collection is fused into quantization.
"""
import types
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(DTYPE).max


@triton.jit
def _quant_kernel(X, S, A, Q, SIZE, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < SIZE, 0).to(tl.bfloat16).to(tl.float32)
    scale = tl.load(S)
    finite = (x == x) & (tl.abs(x) != float('inf'))
    local_max = tl.max(tl.where(finite, tl.abs(x), float('inf')), 0)
    tl.atomic_max(A, local_max, sem='relaxed')
    q = tl.minimum(tl.maximum(x / scale, -448.), 448.)
    tl.store(Q + i, q, i < SIZE)


@torch.library.custom_op('minirain_bench::fp8_quantize', mutates_args={'amax'})
def _quantize(value: torch.Tensor, scale: torch.Tensor, amax: torch.Tensor) -> torch.Tensor:
    # Opaque to AOT, but fully captured as a registered mutating custom op.
    value = value.contiguous()
    out = torch.empty(value.shape, device=value.device, dtype=DTYPE)
    _quant_kernel[(triton.cdiv(value.numel(), 2048),)](
        value, scale, amax, out, value.numel(), 2048)
    return out


@_quantize.register_fake
def _(value, scale, amax):
    return torch.empty(value.shape, device=value.device, dtype=DTYPE)


def _mm(a, b, scale_a, scale_b):
    with torch.autocast('cuda', enabled=False):
        return torch._scaled_mm(a.contiguous(), b.t().contiguous().t(),
            scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16, use_fast_accum=False)


class QuantState(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.register_buffer('scales', torch.ones(3, device=device))
        self.register_buffer('amax', torch.zeros(3, device=device))
        self.calibrating = True


def _observe(value, accumulator):
    # Only used during the untimed BF16 calibration pass.
    accumulator.copy_(torch.maximum(accumulator, value.detach().float().abs().amax()))


class _FP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, state):
        shape = x.shape
        m, k, n = x.numel() // shape[-1], shape[-1], weight.shape[0]
        mp, kp, np = ((v + 15) // 16 * 16 for v in (m, k, n))
        a = F.pad(x.reshape(m, k).to(torch.bfloat16), (0, kp-k, 0, mp-m))
        w = F.pad(weight.to(torch.bfloat16), (0, kp-k, 0, np-n))
        sx, sw, sg = state.scales.unbind()
        ax, aw, ag = state.amax.unbind()
        if state.calibrating:
            _observe(a, ax); _observe(w, aw)
            out = a @ w.t()
            aq, wq = a, w
        else:
            aq, wq = _quantize(a, sx, ax), _quantize(w, sw, aw)
            out = _mm(aq, wq.t(), sx, sw)
        out = out[:m, :n]
        if bias is not None: out = out + bias.to(torch.bfloat16)
        ctx.save_for_backward(aq, wq, sx, sw, sg, ag)
        ctx.shape, ctx.sizes = shape, (m, k, n, mp, kp, np)
        ctx.input_dtype, ctx.weight_dtype = x.dtype, weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.calibrating = state.calibrating
        return out.reshape(*shape[:-1], n)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        aq, wq, sx, sw, sg, ag = ctx.saved_tensors
        m, k, n, mp, kp, np = ctx.sizes
        g = grad_output.reshape(m, n).to(torch.bfloat16)
        padded = F.pad(g, (0, np-n, 0, mp-m))
        if ctx.calibrating:
            _observe(padded, ag)
            dx, dw = padded @ wq, padded.t() @ aq
        else:
            gq = _quantize(padded, sg, ag)
            dx = _mm(gq, wq, sg, sw)
            dw = _mm(gq.t(), aq, sg, sx)
        db = g.sum(0).to(ctx.bias_dtype) if ctx.bias_dtype is not None else None
        return dx[:m, :k].reshape(ctx.shape).to(ctx.input_dtype), dw[:n, :k].to(ctx.weight_dtype), db, None


class FP8Controller:
    def __init__(self, device, scale_interval=32, check_interval=100):
        self.states = []
        self.scale_interval, self.check_interval = scale_interval, check_interval
        self.completed_steps, self.scale_updates, self.host_checks = 0, 0, 0
        self.health = torch.ones((), device=device, dtype=torch.bool)
        self.device = device

    def new_state(self):
        state = QuantState(self.device)
        self.states.append(state)
        return state

    @torch.no_grad()
    def update_scales(self):
        observed = torch.stack([s.amax for s in self.states])
        self.health.logical_and_(torch.isfinite(observed).all())
        safe = torch.where(torch.isfinite(observed), observed, torch.zeros_like(observed))
        # 2x headroom for unseen values between delayed updates.
        scales = (safe * (2. / FP8_MAX)).clamp_min(1e-12)
        for state, new_scale in zip(self.states, scales.unbind()):
            state.scales.copy_(new_scale)
            state.amax.zero_()
        self.scale_updates += 1

    @torch.no_grad()
    def finish_calibration(self):
        self.update_scales()
        for state in self.states: state.calibrating = False

    @torch.no_grad()
    def after_step(self, loss, model):
        self.completed_steps += 1
        self.health.logical_and_(torch.isfinite(loss.detach()).all())
        if self.completed_steps % self.scale_interval == 0: self.update_scales()
        if self.completed_steps % self.check_interval == 0: self.check_now(loss, model)

    @torch.no_grad()
    def check_now(self, loss=None, model=None):
        flags = [self.health, torch.isfinite(torch.stack([s.amax for s in self.states])).all()]
        if loss is not None: flags.append(torch.isfinite(loss.detach()).all())
        if model is not None:
            for p in model.parameters():
                flags.append(torch.isfinite(p).all())
                flags.append(torch.zeros((),device=p.device,dtype=torch.bool) if p.grad is None else torch.isfinite(p.grad).all())
        # Exactly one host read for this periodic/endpoint check.
        valid = torch.stack(flags).all().item()
        self.host_checks += 1
        if not valid: raise FloatingPointError('FP8 delayed amax/loss/gradient/master parameters contain inf/nan or missing gradients')

    def report(self):
        return dict(completed_steps=self.completed_steps,scale_updates=self.scale_updates,
                    host_checks=self.host_checks,scale_interval=self.scale_interval,
                    check_interval=self.check_interval,quantizer='fused GPU amax+cast, no host reads',
                    margin=2.)


class FP8Linear(torch.nn.Linear):
    @classmethod
    def from_linear(cls, source, controller):
        result=cls(source.in_features,source.out_features,source.bias is not None,device='meta')
        result.weight,result.bias=source.weight,source.bias
        result.fp8_state=controller.new_state()
        result.train(source.training)
        return result

    def forward(self,x): return _FP8Linear.apply(x,self.weight,self.bias,self.fp8_state)


def _swiglu_forward(self,x):
    packed=torch.cat([self.gate.weight,self.up.weight],0)
    gate,up=_FP8Linear.apply(x,packed,None,self.fp8_state).chunk(2,-1)
    return self.down(F.silu(gate)*up)


def enable_fp8(model,scale_interval=32,check_interval=100):
    from components.mlp import FusedSwiGLU
    controller=FP8Controller(next(model.parameters()).device,scale_interval,check_interval)
    converted,skipped=[],[]
    def convert(parent,prefix=''):
        fused=isinstance(parent,FusedSwiGLU)
        if fused:
            parent.fp8_state=controller.new_state()
            parent.forward=types.MethodType(_swiglu_forward,parent)
            converted.append(prefix+'.gate_up(functional)')
        for name,child in list(parent.named_children()):
            full=prefix+'.'+name if prefix else name
            if name=='fp8_state' or (fused and name in ('gate','up')): continue
            if isinstance(child,torch.nn.Linear):
                if full=='lm_head' or min(child.in_features,child.out_features)<16: skipped.append(full)
                else:
                    setattr(parent,name,FP8Linear.from_linear(child,controller));converted.append(full)
            else: convert(child,full)
    convert(model)
    return dict(_controller=controller,converted=converted,bf16_linears=skipped,
                recipe='e4m3fn delayed per-tensor; GPU amax+cast; fwd/dgrad/wgrad')


def probe_backend(device):
    a=torch.eye(16,device=device).to(DTYPE)
    s=torch.ones((),device=device)
    out=_mm(a,a.t(),s,s)
    torch.testing.assert_close(out.float(),torch.eye(16,device=device),atol=0.02,rtol=0.02)
    return dict(torch_version=torch.__version__,capability=torch.cuda.get_device_capability(device),scaled_mm_probe_passed=True)
