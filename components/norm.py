from dataclasses import dataclass
import torch.nn.functional as F

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _rms_norm_kernel(X, W, Y, stride, N, eps, BLOCK: tl.constexpr):
        """单 kernel RMSNorm：每行一个 program，fp32 累加，输出回原 dtype。"""
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
        tl.store(Y + row * stride + cols, (x * rstd * w).to(Y.dtype.element_ty), mask=mask)

    def _triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """对最后一维做 RMSNorm，支持任意前置维度。"""
        n = x.shape[-1]
        x2d = x.reshape(-1, n)
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()
        y = torch.empty_like(x2d)
        block = triton.next_power_of_2(n)
        _rms_norm_kernel[(x2d.shape[0],)](
            x2d, weight, y, x2d.stride(0), n, eps,
            BLOCK=block, num_warps=8 if block >= 2048 else 4,
        )
        return y.view(x.shape)

@dataclass
class NormArgs:
    """
    归一化参数。
    """
    dim: int
    eps: float


class RMSNorm(nn.Module):
    name = "rms"
    def __init__(self, args: NormArgs):
        super().__init__()
        self.eps = args.eps
        self.weight = nn.Parameter(torch.ones(args.dim))

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._rms(x.float()).type_as(x) * self.weight


class TorchRMSNorm(nn.Module):
    name = "rms"
    def __init__(self, args: NormArgs):
        super().__init__()
        self.eps = args.eps
        self.weight = nn.Parameter(torch.ones(args.dim))
        self.dim = args.dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (HAS_TRITON and x.is_cuda and not x.requires_grad
                and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
                and x.shape[-1] == self.dim
                and self.weight.is_cuda
                and self.weight.device == x.device
                and (x.device.index or 0) == torch.cuda.current_device()):
            return _triton_rms_norm(x, self.weight, self.eps)
        return F.rms_norm(x, (self.dim,), self.weight.to(x.device), self.eps)



if __name__ == '__main__':
    x = torch.randn(4, 128, 1024, device="cuda", dtype=torch.bfloat16)
    old = RMSNorm(NormArgs(eps=1e-6, dim=1024)).cuda()
    new = TorchRMSNorm(NormArgs(eps=1e-6, dim=1024)).cuda()
    new.weight.data.copy_(old.weight.data)
    a, b = old(x), new(x.detach())
    print(f"triton={HAS_TRITON}  old-vs-new max diff={(a - b).abs().max().item():.3e}  dtype={b.dtype}")
    if HAS_TRITON:
        ref = F.rms_norm(x, (1024,), new.weight, 1e-6)
        print(f"triton-vs-composite max diff={(ref - b).abs().max().item():.3e}")