from dataclasses import dataclass
import torch.nn.functional as F

import torch
import torch.nn as nn

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
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)



if __name__ == '__main__':
    x = torch.randn(4, 128, 1024, device="cuda", dtype=torch.bfloat16)
    old = RMSNorm(NormArgs(eps=1e-6, dim=1024)).cuda()
    new = TorchRMSNorm(NormArgs(eps=1e-6, dim=1024)).cuda()
    new.weight.data.copy_(old.weight.data)
    a, b = old(x), new(x)
    print((a - b).abs().max(), a.dtype)