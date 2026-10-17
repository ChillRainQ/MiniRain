from dataclasses import dataclass
from abc import ABC, abstractmethod

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

