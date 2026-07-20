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
    """Root Mean Square Layer Normalization。

    RMSNorm 将输入按 RMS 归一化，相比 LayerNorm 省去了均值中心化
    的计算，在 LLM（如 Llama、Mistral）中被广泛采用。
    """
    name = "rms"
    def __init__(self, args: NormArgs):
        super().__init__()
        self.eps = args.eps
        self.weight = nn.Parameter(torch.ones(args.dim))

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._rms(x.float()).type_as(x) * self.weight

