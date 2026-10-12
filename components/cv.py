import torch.nn as nn
class ConvStem(nn.Module):
    """16× 降采样卷积干，替代单层 patch conv。仍然是 encoder-free：随机初始化，和主干一起从头训。"""
    def __init__(self, hidden: int):
        super().__init__()
        chs = [3, 128, 256, 512, hidden]     # 4 层 stride2 = 16×
        blocks = []
        for i in range(4):
            blocks += [
                nn.Conv2d(chs[i], chs[i+1], kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(32, chs[i+1]) if i < 3 else nn.Identity(),  # 最后一层用 img_norm 归一
                nn.SiLU(inplace=True) if i < 3 else nn.Identity(),
            ]
        self.stem = nn.Sequential(*blocks)

    def forward(self, x):
        return self.stem(x)