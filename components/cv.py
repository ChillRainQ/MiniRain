import torch.nn as nn


def _gn(ch: int) -> nn.GroupNorm:
    """通道数不被32整除时自动降级"""
    return nn.GroupNorm(min(32, ch), ch)


class ResBlock(nn.Module):
    """下采样的残差"""
    def __init__(self, cin: int, cout: int, stride: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.norm1 = _gn(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.norm2 = _gn(cout)
        self.act = nn.SiLU(inplace=True)
        self.shortcut = nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False),
            _gn(cout),
        ) if (stride != 1 or cin != cout) else nn.Identity()

    def forward(self, x):
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class ConvStem(nn.Module):
    """
    16x 降采样卷积干。encoder-free 优化版：轻量通道 + 残差 + 显式初始化。
    最后一层后不接 norm/act，由下游 img_norm 处理（与原版约定一致）。
    """
    def __init__(self, hidden: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False),   # /2
            _gn(64),
            nn.SiLU(inplace=True),
            ResBlock(64, 128),                        # /4
            ResBlock(128, 256),                       # /8
            nn.Conv2d(256, hidden, 3, 2, 1, bias=False),  # /16
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # 最后一层无 norm 直接输出，截断初始化防激活值漂移
        nn.init.trunc_normal_(self.stem[-1].weight, std=0.02)
        if hasattr(self.stem[-1], 'bias') and self.stem[-1].bias is not None:
            nn.init.zeros_(self.stem[-1].bias)

    def forward(self, x):
        return self.stem(x)
# class ConvStem(nn.Module):
#     """16× 降采样卷积干，替代单层 patch conv。仍然是 encoder-free：随机初始化，和主干一起从头训。"""
#     def __init__(self, hidden: int):
#         super().__init__()
#         chs = [3, 128, 256, 512, hidden]     # 4 层 stride2 = 16×
#         blocks = []
#         for i in range(4):
#             blocks += [
#                 nn.Conv2d(chs[i], chs[i+1], kernel_size=3, stride=2, padding=1, bias=False),
#                 nn.GroupNorm(32, chs[i+1]) if i < 3 else nn.Identity(),  # 最后一层用 img_norm 归一
#                 nn.SiLU(inplace=True) if i < 3 else nn.Identity(),
#             ]
#         self.stem = nn.Sequential(*blocks)
#
#     def forward(self, x):
#         return self.stem(x)