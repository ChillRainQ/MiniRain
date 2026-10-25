# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    _HAS_LIGER = True
except ImportError:
    _HAS_LIGER = False


class _ChunkedCE(torch.autograd.Function):
    """前向逐块算 loss（不建图），反向逐块重算 logits 求梯度。
    峰值显存 ≈ 全量 logits 的 1/num_chunks，梯度与全量 CE 数学等价。"""
    @staticmethod
    def forward(ctx, weight, hidden, target, num_chunks):
        h = hidden.reshape(-1, hidden.shape[-1])
        t = target.reshape(-1)
        chunks = torch.chunk(torch.arange(h.shape[0], device=h.device), num_chunks)
        total_loss, total_cnt = 0.0, 0
        with torch.no_grad():
            for idx in chunks:
                logits = F.linear(h[idx], weight).float()      # [C, V]
                loss = F.cross_entropy(logits, t[idx], ignore_index=-100, reduction="sum")
                total_loss += loss.item()
                total_cnt += (t[idx] != -100).sum().item()
        ctx.save_for_backward(weight, h, t)
        ctx.chunks = chunks
        ctx.denominator = max(total_cnt, 1)
        ctx.hidden_shape = hidden.shape
        return hidden.new_tensor(total_loss / ctx.denominator)

    @staticmethod
    def backward(ctx, grad_out):
        weight, h, t = ctx.saved_tensors
        grad_h = torch.zeros_like(h)
        grad_w = torch.zeros_like(weight)
        scale = grad_out / ctx.denominator
        for idx in ctx.chunks:
            with torch.enable_grad():
                hc = h[idx].detach().requires_grad_(True)
                wc = weight.detach().requires_grad_(True)
                logits = F.linear(hc, wc).float()
                loss = F.cross_entropy(logits, t[idx], ignore_index=-100, reduction="sum")
                gh, gw = torch.autograd.grad(loss, (hc, wc))
            grad_h[idx] = gh.to(h.dtype)
            grad_w += gw.to(weight.dtype)
        grad_h = (grad_h * scale).reshape(ctx.hidden_shape)
        return grad_w * scale, grad_h, None, None


class ChunkedCrossEntropy(nn.Module):
    """分块算 logits + CE，峰值显存约为 1/num_chunks（梯度正确）"""
    def __init__(self, num_chunks: int = 4):
        super().__init__()
        self.num_chunks = num_chunks

    def forward(self, weight, hidden, target):
        return _ChunkedCE.apply(weight, hidden, target, self.num_chunks)


def get_loss_fn(num_chunks: int = 4):
    if _HAS_LIGER:
        print("[loss] 使用 Liger Fused Linear CE")
        return LigerFusedLinearCrossEntropyLoss()
    print(f"[loss] 无 Liger，使用分块 CE（{num_chunks} 块）")
    return ChunkedCrossEntropy(num_chunks)