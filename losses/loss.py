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
        # Autograd backward does not inherit the caller's autocast context.
        # Explicit FP32 recomputation supports BF16 activations + FP32 weights.
        with torch.no_grad(), torch.autocast(device_type=h.device.type, enabled=False):
            for idx in chunks:
                logits = F.linear(h[idx].float(), weight.float())  # [C, V]
                loss = F.cross_entropy(logits, t[idx], ignore_index=-100, reduction="sum")
                total_loss += loss.item()
                total_cnt += (t[idx] != -100).sum().item()
        ctx.save_for_backward(weight, h, t)
        ctx.chunks = chunks
        ctx.denominator = max(total_cnt, 1)
        ctx.hidden_shape = hidden.shape
        return hidden.new_tensor(total_loss / ctx.denominator, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_out):
        weight, h, t = ctx.saved_tensors
        grad_h = torch.zeros_like(h, dtype=torch.float32)
        grad_w = torch.zeros_like(weight, dtype=torch.float32)
        scale = grad_out.float() / ctx.denominator
        for idx in ctx.chunks:
            with torch.enable_grad(), torch.autocast(device_type=h.device.type, enabled=False):
                hc = h[idx].detach().float().requires_grad_(True)
                wc = weight.detach().float().requires_grad_(True)
                logits = F.linear(hc, wc)
                loss = F.cross_entropy(logits, t[idx], ignore_index=-100, reduction="sum")
                gh, gw = torch.autograd.grad(loss, (hc, wc))
            grad_h[idx] = gh
            grad_w += gw
        grad_h = (grad_h * scale).to(h.dtype).reshape(ctx.hidden_shape)
        return (grad_w * scale).to(weight.dtype), grad_h, None, None


class ChunkedCrossEntropy(nn.Module):
    """分块算 logits + CE，峰值显存约为 1/num_chunks（梯度正确）"""
    def __init__(self, num_chunks: int = 4):
        super().__init__()
        self.num_chunks = num_chunks

    def forward(self, weight, hidden, target):
        return _ChunkedCE.apply(weight, hidden, target, self.num_chunks)


def get_loss_fn(num_chunks: int = 4, backend: str = 'auto'):
    if backend not in ('auto', 'liger', 'chunked'):
        raise ValueError(f'Unknown loss backend: {backend}')
    if backend == 'liger' and not _HAS_LIGER:
        raise RuntimeError('Liger loss requested but liger-kernel is not installed')
    if backend != 'chunked' and _HAS_LIGER:
        print("[loss] 使用 Liger Fused Linear CE")
        return LigerFusedLinearCrossEntropyLoss()
    print(f"[loss] 使用 PyTorch 分块 CE（{num_chunks} 块，backend={backend}）")
    return ChunkedCrossEntropy(num_chunks)
