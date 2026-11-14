"""Failure-only diagnostics; no model weights are duplicated on GPU."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def diagnose_loss(model, output, input_ids, labels):
    hidden = getattr(output, 'hidden_states', None)
    valid = labels[:, 1:].reshape(-1) != -100
    report = dict(
        loss=str(output.loss.detach().float().cpu().tolist()) if output.loss is not None else 'None',
        shape=list(input_ids.shape), supervised_tokens=int(valid.sum()),
        input_min=int(input_ids.min()), input_max=int(input_ids.max()),
        parameter_dtype=str(next(model.parameters()).dtype),
        nonfinite_parameters=[name for name, p in model.named_parameters()
                              if not torch.isfinite(p).all()],
    )
    if not isinstance(hidden, torch.Tensor):
        report['hidden_available'] = False
        return report
    report['hidden_dtype'] = str(hidden.dtype)
    report['hidden_nonfinite'] = int((~torch.isfinite(hidden)).sum())
    report['hidden_abs_max'] = str(hidden.abs().max().float().item())
    # Independent, bounded-memory FP32 loss on precisely the supervised positions.
    if not report['hidden_nonfinite'] and not report['nonfinite_parameters'] and valid.any():
        h = hidden[:, :-1, :].reshape(-1, hidden.shape[-1])[valid]
        y = labels[:, 1:].reshape(-1)[valid]
        total = 0.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            weight = model.lm_head.weight.float()
            for start in range(0, len(y), 64):
                logits = F.linear(h[start:start + 64].float(), weight)
                total += F.cross_entropy(logits, y[start:start + 64], reduction='sum').item()
        report['reference_fp32_ce'] = str(total / len(y))
    return report
