"""Shared pieces for JSON-configured dense text pretraining."""
import torch

from losses.loss import get_loss_fn
from train.train_util import get_muon_params, get_adamw_params


class TextCollate:
    def __init__(self, pad_id, max_length):
        self.pad_id, self.max_length = pad_id, max_length

    def __call__(self, items):
        ids = torch.full((len(items), self.max_length), self.pad_id, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        for row, item in enumerate(items):
            if not 2 <= len(item) <= self.max_length:
                raise ValueError('Dataset record does not fit the configured sequence length')
            ids[row, :len(item)] = item
            labels[row, :len(item)] = item
        return ids, labels


class TextTrainingLoss(torch.nn.Module):
    def __init__(self, model, backend, chunks):
        super().__init__()
        self.model = model
        self.loss_fn = None if backend == 'native' else get_loss_fn(chunks, backend)

    def forward(self, ids, labels):
        # Right padding comes after every valid token. Causality isolates valid
        # outputs from trailing PAD; labels mask PAD loss. No cache is retained.
        if self.loss_fn is None:
            return self.model(ids, labels=labels, use_cache=False).loss
        hidden, _, _ = self.model.model(ids, use_cache=False)
        return self.external_loss(hidden, labels)

    @torch.compiler.disable
    def external_loss(self, hidden, labels):
        return self.loss_fn(self.model.lm_head.weight,
            hidden[:, :-1].reshape(-1, hidden.size(-1)).contiguous(),
            labels[:, 1:].reshape(-1).contiguous())


def build_optimizer(model, settings):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if settings['optimizer'] == 'adamw':
        return torch.optim.AdamW(parameters, lr=settings['lr'], betas=(0.9, 0.95),
            weight_decay=settings['weight_decay'], fused=parameters[0].is_cuda)
    from muon import SingleDeviceMuonWithAuxAdam
    muon = get_muon_params(model, matrix_only=True, exclude_norm=True)
    adam = get_adamw_params(model, matrix_only=True, exclude_norm=True)
    identities = [id(p) for p in muon + adam]
    if not muon or len(set(identities)) != len(identities) or set(identities) != {id(p) for p in parameters}:
        raise ValueError('Muon/AdamW groups must cover every trainable parameter exactly once')
    print(f'[Optimizer] Muon={sum(p.numel() for p in muon):,}; AdamW={sum(p.numel() for p in adam):,}', flush=True)
    return SingleDeviceMuonWithAuxAdam([
        dict(params=muon, use_muon=True, lr=settings['lr'] * settings['muon_lr_factor'],
             momentum=settings['muon_momentum'], weight_decay=settings['weight_decay']),
        dict(params=adam, use_muon=False, lr=settings['lr'], betas=(0.9, 0.95),
             eps=1e-8, weight_decay=settings['weight_decay'])])
