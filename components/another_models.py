from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class MemoryRead:
    residual: torch.Tensor
    values: torch.Tensor
    layer_weights: torch.Tensor
    gate: torch.Tensor


class LearnedMemoryHead(nn.Module):
    """A layer's learned reader, address/value writer, and independent gates."""
    def __init__(self, hidden_size, num_layers, key_size, value_size):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.query = nn.Linear(hidden_size, key_size, bias=False)
        self.key = nn.Linear(hidden_size, key_size, bias=False)
        self.value = nn.Linear(hidden_size, value_size, bias=False)
        self.layer_mix = nn.Linear(hidden_size, num_layers)
        self.output = nn.Linear(value_size, hidden_size, bias=False)
        self.read_gate = nn.Linear(hidden_size, 1)
        self.write_gate = nn.Linear(hidden_size, value_size)
        self.erase_gate = nn.Linear(hidden_size, value_size)
        # Small, nonzero read branch: writers can receive future-token gradients
        # immediately, without strongly perturbing an existing backbone.
        self.reset_memory_branch()

    def reset_memory_branch(self):
        nn.init.normal_(self.output.weight, std=1e-3)
        for gate in (self.read_gate, self.write_gate, self.erase_gate):
            nn.init.constant_(gate.bias, -2.)

    def read(self, hidden, snapshot):
        normalized = self.norm(hidden)
        query = F.normalize(self.query(normalized).float(), dim=-1, eps=1e-6)
        weights = self.layer_mix(normalized).float().softmax(-1)
        per_layer = torch.einsum('bk,blkv->blv', query, snapshot)
        values = (weights.unsqueeze(-1) * per_layer).sum(1)
        gate = self.read_gate(normalized).sigmoid()
        residual = gate * self.output(values.to(normalized.dtype))
        return MemoryRead(residual, values, weights, gate)

    def write(self, hidden, old_block, max_column_norm):
        normalized = self.norm(hidden)
        key = F.normalize(self.key(normalized).float(), dim=-1, eps=1e-6)
        value = self.value(normalized).float().tanh()
        write = self.write_gate(normalized).float().sigmoid()
        erase = self.erase_gate(normalized).float().sigmoid()
        prediction = torch.einsum('bk,bkv->bv', key, old_block)
        new_block = old_block + key.unsqueeze(-1) * (write * value - erase * prediction).unsqueeze(-2)
        # Independent write/erase gates otherwise permit unbounded accumulation.
        norm = new_block.norm(dim=-2, keepdim=True)
        return new_block * (max_column_norm / norm.clamp_min(1e-6)).clamp(max=1.)


class PersistentLayerMemory(nn.Module):
    def __init__(self, hidden_size, num_layers, key_size=64, value_size=64,
                 max_column_norm=10.):
        super().__init__()
        if min(hidden_size, num_layers, key_size, value_size) <= 0:
            raise ValueError('Memory dimensions and layer count must be positive')
        if not math.isfinite(max_column_norm) or max_column_norm <= 0:
            raise ValueError('max_column_norm must be finite and positive')
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.key_size, self.value_size = key_size, value_size
        self.max_column_norm = max_column_norm
        self.heads = nn.ModuleList([LearnedMemoryHead(hidden_size, num_layers, key_size, value_size)
                                    for _ in range(num_layers)])
        self.register_buffer('_memory', torch.empty(0, dtype=torch.float32), persistent=False)
        self.steps = 0
        self._snapshot = None
        self._pending = {}
        self._valid = None

    @property
    def active(self):
        return self._snapshot is not None

    def _idle(self):
        if self.active:
            raise RuntimeError('Finish or abort the active memory step first')

    def _apply(self, fn, recurse=True):
        self._idle()
        result = super()._apply(fn, recurse)
        self._memory = self._memory.float()
        return result

    def reset_memory(self, batch_size=1):
        self._idle()
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        self._memory = torch.zeros(batch_size, self.num_layers, self.key_size, self.value_size,
                                   device=self.heads[0].query.weight.device, dtype=torch.float32)
        self.steps = 0

    def detach_memory(self):
        self._idle()
        self._memory = self._memory.detach()

    def begin_step(self, batch_size=None, valid_mask=None):
        """Start a single-token step; every layer sees the same old snapshot."""
        self._idle()
        if self._memory.numel() == 0:
            self.reset_memory(1 if batch_size is None else batch_size)
        if batch_size is not None and batch_size != self._memory.shape[0]:
            raise ValueError('Batch identities changed; explicitly reset or import the correct memory')
        if torch.is_grad_enabled() and self._memory.is_inference():
            # A persisted inference-mode tensor cannot be saved for backward.
            self._memory = self._memory.clone()
        batch = self._memory.shape[0]
        if valid_mask is None:
            valid_mask = torch.ones(batch, device=self._memory.device, dtype=torch.bool)
        if valid_mask.shape != (batch,) or valid_mask.dtype != torch.bool or valid_mask.device != self._memory.device:
            raise ValueError('valid_mask must be a bool [batch] tensor on the memory device')
        self._snapshot, self._valid = self._memory, valid_mask.clone()
        self._pending = {}

    def _validate(self, layer_id, hidden):
        if not self.active:
            raise RuntimeError('Call begin_step() before reading or writing')
        if not isinstance(layer_id, int) or not 0 <= layer_id < self.num_layers:
            raise ValueError('Invalid layer owner')
        if hidden.ndim != 2 or hidden.shape != (self._snapshot.shape[0], self.hidden_size):
            raise ValueError('hidden must be [batch, hidden_size] for ONE token; do not pass a sequence')
        if hidden.device != self._snapshot.device or not hidden.is_floating_point():
            raise ValueError('hidden must be floating point on the memory device')

    def read(self, layer_id, hidden):
        self._validate(layer_id, hidden)
        read = self.heads[layer_id].read(hidden, self._snapshot)
        read.residual = torch.where(self._valid[:, None], read.residual, torch.zeros_like(read.residual))
        read.values = torch.where(self._valid[:, None], read.values, torch.zeros_like(read.values))
        read.gate = torch.where(self._valid[:, None], read.gate, torch.zeros_like(read.gate))
        return read

    def write(self, layer_id, hidden):
        self._validate(layer_id, hidden)
        if layer_id in self._pending:
            raise RuntimeError('Each layer may stage at most one write per token')
        old = self._snapshot[:, layer_id]
        candidate = self.heads[layer_id].write(hidden, old, self.max_column_norm)
        self._pending[layer_id] = torch.where(self._valid[:, None, None], candidate, old)

    def forward(self, layer_id, hidden):
        """Read-gated residual injection and learned write to the owned block."""
        output = hidden + self.read(layer_id, hidden).residual
        self.write(layer_id, output)
        return output

    def commit_step(self, detach=None):
        if not self.active:
            raise RuntimeError('No active memory step')
        if detach is None:
            detach = not self.training
        if not isinstance(detach, bool):
            raise ValueError('detach must be bool or None')
        new = torch.stack([self._pending.get(i, self._snapshot[:, i])
                           for i in range(self.num_layers)], dim=1)
        if not torch.isfinite(new).all():
            self.abort_step()
            raise FloatingPointError('Nonfinite memory update; previous committed memory retained')
        self._memory = new.detach() if detach else new
        self.steps += 1
        self.abort_step()

    def abort_step(self):
        self._snapshot, self._valid, self._pending = None, None, {}

    def memory_snapshot(self):
        """A clone, so callers cannot mutate the committed matrix accidentally."""
        return self._memory.detach().clone()

    def export_memory(self):
        self._idle()
        if self._memory.numel() == 0:
            raise RuntimeError('Memory is not initialized')
        return dict(matrix=self._memory.detach().cpu().clone(), steps=self.steps)

    def import_memory(self, state):
        self._idle()
        matrix, steps = state['matrix'], state['steps']
        if (matrix.ndim != 4 or matrix.shape[0] <= 0 or
                matrix.shape[1:] != (self.num_layers, self.key_size, self.value_size) or
                not isinstance(steps, int) or steps < 0):
            raise ValueError('Incompatible memory state')
        if not matrix.is_floating_point() or not torch.isfinite(matrix).all():
            raise ValueError('Memory state must contain finite floating point values')
        if (matrix.float().norm(dim=-2) > self.max_column_norm + 1e-5).any():
            raise ValueError('Imported memory exceeds max_column_norm')
        self._memory = matrix.to(device=self.heads[0].query.weight.device, dtype=torch.float32).detach().clone()
        self.steps = steps
