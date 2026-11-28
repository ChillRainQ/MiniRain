"""Read RainPocket SFT rows directly from the original JSONL files."""

from collections import OrderedDict
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.compile_sft import (
    assistant_mask_template, encode, pad_sft_row, prepare_sft_sources, record_sft_row, select_sft_rows,
)


class SftJsonlDataset(Dataset):
    def __init__(self, data_dir, tokenizer, *, tokens, seq_length, seed):
        self.data_dir = Path(data_dir).resolve()
        self.tokenizer = tokenizer
        self.template = assistant_mask_template(tokenizer.get_chat_template())
        self.max_length = seq_length
        self.pad_token_id = tokenizer.pad_token_id
        self.vocab_size = len(tokenizer)
        self._handles = OrderedDict()
        self._pid = os.getpid()

        self.sources, self.groups, self.identity, stats = prepare_sft_sources(
            self.data_dir, self.tokenizer, self.template, tokens, seq_length, seed,
        )
        rows = []
        exhausted = select_sft_rows(
            self.sources, self.groups, self.tokenizer, self.template, seq_length, seed, stats,
            write=lambda source, offset, _item: rows.append((source, offset)),
        )
        identity_count = len(self.identity)
        positions = (2 * np.arange(identity_count) + 1) * len(rows) // (2 * identity_count)
        positions += np.arange(identity_count)
        self.offsets = np.empty((len(rows) + identity_count, 2), dtype=np.int64)
        is_corpus = np.ones(len(self.offsets), dtype=bool)
        is_corpus[positions] = False
        self.offsets[is_corpus] = rows
        self.offsets[positions, 0] = -1
        self.offsets[positions, 1] = np.arange(identity_count)
        for item in self.identity:
            record_sft_row(stats["self"], item)
        self.supervised_tokens = sum(group["supervised_tokens"] for group in stats.values())
        self.manifest = dict(layout="sft_jsonl", groups=stats, exhausted_groups=exhausted,
                             sequence_length=seq_length, target_supervised_tokens=tokens)
        self.files = (*self.sources, self.data_dir / "self/identity.jsonl")

    def __len__(self):
        return len(self.offsets)

    def validation_candidates(self):
        return np.flatnonzero(self.offsets[:, 0] >= 0)

    def __getitem__(self, index):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        source, offset = self.offsets[index]
        if source == -1:
            item = self.identity[offset]
        else:
            path = self.sources[source]
            handle = self._handles.pop(source, None)
            if handle is None:
                if len(self._handles) == 16:
                    self._handles.popitem(last=False)[1].close()
                handle = path.open("rb")
            self._handles[source] = handle
            handle.seek(int(offset))
            item, reason = encode(json.loads(handle.readline()), self.tokenizer, self.template,
                                  self.groups[source], self.max_length)
            if reason:
                raise ValueError(f"Indexed SFT row no longer usable at {path}:{offset} ({reason})")
        ids, mask = pad_sft_row(item, self.max_length, self.pad_token_id)
        ids = torch.from_numpy(ids.astype(np.int64))
        labels = ids.masked_fill(torch.from_numpy(mask == 0), -100)
        return ids, labels

    def __getstate__(self):
        return dict(self.__dict__, _handles=OrderedDict())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
