import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from dataset.datasets import PretrainBinDataset


DATA_DIR = ROOT / "data" / "pretrain_bins"


dataset = PretrainBinDataset(
    bin_path=DATA_DIR / "p1_text.bin",
    idx_path=DATA_DIR / "p1_text.idx",
    max_length=4096,
    pad_token_id=0,
)


print()
print("=" * 60)
print("P1 DATASET TEST")
print("=" * 60)

print("records:", len(dataset))

for i in [0, 1, 100, len(dataset) - 1]:
    input_ids, labels = dataset[i]

    valid = labels != -100

    print(
        f"sample={i:,} "
        f"shape={tuple(input_ids.shape)} "
        f"tokens={valid.sum().item():,} "
        f"BOS={input_ids[0].item()} "
        f"EOS={input_ids[valid.sum() - 1].item()}"
    )

    assert input_ids.shape == (4096,)
    assert labels.shape == (4096,)

    # MiniRain BOS/EOS
    assert input_ids[0].item() == 1

    length = valid.sum().item()

    assert input_ids[length - 1].item() == 2

    if length < 4096:
        assert torch.all(
            input_ids[length:] == 0
        )

        assert torch.all(
            labels[length:] == -100
        )

print("=" * 60)
print("PASS")
dataset.close()
