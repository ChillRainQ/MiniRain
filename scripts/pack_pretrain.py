
#!/usr/bin/env python3
"""Pack MiniRain's uint16 BIN + uint64 IDX into fixed-length token blocks."""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


def pack(src_bin: Path, src_idx: Path, dst_bin: Path,
         dst_idx: Path, seq_len: int = 4096):

    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")

    if src_bin.resolve() == dst_bin.resolve():
        raise ValueError("Refusing to overwrite source BIN")
    if src_idx.resolve() == dst_idx.resolve():
        raise ValueError("Refusing to overwrite source IDX")

    for p in (dst_bin, dst_idx):
        if p.exists():
            raise FileExistsError(f"Output already exists: {p}")
        p.parent.mkdir(parents=True, exist_ok=True)

    # Existing MiniRain binary format
    tokens = np.memmap(src_bin, dtype="<u2", mode="r")
    offsets = np.memmap(src_idx, dtype="<u8", mode="r")

    if len(offsets) < 2:
        raise ValueError("Empty index")
    if int(offsets[0]) != 0:
        raise ValueError("IDX must start with zero")
    if int(offsets[-1]) != len(tokens):
        raise ValueError("BIN/IDX length mismatch")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("IDX offsets are not monotonic")

    total_tokens = len(tokens)
    total_docs = len(offsets) - 1
    expected_blocks = math.ceil(total_tokens / seq_len)

    # Temporary names prevent partially written outputs
    # from being mistaken for completed ones.
    temp_bin = Path(str(dst_bin) + ".tmp")
    temp_idx = Path(str(dst_idx) + ".tmp")

    for p in (temp_bin, temp_idx):
        if p.exists():
            raise FileExistsError(f"Temporary file exists: {p}")

    written = 0
    blocks = 0

    try:
        with open(temp_bin, "wb", buffering=8 * 1024 * 1024) as fb, \
             open(temp_idx, "wb", buffering=1024 * 1024) as fi:

            # Starting offset of first block.
            fi.write(np.asarray([0], dtype="<u8").tobytes())

            # Input is already a continuous sequence of document
            # token spans, each containing its own BOS/EOS.
            # A chunk may cross document boundaries.
            for start in range(0, total_tokens, seq_len):
                end = min(start + seq_len, total_tokens)

                block = tokens[start:end]
                fb.write(block.tobytes())

                written += end - start
                blocks += 1

                fi.write(
                    np.asarray([written], dtype="<u8").tobytes()
                )

                if blocks % 100_000 == 0:
                    print(
                        f"\rPacked {blocks:,}/{expected_blocks:,}"
                        f" | {written / 1e9:.3f}B tokens",
                        end="",
                        flush=True,
                    )

            fb.flush()
            fi.flush()
            os.fsync(fb.fileno())
            os.fsync(fi.fileno())

        if written != total_tokens:
            raise RuntimeError("Token count changed unexpectedly")

        if temp_bin.stat().st_size != total_tokens * 2:
            raise RuntimeError("Packed BIN size mismatch")

        if temp_idx.stat().st_size != (blocks + 1) * 8:
            raise RuntimeError("Packed IDX size mismatch")

        os.replace(temp_bin, dst_bin)
        os.replace(temp_idx, dst_idx)

    finally:
        # Leave temporary files for inspection if interrupted.
        pass

    result = {
        "source_documents": total_docs,
        "source_tokens": total_tokens,
        "packed_blocks": blocks,
        "sequence_length": seq_len,
        "last_block_length": (
            total_tokens - (blocks - 1) * seq_len
        ),
        "packed_tokens": written,
        "document_boundaries": "EOS/BOS retained",
        "cross_document_attention": True,
        "format": {
            "bin": "uint16 little-endian",
            "idx": "uint64 little-endian",
        },
    }

    print()
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--stage", choices=["p1", "p2", "p3", "p4"],
                   default="p1")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    output_dir = args.output_dir or args.data_dir

    src_bin = args.data_dir / f"{args.stage}_text.bin"
    src_idx = args.data_dir / f"{args.stage}_text.idx"
    dst_bin = output_dir / f"{args.stage}_packed.bin"
    dst_idx = output_dir / f"{args.stage}_packed.idx"

    result = pack(src_bin, src_idx, dst_bin, dst_idx,
                  args.seq_len)

    manifest = output_dir / f"{args.stage}_packed_manifest.json"
    manifest.write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
