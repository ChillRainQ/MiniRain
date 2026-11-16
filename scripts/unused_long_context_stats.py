
import argparse
import csv
import json
from pathlib import Path

from transformers import AutoTokenizer

THRESHOLDS = (4096, 8192, 16384, 32768, 65536)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("data/ultradata")
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("data/pretrain_bins/manifest.json")
    )
    parser.add_argument(
        "--tokenizer", type=Path,
        default=Path("minirain_tokenizer")
    )
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output", type=Path,
        default=Path("unused_long_context_stats.csv")
    )
    args = parser.parse_args()

    if args.stride < 1 or args.batch_size < 1:
        raise ValueError("stride and batch-size must be positive")

    manifest = json.loads(
        args.manifest.read_text(encoding="utf-8")
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), use_fast=True
    )

    rows = []

    for filename, info in manifest["sources"].items():
        path = args.data_dir / filename

        if not path.exists():
            raise FileNotFoundError(path)

        skip = int(info["lines_read"])

        total_rows = 0
        sampled_rows = 0
        sampled_tokens = 0
        counts = {t: 0 for t in THRESHOLDS}
        tokens_in_long_docs = {t: 0 for t in THRESHOLDS}

        pending = []

        def flush():
            nonlocal sampled_rows, sampled_tokens

            if not pending:
                return

            outputs = tokenizer(
                pending,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )

            for ids in outputs["input_ids"]:
                length = len(ids)
                sampled_rows += 1
                sampled_tokens += length

                for threshold in THRESHOLDS:
                    if length >= threshold:
                        counts[threshold] += 1
                        tokens_in_long_docs[threshold] += length

            pending.clear()

        print(f"[SCAN] {filename}, skipping {skip:,} rows")

        with path.open("r", encoding="utf-8") as f:
            for _ in range(skip):
                if not f.readline():
                    raise RuntimeError(
                        f"Source shorter than manifest: {filename}"
                    )

            for line in f:
                total_rows += 1

                if (total_rows - 1) % args.stride != 0:
                    continue

                obj = json.loads(line)
                pending.append(str(obj["text"]))

                if len(pending) >= args.batch_size:
                    flush()

        flush()

        row = {
            "file": filename,
            "unused_rows": total_rows,
            "sampled_rows": sampled_rows,
            "estimated_unused_tokens": round(
                sampled_tokens / sampled_rows * total_rows
            ) if sampled_rows else 0,
        }

        for threshold in THRESHOLDS:
            row[f"sample_ge_{threshold}"] = counts[threshold]
            row[f"estimated_ge_{threshold}"] = (
                round(counts[threshold] / sampled_rows * total_rows)
                if sampled_rows else 0
            )
            row[f"estimated_tokens_ge_{threshold}"] = (
                round(
                    tokens_in_long_docs[threshold]
                    / sampled_rows * total_rows
                )
                if sampled_rows else 0
            )

        rows.append(row)

        print(
            f"  Unused rows={total_rows:,}, "
            f"sampled={sampled_rows:,}, "
            f">=8K={row['estimated_ge_8192']:,}, "
            f">=16K={row['estimated_ge_16384']:,}, "
            f">=32K={row['estimated_ge_32768']:,}"
        )

    with args.output.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
