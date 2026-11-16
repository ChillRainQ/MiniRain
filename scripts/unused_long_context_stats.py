#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compile unused original UltraData documents into P1-Long 8K/16K/32K.

Only source JSONL rows after manifest.sources[filename].lines_read are read.
No document packing, no concatenation, no splitting, no truncation.
Length thresholds INCLUDE BOS and EOS:
    [ 6144,  8192] ->  8K
    [12288, 16384] -> 16K
    [20480, 32768] -> 32K
Other lengths, including >32K, are skipped and counted.
Default effective-token (BOS+body+EOS, excluding PAD) budgets:
    8K=50M, 16K=100M, 32K=50M. No repeats; no partial records.
If a tier lacks enough qualifying documents, output its actual available amount.
Each stored record is [BOS] + original document tokens + [EOS].
Padding is NOT stored; let the training Collate add PAD with labels=-100.
BIN = little-endian uint16; IDX = little-endian uint64 token offsets, N+1.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

TIERS = ((6144, 8192, "8k"), (12288, 16384, "16k"), (20480, 32768, "32k"))
DEFAULT_BUDGETS = {"8k": 50_000_000, "16k": 100_000_000, "32k": 50_000_000}


def tier_for_length(length: int) -> str | None:
    for lo, hi, label in TIERS:
        if lo <= length <= hi:
            return label
    return None


class Writer:
    def __init__(self, out: Path, label: str, max_len: int, flush_tokens: int):
        self.label, self.max_len, self.flush_tokens = label, max_len, flush_tokens
        self.bin_name = f"p1_long_{label}_text.bin"
        self.idx_name = f"p1_long_{label}_text.idx"
        self.tmp_bin = out / (self.bin_name + ".tmp")
        self.tmp_idx = out / (self.idx_name + ".tmp")
        self.bin_f = self.tmp_bin.open("wb")
        self.idx_f = self.tmp_idx.open("wb")
        self.buf = array("H")
        self.offsets = array("Q", [0])
        self.records = self.body_tokens = self.stored_tokens = 0
        self.min_length = None
        self.max_length = 0
        self.by_source = defaultdict(lambda: {"records": 0, "body_tokens": 0})

    def add(self, ids: list[int], bos: int, eos: int, source: str):
        length = len(ids) + 2
        if tier_for_length(length) != self.label or length > self.max_len:
            raise ValueError(f"Unexpected {self.label} length: {length}")
        record = [bos, *ids, eos]
        if min(record) < 0 or max(record) > 65535:
            raise ValueError("Token id is outside uint16 range")
        self.buf.extend(record)
        self.records += 1
        self.body_tokens += len(ids)
        self.stored_tokens += length
        self.min_length = length if self.min_length is None else min(length, self.min_length)
        self.max_length = max(self.max_length, length)
        self.offsets.append(self.stored_tokens)
        self.by_source[source]["records"] += 1
        self.by_source[source]["body_tokens"] += len(ids)
        if len(self.buf) >= self.flush_tokens:
            self.buf.tofile(self.bin_f)
            self.buf = array("H")
        if len(self.offsets) >= 100_000:
            self.offsets.tofile(self.idx_f)
            self.offsets = array("Q")

    def close(self):
        if not self.bin_f.closed:
            self.buf.tofile(self.bin_f)
            self.offsets.tofile(self.idx_f)
            self.bin_f.flush()
            self.idx_f.flush()
            os.fsync(self.bin_f.fileno())
            os.fsync(self.idx_f.fileno())
            self.bin_f.close()
            self.idx_f.close()
        if self.tmp_bin.stat().st_size != self.stored_tokens * 2:
            raise RuntimeError(f"BIN size mismatch: {self.label}")
        if self.tmp_idx.stat().st_size != (self.records + 1) * 8:
            raise RuntimeError(f"IDX size mismatch: {self.label}")

    def publish(self, out: Path):
        os.replace(self.tmp_bin, out / self.bin_name)
        os.replace(self.tmp_idx, out / self.idx_name)

    def info(self):
        return {"bin": self.bin_name, "idx": self.idx_name,
                "max_seq_len": self.max_len, "records": self.records,
                "body_tokens": self.body_tokens, "stored_tokens": self.stored_tokens,
                "min_record_length": self.min_length or 0,
                "max_record_length": self.max_length,
                "eos_records": self.records,
                "by_source": dict(self.by_source)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/ultradata"))
    p.add_argument("--manifest", type=Path, default=Path("data/pretrain_bins/manifest.json"))
    p.add_argument("--tokenizer", type=Path, default=Path("minirain_tokenizer"))
    p.add_argument("--output-dir", type=Path, default=Path("data/pretrain_long"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--flush-tokens", type=int, default=1_000_000)
    p.add_argument("--target-8k", type=int, default=DEFAULT_BUDGETS["8k"],
                   help="Effective tokens (including BOS/EOS, excluding PAD); 0 means unlimited")
    p.add_argument("--target-16k", type=int, default=DEFAULT_BUDGETS["16k"])
    p.add_argument("--target-32k", type=int, default=DEFAULT_BUDGETS["32k"])
    p.add_argument("--dry-run", action="store_true", help="Only check manifest and input paths")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    budgets = {"8k": args.target_8k, "16k": args.target_16k, "32k": args.target_32k}
    if any(v < 0 for v in budgets.values()):
        p.error("Token targets must be >= 0 (0=unlimited)")
    if args.batch_size < 1 or args.flush_tokens < 1:
        p.error("batch-size and flush-tokens must be positive")
    if sys.byteorder != "little" or array("H").itemsize != 2 or array("Q").itemsize != 8:
        raise RuntimeError("Requires little-endian uint16/uint64 native arrays")
    src_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sources = src_manifest.get("sources", {})
    if not sources:
        raise ValueError("Missing sources in manifest")
    if int(src_manifest.get("max_length", 0)) != 4096:
        raise ValueError("Expected source manifest compiled with max_length=4096")
    for name, details in sources.items():
        if Path(name).name != name or not (args.data_dir / name).is_file():
            raise FileNotFoundError(args.data_dir / name)
        if int(details["lines_read"]) < 0:
            raise ValueError(f"Invalid lines_read for {name}")
    out = args.output_dir
    names = [f"p1_long_{label}_text.{ext}" for _, _, label in TIERS for ext in ("bin", "idx")]
    names.append("manifest_long.json")
    if not args.overwrite and any((out / n).exists() for n in names):
        raise FileExistsError("Output exists: specify another --output-dir or --overwrite")
    if args.dry_run:
        for name, details in sources.items():
            print(f"{name}: skip {details['lines_read']:,} lines")
        return

    # Import after dry-run, so path checks don't require tokenizer loading.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.tokenizer), use_fast=True)
    if tok.bos_token_id is None or tok.eos_token_id is None:
        raise ValueError("Tokenizer must provide BOS and EOS")
    if len(tok) > 65536:
        raise ValueError("Tokenizer vocabulary exceeds uint16")
    tok.model_max_length = 10**12  # warning suppression only; never truncate
    bos, eos = int(tok.bos_token_id), int(tok.eos_token_id)
    out.mkdir(parents=True, exist_ok=True)
    writers = {label: Writer(out, label, hi, args.flush_tokens)
               for _, hi, label in TIERS}
    all_stats = {}
    started = time.monotonic()

    try:
        for name, details in sources.items():
            skip = int(details["lines_read"])
            stat = {"skipped_previously_read_lines": skip,
                    "unread_documents": 0, "unread_body_tokens": 0,
                    "included_documents": 0, "included_body_tokens": 0,
                    "skipped_by_length": Counter(),
                    "skipped_by_budget": Counter(),
                    "included_by_tier": Counter()}
            batch = []

            def flush():
                if not batch:
                    return
                enc = tok(batch, add_special_tokens=False, truncation=False,
                          padding=False, return_attention_mask=False,
                          return_token_type_ids=False)["input_ids"]
                for ids in enc:
                    n = len(ids)
                    stat["unread_body_tokens"] += n
                    label = tier_for_length(n + 2)
                    if label is None:
                        if n + 2 < 6144:
                            reason = "below_6k"
                        elif n + 2 < 12288:
                            reason = "between_8k_and_12k"
                        elif n + 2 < 20480:
                            reason = "between_16k_and_20k"
                        else:
                            reason = "above_32k"
                        stat["skipped_by_length"][reason] += 1
                        continue
                    # Do not exceed a target; never truncate or split a document.
                    # If a candidate does not fit, scan ahead for a smaller one.
                    budget = budgets[label]
                    length = n + 2
                    if budget and writers[label].stored_tokens + length > budget:
                        stat["skipped_by_budget"][label] += 1
                        continue
                    writers[label].add(ids, bos, eos, name)
                    stat["included_documents"] += 1
                    stat["included_body_tokens"] += n
                    stat["included_by_tier"][label] += 1
                batch.clear()

            print(f"[SCAN] {name}: skipping {skip:,} compiled-source lines", flush=True)
            with (args.data_dir / name).open("r", encoding="utf-8") as f:
                for _ in range(skip):
                    if not f.readline():
                        raise RuntimeError(f"Manifest lines_read exceeds {name} length")
                for line_number, line in enumerate(f, skip + 1):
                    stat["unread_documents"] += 1
                    try:
                        obj = json.loads(line)
                        if not isinstance(obj, dict) or "text" not in obj:
                            raise ValueError("Missing text field")
                        batch.append(str(obj["text"]))
                    except (json.JSONDecodeError, ValueError) as exc:
                        raise RuntimeError(f"Invalid record {name}:{line_number}: {exc}") from exc
                    if len(batch) >= args.batch_size:
                        flush()
                    if stat["unread_documents"] % 100_000 == 0:
                        print(f"  {stat['unread_documents']:,} rows; "
                              f"{stat['included_documents']:,} accepted", flush=True)
                flush()
            stat["skipped_by_length"] = dict(stat["skipped_by_length"])
            stat["skipped_by_budget"] = dict(stat["skipped_by_budget"])
            stat["included_by_tier"] = dict(stat["included_by_tier"])
            all_stats[name] = stat

        for w in writers.values():
            w.close()
        report = {
            "project": "MiniRain-Vision P1-Long",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "original_manifest": str(args.manifest),
            "tokenizer": str(args.tokenizer),
            "source_rule": "line_number > original_manifest.sources[filename].lines_read",
            "thresholds_inclusive_of_bos_eos": {
                label: [low, hi] for low, hi, label in TIERS},
            "token_budget": budgets,
            "budget_unit": "effective tokens = body + BOS + EOS, excluding PAD",
            "budget_policy": "no overrun, no repeat, no truncation; scan other documents if next does not fit",
            "padding_stored": False,
            "packing": False, "split_long_documents": False,
            "truncation": False,
            "special_tokens": {"bos": bos, "eos": eos, "every_record_has_bos_eos": True},
            "format": {"bin": "uint16 little-endian", "idx": "uint64 little-endian offsets N+1"},
            "tiers": {label: {**w.info(), "target_effective_tokens": budgets[label],
                               "target_reached": (budgets[label] > 0 and
                                                  w.stored_tokens >= budgets[label] - w.max_len),
                               "remaining_to_target": max(0, budgets[label] - w.stored_tokens)
                               if budgets[label] > 0 else None}
                      for label, w in writers.items()},
            "sources": all_stats,
            "duration_seconds": round(time.monotonic() - started, 2),
        }
        tmp_report = out / "manifest_long.json.tmp"
        with tmp_report.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        for w in writers.values():
            w.publish(out)
        os.replace(tmp_report, out / "manifest_long.json")
    finally:
        for w in writers.values():
            for fd in (w.bin_f, w.idx_f):
                if not fd.closed:
                    fd.close()

    print("\nP1-Long compilation finished")
    for label, w in writers.items():
        target = budgets[label]
        pct = f"{w.stored_tokens / target * 100:.2f}%" if target else "unlimited"
        print(f"  {label}: {w.records:,} documents / {w.stored_tokens:,} effective tokens "
              f"({pct} of {target:,} target) / {w.body_tokens:,} body tokens / EOS={w.records:,}")
    print("Manifest:", out / "manifest_long.json")


if __name__ == "__main__":
    main()
