
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MiniRain-Vision Multi-Stage Pretraining Compiler

将现有 UltraData JSONL 编译为 P1/P2/P3/P4 独立二进制语料。

严格匹配原 PretrainDataset：

    tokens = tokenizer(
        text,
        add_special_tokens=False,
        max_length=max_length - 2,
        truncation=True
    ).input_ids

    tokens = [BOS] + tokens + [EOS]

不同点：
    不在磁盘存 Padding，交给后续 PretrainBinDataset 动态处理。

输出：
    p1_text.bin / p1_text.idx
    p2_text.bin / p2_text.idx
    p3_text.bin / p3_text.idx
    p4_text.bin / p4_text.idx
    manifest.json

BIN:
    uint16 little-endian

IDX:
    uint64 little-endian
    N+1 个 offset，最后一个是 BIN 的结束位置。

不跨文档拼接。
不重复分配记录。
"""

import os
import sys
import json
import time
import argparse

from pathlib import Path
from array import array
from datetime import datetime
from collections import defaultdict

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

from transformers import AutoTokenizer


# ============================================================
# 1. 路径
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_DIR = ROOT / "data" / "ultradata"
DEFAULT_TOKENIZER = ROOT / "minirain_tokenizer"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "pretrain_bins"


# ============================================================
# 2. 原始 Token 统计
#
# 仅用于各 JSONL 文件之间的配额分配。
# 实际写入量使用当前 Tokenizer 重新计算。
# ============================================================

SOURCE_TOKENS = {
    "fineweb_l2_zh.jsonl": 5329265009,
    "fineweb_l2_en.jsonl": 2103330135,

    "fineweb_l3_zh_qa.jsonl": 2603250647,
    "fineweb_l3_zh_multi_style.jsonl": 2364288008,
    "fineweb_l3_en_qa.jsonl": 1113278283,
    "fineweb_l3_en_multi_style.jsonl": 987964140,

    "code_l2_py.jsonl": 983746682,
    "code_l2_sh.jsonl": 378024574,
    "code_l3_py.jsonl": 996596252,
    "code_l3_sh.jsonl": 319090907,

    "windows_batchfile.jsonl": 358704163,
    "windows_powershell.jsonl": 467614021,

    "math_l3_0.jsonl": 668557288,
    "math_l3_1.jsonl": 719901416,
    "math_l3_2.jsonl": 744400164,
    "math_l3_3.jsonl": 719399266,
}

CURRENT_TOTAL = sum(SOURCE_TOKENS.values())


# ============================================================
# 3. 数据分组
# ============================================================

GROUP_FILES = {
    "l2": [
        "fineweb_l2_zh.jsonl",
        "fineweb_l2_en.jsonl",
    ],

    "l3": [
        "fineweb_l3_zh_qa.jsonl",
        "fineweb_l3_zh_multi_style.jsonl",
        "fineweb_l3_en_qa.jsonl",
        "fineweb_l3_en_multi_style.jsonl",
    ],

    "code": [
        "code_l2_py.jsonl",
        "code_l2_sh.jsonl",
        "code_l3_py.jsonl",
        "code_l3_sh.jsonl",
        "windows_batchfile.jsonl",
        "windows_powershell.jsonl",
    ],

    "math": [
        "math_l3_0.jsonl",
        "math_l3_1.jsonl",
        "math_l3_2.jsonl",
        "math_l3_3.jsonl",
    ],
}

STAGES = ["p1", "p2", "p3", "p4"]


# ============================================================
# 4. 原始 12B 训练方案
#
# P1: 3.0B 文本
# P2: 0.6B 文本 + 0.9B 视觉
# P3: 1.8B 文本 + 4.2B 视觉
# P4: 0.3B 文本 + 1.2B 视觉
#
# 按当前 20.857B 规模等比例放大。
# ============================================================

ORIGINAL_PLAN = {
    "p1": {
        "l2": 1_200_000_000,
        "l3": 1_050_000_000,
        "code": 450_000_000,
        "math": 300_000_000,
    },

    "p2": {
        "l2": 200_000_000,
        "l3": 250_000_000,
        "code": 90_000_000,
        "math": 60_000_000,
    },

    "p3": {
        "l2": 540_000_000,
        "l3": 720_000_000,
        "code": 270_000_000,
        "math": 270_000_000,
    },

    "p4": {
        "l2": 60_000_000,
        "l3": 120_000_000,
        "code": 45_000_000,
        "math": 75_000_000,
    },
}

ORIGINAL_TOTAL = 12_000_000_000

SCALE_NUM = CURRENT_TOTAL
SCALE_DEN = ORIGINAL_TOTAL


# ============================================================
# 5. 整数配额分配
# ============================================================

def largest_remainder(total, weights):
    """
    最大余数法。
    全程整数计算，避免浮点误差。
    """

    denominator = sum(weights.values())

    if denominator <= 0:
        raise ValueError("Invalid weights")

    result = {}
    remainders = {}

    for key, weight in weights.items():

        q, r = divmod(
            total * weight,
            denominator,
        )

        result[key] = q
        remainders[key] = r

    remaining = total - sum(result.values())

    order = sorted(
        weights,
        key=lambda k: (remainders[k], str(k)),
        reverse=True,
    )

    for key in order[:remaining]:
        result[key] += 1

    return result


def build_plan():
    """
    生成每个阶段、每个源文件的 Token 配额。

    文件间按原始 Token 数量比例分配。
    阶段间按原始训练预算比例分配。
    """

    plan = {
        stage: {}
        for stage in STAGES
    }

    stage_group_targets = {}

    for stage in STAGES:

        original = ORIGINAL_PLAN[stage]

        original_total = sum(original.values())

        target_total = (
            original_total * SCALE_NUM
            + SCALE_DEN // 2
        ) // SCALE_DEN

        group_targets = largest_remainder(
            target_total,
            original,
        )

        stage_group_targets[stage] = group_targets

        for group, group_target in group_targets.items():

            files = GROUP_FILES[group]

            file_weights = {
                name: SOURCE_TOKENS[name]
                for name in files
            }

            file_targets = largest_remainder(
                group_target,
                file_weights,
            )

            plan[stage].update(file_targets)

    # 检查各源文件总配额
    for filename, capacity in SOURCE_TOKENS.items():

        allocated = sum(
            plan[stage][filename]
            for stage in STAGES
        )

        if allocated > capacity:
            raise RuntimeError(
                f"Quota exceeds source capacity: {filename}"
            )

    return plan, stage_group_targets


# ============================================================
# 6. 显示计划
# ============================================================

def human(n):
    return f"{n / 1e9:.4f}B"


def print_plan(plan, group_targets):

    print("\n" + "=" * 85)
    print("MINIRAIN PRETRAIN COMPILATION PLAN")
    print("=" * 85)

    print(f"Source tokens : {CURRENT_TOTAL:,}")
    print(f"Scale factor  : {SCALE_NUM / SCALE_DEN:.8f}")
    print()

    print(
        f"{'Stage':<10}"
        f"{'L2':>13}"
        f"{'L3':>13}"
        f"{'Code':>13}"
        f"{'Math':>13}"
        f"{'Total':>15}"
    )

    for stage in STAGES:

        values = group_targets[stage]

        total = sum(values.values())

        print(
            f"{stage.upper():<10}"
            f"{human(values['l2']):>13}"
            f"{human(values['l3']):>13}"
            f"{human(values['code']):>13}"
            f"{human(values['math']):>13}"
            f"{human(total):>15}"
        )

    print("-" * 85)

    for filename in SOURCE_TOKENS:

        values = [
            plan[stage][filename]
            for stage in STAGES
        ]

        print(
            f"{filename:<38}"
            + " ".join(f"{human(x):>11}" for x in values)
        )

    print("=" * 85)


# ============================================================
# 7. 二进制 Writer
# ============================================================

class BinWriter:

    def __init__(
        self,
        stage,
        output_dir,
        flush_tokens=1_000_000,
    ):

        self.stage = stage

        self.bin_path = output_dir / f"{stage}_text.bin"
        self.idx_path = output_dir / f"{stage}_text.idx"

        self.bin_tmp = output_dir / f"{stage}_text.bin.tmp"
        self.idx_tmp = output_dir / f"{stage}_text.idx.tmp"

        self.bin_file = self.bin_tmp.open("wb")
        self.idx_file = self.idx_tmp.open("wb")

        self.tokens_buffer = array("H")

        # N+1 offsets，从 0 开始
        self.offset_buffer = array("Q", [0])

        self.flush_tokens = flush_tokens

        self.text_tokens = 0
        self.stored_tokens = 0
        self.records = 0
        self.at_max_length = 0

        self.closed = False

    def add(self, body_ids, bos_id, eos_id, max_body):

        if len(body_ids) > max_body:
            raise RuntimeError("Tokenizer truncation failed")

        if body_ids and (
            min(body_ids) < 0 or max(body_ids) > 65535
        ):
            raise RuntimeError("Token ID exceeds uint16")

        # 精确匹配旧 Dataset
        sequence = [bos_id] + body_ids + [eos_id]

        self.tokens_buffer.extend(sequence)

        self.text_tokens += len(body_ids)
        self.stored_tokens += len(sequence)
        self.records += 1

        if len(body_ids) == max_body:
            self.at_max_length += 1

        # 记录下一个样本的起始位置
        self.offset_buffer.append(
            self.stored_tokens
        )

        if len(self.tokens_buffer) >= self.flush_tokens:
            self.flush()

        if len(self.offset_buffer) >= 100_000:
            self.flush_index()

    def flush(self):

        if self.tokens_buffer:

            self.tokens_buffer.tofile(self.bin_file)

            self.tokens_buffer = array("H")

    def flush_index(self):

        if self.offset_buffer:

            self.offset_buffer.tofile(self.idx_file)

            self.offset_buffer = array("Q")

    def finalize(self):

        if self.closed:
            return

        self.flush()
        self.flush_index()

        self.bin_file.flush()
        self.idx_file.flush()

        os.fsync(self.bin_file.fileno())
        os.fsync(self.idx_file.fileno())

        self.bin_file.close()
        self.idx_file.close()

        self.closed = True

    def abort(self):

        if not self.closed:

            self.bin_file.close()
            self.idx_file.close()

            self.closed = True

        for path in (self.bin_tmp, self.idx_tmp):

            if path.exists():
                path.unlink()


# ============================================================
# 8. 单文件编译
# ============================================================

def compile_file(
    filename,
    data_dir,
    tokenizer,
    plan,
    writers,
    max_length,
    batch_size,
):

    path = data_dir / filename

    quotas = {
        stage: plan[stage][filename]
        for stage in STAGES
    }

    progress = {
        stage: 0
        for stage in STAGES
    }

    stage_records = defaultdict(int)

    max_body = max_length - 2

    stats = {
        "lines_read": 0,
        "invalid_json": 0,
        "records_assigned": 0,
        "unassigned_records": 0,
        "at_max_length": 0,
        "stage_tokens": defaultdict(int),
    }

    if sum(quotas.values()) == 0:
        return stats

    print(f"\n[FILE] {filename}")

    batch = []
    start = time.monotonic()

    def all_finished():
        return all(
            progress[s] >= quotas[s]
            for s in STAGES
        )

    def process_batch():

        if not batch:
            return

        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            max_length=max_body,
            truncation=True,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        for body_ids in encoded["input_ids"]:

            if all_finished():
                stats["unassigned_records"] += 1
                continue

            # 找到进度比例最低的阶段
            active = [
                stage
                for stage in STAGES
                if progress[stage] < quotas[stage]
            ]

            stage = min(
                active,
                key=lambda s: (
                    progress[s] / quotas[s],
                    stage_records[s],
                ),
            )

            n = len(body_ids)

            writers[stage].add(
                body_ids,
                tokenizer.bos_token_id,
                tokenizer.eos_token_id,
                max_body,
            )

            progress[stage] += n

            stage_records[stage] += 1

            stats["stage_tokens"][stage] += n
            stats["records_assigned"] += 1

            if n == max_body:
                stats["at_max_length"] += 1

        batch.clear()

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_no, line in enumerate(f, 1):

            if all_finished() and not batch:
                break

            stats["lines_read"] += 1

            try:
                obj = json.loads(line)

            except json.JSONDecodeError as exc:

                raise RuntimeError(
                    f"Invalid JSON: {filename}:{line_no}"
                ) from exc

            if not isinstance(obj, dict) or "text" not in obj:

                raise RuntimeError(
                    f"Missing text field: {filename}:{line_no}"
                )

            # 与原 PretrainDataset 一致
            text = str(obj["text"])

            batch.append(text)

            if len(batch) >= batch_size:

                process_batch()

                if stats["lines_read"] % 100_000 < batch_size:

                    elapsed = time.monotonic() - start

                    total = sum(progress.values())

                    print(
                        f"\r  Records: {stats['records_assigned']:,}"
                        f" | Tokens: {total:,}"
                        f" | Time: {elapsed/60:.1f} min",
                        end="",
                        flush=True,
                    )

        process_batch()

    # 检查是否达到所有文件配额
    unfinished = {
        stage: quotas[stage] - progress[stage]
        for stage in STAGES
        if progress[stage] < quotas[stage]
    }

    if unfinished:

        raise RuntimeError(
            f"Source exhausted before quotas reached: "
            f"{filename}: {unfinished}"
        )

    print()

    for stage in STAGES:

        if quotas[stage] > 0:

            print(
                f"  {stage.upper()}: "
                f"target={quotas[stage]:,} "
                f"actual={progress[stage]:,} "
                f"records={stage_records[stage]:,}"
            )

    elapsed = time.monotonic() - start

    print(f"  Finished in {elapsed/60:.2f} min")

    stats["stage_records"] = dict(stage_records)
    stats["stage_tokens"] = dict(stats["stage_tokens"])
    stats["target_tokens"] = quotas

    return stats


# ============================================================
# 9. Manifest
# ============================================================

def save_manifest(
    output_dir,
    tokenizer_dir,
    tokenizer,
    max_length,
    plan,
    group_targets,
    writers,
    source_results,
):

    manifest = {
        "project": "MiniRain-Vision",

        "updated_at": datetime.now().isoformat(),

        "tokenizer_dir": str(tokenizer_dir),

        "vocab_size": len(tokenizer),

        "max_length": max_length,

        "max_body_tokens": max_length - 2,

        "special_tokens": {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },

        "format": {
            "bin": "uint16 little-endian",
            "idx": "uint64 little-endian",
            "offset_count": "records + 1",
            "padding": "not stored",
            "document_concatenation": False,
        },

        "allocation": {
            "original_total": ORIGINAL_TOTAL,
            "source_total": CURRENT_TOTAL,
            "scale_factor": SCALE_NUM / SCALE_DEN,
            "stage_group_targets": group_targets,
            "source_file_quotas": plan,
        },

        "stages": {},

        "sources": source_results,

        "vision": {
            "available": False,
            "compiled": False,
        },
    }

    for stage in STAGES:

        writer = writers[stage]

        manifest["stages"][stage] = {
            "bin": f"{stage}_text.bin",
            "idx": f"{stage}_text.idx",

            "target_text_tokens": sum(
                plan[stage].values()
            ),

            "actual_text_tokens": writer.text_tokens,

            "stored_tokens": writer.stored_tokens,

            "records": writer.records,

            "at_max_body_length": writer.at_max_length,

            "bin_bytes": writer.stored_tokens * 2,

            "idx_bytes": (writer.records + 1) * 8,
        }

    tmp = output_dir / "manifest.json.tmp"
    final = output_dir / "manifest.json"

    with tmp.open("w", encoding="utf-8") as f:

        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, final)


# ============================================================
# 10. Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="MiniRain Multi-Stage Pretraining Compiler"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )

    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=131072,
        help="Must match training args.max_seq_len",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--flush-tokens",
        type=int,
        default=1_000_000,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    tokenizer_dir = args.tokenizer.resolve()
    output_dir = args.output_dir.resolve()

    if args.max_length < 3:
        parser.error("max-length must be >= 3")

    if args.batch_size <= 0:
        parser.error("batch-size must be positive")

    if args.flush_tokens <= 0:
        parser.error("flush-tokens must be positive")

    if sys.byteorder != "little":
        raise RuntimeError("Little-endian machine required")

    if array("H").itemsize != 2:
        raise RuntimeError("uint16 is not 2 bytes")

    if array("Q").itemsize != 8:
        raise RuntimeError("uint64 is not 8 bytes")

    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)

    # --------------------------------------------------------
    # Build plan
    # --------------------------------------------------------

    plan, group_targets = build_plan()

    print_plan(plan, group_targets)

    if args.dry_run:
        print("\n[DRY RUN] No files generated.")
        return

    # --------------------------------------------------------
    # Check sources
    # --------------------------------------------------------

    missing = [
        filename
        for filename in SOURCE_TOKENS
        if not (data_dir / filename).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing source files:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # Check output
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_names = ["manifest.json"]

    for stage in STAGES:
        output_names.extend([
            f"{stage}_text.bin",
            f"{stage}_text.idx",
        ])

    existing = [
        name for name in output_names
        if (output_dir / name).exists()
    ]

    if existing and not args.overwrite:
        raise FileExistsError(
            "Output already exists. Use --overwrite:\n"
            + "\n".join(existing)
        )

    temp_files = list(output_dir.glob("*.tmp"))

    if temp_files:
        raise RuntimeError(
            "Temporary files found. Inspect/remove them first:\n"
            + "\n".join(str(x) for x in temp_files)
        )

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print("\n[LOAD] Tokenizer")

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        use_fast=True,
        local_files_only=True,
    )

    if not tokenizer.is_fast:
        raise RuntimeError("Fast tokenizer required")

    if tokenizer.bos_token_id is None:
        raise RuntimeError("BOS token missing")

    if tokenizer.eos_token_id is None:
        raise RuntimeError("EOS token missing")

    if tokenizer.pad_token_id is None:
        raise RuntimeError("PAD token missing")

    if len(tokenizer) > 65536:
        raise RuntimeError("Vocabulary exceeds uint16 capacity")

    print(f"Vocab       : {len(tokenizer)}")
    print(f"BOS ID      : {tokenizer.bos_token_id}")
    print(f"EOS ID      : {tokenizer.eos_token_id}")
    print(f"PAD ID      : {tokenizer.pad_token_id}")
    print(f"Max length  : {args.max_length}")
    print(f"Body limit  : {args.max_length - 2}")

    # --------------------------------------------------------
    # Create writers
    # --------------------------------------------------------

    writers = {}
    source_results = {}

    try:

        for stage in STAGES:

            writers[stage] = BinWriter(
                stage,
                output_dir,
                flush_tokens=args.flush_tokens,
            )

        # ----------------------------------------------------
        # Process source files by group
        # ----------------------------------------------------

        print("\n" + "=" * 85)
        print("START COMPILATION")
        print("=" * 85)

        file_index = 0
        total_files = len(SOURCE_TOKENS)

        for group, filenames in GROUP_FILES.items():

            print(f"\n\n### GROUP: {group.upper()}")

            for filename in filenames:

                file_index += 1

                print(
                    f"\n[{file_index}/{total_files}]"
                )

                result = compile_file(
                    filename=filename,
                    data_dir=data_dir,
                    tokenizer=tokenizer,
                    plan=plan,
                    writers=writers,
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                )

                source_results[filename] = result

        # ----------------------------------------------------
        # Finalize all temporary files
        # ----------------------------------------------------

        for writer in writers.values():
            writer.finalize()

        # ----------------------------------------------------
        # Commit BIN and IDX
        # ----------------------------------------------------

        for stage in STAGES:

            writer = writers[stage]

            os.replace(
                writer.bin_tmp,
                writer.bin_path,
            )

            os.replace(
                writer.idx_tmp,
                writer.idx_path,
            )

        # ----------------------------------------------------
        # Manifest last
        # ----------------------------------------------------

        save_manifest(
            output_dir=output_dir,
            tokenizer_dir=tokenizer_dir,
            tokenizer=tokenizer,
            max_length=args.max_length,
            plan=plan,
            group_targets=group_targets,
            writers=writers,
            source_results=source_results,
        )

    except BaseException:

        for writer in writers.values():
            writer.abort()

        raise

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 85)
    print("COMPILATION COMPLETE")
    print("=" * 85)

    total_text_tokens = 0
    total_records = 0

    for stage in STAGES:

        writer = writers[stage]

        total_text_tokens += writer.text_tokens
        total_records += writer.records

        print(
            f"{stage.upper():<8}"
            f" Tokens={writer.text_tokens:,}"
            f" Records={writer.records:,}"
            f" BIN={writer.stored_tokens * 2 / 1e9:.3f} GB"
        )

    print("-" * 85)
    print(f"Total text tokens : {total_text_tokens:,}")
    print(f"Total records     : {total_records:,}")
    print(f"Output directory  : {output_dir}")
    print("=" * 85)


if __name__ == "__main__":
    main()
