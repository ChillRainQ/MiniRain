
"""
MiniRain-Vision UltraData Downloader
=====================================

Source:
    HF-Mirror: https://hf-mirror.com

Datasets:
    openbmb/Ultra-FineWeb
    openbmb/Ultra-FineWeb-L3
    openbmb/UltraData-Code
    openbmb/UltraData-Math

Features:
    - HF-Mirror endpoint
    - No HF authentication for public datasets
    - Streaming download
    - Sequential reading (no shuffle)
    - GB capacity allocation
    - Resume support
    - Pinned dataset revisions
    - Atomic checkpoints
    - Automatic orphan recovery
    - Script-relative output directory
    - First-sample heartbeat
"""

# ============================================================
# 1. Environment configuration
# IMPORTANT: before importing datasets/huggingface_hub
# ============================================================

import os

HF_ENDPOINT = "https://hf-mirror.com"

os.environ["HF_ENDPOINT"] = HF_ENDPOINT

# Avoid Xet requests that may bypass or be unsupported by mirrors.
os.environ["HF_HUB_DISABLE_XET"] = "1"

# Network timeout settings (seconds)
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"


# ============================================================
# 2. Imports
# ============================================================

import json
import time
import argparse
import threading

from itertools import chain
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset


# ============================================================
# 3. Global configuration
# ============================================================

GB = 1_000_000_000

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR / "data" / "ultradata"
)


# Dataset revisions previously resolved successfully.
DATASET_REVISIONS = {

    "openbmb/Ultra-FineWeb":
        "02c85641e3d19a854be2e09139c25adaa9518063",

    "openbmb/Ultra-FineWeb-L3":
        "bc3b1ba986fcaef6871b9790a413b16267c2de0f",

    "openbmb/UltraData-Code":
        "8ce985cfd4522832c8b986e05968c87fffa0c66a",

    "openbmb/UltraData-Math":
        "fe10db8efd35597fd7fcff8ff576b5ec4ea5ff87",
}


# Text dataset allocation weights.
GROUP_WEIGHTS = {
    "l2": 2.000,
    "l3": 2.140,
    "code": 0.855,
    "math": 0.705,
}


CODE_LANG_WEIGHTS = {
    "py": 0.40,
    "cpp": 0.12,
    "cs": 0.05,
    "go": 0.08,
    "java": 0.10,
    "js": 0.10,
    "php": 0.03,
    "r": 0.01,
    "rb": 0.01,
    "rust": 0.08,
    "sh": 0.02,
}


MATH_CONFIGS = [
    "UltraData-Math-L3-Conversation-Synthetic",
    "UltraData-Math-L3-Multi-Style-Synthetic",
    "UltraData-Math-L3-QA-Synthetic",
    "UltraData-Math-L3-Textbook-Exercise-Synthetic",
]


# ============================================================
# 4. Job definition
# ============================================================

@dataclass
class Job:

    name: str
    group: str
    repo: str
    config: str
    split: str
    weight: float
    language: str = ""


# ============================================================
# 5. Build download jobs
# ============================================================

def build_jobs(args):

    jobs = []

    # --------------------------------------------------------
    # Ultra-FineWeb L2
    # --------------------------------------------------------

    jobs.append(
        Job(
            name="fineweb_l2_zh",
            group="l2",
            repo="openbmb/Ultra-FineWeb",
            config="default",
            split="zh",
            weight=args.l2_zh_ratio,
            language="zh",
        )
    )

    jobs.append(
        Job(
            name="fineweb_l2_en",
            group="l2",
            repo="openbmb/Ultra-FineWeb",
            config="default",
            split="en",
            weight=1.0 - args.l2_zh_ratio,
            language="en",
        )
    )

    # --------------------------------------------------------
    # Ultra-FineWeb L3
    # --------------------------------------------------------

    for lang, lang_weight in [
        ("zh", args.l3_zh_ratio),
        ("en", 1.0 - args.l3_zh_ratio),
    ]:

        for style, style_weight in [
            ("QA", args.l3_qa_ratio),
            ("Multi-Style", 1.0 - args.l3_qa_ratio),
        ]:

            config = (
                f"Ultra-FineWeb-L3-{lang}-"
                f"{style}-Synthetic"
            )

            name = (
                f"fineweb_l3_{lang}_"
                f"{style.lower().replace('-', '_')}"
            )

            jobs.append(
                Job(
                    name=name,
                    group="l3",
                    repo="openbmb/Ultra-FineWeb-L3",
                    config=config,
                    split="train",
                    weight=lang_weight * style_weight,
                    language=lang,
                )
            )

    # --------------------------------------------------------
    # UltraData-Code L2 / L3
    # --------------------------------------------------------

    languages = [
        x.strip()
        for x in args.code_languages.split(",")
        if x.strip()
    ]

    invalid = set(languages) - set(CODE_LANG_WEIGHTS)

    if invalid:
        raise ValueError(
            f"Unsupported Code languages: {invalid}"
        )

    if not languages:
        raise ValueError("Code languages cannot be empty")

    lang_sum = sum(
        CODE_LANG_WEIGHTS[x]
        for x in languages
    )

    for tier, tier_weight in [
        ("L2", args.code_l2_ratio),
        ("L3", 1.0 - args.code_l2_ratio),
    ]:

        if tier_weight <= 0:
            continue

        for lang in languages:

            jobs.append(
                Job(
                    name=f"code_{tier.lower()}_{lang}",
                    group="code",
                    repo="openbmb/UltraData-Code",
                    config=f"UltraData-Code-{tier}",
                    split=lang,
                    weight=(
                        tier_weight
                        * CODE_LANG_WEIGHTS[lang]
                        / lang_sum
                    ),
                    language=lang,
                )
            )

    # --------------------------------------------------------
    # UltraData-Math L3
    # --------------------------------------------------------

    for i, config in enumerate(MATH_CONFIGS):

        jobs.append(
            Job(
                name=f"math_l3_{i}",
                group="math",
                repo="openbmb/UltraData-Math",
                config=config,
                split="train",
                weight=0.25,
            )
        )

    return jobs


# ============================================================
# 6. Capacity allocation
# ============================================================

def allocate_group_quotas(args):

    values = {
        "l2": args.l2_gb,
        "l3": args.l3_gb,
        "code": args.code_gb,
        "math": args.math_gb,
    }

    if args.total_gb is not None:

        if any(v is not None for v in values.values()):
            raise ValueError(
                "Do not combine --total-gb "
                "with individual capacities."
            )

        if args.total_gb <= 0:
            raise ValueError("--total-gb must be positive")

        total = int(args.total_gb * GB)

        weight_sum = sum(GROUP_WEIGHTS.values())

        result = {}
        used = 0

        groups = list(GROUP_WEIGHTS.items())

        for i, (group, weight) in enumerate(groups):

            if i == len(groups) - 1:
                amount = total - used
            else:
                amount = int(
                    total * weight / weight_sum
                )

            result[group] = amount
            used += amount

        return result

    if not all(v is not None for v in values.values()):

        raise ValueError(
            "Specify --total-gb OR all four individual capacities."
        )

    if any(v < 0 for v in values.values()):
        raise ValueError("Capacity cannot be negative")

    return {
        group: int(value * GB)
        for group, value in values.items()
    }


def allocate_job_quotas(jobs, group_quotas):

    quotas = {}

    for group, group_bytes in group_quotas.items():

        group_jobs = [
            job for job in jobs
            if job.group == group
        ]

        if group_bytes <= 0:
            continue

        if not group_jobs:
            raise ValueError(
                f"No jobs configured for group {group}"
            )

        weight_sum = sum(
            job.weight for job in group_jobs
        )

        if weight_sum <= 0:
            raise ValueError(
                f"Invalid weight for group {group}"
            )

        used = 0

        for i, job in enumerate(group_jobs):

            if i == len(group_jobs) - 1:
                amount = group_bytes - used
            else:
                amount = int(
                    group_bytes * job.weight / weight_sum
                )

            quotas[job.name] = amount
            used += amount

    return quotas


# ============================================================
# 7. Atomic JSON state
# ============================================================

def save_json_atomic(path, data):

    temp = path.with_name(path.name + ".tmp")

    with open(temp, "w", encoding="utf-8") as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        f.flush()
        os.fsync(f.fileno())

    os.replace(temp, path)


def load_state(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_signature(job, revision):

    return {
        "repo": job.repo,
        "config": job.config,
        "split": job.split,
        "revision": revision,

        # Explicitly identifies sequential iteration.
        "stream_order": "sequential-v1",
    }


def make_state(
    signature,
    source_rows,
    records,
    bytes_done,
    target_bytes,
    exhausted=False,
    limit_reached=False,
):

    return {
        **signature,
        "source_rows": source_rows,
        "records": records,
        "bytes": bytes_done,
        "target_bytes": target_bytes,
        "exhausted": exhausted,
        "limit_reached": limit_reached,
    }


# ============================================================
# 8. Backup and recovery
# ============================================================

def backup_file(path, suffix="backup"):

    if not path.exists():
        return

    stamp = (
        time.strftime("%Y%m%d_%H%M%S")
        + f"_{time.time_ns() % 1_000_000:06d}"
    )

    backup = path.with_name(
        f"{path.stem}.{suffix}_{stamp}{path.suffix}"
    )

    os.replace(path, backup)

    print(f"[BACKUP] {backup.name}")


def backup_pair(output_path, state_path, suffix="stale"):

    backup_file(output_path, suffix)
    backup_file(state_path, suffix)


def recover_orphan_files(output_path, state_path):

    if output_path.exists() == state_path.exists():
        return

    print(
        f"[RECOVERY] Incomplete file pair: "
        f"{output_path.stem}"
    )

    backup_pair(
        output_path,
        state_path,
        suffix="orphan",
    )


# ============================================================
# 9. Text extraction
# ============================================================

def value_to_text(value):

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, (dict, list, tuple)):

        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    return str(value)


def extract_text(row, job):

    # --------------------------------------------------------
    # Code L3
    # --------------------------------------------------------

    if (
        job.group == "code"
        and job.config == "UltraData-Code-L3"
    ):

        for key in (
            "full_content",
            "content",
            "raw_content",
        ):

            value = row.get(key)

            if value is not None:

                text = value_to_text(value)

                if text.strip():
                    return text

        parts = []

        for key in (
            "task",
            "analysis",
            "solution",
            "test",
        ):

            value = row.get(key)

            if value is not None:

                text = value_to_text(value)

                if text.strip():
                    parts.append(
                        f"## {key.upper()}\n{text}"
                    )

        if parts:
            return "\n\n".join(parts)

    # --------------------------------------------------------
    # General text fields
    # --------------------------------------------------------

    for key in (
        "content",
        "text",
        "full_content",
        "raw_content",
        "body",
        "messages",
        "conversations",
        "conversation",
        "question",
        "problem",
        "analysis",
        "solution",
        "answer",
        "response",
    ):

        value = row.get(key)

        if value is not None:

            text = value_to_text(value)

            if text.strip():
                return text

    # --------------------------------------------------------
    # Generic fallback:
    # preserve the complete row instead of silently discarding
    # an unfamiliar schema.
    # --------------------------------------------------------

    if row:

        return json.dumps(
            row,
            ensure_ascii=False,
            default=str,
        )

    return ""


# ============================================================
# 10. Metadata
# ============================================================

def extract_metadata(row):

    keys = (
        "uuid",
        "source",
        "url",
        "score",
        "quality_score",
        "algo_rel_score",
        "repo_name",
        "relative_path",
        "category",
        "content_format",
        "full_content_format",
        "style",
        "uid",
    )

    metadata = {}

    for key in keys:

        value = row.get(key)

        if value is not None:
            metadata[key] = value

    return metadata


# ============================================================
# 11. JSONL record
# ============================================================

def make_record(row, job, min_chars):

    text = extract_text(row, job)

    if not text or len(text.strip()) < min_chars:
        return None

    return {
        "text": text,
        "dataset": job.repo,
        "config": job.config,
        "split": job.split,
        "language": job.language,
        "metadata": extract_metadata(row),
    }


# ============================================================
# 12. First sample heartbeat
# ============================================================

def next_with_heartbeat(iterator, label, interval=15):

    stop_event = threading.Event()

    def heartbeat():

        while not stop_event.wait(interval):

            print(
                f"[STREAM] Still waiting for data: {label}",
                flush=True,
            )

    thread = threading.Thread(
        target=heartbeat,
        daemon=True,
    )

    thread.start()

    try:
        return next(iterator)

    finally:
        stop_event.set()
        thread.join(timeout=1)


# ============================================================
# 13. Download one dataset
# ============================================================

def download_job(
    job,
    quota,
    output_dir,
    revision,
    args,
):

    output_path = output_dir / f"{job.name}.jsonl"

    state_path = output_dir / f"{job.name}.state.json"

    if quota <= 0:
        return None

    signature = make_signature(
        job,
        revision,
    )

    # --------------------------------------------------------
    # Explicit overwrite
    # --------------------------------------------------------

    if args.overwrite:

        output_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)

    # --------------------------------------------------------
    # Orphan recovery
    # --------------------------------------------------------

    recover_orphan_files(
        output_path,
        state_path,
    )

    # --------------------------------------------------------
    # Resume variables
    # --------------------------------------------------------

    source_rows = 0
    records = 0
    bytes_done = 0

    exhausted = False
    limit_reached = False
    old_target = quota

    # --------------------------------------------------------
    # Existing files
    # --------------------------------------------------------

    if output_path.exists():

        try:

            state = load_state(state_path)

            valid_signature = all(
                state.get(key) == value
                for key, value in signature.items()
            )

            if not valid_signature:

                print(
                    f"[RECOVERY] Old download settings detected: "
                    f"{job.name}"
                )

                backup_pair(
                    output_path,
                    state_path,
                    suffix="stale",
                )

            else:

                source_rows = int(
                    state["source_rows"]
                )

                records = int(
                    state["records"]
                )

                bytes_done = int(
                    state["bytes"]
                )

                exhausted = bool(
                    state.get("exhausted", False)
                )

                limit_reached = bool(
                    state.get("limit_reached", False)
                )

                old_target = int(
                    state.get("target_bytes", quota)
                )

                actual_size = output_path.stat().st_size

                if actual_size < bytes_done:

                    print(
                        "[RECOVERY] File shorter than checkpoint."
                    )

                    backup_pair(
                        output_path,
                        state_path,
                        suffix="corrupt",
                    )

                    source_rows = 0
                    records = 0
                    bytes_done = 0
                    exhausted = False
                    limit_reached = False

                elif bytes_done > quota:

                    raise RuntimeError(
                        f"{job.name}: existing file exceeds "
                        "the requested quota."
                    )

                else:

                    # Remove uncommitted bytes after the last
                    # valid checkpoint.
                    with open(output_path, "r+b") as f:
                        f.truncate(bytes_done)

                    print(
                        f"[RESUME] {job.name}: "
                        f"{bytes_done / GB:.3f} GB, "
                        f"{records:,} records, "
                        f"{source_rows:,} source rows"
                    )

        except (ValueError, KeyError, json.JSONDecodeError):

            print(
                f"[RECOVERY] Invalid state: {job.name}"
            )

            backup_pair(
                output_path,
                state_path,
                suffix="invalid",
            )

            source_rows = 0
            records = 0
            bytes_done = 0
            exhausted = False
            limit_reached = False

    # --------------------------------------------------------
    # Create fresh pair
    # --------------------------------------------------------

    if not output_path.exists():

        output_path.touch()

        save_json_atomic(
            state_path,
            make_state(
                signature,
                0,
                0,
                0,
                quota,
            ),
        )

        print(f"[NEW] {job.name}")

    # --------------------------------------------------------
    # Completed checks
    # --------------------------------------------------------

    if exhausted:

        print(f"[SKIP] {job.name}: source exhausted")

        return {
            "name": job.name,
            "target_bytes": quota,
            "downloaded_bytes": bytes_done,
            "records": records,
            "source_rows": source_rows,
            "exhausted": True,
        }

    if bytes_done >= quota:

        print(f"[SKIP] {job.name}: quota reached")

        return {
            "name": job.name,
            "target_bytes": quota,
            "downloaded_bytes": bytes_done,
            "records": records,
            "source_rows": source_rows,
            "exhausted": False,
        }

    if limit_reached and old_target == quota:

        print(f"[SKIP] {job.name}: target already reached")

        return {
            "name": job.name,
            "target_bytes": quota,
            "downloaded_bytes": bytes_done,
            "records": records,
            "source_rows": source_rows,
            "exhausted": False,
        }

    # --------------------------------------------------------
    # Load streaming dataset
    # --------------------------------------------------------

    print(f"\n[LOAD] {job.name}")
    print(f"  Repo: {job.repo}")
    print(f"  Config: {job.config}")
    print(f"  Split: {job.split}")
    print(f"  Revision: {revision}")
    print(f"  Target: {quota / GB:.3f} GB")

    print(
        f"[MIRROR] Endpoint: {HF_ENDPOINT}",
        flush=True,
    )

    print(
        "[STREAM] Calling load_dataset...",
        flush=True,
    )

    ds = load_dataset(
        path=job.repo,
        name=job.config,
        split=job.split,
        streaming=True,
        revision=revision,
        token=False,
    )

    print(
        "[STREAM] load_dataset returned.",
        flush=True,
    )

    print(
        "[STREAM] Features:",
        ds.features,
        flush=True,
    )

    # Sequential streaming. No shuffle.
    print(
        "[STREAM] Sequential reading enabled.",
        flush=True,
    )

    if source_rows:

        print(
            f"[STREAM] Replaying {source_rows:,} rows...",
            flush=True,
        )

        ds = ds.skip(source_rows)

    iterator = iter(ds)

    # --------------------------------------------------------
    # Download loop
    # --------------------------------------------------------

    start_time = time.time()
    start_bytes = bytes_done
    last_report = start_time

    stopped_by_size = False
    skipped_oversize = 0

    def persist(
        f,
        is_exhausted=False,
        is_limit_reached=False,
    ):

        f.flush()
        os.fsync(f.fileno())

        save_json_atomic(
            state_path,
            make_state(
                signature,
                source_rows,
                records,
                bytes_done,
                quota,
                exhausted=is_exhausted,
                limit_reached=is_limit_reached,
            ),
        )

    with open(output_path, "r+b") as f:

        f.truncate(bytes_done)
        f.seek(bytes_done)

        try:

            print(
                "[STREAM] Waiting for first sample...",
                flush=True,
            )

            try:

                first_row = next_with_heartbeat(
                    iterator,
                    job.name,
                    interval=15,
                )

            except StopIteration:

                print(
                    "[STREAM] Dataset is empty.",
                    flush=True,
                )

                persist(
                    f,
                    is_exhausted=True,
                )

                return {
                    "name": job.name,
                    "target_bytes": quota,
                    "downloaded_bytes": bytes_done,
                    "records": records,
                    "source_rows": source_rows,
                    "exhausted": True,
                }

            print(
                "[STREAM] First sample received.",
                flush=True,
            )

            print(
                "[STREAM] Keys:",
                list(first_row.keys()),
                flush=True,
            )

            for row in chain(
                (first_row,),
                iterator,
            ):

                record = make_record(
                    row,
                    job,
                    args.min_chars,
                )

                if record is None:

                    source_rows += 1

                else:

                    line = (
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            default=str,
                        ) + "\n"
                    ).encode("utf-8")

                    line_size = len(line)

                    # Skip a single record that cannot fit
                    # into the entire quota.
                    if (
                        bytes_done == 0
                        and line_size > quota
                    ):

                        source_rows += 1
                        skipped_oversize += 1
                        continue

                    # Do not consume a row that does not fit.
                    if bytes_done + line_size > quota:

                        stopped_by_size = True
                        break

                    f.write(line)

                    bytes_done += line_size
                    records += 1
                    source_rows += 1

                    if bytes_done >= quota:

                        stopped_by_size = True
                        break

                # ------------------------------------------------
                # Periodic checkpoint
                # ------------------------------------------------

                if (
                    source_rows > 0
                    and source_rows % args.save_every_rows == 0
                ):

                    persist(f)

                # ------------------------------------------------
                # Progress
                # ------------------------------------------------

                now = time.time()

                if now - last_report >= 5:

                    elapsed = max(
                        now - start_time,
                        0.001,
                    )

                    speed = (
                        bytes_done - start_bytes
                    ) / elapsed / (1024 ** 2)

                    print(
                        f"\r"
                        f"{bytes_done / GB:.3f} / "
                        f"{quota / GB:.3f} GB"
                        f" | records={records:,}"
                        f" | source_rows={source_rows:,}"
                        f" | speed={speed:.2f} MiB/s",
                        end="",
                        flush=True,
                    )

                    last_report = now

        except KeyboardInterrupt:

            print(
                "\n[INTERRUPTED] Saving checkpoint...",
                flush=True,
            )

            persist(f)
            raise

        except Exception:

            print(
                "\n[ERROR] Saving checkpoint...",
                flush=True,
            )

            persist(f)
            raise

        else:

            persist(
                f,
                is_exhausted=not stopped_by_size,
                is_limit_reached=stopped_by_size,
            )

    elapsed = time.time() - start_time

    print(
        f"\n[DONE] {job.name}\n"
        f"  Target: {quota / GB:.3f} GB\n"
        f"  Saved: {bytes_done / GB:.3f} GB\n"
        f"  Records: {records:,}\n"
        f"  Source rows: {source_rows:,}\n"
        f"  Oversized skipped: {skipped_oversize:,}\n"
        f"  Time: {elapsed:.1f}s",
        flush=True,
    )

    if bytes_done < quota and not stopped_by_size:

        print(
            f"[WARN] {job.name}: source exhausted "
            "before reaching target."
        )

    return {
        "name": job.name,
        "target_bytes": quota,
        "downloaded_bytes": bytes_done,
        "records": records,
        "source_rows": source_rows,
        "exhausted": not stopped_by_size,
    }


# ============================================================
# 14. Manifest
# ============================================================

def save_manifest(
    output_dir,
    group_quotas,
    results,
):

    manifest = {

        "source": HF_ENDPOINT,

        "unit": "decimal GB",

        "group_quotas": {
            key: value / GB
            for key, value in group_quotas.items()
        },

        "revisions": DATASET_REVISIONS,

        "results": results,

        "total_target_bytes": sum(
            group_quotas.values()
        ),

        "total_saved_bytes": sum(
            item["downloaded_bytes"]
            for item in results
        ),

        "updated_at": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }

    save_json_atomic(
        output_dir / "manifest.json",
        manifest,
    )


# ============================================================
# 15. CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="MiniRain-Vision HF-Mirror Downloader"
    )

    parser.add_argument(
        "--total-gb",
        type=float,
        default=None,
    )

    parser.add_argument("--l2-gb", type=float)
    parser.add_argument("--l3-gb", type=float)
    parser.add_argument("--code-gb", type=float)
    parser.add_argument("--math-gb", type=float)

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory",
    )

    parser.add_argument(
        "--code-languages",
        type=str,
        default="py,cpp,cs,go,java,js,php,r,rb,rust,sh",
    )

    parser.add_argument(
        "--code-l2-ratio",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--l2-zh-ratio",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--l3-zh-ratio",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--l3-qa-ratio",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--save-every-rows",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--min-chars",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# 16. Main
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if args.save_every_rows <= 0:
        raise ValueError("--save-every-rows must be positive")

    if args.min_chars < 0:
        raise ValueError("--min-chars cannot be negative")

    for name, value in [
        ("l2_zh_ratio", args.l2_zh_ratio),
        ("l3_zh_ratio", args.l3_zh_ratio),
        ("l3_qa_ratio", args.l3_qa_ratio),
        ("code_l2_ratio", args.code_l2_ratio),
    ]:

        if not 0 <= value <= 1:
            raise ValueError(
                f"{name} must be between 0 and 1"
            )

    # --------------------------------------------------------
    # Jobs and capacity
    # --------------------------------------------------------

    jobs = build_jobs(args)

    group_quotas = allocate_group_quotas(args)

    job_quotas = allocate_job_quotas(
        jobs,
        group_quotas,
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else DEFAULT_OUTPUT_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Revisions
    # --------------------------------------------------------

    revisions = DATASET_REVISIONS

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    total_target = sum(group_quotas.values())

    print("\n" + "=" * 65)
    print("MiniRain-Vision UltraData Downloader")
    print("=" * 65)

    print(f"[MIRROR] Endpoint: {HF_ENDPOINT}")
    print("[MIRROR] Authentication: disabled (public datasets)")
    print("[STREAM] Sequential reading: enabled")
    print("[STREAM] Shuffle: disabled")

    print("\n[REVISION] Using pinned versions")

    for repo, sha in revisions.items():
        print(f"  {repo}: {sha}")

    print("\n[CAPACITY]")

    for group, quota in group_quotas.items():

        print(
            f"{group:8s}: {quota / GB:10.3f} GB"
        )

    print("-" * 65)
    print(
        f"{'TOTAL':8s}: "
        f"{total_target / GB:10.3f} GB"
    )

    print(f"Output: {output_dir}")
    print(f"Checkpoint interval: {args.save_every_rows} rows")

    print("=" * 65)

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    results = []

    try:

        for job in jobs:

            quota = job_quotas.get(
                job.name,
                0,
            )

            if quota <= 0:
                continue

            result = download_job(
                job=job,
                quota=quota,
                output_dir=output_dir,
                revision=revisions[job.repo],
                args=args,
            )

            if result is not None:
                results.append(result)

            # Save manifest after each job.
            save_manifest(
                output_dir,
                group_quotas,
                results,
            )

    except KeyboardInterrupt:

        print("\n[STOPPED] Checkpoints saved.")
        print(
            "Run the same command again to resume."
        )

        return

    except Exception as exc:

        print(
            f"\n[FAILED] {type(exc).__name__}: {exc}"
        )

        print(
            "Existing data and checkpoints are preserved."
        )

        raise

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    actual = sum(
        item["downloaded_bytes"]
        for item in results
    )

    print("\n" + "=" * 65)
    print("DOWNLOAD SUMMARY")
    print("=" * 65)

    for item in results:

        print(
            f"{item['name']:35s}"
            f"{item['downloaded_bytes'] / GB:9.3f} GB"
        )

    print("-" * 65)
    print(f"Target: {total_target / GB:.3f} GB")
    print(f"Saved:  {actual / GB:.3f} GB")
    print(f"Output: {output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
