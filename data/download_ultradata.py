
#!/usr/bin/env python3
"""
MiniRain-Vision UltraData Downloader

HF-Mirror streaming downloader.

Default budget: 80 GB
    Ultra-FineWeb L2 : 28.070 GB
    Ultra-FineWeb L3 : 30.035 GB
    UltraData-Code   : 12.000 GB
        Python       :  7.200 GB
        Shell        :  2.400 GB
        CMD reserved :  2.400 GB
    UltraData-Math   :  9.895 GB

Only Python and sh are downloaded from UltraData-Code.
The reserved CMD budget is not downloaded.

Features:
    - HF-Mirror
    - Streaming
    - Sequential reading
    - Pinned revisions
    - Resume from existing state
    - Atomic checkpoint
    - GB quota allocation
    - Dry run
    - No overwrite of original manifest.json
"""

# ============================================================
# 1. Environment
# ============================================================

import os

HF_ENDPOINT = "https://hf-mirror.com"

# Set before importing datasets / huggingface_hub.
os.environ["HF_ENDPOINT"] = HF_ENDPOINT
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"


# ============================================================
# 2. Imports
# ============================================================

import json
import time
import argparse
import threading

from dataclasses import dataclass
from itertools import chain
from pathlib import Path

from datasets import load_dataset


# ============================================================
# 3. Global configuration
# ============================================================

GB = 1_000_000_000

SCRIPT_DIR = Path(__file__).resolve().parent

# data/download_ultradata.py
# data/ultradata/
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "ultradata"


# ============================================================
# 4. Dataset revisions
# ============================================================

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


# ============================================================
# 5. Dataset weights
# ============================================================

# Keep original 80GB allocation proportions.
GROUP_WEIGHTS = {
    "l2": 2.000,
    "l3": 2.140,
    "code": 0.855,
    "math": 0.705,
}

# Code ratios are relative to the ENTIRE Code budget.
#
# Python: 60%
# Shell : 20%
# CMD   : 20% reserved, not downloaded.
#
# Do NOT normalize these to sum to 1.

CODE_LANG_WEIGHTS = {
    "py": 0.60,
    "sh": 0.20,
}

CMD_RESERVED_RATIO = 0.20


MATH_CONFIGS = [
    "UltraData-Math-L3-Conversation-Synthetic",
    "UltraData-Math-L3-Multi-Style-Synthetic",
    "UltraData-Math-L3-QA-Synthetic",
    "UltraData-Math-L3-Textbook-Exercise-Synthetic",
]


# ============================================================
# 6. Job definition
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
# 7. Build jobs
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
    # UltraData-Code
    #
    # Python: 60% of Code budget
    # Shell : 20% of Code budget
    # CMD   : 20% reserved
    #
    # L2 / L3 split is controlled by code_l2_ratio.
    # --------------------------------------------------------

    for tier, tier_weight in [
        ("L2", args.code_l2_ratio),
        ("L3", 1.0 - args.code_l2_ratio),
    ]:

        if tier_weight <= 0:
            continue

        for lang, lang_weight in CODE_LANG_WEIGHTS.items():

            jobs.append(
                Job(
                    name=f"code_{tier.lower()}_{lang}",
                    group="code",
                    repo="openbmb/UltraData-Code",
                    config=f"UltraData-Code-{tier}",
                    split=lang,
                    weight=tier_weight * lang_weight,
                    language=lang,
                )
            )

    # --------------------------------------------------------
    # UltraData-Math
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
# 8. Allocate group quotas
# ============================================================

def allocate_group_quotas(args):

    values = {
        "l2": args.l2_gb,
        "l3": args.l3_gb,
        "code": args.code_gb,
        "math": args.math_gb,
    }

    # --------------------------------------------------------
    # Total GB mode
    # --------------------------------------------------------

    if args.total_gb is not None:

        if any(v is not None for v in values.values()):
            raise ValueError(
                "Cannot combine --total-gb with "
                "individual group capacities."
            )

        if args.total_gb <= 0:
            raise ValueError(
                "--total-gb must be positive."
            )

        total = round(args.total_gb * GB)

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

    # --------------------------------------------------------
    # Individual group capacities
    # --------------------------------------------------------

    if not all(v is not None for v in values.values()):
        raise ValueError(
            "Specify --total-gb OR all four "
            "individual capacities."
        )

    if any(v < 0 for v in values.values()):
        raise ValueError(
            "Capacity cannot be negative."
        )

    return {
        group: round(value * GB)
        for group, value in values.items()
    }


# ============================================================
# 9. Allocate individual job quotas
# ============================================================

def allocate_job_quotas(jobs, group_quotas, args):

    quotas = {}

    for group, group_bytes in group_quotas.items():

        if group_bytes <= 0:
            continue

        group_jobs = [
            job for job in jobs
            if job.group == group
        ]

        # ----------------------------------------------------
        # Code: absolute ratios, no normalization
        # ----------------------------------------------------

        if group == "code":

            for lang, ratio in CODE_LANG_WEIGHTS.items():

                lang_total = round(group_bytes * ratio)

                l2_bytes = round(
                    lang_total * args.code_l2_ratio
                )

                l3_bytes = lang_total - l2_bytes

                quotas[f"code_l2_{lang}"] = l2_bytes
                quotas[f"code_l3_{lang}"] = l3_bytes

            # Remaining budget belongs to CMD.
            # No dataset job is created for it.
            continue

        # ----------------------------------------------------
        # Other groups: original normalized weights
        # ----------------------------------------------------

        if not group_jobs:
            raise ValueError(
                f"No jobs configured for group: {group}"
            )

        weight_sum = sum(
            job.weight for job in group_jobs
        )

        if weight_sum <= 0:
            raise ValueError(
                f"Invalid weight in group: {group}"
            )

        used = 0

        for i, job in enumerate(group_jobs):

            if i == len(group_jobs) - 1:
                amount = group_bytes - used
            else:
                amount = int(
                    group_bytes
                    * job.weight
                    / weight_sum
                )

            quotas[job.name] = amount
            used += amount

    return quotas


# ============================================================
# 10. Atomic state saving
# ============================================================

def save_json_atomic(path, data):

    temp = path.with_name(
        path.name + ".tmp"
    )

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


# ============================================================
# 11. State signature
#
# Keep this consistent with original sequential downloader.
# ============================================================

def make_signature(job, revision):

    return {
        "repo": job.repo,
        "config": job.config,
        "split": job.split,
        "revision": revision,
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
# 12. Text conversion
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


# ============================================================
# 13. Text extraction
# ============================================================

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
    # Generic fallback
    # --------------------------------------------------------

    if row:
        return json.dumps(
            row,
            ensure_ascii=False,
            default=str,
        )

    return ""


# ============================================================
# 14. Metadata extraction
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
# 15. JSONL record
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
# 16. Wait heartbeat
# ============================================================

def next_with_heartbeat(
    iterator,
    label,
    interval=15,
):

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
# 17. Prepare existing files
#
# Important:
# - Never silently discard an existing dataset.
# - Never reset a mismatching checkpoint.
# - Never shrink a dataset to a smaller target.
# ============================================================

def prepare_job(job, quota, output_dir, revision):

    data_path = output_dir / f"{job.name}.jsonl"
    state_path = output_dir / f"{job.name}.state.json"

    signature = make_signature(job, revision)

    if data_path.exists() != state_path.exists():

        raise RuntimeError(
            f"{job.name}: JSONL/state file mismatch. "
            "Both files must exist for resuming."
        )

    if not data_path.exists():

        state = make_state(
            signature,
            source_rows=0,
            records=0,
            bytes_done=0,
            target_bytes=quota,
        )

        return data_path, state_path, state, False

    state = load_state(state_path)

    # Check original revision and stream order.
    for key, value in signature.items():

        if state.get(key) != value:

            raise RuntimeError(
                f"{job.name}: checkpoint mismatch: {key}. "
                f"Existing={state.get(key)!r}, "
                f"Expected={value!r}. "
                "Refusing unsafe resume."
            )

    for key in ("source_rows", "records", "bytes"):

        if key not in state:
            raise RuntimeError(
                f"{job.name}: missing state field {key}"
            )

        if int(state[key]) < 0:
            raise RuntimeError(
                f"{job.name}: invalid state field {key}"
            )

    recorded_bytes = int(state["bytes"])
    actual_bytes = data_path.stat().st_size

    if actual_bytes < recorded_bytes:

        raise RuntimeError(
            f"{job.name}: JSONL smaller than checkpoint. "
            "Refusing unsafe resume."
        )

    if recorded_bytes > quota:

        raise RuntimeError(
            f"{job.name}: existing saved data "
            f"({recorded_bytes / GB:.3f} GB) "
            f"exceeds target ({quota / GB:.3f} GB). "
            "Existing data will not be truncated."
        )

    return data_path, state_path, state, True


# ============================================================
# 18. Download one job
# ============================================================

def download_job(
    job,
    quota,
    output_dir,
    revision,
    args,
):

    if quota <= 0:
        return None

    (
        data_path,
        state_path,
        state,
        exists,
    ) = prepare_job(
        job,
        quota,
        output_dir,
        revision,
    )

    signature = make_signature(job, revision)

    source_rows = int(state["source_rows"])
    records = int(state["records"])
    bytes_done = int(state["bytes"])

    old_target = int(
        state.get("target_bytes", quota)
    )

    exhausted = bool(
        state.get("exhausted", False)
    )

    limit_reached = bool(
        state.get("limit_reached", False)
    )

    actual_size = (
        data_path.stat().st_size
        if exists else 0
    )

    extra_bytes = actual_size - bytes_done

    # --------------------------------------------------------
    # Progress plan
    # --------------------------------------------------------

    remaining = max(
        0,
        quota - bytes_done,
    )

    print(
        f"[PLAN] {job.name:30s} "
        f"{bytes_done / GB:.3f} / "
        f"{quota / GB:.3f} GB "
        f"| remaining={remaining / GB:.3f} GB "
        f"| rows={source_rows:,}",
        flush=True,
    )

    def result(is_exhausted):

        return {
            "name": job.name,
            "target_bytes": quota,
            "downloaded_bytes": bytes_done,
            "records": records,
            "source_rows": source_rows,
            "exhausted": is_exhausted,
        }

    # --------------------------------------------------------
    # Dry run: no file modification
    # --------------------------------------------------------

    if args.dry_run:

        if extra_bytes > 0:

            print(
                f"[CHECK] {job.name}: "
                f"{extra_bytes} bytes exist beyond "
                "the last checkpoint."
            )

        return result(exhausted)

    # --------------------------------------------------------
    # Create new file pair
    # --------------------------------------------------------

    if not exists:

        with data_path.open("xb"):
            pass

        save_json_atomic(
            state_path,
            state,
        )

        print(
            f"[NEW] {job.name}",
            flush=True,
        )

    # --------------------------------------------------------
    # Recover uncommitted data
    # --------------------------------------------------------

    elif extra_bytes > 0:

        # The checkpoint is the committed boundary.
        with data_path.open("r+b") as f:
            f.truncate(bytes_done)

        print(
            f"[RECOVER] {job.name}: "
            f"discarded {extra_bytes} "
            "uncommitted tail bytes.",
            flush=True,
        )

    # --------------------------------------------------------
    # Completion checks
    # --------------------------------------------------------

    if exhausted:

        print(
            f"[SKIP] {job.name}: source exhausted.",
            flush=True,
        )

        return result(True)

    if bytes_done >= quota:

        print(
            f"[SKIP] {job.name}: quota reached.",
            flush=True,
        )

        return result(False)

    # Existing target already ended on a row boundary.
    # If target is increased, continue normally.
    if (
        limit_reached
        and old_target == quota
    ):

        print(
            f"[SKIP] {job.name}: "
            "same target already reached.",
            flush=True,
        )

        return result(False)

    # --------------------------------------------------------
    # Load streaming dataset
    # --------------------------------------------------------

    print(
        f"\n[LOAD] {job.name}",
        flush=True,
    )

    print(f"  Repo: {job.repo}")
    print(f"  Config: {job.config}")
    print(f"  Split: {job.split}")
    print(f"  Revision: {revision}")
    print(f"  Target: {quota / GB:.3f} GB")
    print(f"  Endpoint: {HF_ENDPOINT}")

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

    # --------------------------------------------------------
    # Sequential reading
    # --------------------------------------------------------

    if source_rows > 0:

        print(
            f"[RESUME] Skipping "
            f"{source_rows:,} source rows...",
            flush=True,
        )

        ds = ds.skip(source_rows)

    iterator = iter(ds)

    # --------------------------------------------------------
    # Timing and counters
    # --------------------------------------------------------

    start_time = time.monotonic()
    start_bytes = bytes_done
    last_report = start_time

    stopped_by_size = False
    end_of_source = False
    skipped_oversize = 0

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

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
                signature=signature,
                source_rows=source_rows,
                records=records,
                bytes_done=bytes_done,
                target_bytes=quota,
                exhausted=is_exhausted,
                limit_reached=is_limit_reached,
            ),
        )

    # --------------------------------------------------------
    # Open output
    # --------------------------------------------------------

    with data_path.open("r+b") as f:

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

                end_of_source = True

                persist(
                    f,
                    is_exhausted=True,
                )

                print(
                    f"[EMPTY] {job.name}: no more rows.",
                    flush=True,
                )

                return result(True)

            print(
                "[STREAM] First sample received.",
                flush=True,
            )

            print(
                "[STREAM] Keys:",
                list(first_row.keys()),
                flush=True,
            )

            # ------------------------------------------------
            # Stream rows
            # ------------------------------------------------

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

                    # Skip a single document larger than
                    # the entire capacity.
                    if (
                        bytes_done == 0
                        and line_size > quota
                    ):

                        source_rows += 1
                        skipped_oversize += 1
                        continue

                    # Do not count the row when it
                    # does not fit in the remaining quota.
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

                # --------------------------------------------
                # Periodic checkpoint
                # --------------------------------------------

                if (
                    source_rows > 0
                    and source_rows % args.save_every_rows == 0
                ):

                    persist(f)

                # --------------------------------------------
                # Progress
                # --------------------------------------------

                now = time.monotonic()

                if now - last_report >= 5:

                    elapsed = max(
                        now - start_time,
                        0.001,
                    )

                    speed = (
                        bytes_done - start_bytes
                    ) / elapsed / (1024 ** 2)

                    print(
                        f"[PROGRESS] {job.name}: "
                        f"{bytes_done / GB:.3f} / "
                        f"{quota / GB:.3f} GB"
                        f" | records={records:,}"
                        f" | source_rows={source_rows:,}"
                        f" | speed={speed:.2f} MiB/s",
                        flush=True,
                    )

                    last_report = now

            else:

                end_of_source = True

        # ----------------------------------------------------
        # Interrupted / error
        # ----------------------------------------------------

        except BaseException:

            print(
                f"\n[INTERRUPTED] {job.name}: "
                "saving checkpoint...",
                flush=True,
            )

            persist(f)
            raise

        # ----------------------------------------------------
        # Normal completion
        # ----------------------------------------------------

        persist(
            f,
            is_exhausted=end_of_source,
            is_limit_reached=stopped_by_size,
        )

    elapsed = time.monotonic() - start_time

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

    if end_of_source and bytes_done < quota:

        print(
            f"[WARN] {job.name}: "
            "source exhausted before target.",
            flush=True,
        )

    return result(end_of_source)


# ============================================================
# 19. Manifest
# ============================================================

def save_manifest(
    output_dir,
    group_quotas,
    job_quotas,
    results,
):

    total_budget = sum(
        group_quotas.values()
    )

    actual_target = sum(
        job_quotas.values()
    )

    reserved = total_budget - actual_target

    manifest = {
        "source": HF_ENDPOINT,
        "unit": "decimal GB",

        "description": (
            "MiniRain UltraData download plan: "
            "Python 60%, sh 20%, "
            "CMD 20% reserved in Code budget."
        ),

        "group_quotas": {
            k: v / GB
            for k, v in group_quotas.items()
        },

        "job_quotas": {
            k: v / GB
            for k, v in job_quotas.items()
        },

        "code_ratios": {
            "py": 0.60,
            "sh": 0.20,
            "cmd_reserved": 0.20,
        },

        "revisions": DATASET_REVISIONS,

        "results": results,

        "total_budget_bytes": total_budget,
        "total_target_bytes": actual_target,
        "reserved_bytes": reserved,

        "total_saved_bytes": sum(
            x["downloaded_bytes"]
            for x in results
        ),

        "updated_at": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }

    # Do not overwrite the original manifest.json.
    save_json_atomic(
        output_dir / "manifest_py_sh.json",
        manifest,
    )


# ============================================================
# 20. CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "MiniRain-Vision UltraData Downloader "
            "(Python + Shell)"
        )
    )

    parser.add_argument(
        "--total-gb",
        type=float,
        default=80.0,
        help="Total budget including reserved CMD.",
    )

    parser.add_argument(
        "--l2-gb",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--l3-gb",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--code-gb",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--math-gb",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output directory. Default: "
            "ultradata beside this script."
        ),
    )

    parser.add_argument(
        "--code-l2-ratio",
        type=float,
        default=0.50,
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
        "--dry-run",
        action="store_true",
        help="Show download plan without writing files.",
    )

    return parser.parse_args()


# ============================================================
# 21. Validation
# ============================================================

def validate_args(args):

    if args.save_every_rows <= 0:
        raise ValueError(
            "--save-every-rows must be positive."
        )

    if args.min_chars < 0:
        raise ValueError(
            "--min-chars cannot be negative."
        )

    for name, value in [
        ("l2_zh_ratio", args.l2_zh_ratio),
        ("l3_zh_ratio", args.l3_zh_ratio),
        ("l3_qa_ratio", args.l3_qa_ratio),
        ("code_l2_ratio", args.code_l2_ratio),
    ]:

        if not 0 <= value <= 1:
            raise ValueError(
                f"{name} must be between 0 and 1."
            )

    if abs(
        sum(CODE_LANG_WEIGHTS.values())
        + CMD_RESERVED_RATIO
        - 1.0
    ) > 1e-9:

        raise ValueError(
            "Code weights plus CMD reservation "
            "must sum to 1."
        )


# ============================================================
# 22. Main
# ============================================================

def main():

    args = parse_args()

    validate_args(args)

    # --------------------------------------------------------
    # Build download jobs
    # --------------------------------------------------------

    jobs = build_jobs(args)

    group_quotas = allocate_group_quotas(args)

    job_quotas = allocate_job_quotas(
        jobs,
        group_quotas,
        args,
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else DEFAULT_OUTPUT_DIR
    )

    if args.dry_run:

        if not output_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset directory not found: {output_dir}"
            )

    else:

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # Capacity
    # --------------------------------------------------------

    total_budget = sum(
        group_quotas.values()
    )

    actual_target = sum(
        job_quotas.values()
    )

    reserved = total_budget - actual_target

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("MiniRain-Vision UltraData Downloader")
    print("=" * 65)

    print(f"[MIRROR] {HF_ENDPOINT}")
    print("[AUTH] Public datasets, no HF token")
    print("[STREAM] Sequential reading")
    print(f"[OUTPUT] {output_dir}")
    print(f"[DRY RUN] {args.dry_run}")

    print("\n[REVISION]")

    for repo, sha in DATASET_REVISIONS.items():
        print(f"  {repo}: {sha}")

    print("\n[GROUP CAPACITY]")

    for group, quota in group_quotas.items():

        print(
            f"  {group:8s}: "
            f"{quota / GB:10.3f} GB"
        )

    print("\n[CODE PLAN]")

    code_bytes = group_quotas.get("code", 0)

    for lang, ratio in CODE_LANG_WEIGHTS.items():

        print(
            f"  {lang:8s}: "
            f"{code_bytes * ratio / GB:.3f} GB"
        )

    print(
        f"  {'CMD':8s}: "
        f"{code_bytes * CMD_RESERVED_RATIO / GB:.3f} GB"
        " (reserved)"
    )

    print("\n[TOTAL]")

    print(
        f"  Budget:   {total_budget / GB:.3f} GB"
    )

    print(
        f"  Download: {actual_target / GB:.3f} GB"
    )

    print(
        f"  Reserved: {reserved / GB:.3f} GB"
    )

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
                revision=DATASET_REVISIONS[job.repo],
                args=args,
            )

            if result is not None:
                results.append(result)

            if not args.dry_run:

                save_manifest(
                    output_dir,
                    group_quotas,
                    job_quotas,
                    results,
                )

    except KeyboardInterrupt:

        print(
            "\n[STOPPED] Checkpoint saved. "
            "Run again to resume.",
            flush=True,
        )

        return

    except Exception as exc:

        print(
            f"\n[FAILED] "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        print(
            "Existing checkpoint files are preserved.",
            flush=True,
        )

        raise

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    actual = sum(
        x["downloaded_bytes"]
        for x in results
    )

    print("\n" + "=" * 65)
    print("DOWNLOAD SUMMARY")
    print("=" * 65)

    for result in results:

        print(
            f"{result['name']:35s}"
            f"{result['downloaded_bytes'] / GB:9.3f} GB"
        )

    print("-" * 65)

    print(
        f"Budget:   {total_budget / GB:.3f} GB"
    )

    print(
        f"Target:   {actual_target / GB:.3f} GB"
    )

    print(
        f"Saved:    {actual / GB:.3f} GB"
    )

    print(
        f"Reserved: {reserved / GB:.3f} GB"
    )

    print(
        f"Output:   {output_dir}"
    )

    print("=" * 65)


# ============================================================
# 23. Entry point
# ============================================================

if __name__ == "__main__":
    main()
