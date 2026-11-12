
"""
MiniRain-Vision UltraData Downloader
====================================

Datasets:
    openbmb/Ultra-FineWeb
    openbmb/Ultra-FineWeb-L3
    openbmb/UltraData-Code
    openbmb/UltraData-Math

Features:
    - Hardcoded Hugging Face token
    - Authentication validation
    - Streaming download
    - GB capacity allocation
    - Resume support
    - Immutable dataset revisions
    - JSONL output
    - Automatic orphan file recovery
    - Script-relative output directory
"""

# ============================================================
# 1. Token configuration
# ============================================================

import os

# Replace with your own Hugging Face read token.
HF_TOKEN = "xxx"

if HF_TOKEN and not HF_TOKEN.startswith("hf_"):
    raise ValueError("Invalid HF_TOKEN format")

# Set environment variables before importing HF libraries.
if HF_TOKEN and "REPLACE_WITH_YOUR_TOKEN" not in HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN


# ============================================================
# 2. Imports
# ============================================================

import json
import time
import argparse
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi


# ============================================================
# 3. Global configuration
# ============================================================

GB = 1_000_000_000

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "ultradata"

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
# 5. Authentication
# ============================================================

def authenticate():

    token = HF_TOKEN.strip()

    if (
        not token
        or not token.startswith("hf_")
        or "REPLACE_WITH_YOUR_TOKEN" in token
    ):
        raise RuntimeError(
            "Please configure your HF_TOKEN at the top of this script."
        )

    print("\n[AUTH] Checking Hugging Face authentication...")

    api = HfApi(token=token)

    try:
        user = api.whoami()
    except Exception as exc:
        raise RuntimeError(
            f"Hugging Face authentication failed: {exc}"
        ) from exc

    username = user.get("name", "unknown")

    print(f"[AUTH] Authenticated as: {username}")
    print("[AUTH] Explicit token authentication enabled.")

    return token, api


# ============================================================
# 6. Build jobs
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
    # UltraData Code L2 / L3
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
        CODE_LANG_WEIGHTS[x] for x in languages
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
    # UltraData Math L3
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
# 7. Capacity allocation
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
                amount = int(total * weight / weight_sum)

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

        if group_bytes <= 0 or not group_jobs:
            continue

        weight_sum = sum(
            job.weight for job in group_jobs
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
# 8. Resolve immutable dataset revisions
# ============================================================

def resolve_revisions(jobs, api, requested_revision):

    revisions = {}

    repositories = sorted({
        job.repo for job in jobs
    })

    print("\n[REVISION] Resolving dataset versions")

    for repo in repositories:

        info = api.dataset_info(
            repo_id=repo,
            revision=requested_revision,
        )

        revisions[repo] = info.sha

        print(f"  {repo}: {info.sha}")

    return revisions


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

    # Code L3: prefer full_content.
    if (
        job.group == "code"
        and job.config == "UltraData-Code-L3"
    ):

        for key in ("full_content", "content", "raw_content"):

            value = row.get(key)

            if value is not None:

                text = value_to_text(value)

                if text.strip():
                    return text

        # Structured fallback.
        parts = []

        for key in ("task", "analysis", "solution", "test"):

            value = row.get(key)

            if value is not None:

                text = value_to_text(value)

                if text.strip():
                    parts.append(
                        f"## {key.upper()}\n{text}"
                    )

        return "\n\n".join(parts)

    # FineWeb / Math / Code L2.
    for key in (
        "content",
        "text",
        "full_content",
        "raw_content",
        "body",
        "messages",
        "conversations",
        "conversation",
    ):

        value = row.get(key)

        if value is not None:

            text = value_to_text(value)

            if text.strip():
                return text

    # Generic structured fallback.
    parts = []

    for key in (
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
                parts.append(f"{key}:\n{text}")

    return "\n\n".join(parts)


# ============================================================
# 10. Metadata extraction
# ============================================================

def extract_metadata(row, job):

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
# 11. Build JSONL record
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
        "metadata": extract_metadata(row, job),
    }


# ============================================================
# 12. Atomic state saving
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


def make_signature(job, revision, args):

    return {
        "repo": job.repo,
        "config": job.config,
        "split": job.split,
        "revision": revision,
        "seed": args.seed,
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
# 13. Orphan file recovery
# ============================================================

def recover_orphan_files(output_path, state_path):

    if output_path.exists() == state_path.exists():
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    print(
        f"\n[RECOVERY] Incomplete file pair detected: "
        f"{output_path.stem}"
    )

    if output_path.exists():

        backup = output_path.with_name(
            f"{output_path.stem}.orphan_{timestamp}.jsonl"
        )

        os.replace(output_path, backup)

        print(f"[RECOVERY] Orphan data backed up: {backup.name}")

    if state_path.exists():

        backup = state_path.with_name(
            f"{state_path.stem}.orphan_{timestamp}.json"
        )

        os.replace(state_path, backup)

        print(f"[RECOVERY] Orphan state backed up: {backup.name}")


# ============================================================
# 14. Download one dataset split
# ============================================================

def download_job(
    job,
    quota,
    output_dir,
    revision,
    args,
    token,
):

    output_path = output_dir / f"{job.name}.jsonl"
    state_path = output_dir / f"{job.name}.state.json"

    if quota <= 0:
        return None

    signature = make_signature(
        job,
        revision,
        args,
    )

    # --------------------------------------------------------
    # Overwrite
    # --------------------------------------------------------

    if args.overwrite:

        output_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)

    # --------------------------------------------------------
    # Recover orphan files
    # --------------------------------------------------------

    recover_orphan_files(
        output_path,
        state_path,
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    source_rows = 0
    records = 0
    bytes_done = 0
    exhausted = False
    limit_reached = False
    old_target = quota

    if output_path.exists():

        state = load_state(state_path)

        for key, value in signature.items():

            if state.get(key) != value:

                raise RuntimeError(
                    f"{job.name}: resume setting changed: {key}. "
                    "Use original settings or --overwrite."
                )

        source_rows = int(state["source_rows"])
        records = int(state["records"])
        bytes_done = int(state["bytes"])
        exhausted = bool(state.get("exhausted", False))
        limit_reached = bool(state.get("limit_reached", False))
        old_target = int(state.get("target_bytes", quota))

        actual_size = output_path.stat().st_size

        if actual_size < bytes_done:

            raise RuntimeError(
                f"{job.name}: output file is smaller than state."
            )

        if bytes_done > quota:

            raise RuntimeError(
                f"{job.name}: existing output exceeds new quota. "
                "Increase quota or use --overwrite."
            )

        # Discard data written after the last checkpoint.
        with open(output_path, "r+b") as f:
            f.truncate(bytes_done)

        print(
            f"[RESUME] {job.name}: "
            f"{bytes_done / GB:.3f} GB, "
            f"{records:,} records"
        )

        # Same target, previously stopped because next row
        # would exceed remaining space.
        if limit_reached and old_target == quota:

            print(f"[SKIP] {job.name}: target already reached")
            return {
                "name": job.name,
                "target_bytes": quota,
                "downloaded_bytes": bytes_done,
                "records": records,
                "source_rows": source_rows,
                "exhausted": exhausted,
            }

    else:

        # Create both files and immediately save initial state.
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

    if exhausted or bytes_done >= quota:

        print(f"[SKIP] {job.name}: completed")

        return {
            "name": job.name,
            "target_bytes": quota,
            "downloaded_bytes": bytes_done,
            "records": records,
            "source_rows": source_rows,
            "exhausted": exhausted,
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

    print("[STREAM] Calling load_dataset...", flush=True)

    ds = load_dataset(
        path=job.repo,
        name=job.config,
        split=job.split,
        streaming=True,
        revision=revision,
        token=token,
    )

    print("[STREAM] load_dataset returned.", flush=True)
    print("[STREAM] Features:", ds.features, flush=True)

    # 下载阶段不进行 Shuffle，顺序读取数据
    print("[STREAM] Sequential streaming enabled.", flush=True)

    if source_rows:
        print(
            f"[STREAM] Skipping {source_rows:,} rows...",
            flush=True,
        )
        ds = ds.skip(source_rows)

    print("[STREAM] Waiting for first sample...", flush=True)

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    mode = "r+b" if output_path.exists() else "w+b"

    start_time = time.time()
    start_bytes = bytes_done
    last_report = start_time

    stopped_by_size = False
    first_sample = True
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

    with open(output_path, mode) as f:

        f.truncate(bytes_done)
        f.seek(bytes_done)

        try:

            for row in ds:

                if first_sample:

                    print(
                        "[STREAM] First sample received. "
                        f"Keys: {list(row.keys())}",
                        flush=True,
                    )

                    first_sample = False

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

                    # Skip an individual record larger than
                    # the entire target capacity.
                    if bytes_done == 0 and line_size > quota:

                        source_rows += 1
                        skipped_oversize += 1
                        continue

                    # Do not consume the row when it does
                    # not fit. It will be retried if quota grows.
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

                # Checkpoint
                if (
                    source_rows > 0
                    and source_rows % args.save_every_rows == 0
                ):

                    persist(f)

                # Progress
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
                        f"\r  "
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

            print("\n[INTERRUPTED] Saving checkpoint...")

            persist(f)
            raise

        except Exception:

            print("\n[ERROR] Saving checkpoint...")

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
        f"  Time: {elapsed:.1f}s"
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
# 15. Manifest
# ============================================================

def save_manifest(
    output_dir,
    group_quotas,
    results,
    revisions,
):

    manifest = {
        "unit": "decimal GB",
        "group_quotas": {
            k: v / GB
            for k, v in group_quotas.items()
        },
        "revisions": revisions,
        "results": results,
        "total_target_bytes": sum(
            group_quotas.values()
        ),
        "total_saved_bytes": sum(
            x["downloaded_bytes"]
            for x in results
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
# 16. CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="MiniRain-Vision UltraData Downloader"
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
        help="Output directory; default is script_dir/data/ultradata",
    )

    parser.add_argument(
        "--revision",
        type=str,
        default="main",
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
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=1000,
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
# 17. Main
# ============================================================

def main():

    args = parse_args()

    if args.save_every_rows <= 0:
        raise ValueError("--save-every-rows must be positive")

    if args.shuffle_buffer <= 0:
        raise ValueError("--shuffle-buffer must be positive")

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
    # Authentication FIRST
    # --------------------------------------------------------

    token, api = authenticate()

    # --------------------------------------------------------
    # Build jobs
    # --------------------------------------------------------

    jobs = build_jobs(args)

    group_quotas = allocate_group_quotas(args)

    job_quotas = allocate_job_quotas(
        jobs,
        group_quotas,
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = (
        Path(args.out).resolve()
        if args.out
        else DEFAULT_OUTPUT_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Resolve revisions
    # --------------------------------------------------------

    revisions = resolve_revisions(
        jobs,
        api,
        args.revision,
    )

    # --------------------------------------------------------
    # Print plan
    # --------------------------------------------------------

    total_target = sum(group_quotas.values())

    print("\n" + "=" * 65)
    print("MiniRain-Vision UltraData Downloader")
    print("=" * 65)

    for group, quota in group_quotas.items():

        print(
            f"{group:8s}: {quota / GB:10.3f} GB"
        )

    print("-" * 65)
    print(f"{'TOTAL':8s}: {total_target / GB:10.3f} GB")

    print(f"Output: {output_dir}")
    print(f"Seed: {args.seed}")
    print(f"Shuffle buffer: {args.shuffle_buffer}")
    print(f"Authentication: verified")
    print("=" * 65)

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    results = []

    try:

        for job in jobs:

            quota = job_quotas.get(job.name, 0)

            if quota <= 0:
                continue

            result = download_job(
                job=job,
                quota=quota,
                output_dir=output_dir,
                revision=revisions[job.repo],
                args=args,
                token=token,
            )

            if result is not None:
                results.append(result)

            save_manifest(
                output_dir,
                group_quotas,
                results,
                revisions,
            )

    except KeyboardInterrupt:

        print("\n[STOPPED] Checkpoints saved.")
        print("Run the same command again to resume.")
        return

    except Exception as exc:

        print(
            f"\n[FAILED] {type(exc).__name__}: {exc}"
        )

        print(
            "Existing data and checkpoints have been preserved."
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
    print(f"Target: {total_target / GB:.3f} GB")
    print(f"Saved:  {actual / GB:.3f} GB")
    print(f"Output: {output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
