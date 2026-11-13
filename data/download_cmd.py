
#!/usr/bin/env python3
"""
MiniRain Windows CMD / PowerShell Downloader

Dataset:
    bigcode/the-stack

Endpoint:
    https://hf-mirror.com

Download:
    Batchfile   : ALL records (unlimited)
    PowerShell  : 1.6 GB by default

Output:
    data/ultradata/windows_batchfile.jsonl
    data/ultradata/windows_powershell.jsonl

Features:
    - HF-Mirror
    - Hugging Face token authentication
    - Streaming
    - Checkpoint resume
    - Atomic state saving
    - Progress tracking
    - No shuffle
    - Safe quota expansion
    - Dry run

Requires:
    pip install -U datasets huggingface_hub pyarrow
"""

# ============================================================
# 1. Environment
# ============================================================

import os

# Must be configured before importing HF libraries.
HF_ENDPOINT = "https://hf-mirror.com"

os.environ["HF_ENDPOINT"] = HF_ENDPOINT
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")


# ============================================================
# 2. Imports
# ============================================================

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import get_token


# ============================================================
# 3. Global configuration
# ============================================================

GB = 1_000_000_000

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "ultradata"

REPO = "bigcode/the-stack"

# Optional direct token.
# Prefer HF_TOKEN environment variable or hf auth login.
#
# Do not commit a real token to Git.
HF_TOKEN = None


# ============================================================
# 4. Job definition
# ============================================================

@dataclass(frozen=True)
class Job:
    name: str
    language: str
    limit_bytes: int | None


# ============================================================
# 5. Authentication
# ============================================================

def get_auth_token():

    token = (
        HF_TOKEN
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or get_token()
    )

    if not token:
        raise RuntimeError(
            "Hugging Face token not found.\n"
            "Set HF_TOKEN or run 'hf auth login'.\n"
            "The Stack requires approved access."
        )

    print(
        "[AUTH] Hugging Face token found.",
        flush=True,
    )

    print(
        "[AUTH] Using token with HF-Mirror.",
        flush=True,
    )

    return token


# ============================================================
# 6. Atomic state saving
# ============================================================

def atomic_json(path, data):

    temp_path = path.with_name(
        path.name + ".tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_path, path)


def read_json(path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# 7. Checkpoint signature
# ============================================================

def signature(job, revision):

    return {
        "repo": REPO,
        "language": job.language,
        "data_dir": f"data/{job.language}",
        "split": "train",
        "revision": revision,
        "stream_order": "sequential-v1",
    }


def make_state(
    job,
    revision,
    rows,
    records,
    bytes_done,
    completed=False,
    completion_reason=None,
):

    return {
        **signature(job, revision),

        "rows": rows,
        "records": records,
        "bytes": bytes_done,

        "completed": completed,
        "completion_reason": completion_reason,

        "updated_at": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }


# ============================================================
# 8. Text extraction and record encoding
# ============================================================

def encode_record(row, job):

    content = row.get("content")

    if not isinstance(content, str):
        return None

    if not content.strip():
        return None

    metadata_keys = (
        "repository_name",
        "path",
        "licenses",
        "hexsha",
        "ext",
        "lang",
        "size",
        "avg_line_length",
        "max_line_length",
        "alphanum_fraction",
    )

    metadata = {
        key: row[key]
        for key in metadata_keys
        if row.get(key) is not None
    }

    record = {
        "text": content,
        "dataset": REPO,
        "language": job.language,
        "metadata": metadata,
    }

    line = json.dumps(
        record,
        ensure_ascii=False,
        default=str,
    ) + "\n"

    return line.encode("utf-8")


# ============================================================
# 9. Prepare existing files
# ============================================================

def prepare(job, output_dir, revision):

    data_path = (
        output_dir / f"{job.name}.jsonl"
    )

    state_path = (
        output_dir / f"{job.name}.state.json"
    )

    data_exists = data_path.exists()
    state_exists = state_path.exists()

    if data_exists != state_exists:

        raise RuntimeError(
            f"{job.name}: JSONL/state mismatch.\n"
            "Both files are required for resuming.\n"
            "Existing files will not be overwritten."
        )

    if not data_exists:

        state = make_state(
            job=job,
            revision=revision,
            rows=0,
            records=0,
            bytes_done=0,
        )

        return (
            data_path,
            state_path,
            state,
            False,
        )

    state = read_json(state_path)

    expected = signature(
        job,
        revision,
    )

    for key, expected_value in expected.items():

        actual_value = state.get(key)

        if actual_value != expected_value:

            raise RuntimeError(
                f"{job.name}: checkpoint mismatch.\n"
                f"Field: {key}\n"
                f"Actual: {actual_value!r}\n"
                f"Expected: {expected_value!r}\n"
                "Refusing unsafe resume."
            )

    for key in (
        "rows",
        "records",
        "bytes",
    ):

        if key not in state:
            raise RuntimeError(
                f"{job.name}: missing state field: {key}"
            )

        if int(state[key]) < 0:
            raise RuntimeError(
                f"{job.name}: invalid state field: {key}"
            )

    actual_size = data_path.stat().st_size
    saved_size = int(state["bytes"])

    if actual_size < saved_size:

        raise RuntimeError(
            f"{job.name}: JSONL shorter than checkpoint."
        )

    return (
        data_path,
        state_path,
        state,
        True,
    )


# ============================================================
# 10. Download one job
# ============================================================

def download_job(
    job,
    output_dir,
    revision,
    token,
    args,
):

    (
        data_path,
        state_path,
        state,
        exists,
    ) = prepare(
        job,
        output_dir,
        revision,
    )

    rows = int(state["rows"])
    records = int(state["records"])
    bytes_done = int(state["bytes"])

    completed = bool(
        state.get("completed", False)
    )

    completion_reason = state.get(
        "completion_reason"
    )

    actual_size = (
        data_path.stat().st_size
        if exists else 0
    )

    extra_bytes = actual_size - bytes_done

    target_text = (
        "ALL"
        if job.limit_bytes is None
        else f"{job.limit_bytes / GB:.3f} GB"
    )

    print("\n" + "=" * 65)
    print(f"[JOB] {job.name}")
    print(f"[LANGUAGE] {job.language}")
    print(f"[TARGET] {target_text}")
    print(f"[EXISTING] {bytes_done / GB:.3f} GB")
    print(f"[ROWS] {rows:,}")
    print("=" * 65, flush=True)

    # Never shrink existing data.
    if (
        job.limit_bytes is not None
        and bytes_done > job.limit_bytes
    ):

        raise RuntimeError(
            f"{job.name}: existing data exceeds "
            "the new target. Refusing to shrink."
        )

    def result(reason=None):

        return {
            "name": job.name,
            "language": job.language,
            "target_bytes": job.limit_bytes,
            "bytes": bytes_done,
            "rows": rows,
            "records": records,
            "completion_reason": reason,
        }

    # --------------------------------------------------------
    # Dry run
    # --------------------------------------------------------

    if args.dry_run:

        if extra_bytes > 0:
            print(
                f"[CHECK] {extra_bytes} bytes beyond "
                "the last checkpoint."
            )

        print(
            f"[CHECK] Completion reason: "
            f"{completion_reason}"
        )

        return result(completion_reason)

    # --------------------------------------------------------
    # Initialize files
    # --------------------------------------------------------

    if not exists:

        with data_path.open("xb"):
            pass

        atomic_json(
            state_path,
            state,
        )

        print(
            f"[NEW] {job.name}",
            flush=True,
        )

    elif extra_bytes > 0:

        # Only remove bytes beyond the committed checkpoint.
        with data_path.open("r+b") as f:
            f.truncate(bytes_done)

        print(
            f"[RECOVERY] Truncated "
            f"{extra_bytes} uncommitted bytes.",
            flush=True,
        )

    # --------------------------------------------------------
    # Completion checks
    # --------------------------------------------------------

    # A fully exhausted dataset cannot be extended.
    if completion_reason == "source_exhausted":

        print(
            f"[SKIP] {job.name}: source exhausted.",
            flush=True,
        )

        return result(completion_reason)

    # Existing data is at the current target.
    if (
        job.limit_bytes is not None
        and bytes_done >= job.limit_bytes
    ):

        print(
            f"[SKIP] {job.name}: quota reached.",
            flush=True,
        )

        return result("quota_reached")

    # A completed quota may be expanded later.
    if completion_reason == "quota_reached":

        print(
            f"[RESUME] Expanding previous quota.",
            flush=True,
        )

    # Backwards compatibility with the earlier script:
    # old states only had a "completed" boolean.
    if (
        completed
        and completion_reason is None
    ):

        # For a capped job, the old completed flag may
        # mean that its quota was reached.
        # Allow continuing when the new quota is larger.
        if job.limit_bytes is not None:

            print(
                "[RESUME] Legacy completed state; "
                "checking for additional quota.",
                flush=True,
            )

        else:

            raise RuntimeError(
                f"{job.name}: old state marks completed "
                "without recording the reason.\n"
                "Cannot safely distinguish quota "
                "completion from source exhaustion."
            )

    # --------------------------------------------------------
    # Connect to HF-Mirror
    # --------------------------------------------------------

    print(
        f"[MIRROR] {HF_ENDPOINT}",
        flush=True,
    )

    print(
        f"[LOAD] {REPO}/data/{job.language}",
        flush=True,
    )

    print(
        "[STREAM] Calling load_dataset...",
        flush=True,
    )

    ds = load_dataset(
        path=REPO,
        data_dir=f"data/{job.language}",
        split="train",
        streaming=True,
        revision=revision,
        token=token,
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
    # Sequential resume
    # --------------------------------------------------------

    if rows > 0:

        print(
            f"[RESUME] Skipping {rows:,} "
            "source rows...",
            flush=True,
        )

        ds = ds.skip(rows)

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    start_time = time.monotonic()
    start_bytes = bytes_done
    last_report = start_time

    source_exhausted = False
    quota_reached = False

    # --------------------------------------------------------
    # Checkpoint function
    # --------------------------------------------------------

    def checkpoint(
        file,
        reason=None,
    ):

        file.flush()
        os.fsync(file.fileno())

        atomic_json(
            state_path,
            make_state(
                job=job,
                revision=revision,
                rows=rows,
                records=records,
                bytes_done=bytes_done,
                completed=(reason is not None),
                completion_reason=reason,
            ),
        )

    # --------------------------------------------------------
    # Stream data
    # --------------------------------------------------------

    with data_path.open("r+b") as f:

        f.seek(bytes_done)

        try:

            print(
                "[STREAM] Waiting for first sample...",
                flush=True,
            )

            first_received = False

            for row in ds:

                if not first_received:

                    print(
                        "[STREAM] First sample received.",
                        flush=True,
                    )

                    print(
                        "[STREAM] Keys:",
                        list(row.keys()),
                        flush=True,
                    )

                    first_received = True

                raw = encode_record(
                    row,
                    job,
                )

                if raw is not None:

                    # Stop before exceeding a capped quota.
                    # Do not count the uncommitted row.
                    if (
                        job.limit_bytes is not None
                        and bytes_done + len(raw)
                        > job.limit_bytes
                    ):

                        quota_reached = True
                        break

                    f.write(raw)

                    bytes_done += len(raw)
                    records += 1

                rows += 1

                # --------------------------------------------
                # Periodic checkpoint
                # --------------------------------------------

                if (
                    rows > 0
                    and rows % args.save_every_rows == 0
                ):

                    checkpoint(f)

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
                        f"[PROGRESS] {job.name}"
                        f" | {bytes_done / GB:.3f} GB"
                        f" | rows={rows:,}"
                        f" | records={records:,}"
                        f" | speed={speed:.2f} MiB/s",
                        flush=True,
                    )

                    last_report = now

                if (
                    job.limit_bytes is not None
                    and bytes_done >= job.limit_bytes
                ):

                    quota_reached = True
                    break

            else:

                # Loop ended naturally.
                source_exhausted = True

        except BaseException:

            print(
                "\n[INTERRUPTED] Saving checkpoint...",
                flush=True,
            )

            checkpoint(f)
            raise

        # ----------------------------------------------------
        # Final checkpoint
        # ----------------------------------------------------

        if source_exhausted:
            reason = "source_exhausted"

        elif quota_reached:
            reason = "quota_reached"

        else:
            reason = None

        checkpoint(
            f,
            reason=reason,
        )

    # --------------------------------------------------------
    # Summary for this job
    # --------------------------------------------------------

    elapsed = time.monotonic() - start_time

    print("\n" + "-" * 65)
    print(f"[DONE] {job.name}")
    print(f"  Saved: {bytes_done / GB:.3f} GB")
    print(f"  Records: {records:,}")
    print(f"  Source rows: {rows:,}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Completion: {reason}")
    print("-" * 65, flush=True)

    return result(reason)


# ============================================================
# 11. CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Download Windows Batchfile and "
            "PowerShell from BigCode using HF-Mirror."
        )
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Default: data/ultradata/",
    )

    parser.add_argument(
        "--powershell-gb",
        type=float,
        default=1.6,
        help=(
            "PowerShell output size in decimal GB. "
            "Use 0 to download all records."
        ),
    )

    parser.add_argument(
        "--revision",
        type=str,
        default="main",
    )

    parser.add_argument(
        "--save-every-rows",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# 12. Manifest
# ============================================================

def write_manifest(
    output_dir,
    revision,
    results,
):

    manifest = {
        "dataset": REPO,
        "endpoint": HF_ENDPOINT,
        "revision": revision,
        "unit": "decimal GB",
        "results": results,

        "total_saved_bytes": sum(
            item["bytes"]
            for item in results
        ),

        "updated_at": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }

    atomic_json(
        output_dir / "manifest_windows.json",
        manifest,
    )


# ============================================================
# 13. Main
# ============================================================

def main():

    args = parse_args()

    if args.powershell_gb < 0:
        raise ValueError(
            "--powershell-gb cannot be negative."
        )

    if args.save_every_rows <= 0:
        raise ValueError(
            "--save-every-rows must be positive."
        )

    output_dir = (
        args.out.expanduser().resolve()
    )

    if args.dry_run:

        if not output_dir.is_dir():
            raise FileNotFoundError(
                f"Directory not found: {output_dir}"
            )

        token = None

    else:

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        token = get_auth_token()

    # --------------------------------------------------------
    # Jobs
    # --------------------------------------------------------

    powershell_limit = (
        None
        if args.powershell_gb == 0
        else round(args.powershell_gb * GB)
    )

    jobs = [
        Job(
            name="windows_batchfile",
            language="batchfile",
            limit_bytes=None,
        ),
        Job(
            name="windows_powershell",
            language="powershell",
            limit_bytes=powershell_limit,
        ),
    ]

    # --------------------------------------------------------
    # Configuration summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("MiniRain Windows Dataset Downloader")
    print("=" * 65)

    print(f"[DATASET] {REPO}")
    print(f"[MIRROR] {HF_ENDPOINT}")
    print(f"[OUTPUT] {output_dir}")
    print(f"[REVISION] {args.revision}")

    print("[CMD] Unlimited / full dataset")

    if powershell_limit is None:
        print("[PowerShell] Unlimited / full dataset")
    else:
        print(
            f"[PowerShell] {args.powershell_gb:.3f} GB"
        )

    print(f"[DRY RUN] {args.dry_run}")
    print("=" * 65)

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    results = []

    try:

        for job in jobs:

            result = download_job(
                job=job,
                output_dir=output_dir,
                revision=args.revision,
                token=token,
                args=args,
            )

            results.append(result)

            if not args.dry_run:

                write_manifest(
                    output_dir,
                    args.revision,
                    results,
                )

    except KeyboardInterrupt:

        print(
            "\n[STOPPED] Checkpoints saved. "
            "Run again to resume.",
            flush=True,
        )

        return

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("WINDOWS DOWNLOAD SUMMARY")
    print("=" * 65)

    for result in results:

        print(
            f"{result['name']:25s}"
            f"{result['bytes'] / GB:9.3f} GB"
        )

    print("-" * 65)

    total = sum(
        result["bytes"]
        for result in results
    )

    print(
        f"Total saved: {total / GB:.3f} GB"
    )

    print(f"Output: {output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
