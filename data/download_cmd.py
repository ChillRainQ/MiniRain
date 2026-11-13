
#!/usr/bin/env python3
"""
MiniRain Windows Code Dataset Downloader

Dataset:
    bigcode/the-stack

Downloads:
    1. Batchfile (CMD): ALL available records
    2. PowerShell: 1.6 GB by default

Features:
    - Hugging Face authentication
    - Streaming
    - Resume from checkpoints
    - Atomic state updates
    - Download progress and speed
    - No shuffle
    - Independent JSONL files
    - No fixed limit for Batchfile

Default directory:
    data/ultradata/
"""

# ============================================================
# 1. Environment (before HF imports)
# ============================================================

import os

# Gated BigCode data: use the official HF endpoint.
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

# Read HF_TOKEN from environment, or use the token
# saved by `hf auth login`.
# Never hardcode credentials in the script.


# ============================================================
# 2. Imports
# ============================================================

import argparse
import json
import time

from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset


# ============================================================
# 3. Configuration
# ============================================================

GB = 1_000_000_000

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "ultradata"

REPO = "bigcode/the-stack"

# Keep the repository revision fixed within each
# checkpoint. By default use main; existing checkpoints
# must match the supplied --revision.


@dataclass(frozen=True)
class Job:
    name: str
    language: str
    limit_bytes: int | None


# ============================================================
# 4. Atomic JSON
# ============================================================

def save_json_atomic(path, data):

    tmp = path.with_name(path.name + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)


def load_json(path):

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 5. State
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
    completed,
):

    return {
        **signature(job, revision),
        "rows": rows,
        "records": records,
        "bytes": bytes_done,
        "completed": completed,
        "updated_at": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }


# ============================================================
# 6. Record conversion
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

    record = {
        "text": content,
        "dataset": REPO,
        "language": job.language,
        "metadata": {
            key: row[key]
            for key in metadata_keys
            if row.get(key) is not None
        },
    }

    return (
        json.dumps(
            record,
            ensure_ascii=False,
            default=str,
        ) + "\n"
    ).encode("utf-8")


# ============================================================
# 7. Prepare existing download
# ============================================================

def prepare(job, output_dir, revision):

    data_path = output_dir / f"{job.name}.jsonl"

    state_path = (
        output_dir / f"{job.name}.state.json"
    )

    data_exists = data_path.exists()
    state_exists = state_path.exists()

    if data_exists != state_exists:

        raise RuntimeError(
            f"{job.name}: JSONL and state file mismatch. "
            "Refusing unsafe resume."
        )

    if not data_exists:

        state = make_state(
            job=job,
            revision=revision,
            rows=0,
            records=0,
            bytes_done=0,
            completed=False,
        )

        return data_path, state_path, state, False

    state = load_json(state_path)

    expected = signature(job, revision)

    for key, value in expected.items():

        if state.get(key) != value:

            raise RuntimeError(
                f"{job.name}: checkpoint mismatch: {key}\n"
                f"Current: {state.get(key)!r}\n"
                f"Expected: {value!r}\n"
                "Existing files were not modified."
            )

    for key in ("rows", "records", "bytes"):

        if key not in state:

            raise RuntimeError(
                f"{job.name}: missing state field {key}"
            )

        if int(state[key]) < 0:

            raise RuntimeError(
                f"{job.name}: invalid state field {key}"
            )

    actual_bytes = data_path.stat().st_size
    recorded_bytes = int(state["bytes"])

    if actual_bytes < recorded_bytes:

        raise RuntimeError(
            f"{job.name}: JSONL smaller than checkpoint."
        )

    return data_path, state_path, state, True


# ============================================================
# 8. Download
# ============================================================

def download(job, output_dir, revision, args):

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

    print()
    print("=" * 65)
    print(f"[JOB] {job.name}")
    print(f"[LANGUAGE] {job.language}")
    print(f"[TARGET] {target_text}")
    print(f"[EXISTING] {bytes_done / GB:.3f} GB")
    print(f"[ROWS] {rows:,}")
    print("=" * 65, flush=True)

    if job.limit_bytes is not None:
        if bytes_done > job.limit_bytes:
            raise RuntimeError(
                f"{job.name}: existing saved data "
                "exceeds target. Refusing to shrink."
            )

    if args.dry_run:

        if extra_bytes:
            print(
                f"[CHECK] {extra_bytes} extra bytes "
                "after checkpoint."
            )

        return {
            "name": job.name,
            "bytes": bytes_done,
            "rows": rows,
            "completed": completed,
        }

    if not exists:

        with data_path.open("xb"):
            pass

        save_json_atomic(
            state_path,
            state,
        )

        print(f"[NEW] {job.name}", flush=True)

    elif extra_bytes > 0:

        # Remove only the uncommitted bytes
        # written after the last checkpoint.
        with data_path.open("r+b") as f:
            f.truncate(bytes_done)

        print(
            f"[RECOVER] Removed {extra_bytes} "
            "uncommitted bytes.",
            flush=True,
        )

    if completed:

        print(
            f"[SKIP] {job.name}: completed.",
            flush=True,
        )

        return {
            "name": job.name,
            "bytes": bytes_done,
            "rows": rows,
            "completed": True,
        }

    if (
        job.limit_bytes is not None
        and bytes_done >= job.limit_bytes
    ):

        print(
            f"[SKIP] {job.name}: quota reached.",
            flush=True,
        )

        return {
            "name": job.name,
            "bytes": bytes_done,
            "rows": rows,
            "completed": True,
        }

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(
        f"[LOAD] {REPO} / data/{job.language}",
        flush=True,
    )

    ds = load_dataset(
        path=REPO,
        data_dir=f"data/{job.language}",
        split="train",
        streaming=True,
        revision=revision,
        token=True,
    )

    print(
        f"[LOAD] Features: {ds.features}",
        flush=True,
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    if rows > 0:

        print(
            f"[RESUME] Skipping {rows:,} source rows...",
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

    def checkpoint(f, is_completed=False):

        f.flush()
        os.fsync(f.fileno())

        save_json_atomic(
            state_path,
            make_state(
                job=job,
                revision=revision,
                rows=rows,
                records=records,
                bytes_done=bytes_done,
                completed=is_completed,
            ),
        )

    # --------------------------------------------------------
    # Stream
    # --------------------------------------------------------

    with data_path.open("r+b") as f:

        f.seek(bytes_done)

        try:

            print(
                "[STREAM] Waiting for first data...",
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

                raw = encode_record(row, job)

                if raw is not None:

                    # For capped datasets, stop before
                    # exceeding the maximum byte size.
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
                # Checkpoint
                # --------------------------------------------

                if rows % args.save_every_rows == 0:
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
                        (bytes_done - start_bytes)
                        / elapsed
                        / (1024 ** 2)
                    )

                    print(
                        f"[PROGRESS] {job.name} "
                        f"| {bytes_done / GB:.3f} GB "
                        f"| rows={rows:,} "
                        f"| records={records:,} "
                        f"| speed={speed:.2f} MiB/s",
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

                source_exhausted = True

        except BaseException:

            print(
                "[INTERRUPTED] Saving checkpoint...",
                flush=True,
            )

            checkpoint(f, is_completed=False)
            raise

        checkpoint(
            f,
            is_completed=(
                source_exhausted or quota_reached
            ),
        )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    elapsed = time.monotonic() - start_time

    print()
    print(f"[DONE] {job.name}")
    print(f"  Saved: {bytes_done / GB:.3f} GB")
    print(f"  Records: {records:,}")
    print(f"  Source rows: {rows:,}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(
        f"  Source exhausted: {source_exhausted}",
        flush=True,
    )

    return {
        "name": job.name,
        "bytes": bytes_done,
        "rows": rows,
        "records": records,
        "completed": (
            source_exhausted or quota_reached
        ),
        "source_exhausted": source_exhausted,
    }


# ============================================================
# 9. CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Download Windows Batchfile and "
            "PowerShell from BigCode The Stack."
        )
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--powershell-gb",
        type=float,
        default=1.6,
        help=(
            "Maximum PowerShell JSONL size. "
            "Use 0 for unlimited."
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
# 10. Main
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

    output_dir = args.out.expanduser().resolve()

    # Dry run must not create directories.
    if args.dry_run:

        if not output_dir.is_dir():

            raise FileNotFoundError(
                f"Directory not found: {output_dir}"
            )

    else:

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # CMD: unlimited
    # PowerShell: 1.6GB by default

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

    print("=" * 65)
    print("MiniRain Windows Code Downloader")
    print("=" * 65)
    print(f"[REPO] {REPO}")
    print(f"[ENDPOINT] {os.environ['HF_ENDPOINT']}")
    print(f"[OUTPUT] {output_dir}")
    print("[CMD] Full dataset")
    print(
        "[PowerShell] "
        + (
            "Full dataset"
            if powershell_limit is None
            else f"{args.powershell_gb:.3f} GB"
        )
    )
    print(f"[DRY RUN] {args.dry_run}")
    print("=" * 65)

    results = []

    try:

        for job in jobs:

            result = download(
                job,
                output_dir,
                args.revision,
                args,
            )

            results.append(result)

            if not args.dry_run:

                save_json_atomic(
                    output_dir / "manifest_windows.json",
                    {
                        "dataset": REPO,
                        "revision": args.revision,
                        "results": results,
                        "updated_at": time.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    },
                )

    except KeyboardInterrupt:

        print(
            "\n[STOPPED] Checkpoints saved. "
            "Run again to resume."
        )

        return

    print()
    print("=" * 65)
    print("WINDOWS DATA DOWNLOAD SUMMARY")
    print("=" * 65)

    for result in results:

        print(
            f"{result['name']:25s}"
            f"{result['bytes'] / GB:9.3f} GB"
        )

    print("=" * 65)


if __name__ == "__main__":
    main()
