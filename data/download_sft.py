"""Stream UltraData SFT JSONL from HF Mirror under a local byte budget."""

import argparse
from decimal import Decimal, InvalidOperation
from getpass import GetPassWarning, getpass
import json
import os
from pathlib import Path, PurePosixPath
import sys
from tempfile import TemporaryDirectory
import warnings


GB = 1_000_000_000
MIRROR = "https://hf-mirror.com"
OUTPUT = Path(__file__).resolve().parent / "sft"
REPOS = {
    "2605": "openbmb/UltraData-SFT-2605",
    "2609": "openbmb/UltraData-SFT-Agent-2609",
}
CORE = {
    "no_think/Chinese-general": 40,
    "no_think/IF": 20,
    "no_think/Knowledge": 10,
    "no_think/Code": 3,
    "no_think/Math": 2,
    "think/Chinese-general": 3,
    "think/Knowledge": 2,
    "think/Code": 7,
    "think/Math": 13,
}
AGENT = ("Tool_Use", "Search_Agent", "General_Agent")


def budget_bytes(value):
    try:
        amount = Decimal(value)
        if amount.is_finite() and amount * GB >= 1:
            return int(amount * GB)
    except InvalidOperation:
        pass
    raise argparse.ArgumentTypeError("--gb must be a positive number (1 GB = 1,000,000,000 bytes)")


def allocate(state, budget, capacities):
    if state == "2609":
        quotas = {}
        for group in AGENT:
            quotas[group] = min(budget, capacities[group])
            budget -= quotas[group]
        return quotas

    quotas = dict.fromkeys(CORE, 0)
    while budget:
        active = [group for group in CORE if quotas[group] < capacities[group]]
        if not active:
            break
        weight = sum(CORE[group] for group in active)
        shares = {group: budget * CORE[group] // weight for group in active}
        shares[active[0]] += budget - sum(shares.values())
        for group, share in shares.items():
            added = min(share, capacities[group] - quotas[group])
            quotas[group] += added
            budget -= added
    return quotas


def write_group(session, shard_urls, output, budget):
    size = rows = 0
    with output.open("wb") as handle:
        for url in shard_urls:
            with session.get(url, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                for line in response.iter_lines(chunk_size=64 * 1024):
                    if not line.strip():
                        continue
                    if not line.lstrip().startswith(b"{"):
                        raise ValueError(f"Expected JSONL objects from {url}")
                    line += b"\n"
                    if len(line) > budget - size:
                        return size, rows
                    handle.write(line)
                    size += len(line)
                    rows += 1
                    if size == budget:
                        return size, rows
    return size, rows


def download(state, budget, target, token):
    from huggingface_hub import HfApi, hf_hub_url
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    repo = REPOS[state]
    api = HfApi(endpoint=MIRROR, token=token or False)
    revision = api.dataset_info(repo).sha
    groups = CORE if state == "2605" else AGENT
    shards = {group: [] for group in groups}
    for entry in api.list_repo_tree(repo, path_in_repo="data", recursive=True,
                                    repo_type="dataset", revision=revision):
        path = PurePosixPath(entry.path)
        if path.suffix == ".jsonl" and (group := path.parent.relative_to("data").as_posix()) in shards:
            shards[group].append((entry.path, entry.size))
    missing = [group for group, files in shards.items() if not files]
    if missing:
        raise RuntimeError(f"Missing JSONL groups on {MIRROR}: {', '.join(missing)}")
    capacities = {group: sum(size for _, size in files) for group, files in shards.items()}
    quotas = allocate(state, budget, capacities)

    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{state}-", dir=target.parent) as temporary, requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504),
        )))
        if token:
            session.headers["Authorization"] = f"Bearer {token}"
        staging = Path(temporary) / "output"
        staging.mkdir()
        manifest = dict(dataset=repo, revision=revision, endpoint=MIRROR, budget_bytes=budget, files=[])
        for group, quota in quotas.items():
            if quota == 0:
                continue
            urls = (hf_hub_url(repo, path, repo_type="dataset", revision=revision, endpoint=MIRROR)
                    for path, _ in sorted(shards[group]))
            relative_path = Path(f"{group}.jsonl")
            output = staging / relative_path
            output.parent.mkdir(parents=True, exist_ok=True)
            print(f"[Download] {group} up to {quota / GB:.3f} GB", flush=True)
            size, rows = write_group(session, urls, output, quota)
            manifest["files"].append(dict(path=relative_path.as_posix(), bytes=size, rows=rows))
            print(f"[Saved] {rows:,} rows, {size:,} bytes", flush=True)
        manifest["bytes"] = sum(entry["bytes"] for entry in manifest["files"])
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(target)
    print(f"[Done] {manifest['bytes'] / GB:.3f} GB -> {target}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, choices=REPOS)
    parser.add_argument("--gb", required=True, type=budget_bytes, help="maximum local JSONL size in decimal GB")
    args = parser.parse_args()
    target = OUTPUT / args.state
    if target.exists():
        parser.exit(1, f"Output already exists: {target}\n")
    token = None
    if args.state == "2605":
        token = os.environ.get("HF_MIRROR_TOKEN", "").strip()
        if not token and sys.stdin.isatty():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", GetPassWarning)
                    token = getpass("Hugging Face token for hf-mirror.com: ").strip()
            except (EOFError, KeyboardInterrupt, GetPassWarning):
                parser.exit(1, "\nToken input cancelled or hidden input unavailable.\n")
        if not token:
            parser.exit(1, "No Hugging Face token received. Run in a terminal or set HF_MIRROR_TOKEN.\n")
    try:
        download(args.state, args.gb, target, token)
    except Exception as error:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        detail = f"HTTP {status}" if status is not None else type(error).__name__
        if type(error) is RuntimeError:
            detail = str(error)
        if args.state == "2605" and status in (401, 403):
            detail += "; verify the dataset access approval and HF_MIRROR_TOKEN"
        parser.exit(1, f"Download failed via {MIRROR}: {detail}\n")


if __name__ == "__main__":
    main()
