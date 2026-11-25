"""Download UltraData SFT from HF Mirror or generate local identity examples."""

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


def generate_identity(target, budget):
    name, version, developer = "RainPocket", "V1", "Chill_Rain"
    topics = (
        (
            (f"我叫 {name}。", f"我的模型名称是 {name}。", f"I’m {name}.", f"My model name is {name}."),
            ("你叫什么名字？", "你的模型名字是什么？", "请告诉我你的名称。", "我该怎么称呼你？"),
            ("What is your name?", "What should I call you?", "Tell me your model name.", "What are you called?"),
        ),
        (
            (f"我的版本是 {version}。", f"当前是 {name} {version}。",
             f"My version is {version}.", f"This is {name} {version}."),
            ("你是什么版本？", "你的版本号是多少？", "这是什么版本的模型？", "请告诉我当前的版本。"),
            ("What version are you?", "What is your model version?",
             "Which release am I talking to?", "Tell me your version."),
        ),
        (
            (f"我的开发者是 {developer}。", f"我是由 {developer} 开发的。",
             f"My developer is {developer}.", f"I was developed by {developer}."),
            ("谁开发了你？", "你的开发者是谁？", "你是由谁做出来的？", "请告诉我这个模型的开发者。"),
            ("Who developed you?", "Who is your developer?", "Who made this model?", "Name your developer."),
        ),
        (
            (f"我是 {name} {version}，由 {developer} 开发。",
             f"我是 {developer} 开发的 {name}，当前版本是 {version}。",
             f"I’m {name} {version}, developed by {developer}.",
             f"I’m {name}, version {version}. My developer is {developer}."),
            ("你是谁？", "简单介绍一下自己。", "请介绍你的名称、版本和开发者。", "你好，能说一下你的模型信息吗？"),
            ("Who are you?", "Introduce yourself briefly.",
             "What is your name, version, and developer?", "Tell me a bit about this model."),
        ),
    )
    conversations = []
    forms_zh = ("{}", "请简短回答：{}", "我想确认一下，{}", "请直接告诉我：{}",
                "用一句话回答：{}", "你好，想问一下，{}")
    forms_en = ("{}", "A quick question: {}", "Could you confirm this? {}", "Please answer directly: {}",
                "Answer in one sentence: {}", "Hi, I wanted to ask: {}")
    for (short_zh, long_zh, short_en, long_en), questions_zh, questions_en in topics:
        for questions, answers, forms in ((questions_zh, (short_zh, long_zh), forms_zh),
                                          (questions_en, (short_en, long_en), forms_en)):
            conversations.extend((form.format(question), answers[(index + style) % len(answers)])
                                 for index, question in enumerate(questions)
                                 for style, form in enumerate(forms))
    conversations.extend((
        ("你叫什么名字？", f"我叫 {name}。", "这个模型是谁开发的？", f"开发者是 {developer}。"),
        ("谁开发了你？", f"我的开发者是 {developer}。", "那现在是什么版本？", f"当前版本是 {version}。"),
        ("What is your name?", f"I’m {name}.", "And which version is this?", f"This is version {version}."),
        ("Which version are you?", f"My version is {version}.", "Who developed it?", f"It was developed by {developer}."),
        ("你是 V1 吗？", f"是的，我的版本是 {version}。", "模型全名是什么？", f"我是 {name} {version}。"),
        ("简单介绍一下自己。", f"我是 {name} {version}，由 {developer} 开发。",
         "单独说一下开发者名字。", developer),
        ("Who made you?", f"I was developed by {developer}.", "What is the model called?", name),
        ("What model is this?", f"This is {name} {version}.", "Please confirm the developer.", developer),
    ))
    payload = "".join(
        json.dumps({"messages": [
            {"role": "user" if index % 2 == 0 else "assistant", "content": text}
            for index, text in enumerate(conversation)
        ]}, ensure_ascii=False) + "\n"
        for conversation in conversations
    ).encode("utf-8")
    if budget is not None and len(payload) > budget:
        raise ValueError(f"Identity dataset needs {len(payload):,} bytes. Increase --gb or omit it for self.")
    if target.exists():
        old_manifest = target / "manifest.json"
        if not old_manifest.is_file() or json.loads(old_manifest.read_text(encoding="utf-8")).get("dataset") != "self":
            raise ValueError(f"Output is not a generated identity dataset: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".self-", dir=target.parent) as temporary:
        staging = Path(temporary) / "output"
        staging.mkdir()
        (staging / "identity.jsonl").write_bytes(payload)
        manifest = dict(
            dataset="self", name=name, version=version, developer=developer,
            budget_bytes=budget, bytes=len(payload),
            files=[dict(path="identity.jsonl", bytes=len(payload), rows=len(conversations))],
        )
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if target.exists():
            for name in ("identity.jsonl", "manifest.json"):
                (staging / name).replace(target / name)
        else:
            staging.rename(target)
    print(f"[Generated] {len(conversations):,} rows, {len(payload):,} bytes -> {target}", flush=True)


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
    parser.add_argument("--state", required=True, choices=(*REPOS, "self"))
    parser.add_argument("--gb", type=budget_bytes, help="maximum JSONL size in decimal GB; optional for self")
    args = parser.parse_args()
    if args.state != "self" and args.gb is None:
        parser.error("--gb is required for 2605 and 2609")
    target = OUTPUT / args.state
    if target.exists() and args.state != "self":
        parser.exit(1, f"Output already exists: {target}\n")
    if args.state == "self":
        try:
            generate_identity(target, args.gb)
        except (OSError, ValueError) as error:
            parser.exit(1, f"Identity generation failed: {error}\n")
        return
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
