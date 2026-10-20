# collect_text.py
"""
纯文本语料收集：zh(世界知识) + en(世界知识) + code + math
零 modelscope/pyarrow 依赖：jsonl 直读，parquet 用 polars
文件列表走 API 自动发现，不用猜分片命名

前置：uv pip install polars requests
gated 仓库：set HF_TOKEN=hf_xxxxx
"""
import argparse
import io
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import requests

MIRROR = "https://hf-mirror.com"
HF_TOKEN = os.environ.get("HF_TOKEN")
AUTH = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


@dataclass
class Source:
    repo: str
    subset: str       # 仓库内目录，如 "zh_cc-high-loss0"、"4_5"、"sample/10BT"
    text_key: str
    n_files: int      # 取多少个分片
    shuffle: bool     # True=随机抽分片，False=顺序取前N个


SOURCES = {
    "zh": [
        Source("BAAI/CCI4.0-M2-Base-v1", "zh_cc-high-loss0", "text", 20, False),
        Source("opencsg/Fineweb-Edu-Chinese-V2.2", "4_5", "text", 3, True),
        Source("opencsg/Fineweb-Edu-Chinese-V2.2", "3_4", "text", 1, True),
        Source("wikimedia/wikipedia", "20231101.zh", "text", 2, True),
    ],
    "en": [
        Source("HuggingFaceFW/fineweb-edu", "sample/10BT", "text", 8, False),
        Source("HuggingFaceFW/fineweb", "sample/10BT", "text", 2, False),
    ],
    "code": [
        Source("bigcode/the-stack", "data/python", "content", 5, False),
    ],
    "math": [
        Source("AI-MO/NuminaMath-1.5", "data", "solution", 2, False),
    ],
}


def list_files(src: Source) -> list[str]:
    """API 列出 subset 目录下的数据文件（jsonl/parquet），按名排序。"""
    u = f"{MIRROR}/api/datasets/{src.repo}/tree/main/{src.subset}"
    r = requests.get(u, headers=AUTH, timeout=60)
    r.raise_for_status()
    files = [f["path"] for f in r.json()
             if f["type"] == "file" and f["path"].endswith((".jsonl", ".parquet"))]
    files.sort()
    if src.shuffle:
        random.Random(42).shuffle(files)
    files = files[:src.n_files]
    if len(files) < src.n_files:
        print(f"  [警告] 只要 {src.n_files} 片，实际只有 {len(files)} 片")
    return files


def download(url: str, dest: Path, max_retry: int = 5):
    if dest.exists():
        print(f"  已存在: {dest.name}")
        return
    tmp = dest.with_suffix(".part")
    for attempt in range(1, max_retry + 1):
        try:
            headers = dict(AUTH)
            got = tmp.stat().st_size if tmp.exists() else 0
            if got:
                headers["Range"] = f"bytes={got}-"
            with requests.get(url, headers=headers, stream=True,
                              timeout=(30, 300)) as r:
                r.raise_for_status()
                with open(tmp, "ab" if got else "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 22):  # 4MB chunk
                        f.write(chunk)
                        got += len(chunk)
                        print(f"\r    {dest.name}: {got/1e9:.2f} GB",
                              end="", flush=True)
            tmp.rename(dest)
            print()
            return
        except Exception as e:
            print(f"\n    第 {attempt} 次失败: {type(e).__name__}: {e}")
            if attempt < max_retry:
                time.sleep(10 * attempt)
    raise RuntimeError(f"下载失败: {url}")


def keep_text(t: str) -> bool:
    return 50 <= len(t) <= 10000 and "\x00" not in t


def dump_jsonl(path: Path, src: Source, out) -> int:
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                text = str(json.loads(line).get(src.text_key, "")).strip()
            except json.JSONDecodeError:
                continue
            if keep_text(text):
                out.write(json.dumps({"text": text, "source": src.repo},
                                     ensure_ascii=False) + "\n")
                n += 1
    return n


def dump_parquet(path: Path, src: Source, out) -> int:
    import polars as pl
    n = 0
    try:
        df = pl.read_parquet(path, columns=[src.text_key])
    except pl.exceptions.ColumnNotFoundError:
        df0 = pl.read_parquet(path).head(1)
        raise RuntimeError(f"{path.name} 没有列 '{src.text_key}'，实际列: {df0.columns}")
    for text in df[src.text_key].to_list():
        text = str(text or "").strip()
        if keep_text(text):
            out.write(json.dumps({"text": text, "source": src.repo},
                                 ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=str, default="zh,en,code,math",
                        help="逗号分隔，可只跑部分")
    parser.add_argument("--work_dir", type=str, default="text_raw")
    parser.add_argument("--out_dir", type=str, default="texts")
    parser.add_argument("--keep_files", action="store_true",
                        help="处理后保留原始分片（默认删除省磁盘）")
    args = parser.parse_args()

    work = Path(args.work_dir); work.mkdir(exist_ok=True)
    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)

    for group in args.groups.split(","):
        for src in SOURCES[group.strip()]:
            tag = f"{group}_{src.repo.split('/')[-1]}_{src.subset.replace('/', '-')}"
            out_path = out_dir / f"{tag}.jsonl"
            print(f"\n{'='*55}\n{src.repo} / {src.subset or '(root)'} → {out_path}\n{'='*55}")
            if out_path.exists():
                print("  已存在，跳过（重收请删除）")
                continue

            files = list_files(src)
            print(f"  分片: {len(files)} 个")

            total = 0
            with open(out_path, "w", encoding="utf-8") as out:
                for fpath in files:
                    local = work / src.repo.split("/")[-1] / fpath.replace("/", "_")
                    local.parent.mkdir(parents=True, exist_ok=True)
                    url = f"{MIRROR}/datasets/{src.repo}/resolve/main/{fpath}"
                    download(url, local)
                    fn = dump_jsonl if fpath.endswith(".jsonl") else dump_parquet
                    n = fn(local, src, out)
                    total += n
                    print(f"    +{n:,} 条（累计 {total:,}）")
                    if not args.keep_files:
                        local.unlink()   # 处理完即删，省磁盘
            print(f"  完成: {total:,} 条，{out_path.stat().st_size/1e9:.2f} GB")

    print("\n全部产物:")
    for p in sorted(out_dir.glob("*.jsonl")):
        print(f"  {p.name:50s} {p.stat().st_size/1e9:.2f} GB")


if __name__ == "__main__":
    main()