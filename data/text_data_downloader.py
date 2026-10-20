#!/usr/bin/env python3
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from datasets import load_dataset
    HAS_HF = True
except ImportError:
    HAS_HF = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("text_dl")


@dataclass
class Source:
    repo: str
    subdir: str = ""
    text_col: str = "text"
    weight_gb: float = 0.0
    is_config: bool = False
    fallback_repo: Optional[str] = None


SOURCES = {
    "zh": [
        Source("BAAI/CCI4.0-M2-Base-v1", "", "text", 20, False),
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
        Source("AI-MO/NuminaMath-1.5", "", "solution", 2, False),
    ],
}


class Downloader:
    def __init__(self, output_dir: str, cache_dir: Optional[str] = None):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.cache = Path(cache_dir) if cache_dir else self.out / ".hf_cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.manifest: Dict = {}

    def _save_dir(self, src: Source, domain: str, idx: int) -> Path:
        safe = src.repo.replace("/", "__")
        tag = src.subdir.replace("/", "__") if src.subdir else "default"
        return self.out / domain / f"{idx:02d}_{safe}__{tag}"

    @staticmethod
    def _parse_size(s: str) -> int:
        s = s.upper().strip()
        for unit, mul in [("TB", 1 << 40), ("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)]:
            if s.endswith(unit):
                return int(float(s[:-len(unit)]) * mul)
        return int(s)

    def download(self, src: Source, domain: str, idx: int,
                 max_shard_size: str = "1GB", streaming: bool = True) -> dict:
        save_dir = self._save_dir(src, domain, idx)
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{domain}] 开始: {src.repo} | subdir={src.subdir!r} | col={src.text_col}")

        ds = None
        repos_to_try = [src.repo]
        if src.fallback_repo:
            repos_to_try.append(src.fallback_repo)

        for repo in repos_to_try:
            try:
                kwargs = {
                    "split": "train",
                    "streaming": streaming,
                    "cache_dir": str(self.cache),
                    "trust_remote_code": True,
                }
                if src.is_config:
                    ds = load_dataset(repo, src.subdir, **kwargs)
                elif src.subdir:
                    ds = load_dataset(repo, data_dir=src.subdir, **kwargs)
                else:
                    ds = load_dataset(repo, **kwargs)
                logger.info(f"  成功加载: {repo}")
                break
            except Exception as e:
                logger.warning(f"  加载失败 {repo}: {e}")
                continue

        if ds is None:
            logger.error(f"[{domain}] 所有 repo 均失败: {src.repo}")
            return {"status": "failed", "repo": src.repo, "error": "all repos failed"}

        max_bytes = self._parse_size(max_shard_size)
        shard_idx, current_bytes = 0, 0
        shard_f = None
        buffer = []
        written = 0

        def flush(force_new: bool = False):
            nonlocal shard_f, shard_idx, current_bytes
            if not buffer:
                return
            if shard_f is None or current_bytes >= max_bytes or force_new:
                if shard_f:
                    shard_f.close()
                shard_idx += 1
                path = save_dir / f"shard_{shard_idx:05d}.jsonl"
                shard_f = open(path, "w", encoding="utf-8")
                current_bytes = 0
            for r in buffer:
                line = json.dumps(r, ensure_ascii=False) + "\n"
                shard_f.write(line)
                current_bytes += len(line.encode("utf-8"))
            buffer.clear()

        try:
            for ex in ds:
                text = ex.get(src.text_col)
                if text is None or not isinstance(text, str) or not text.strip():
                    continue
                buffer.append({"text": text.strip()})
                if len(buffer) >= 5000:
                    flush()
                written += 1
                if written % 50000 == 0:
                    logger.info(f"  ...已处理 {written:,} 条")
            flush()
            if shard_f:
                shard_f.close()
        except Exception as e:
            logger.error(f"[{domain}] 写入中断: {e}")
            return {"status": "error", "repo": src.repo, "error": str(e)}

        logger.info(f"[{domain}] 完成: {written:,} 条 -> {save_dir}")
        return {
            "status": "success",
            "repo": src.repo,
            "subdir": src.subdir,
            "domain": domain,
            "samples": written,
            "path": str(save_dir),
            "text_col": src.text_col,
        }

    def run(self, domains: Optional[List[str]] = None,
            streaming: bool = True, shard_size: str = "1GB"):
        if not HAS_HF:
            logger.error("缺少依赖: pip install datasets")
            sys.exit(1)

        todo = {k: v for k, v in SOURCES.items() if not domains or k in domains}

        for domain, srcs in todo.items():
            self.manifest[domain] = []
            for idx, src in enumerate(srcs):
                meta = self.download(src, domain, idx, streaming=streaming, max_shard_size=shard_size)
                self.manifest[domain].append(meta)

        mf = self.out / "manifest.json"
        with open(mf, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)
        logger.info(f"Manifest 已保存: {mf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载纯文本预训练数据")
    parser.add_argument("-o", "--output", default="./pretrain_text", help="输出目录")
    parser.add_argument("--cache", default=None, help="HF 缓存目录")
    parser.add_argument("-d", "--domains", nargs="+", default=None,
                        choices=["zh", "en", "code", "math"],
                        help="只下载指定域，默认全部")
    parser.add_argument("--no-streaming", action="store_true",
                        help="关闭流式（先把完整数据集下载到本地缓存）")
    parser.add_argument("--shard-size", default="1GB", help="单个 shard 大小")
    args = parser.parse_args()

    dl = Downloader(args.output, args.cache)
    dl.run(domains=args.domains, streaming=not args.no_streaming, shard_size=args.shard_size)
