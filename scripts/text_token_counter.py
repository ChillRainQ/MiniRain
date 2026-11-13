
"""
MiniRain Pretraining Token Counter

统计预训练 JSONL 数据的实际 Token 数量。

功能：
1. 递归扫描 JSONL
2. 自动识别 text/content 等字段
3. 使用 MiniRain Fast Tokenizer
4. 分文件统计 Token 数量
5. 支持追加 EOS 统计
6. 流式处理，不将全部数据加载到内存
7. 输出 JSON 报告

运行：
python scripts/count_pretrain_tokens.py

指定数据目录：
python scripts/count_pretrain_tokens.py --data-dir data/ultradata

追加 EOS：
python scripts/count_pretrain_tokens.py --append-eos
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

from transformers import AutoTokenizer


# ============================================================
# 路径
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_DIRS = [
    ROOT / "scripts" / "data" / "ultradata",
    ROOT / "data" / "ultradata",
]

DEFAULT_TOKENIZER_DIRS = [
    ROOT / "minirain_tokenizer",
    ROOT / "minirain_tokenizers",
]


def detect_directory(candidates):
    """自动选择存在且包含目标文件的目录。"""
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


# ============================================================
# 文本提取
# ============================================================

TEXT_KEYS = [
    "text",
    "content",
    "code",
    "document",
    "problem",
    "question",
    "instruction",
    "input",
    "solution",
    "response",
    "answer",
    "output",
]


def extract_text(obj, field="auto"):
    """
    从 JSONL 记录中提取文本。

    优先级：
    1. 用户指定字段
    2. text
    3. content
    4. 其他常见文本字段
    """

    if isinstance(obj, str):
        return obj

    if not isinstance(obj, dict):
        return None

    if field != "auto":
        value = obj.get(field)
        return value if isinstance(value, str) else None

    # 优先使用统一文本字段
    for key in ("text", "content"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value

    # 处理数学数据、问答数据等多字段记录
    parts = []

    for key in TEXT_KEYS:
        if key in ("text", "content"):
            continue

        value = obj.get(key)

        if isinstance(value, str) and value.strip():
            parts.append(value)

    if parts:
        return "\n".join(parts)

    # 兼容 messages 格式
    messages = obj.get("messages")

    if isinstance(messages, list):
        parts = []

        for message in messages:
            if isinstance(message, dict):
                content = message.get("content")

                if isinstance(content, str):
                    parts.append(content)

        if parts:
            return "\n".join(parts)

    return None


# ============================================================
# Tokenizer
# ============================================================

def load_tokenizer(path):

    if not path.is_dir():
        raise FileNotFoundError(
            f"Tokenizer directory not found: {path}"
        )

    if not (path / "tokenizer.json").is_file():
        raise FileNotFoundError(
            f"tokenizer.json not found: {path}"
        )

    print("=" * 65)
    print("Loading MiniRain Tokenizer")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(
        str(path),
        use_fast=True,
        local_files_only=True,
    )

    if not tokenizer.is_fast:
        raise RuntimeError("需要使用 Fast Tokenizer")

    print(f"Path       : {path}")
    print(f"Class      : {type(tokenizer).__name__}")
    print(f"Vocab size : {len(tokenizer)}")
    print(f"Max length : {tokenizer.model_max_length}")

    return tokenizer


# ============================================================
# 批量 Token 统计
# ============================================================

def encode_batch(backend, texts, append_eos=False):

    if not texts:
        return 0

    # 直接调用 Rust Tokenizers 后端
    encodings = backend.encode_batch(
        texts,
        add_special_tokens=False,
    )

    total = sum(len(enc.ids) for enc in encodings)

    # 如果训练时每条样本都会追加 EOS，则计入
    if append_eos:
        total += len(texts)

    return total


# ============================================================
# 单文件统计
# ============================================================

def count_file(
    path,
    data_root,
    backend,
    batch_size=256,
    text_field="auto",
    append_eos=False,
):

    stats = {
        "name": str(path.relative_to(data_root)),
        "size_bytes": path.stat().st_size,
        "records": 0,
        "valid_records": 0,
        "skipped_records": 0,
        "invalid_json": 0,
        "chars": 0,
        "text_bytes": 0,
        "tokens": 0,
    }

    batch = []

    def flush():
        if not batch:
            return

        stats["tokens"] += encode_batch(
            backend,
            batch,
            append_eos=append_eos,
        )

        batch.clear()

    print(f"\n[FILE] {stats['name']}")
    print(
        f"       Size: "
        f"{stats['size_bytes'] / 1e9:.3f} GB"
    )

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for line in f:

            stats["records"] += 1

            line = line.strip()

            if not line:
                stats["skipped_records"] += 1
                continue

            try:
                obj = json.loads(line)

            except json.JSONDecodeError:
                stats["invalid_json"] += 1

                # 非 JSON 行不默认当作训练文本，
                # 避免把损坏的 JSON 误计入语料。
                stats["skipped_records"] += 1
                continue

            text = extract_text(
                obj,
                field=text_field,
            )

            if not isinstance(text, str) or not text.strip():
                stats["skipped_records"] += 1
                continue

            stats["valid_records"] += 1
            stats["chars"] += len(text)
            stats["text_bytes"] += len(
                text.encode("utf-8")
            )

            batch.append(text)

            if len(batch) >= batch_size:
                flush()

            if stats["records"] % 100000 == 0:
                print(
                    f"\r       Records: "
                    f"{stats['records']:,} | "
                    f"Tokens: {stats['tokens']:,}",
                    end="",
                    flush=True,
                )

    flush()

    print(
        f"\n       Records : {stats['records']:,}"
        f"\n       Valid   : {stats['valid_records']:,}"
        f"\n       Tokens  : {stats['tokens']:,}"
        f"\n       Skipped : {stats['skipped_records']:,}"
    )

    return stats


# ============================================================
# 总统计
# ============================================================

def summarize(results):

    total = defaultdict(int)

    for item in results:

        for key in (
            "size_bytes",
            "records",
            "valid_records",
            "skipped_records",
            "invalid_json",
            "chars",
            "text_bytes",
            "tokens",
        ):
            total[key] += item[key]

    return dict(total)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="MiniRain Pretraining Token Counter"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="预训练 JSONL 数据目录",
    )

    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="MiniRain Tokenizer 目录",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="批量 Tokenize 的文本数量",
    )

    parser.add_argument(
        "--text-field",
        type=str,
        default="auto",
        help="指定文本字段，默认自动识别",
    )

    parser.add_argument(
        "--append-eos",
        action="store_true",
        help="每条有效文本额外计入一个 EOS",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "pretrain_token_count.json",
        help="统计报告输出路径",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size 必须大于 0")

    data_dir = (
        args.data_dir.resolve()
        if args.data_dir
        else detect_directory(DEFAULT_DATA_DIRS)
    )

    tokenizer_dir = (
        args.tokenizer.resolve()
        if args.tokenizer
        else detect_directory(DEFAULT_TOKENIZER_DIRS)
    )

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Training data directory not found: {data_dir}"
        )

    # --------------------------------------------------------
    # 扫描 JSONL
    # --------------------------------------------------------

    files = sorted(data_dir.rglob("*.jsonl"))

    if not files:
        raise RuntimeError(
            f"No JSONL files found in {data_dir}"
        )

    print("\n" + "=" * 65)
    print("MiniRain Pretraining Token Counter")
    print("=" * 65)

    print(f"Data directory : {data_dir}")
    print(f"Tokenizer      : {tokenizer_dir}")
    print(f"JSONL files    : {len(files)}")
    print(f"Batch size     : {args.batch_size}")
    print(f"Text field     : {args.text_field}")
    print(f"Append EOS     : {args.append_eos}")
    print("=" * 65)

    # --------------------------------------------------------
    # 加载 Tokenizer
    # --------------------------------------------------------

    tokenizer = load_tokenizer(tokenizer_dir)

    backend = tokenizer.backend_tokenizer

    # --------------------------------------------------------
    # 逐文件统计
    # --------------------------------------------------------

    results = []

    for i, path in enumerate(files, 1):

        print(
            f"\n[{i}/{len(files)}] Processing"
        )

        result = count_file(
            path=path,
            data_root=data_dir,
            backend=backend,
            batch_size=args.batch_size,
            text_field=args.text_field,
            append_eos=args.append_eos,
        )

        results.append(result)

    # --------------------------------------------------------
    # 汇总
    # --------------------------------------------------------

    total = summarize(results)

    print("\n" + "=" * 65)
    print("PRETRAIN TOKEN COUNT SUMMARY")
    print("=" * 65)

    print(
        f"Files          : {len(results):,}"
    )

    print(
        f"Dataset size   : "
        f"{total['size_bytes'] / 1e9:.3f} GB"
    )

    print(
        f"Text size      : "
        f"{total['text_bytes'] / 1e9:.3f} GB"
    )

    print(
        f"Records        : "
        f"{total['records']:,}"
    )

    print(
        f"Valid records  : "
        f"{total['valid_records']:,}"
    )

    print(
        f"Skipped        : "
        f"{total['skipped_records']:,}"
    )

    print(
        f"Total tokens   : "
        f"{total['tokens']:,}"
    )

    print(
        f"Tokens (B)     : "
        f"{total['tokens'] / 1e9:.4f} B"
    )

    if total["valid_records"] > 0:
        print(
            f"Avg tokens/doc : "
            f"{total['tokens'] / total['valid_records']:.2f}"
        )

    if total["tokens"] > 0:
        print(
            f"Bytes/token    : "
            f"{total['text_bytes'] / total['tokens']:.3f}"
        )

    print("=" * 65)

    # --------------------------------------------------------
    # 保存报告
    # --------------------------------------------------------

    report = {
        "project": "MiniRain",
        "updated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "data_dir": str(data_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "vocab_size": len(tokenizer),
        "model_max_length": tokenizer.model_max_length,
        "batch_size": args.batch_size,
        "text_field": args.text_field,
        "append_eos": args.append_eos,
        "total": total,
        "files": results,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n[SAVED] {args.output}")
    print("\n统计完成。")


if __name__ == "__main__":
    main()
