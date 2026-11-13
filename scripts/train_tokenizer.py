
#!/usr/bin/env python3
"""
MiniRain-Vision BPE Tokenizer Trainer

Training data:
    data/ultradata/
        fineweb_l2_zh.jsonl
        fineweb_l2_en.jsonl
        fineweb_l3_zh_*.jsonl
        fineweb_l3_en_*.jsonl
        code_l2_py.jsonl
        code_l3_py.jsonl
        code_l2_sh.jsonl
        code_l3_sh.jsonl
        math_l3_*.jsonl
        windows_batchfile.jsonl
        windows_powershell.jsonl

Tokenizer:
    ByteLevel BPE

Special tokens:
    IDs 0-35, unchanged

Default vocabulary:
    16384

Default sampling:
    Approximately 3.05 GB

Usage:
    python train_tokenizer.py

    python train_tokenizer.py --vocab-size 32768

    python train_tokenizer.py --sample-scale 0.33

    python train_tokenizer.py --allow-missing
"""

import argparse
import glob
import json
import random
from dataclasses import dataclass
from pathlib import Path

from tokenizers import (
    Tokenizer,
    models,
    trainers,
    pre_tokenizers,
    decoders,
)

GB = 1_000_000_000


# ============================================================
# 1. Special tokens
#
# IMPORTANT:
# Order determines token IDs.
# Do not modify without updating RainConfig.
# ============================================================

SPECIALS = [
    "<|endoftext|>",           # 0  PAD
    "<|im_start|>",            # 1  BOS
    "<|im_end|>",              # 2  EOS
    "<|object_ref_start|>",    # 3
    "<|object_ref_end|>",      # 4
    "<|box_start|>",           # 5
    "<|box_end|>",             # 6
    "<|quad_start|>",          # 7
    "<|quad_end|>",            # 8
    "<|vision_start|>",        # 9
    "<|vision_end|>",          # 10
    "<|vision_pad|>",          # 11
    "<|image_pad|>",           # 12
    "<|video_pad|>",           # 13
    "<|audio_start|>",         # 14
    "<|audio_end|>",           # 15
    "<|audio_pad|>",           # 16
    "<tts_pad>",               # 17
    "<tts_text_bos>",          # 18
    "<tts_text_eod>",          # 19
    "<tts_text_bos_single>",   # 20
    "<tool_call>",             # 21
    "</tool_call>",            # 22
    "<tool_response>",         # 23
    "</tool_response>",        # 24
    "<think>",                 # 25
    "</think>",                # 26
    "<|img_row|>",             # 27
    "<|cpu_call|>",             # 28
    "<|/cpu_call|>",             # 29
    "<|cpu_response|>",             # 30
    "<|/cpu_response|>",             # 31
    "<|buffer6|>",             # 32
    "<|buffer7|>",             # 33
    "<|buffer8|>",             # 34
    "<|buffer9|>",             # 35
]

assert len(SPECIALS) == 36


# ============================================================
# 2. Training data sampling plan
#
# All paths are relative to data/ultradata.
#
# The targets describe tokenizer training samples,
# NOT the total pretraining corpus.
# ============================================================

@dataclass(frozen=True)
class Source:
    name: str
    patterns: tuple[str, ...]
    target_gb: float


SOURCES = [
    # --------------------------------------------------------
    # Chinese
    # --------------------------------------------------------

    Source(
        name="FineWeb L2 Chinese",
        patterns=("fineweb_l2_zh.jsonl",),
        target_gb=0.55,
    ),

    Source(
        name="FineWeb L3 Chinese",
        patterns=("fineweb_l3_zh_*.jsonl",),
        target_gb=0.55,
    ),

    # --------------------------------------------------------
    # English
    # --------------------------------------------------------

    Source(
        name="FineWeb L2 English",
        patterns=("fineweb_l2_en.jsonl",),
        target_gb=0.25,
    ),

    Source(
        name="FineWeb L3 English",
        patterns=("fineweb_l3_en_*.jsonl",),
        target_gb=0.25,
    ),

    # --------------------------------------------------------
    # Python
    # --------------------------------------------------------

    Source(
        name="Python L2",
        patterns=("code_l2_py.jsonl",),
        target_gb=0.30,
    ),

    Source(
        name="Python L3",
        patterns=("code_l3_py.jsonl",),
        target_gb=0.30,
    ),

    # --------------------------------------------------------
    # Linux Shell
    # --------------------------------------------------------

    Source(
        name="Shell L2",
        patterns=("code_l2_sh.jsonl",),
        target_gb=0.10,
    ),

    Source(
        name="Shell L3",
        patterns=("code_l3_sh.jsonl",),
        target_gb=0.10,
    ),

    # --------------------------------------------------------
    # Math
    # --------------------------------------------------------

    Source(
        name="UltraData Math",
        patterns=("math_l3_*.jsonl",),
        target_gb=0.45,
    ),

    # --------------------------------------------------------
    # Windows CMD / Batchfile
    # --------------------------------------------------------

    Source(
        name="Windows CMD",
        patterns=("windows_batchfile.jsonl",),
        target_gb=0.12,
    ),

    # --------------------------------------------------------
    # Windows PowerShell
    # --------------------------------------------------------

    Source(
        name="Windows PowerShell",
        patterns=("windows_powershell.jsonl",),
        target_gb=0.08,
    ),
]


# ============================================================
# 3. Project paths
# ============================================================

def detect_project_root():

    here = Path(__file__).resolve().parent

    # train_tokenizer.py at project root
    if (here / "data" / "ultradata").is_dir():
        return here

    # train_tokenizer.py inside data/
    if (here / "ultradata").is_dir():
        return here.parent

    return here


# ============================================================
# 4. Resolve dataset files
# ============================================================

def resolve_files(data_dir, patterns):

    found = set()

    for pattern in patterns:
        for filename in glob.glob(
            str(data_dir / pattern)
        ):

            path = Path(filename)

            if path.is_file():
                found.add(path.resolve())

    return sorted(found)


# ============================================================
# 5. Prepare sampling plan
# ============================================================

def prepare_sources(
    data_dir,
    sample_scale,
    allow_missing,
):

    prepared = []
    missing = []

    print("\n" + "=" * 72)
    print("MiniRain Tokenizer Data Sources")
    print("=" * 72)

    print(f"Data directory: {data_dir}")
    print(f"Sample scale: {sample_scale:.3f}")
    print()

    for source in SOURCES:

        files = resolve_files(
            data_dir,
            source.patterns,
        )

        total_bytes = sum(
            fp.stat().st_size
            for fp in files
        )

        target_bytes = round(
            source.target_gb
            * sample_scale
            * GB
        )

        if not files or total_bytes == 0:

            missing.append(source.name)

            print(
                f"[MISSING] {source.name}"
            )

            continue

        prepared.append(
            {
                "source": source,
                "files": files,
                "total_bytes": total_bytes,
                "target_bytes": target_bytes,
            }
        )

        print(
            f"[OK] {source.name:23s}"
            f" files={len(files):2d}"
            f" available={total_bytes / GB:7.3f} GB"
            f" sample={target_bytes / GB:6.3f} GB"
        )

        if total_bytes < target_bytes:
            print(
                "     [WARN] Source smaller than "
                "sample target; will use all "
                "available valid records."
            )

    if missing and not allow_missing:

        raise RuntimeError(
            "\nMissing or empty data sources:\n"
            + "\n".join(
                f"  - {name}"
                for name in missing
            )
            + "\n\nRun downloads first, or use "
            "--allow-missing to train without them."
        )

    if not prepared:

        raise RuntimeError(
            "No available JSONL training files."
        )

    total_target = sum(
        item["target_bytes"]
        for item in prepared
    )

    print("-" * 72)

    print(
        f"Total target sampling: "
        f"{total_target / GB:.3f} GB"
    )

    print("=" * 72)

    return prepared


# ============================================================
# 6. JSONL text extraction
# ============================================================

def extract_text(line):

    try:
        record = json.loads(line)
    except (json.JSONDecodeError, UnicodeError):
        return None

    if isinstance(record, str):
        return record

    if not isinstance(record, dict):
        return None

    # All MiniRain download scripts store text here.
    text = record.get("text")

    if isinstance(text, str):
        return text

    # Optional compatibility with original source JSONL.
    for key in (
        "content",
        "full_content",
        "raw_content",
    ):

        text = record.get(key)

        if isinstance(text, str):
            return text

    return None


# ============================================================
# 7. Streaming random sampling
#
# Avoid loading all JSONL data into RAM.
# Shuffle files and sample rows probabilistically.
#
# This is approximate proportional sampling, not
# an exact uniform reservoir sample.
# ============================================================

def iter_source(
    item,
    rng,
    min_chars,
):

    source = item["source"]

    files = list(item["files"])

    target_bytes = item["target_bytes"]
    total_bytes = item["total_bytes"]

    if target_bytes <= 0:
        return

    rng.shuffle(files)

    # Margin compensates for metadata overhead and
    # rejected short / invalid records.
    probability = min(
        1.0,
        target_bytes * 1.20
        / max(total_bytes, 1),
    )

    sampled_bytes = 0
    sampled_records = 0

    for fp in files:

        if sampled_bytes >= target_bytes:
            break

        print(
            f"  [READ] {fp.name}",
            flush=True,
        )

        with fp.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as f:

            for line in f:

                if sampled_bytes >= target_bytes:
                    break

                if rng.random() > probability:
                    continue

                text = extract_text(line)

                if not text:
                    continue

                if len(text.strip()) < min_chars:
                    continue

                # Use text UTF-8 size, not JSONL file size.
                size = len(
                    text.encode("utf-8")
                )

                sampled_bytes += size
                sampled_records += 1

                yield text

    print(
        f"[SAMPLED] {source.name}: "
        f"{sampled_bytes / GB:.3f} GB, "
        f"{sampled_records:,} records",
        flush=True,
    )


# ============================================================
# 8. Combine source iterators
# ============================================================

def iter_training_data(
    prepared,
    seed,
    min_chars,
):

    # Deterministic pseudorandom sampling.
    rng = random.Random(seed)

    # Each source contributes at most its own target.
    # The tokenizer sees all selected source groups.
    for item in prepared:

        print(
            f"\n[SOURCE] {item['source'].name}",
            flush=True,
        )

        yield from iter_source(
            item,
            rng,
            min_chars,
        )


# ============================================================
# 9. Tokenizer construction
# ============================================================

def build_tokenizer(vocab_size):

    # ByteLevel BPE supports arbitrary UTF-8 text.
    tokenizer = Tokenizer(
        models.BPE(
            unk_token=None,
        )
    )

    tokenizer.pre_tokenizer = (
        pre_tokenizers.ByteLevel(
            add_prefix_space=False,
            use_regex=True,
        )
    )

    tokenizer.decoder = (
        decoders.ByteLevel()
    )

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,

        special_tokens=SPECIALS,

        # Cover all 256 byte values, even when
        # rare characters are absent from the sample.
        initial_alphabet=(
            pre_tokenizers.ByteLevel.alphabet()
        ),
    )

    return tokenizer, trainer


# ============================================================
# 10. Special token validation
# ============================================================

def validate_special_tokens(tokenizer):

    print("\n[SPECIAL TOKEN VALIDATION]")

    if tokenizer.get_vocab_size() < len(SPECIALS):
        raise RuntimeError(
            "Vocabulary smaller than special token count."
        )

    for expected_id, token in enumerate(SPECIALS):

        actual_id = tokenizer.token_to_id(token)

        status = (
            "OK"
            if actual_id == expected_id
            else "FAIL"
        )

        print(
            f"{expected_id:2d} "
            f"{token:28s} "
            f"actual={actual_id} "
            f"{status}"
        )

        if actual_id != expected_id:

            raise RuntimeError(
                f"Special token ID mismatch: "
                f"{token}, "
                f"expected={expected_id}, "
                f"actual={actual_id}"
            )

    print("[OK] All 36 special token IDs aligned.")


# ============================================================
# 11. Compression tests
# ============================================================

TEST_CASES = [
    (
        "中文",
        "深度学习是机器学习的一个重要分支，"
        "它使用多层神经网络从数据中学习表示。",
    ),

    (
        "英文",
        "Deep learning is a subset of machine "
        "learning that uses neural networks.",
    ),

    (
        "Python",
        "def forward(self, x):\n"
        "    return self.linear(x)  # comment",
    ),

    (
        "Linux Shell",
        "#!/bin/bash\n"
        "find /var/log -type f -name '*.log' "
        "| xargs grep -n 'error'",
    ),

    (
        "Windows CMD",
        "@echo off\r\n"
        "set PATH=%PATH%;C:\\Tools\r\n"
        "for %%f in (*.txt) do echo %%f\r\n",
    ),

    (
        "PowerShell",
        "Get-ChildItem -Path C:\\Users "
        "-Recurse -Filter *.log "
        "| Select-Object -First 10",
    ),

    (
        "数学",
        "设 $f(x)=x^2+2x+1$，求 "
        "$f'(x)$ 在 $x=0$ 处的值。",
    ),

    (
        "中英混合",
        "使用 Transformer 模型处理 NLP 任务，"
        "learning rate 设为 2e-4。",
    ),
]


def compression_test(tokenizer):

    print("\n[COMPRESSION TEST]")

    for label, text in TEST_CASES:

        ids = tokenizer.encode(text).ids

        chars = len(text)
        utf8_bytes = len(text.encode("utf-8"))

        token_count = len(ids)

        print(
            f"  {label:16s} "
            f"tokens={token_count:4d} "
            f"token/char={token_count/max(chars,1):.3f} "
            f"bytes/token={utf8_bytes/max(token_count,1):.2f}"
        )


# ============================================================
# 12. Round-trip tests
# ============================================================

def roundtrip_test(tokenizer):

    print("\n[ROUND-TRIP TEST]")

    cases = [
        "你好，MiniRain！",
        "Hello World!",
        "def main():\n    print('hello')",
        "@echo off\r\necho hello\r\n",
        "Get-ChildItem C:\\Users",
        "a = x^2 + y^2",
        (
            "<|im_start|>图像序列 "
            "<|vision_start|><|image_pad|>"
            "<|buffer1|><|vision_end|>"
            " 测试123<|im_end|>"
        ),
    ]

    for text in cases:

        encoded = tokenizer.encode(text)

        # IMPORTANT:
        # The default decoder skips special tokens.
        # Disable skipping for exact round-trip checks.
        decoded = tokenizer.decode(
            encoded.ids,
            skip_special_tokens=False,
        )

        if decoded != text:

            print("[FAIL]")
            print("Original:", repr(text))
            print("Decoded: ", repr(decoded))

            raise RuntimeError(
                "Tokenizer round-trip mismatch."
            )

    print("[OK] All round-trip tests passed.")


# ============================================================
# 13. CLI
# ============================================================

def parse_args():

    root = detect_project_root()

    parser = argparse.ArgumentParser(
        description=(
            "Train MiniRain BPE tokenizer "
            "using actual pretraining JSONL data."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "data" / "ultradata",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=root / "minirain_tokenizer.json",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=16384,
    )

    parser.add_argument(
        "--sample-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply all sampling quotas by this factor. "
            "0.33 gives about 1 GB total."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--min-chars",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Allow training if some expected "
            "dataset files are missing."
        ),
    )

    return parser.parse_args()


# ============================================================
# 14. Main
# ============================================================

def main():

    args = parse_args()

    if args.vocab_size < 512:
        raise ValueError(
            "vocab-size must be at least 512."
        )

    if args.sample_scale <= 0:
        raise ValueError(
            "sample-scale must be positive."
        )

    if args.min_chars < 0:
        raise ValueError(
            "min-chars cannot be negative."
        )

    data_dir = (
        args.data_dir.expanduser().resolve()
    )

    output_path = (
        args.output.expanduser().resolve()
    )

    if not data_dir.is_dir():

        raise FileNotFoundError(
            f"Training data directory not found: "
            f"{data_dir}"
        )

    # --------------------------------------------------------
    # Prepare sampling
    # --------------------------------------------------------

    prepared = prepare_sources(
        data_dir=data_dir,
        sample_scale=args.sample_scale,
        allow_missing=args.allow_missing,
    )

    # --------------------------------------------------------
    # Build tokenizer
    # --------------------------------------------------------

    tokenizer, trainer = build_tokenizer(
        args.vocab_size
    )

    print("\n" + "=" * 72)
    print("MiniRain BPE Tokenizer Training")
    print("=" * 72)

    print(f"Vocab size: {args.vocab_size}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_path}")
    print(f"Special tokens: {len(SPECIALS)}")
    print("Algorithm: ByteLevel BPE")
    print("=" * 72)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    tokenizer.train_from_iterator(
        iter_training_data(
            prepared=prepared,
            seed=args.seed,
            min_chars=args.min_chars,
        ),
        trainer=trainer,
    )

    print(
        "\n[TRAINED] Actual vocab size:",
        tokenizer.get_vocab_size(),
    )

    # --------------------------------------------------------
    # Validate before saving
    # --------------------------------------------------------

    validate_special_tokens(tokenizer)

    compression_test(tokenizer)

    roundtrip_test(tokenizer)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save(str(output_path))

    print("\n" + "=" * 72)
    print("[DONE] Tokenizer saved")
    print(output_path)
    print(
        f"Vocabulary: {tokenizer.get_vocab_size()}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
