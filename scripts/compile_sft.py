#!/usr/bin/env python3
"""Compile one shuffled, fixed-length RainPocket SFT mixture.

input_ids.bin and loss_mask.bin are row-major arrays shaped by manifest.json.
Read tokens as uint16 and derive labels with where(loss_mask, input_ids, -100).
"""

import argparse
from contextlib import closing
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compile_pretrain_text import shuffled_rows


MIX = {"no_think": 60, "think": 25, "agent": 15}
AGENT_SOURCES = ("Tool_Use", "Search_Agent", "General_Agent")
MAX_AGENT_CALLS = 4
DEFAULT_SEQUENCE_LENGTH = 4_096
IDENTITY_ROWS = 200


def token_budget(value):
    units = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    suffix = value[-1:].lower()
    try:
        amount = Decimal(value[:-1] if suffix in units else value) * units.get(suffix, 1)
        if amount.is_finite() and amount >= 1 and amount == amount.to_integral_value():
            return int(amount)
    except InvalidOperation:
        pass
    raise argparse.ArgumentTypeError("expected a positive token count, such as 200M")


def assistant_mask_template(template):
    """Mark assistant output (including EOS) without changing rendered text."""
    assistant_start = '{%- elif message.role == "assistant" %}'
    tool_start = '{%- elif message.role == "tool" %}'
    prefix, _, rest = template.partition(assistant_start)
    assistant, _, suffix = rest.partition(tool_start)
    header = r"'<|im_start|>' + message.role + '\n<think>\n'"
    body_start = "{%- if true %}"
    body_end = "{%- endif %}\n        {%- if message.tool_calls %}"
    eos = r"{{- '<|im_end|>\n' }}"
    if not suffix or any(assistant.count(part) != 1 for part in (header, body_start, body_end, eos)):
        raise ValueError("Expected the MiniRain chat template for assistant-only masking")
    assistant = assistant.replace(body_start, r"{{- '<|im_start|>assistant\n' }}" + "{%- set assistant_body %}")
    assistant = assistant.replace(header, r"'<think>\n'")
    assistant = assistant.replace(body_end, "{%- if message.tool_calls %}")
    assistant = assistant.replace(eos, (
        "{{- '<|im_end|>' }}{%- endset %}"
        "{%- if message.sft_target %}{% generation %}{{- assistant_body -}}{% endgeneration %}"
        "{%- else %}{{- assistant_body -}}{%- endif %}{{- '\\n' }}"
    ))
    return prefix + assistant_start + assistant + tool_start + suffix


def encode(row, tokenizer, template, group, seq_length):
    if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
        return None, "invalid"
    messages = []
    tool_calls = 0
    for original in row["messages"]:
        if not isinstance(original, dict) or original.get("role") not in ("system", "user", "assistant", "tool"):
            return None, "invalid"
        message = dict(original)
        role = message["role"]
        content = message.get("content")
        if role == "tool" and isinstance(content, (dict, list)):
            content = json.dumps(content, ensure_ascii=False)
        if content is not None and not isinstance(content, str):
            return None, "invalid"
        message["content"] = content or ""
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list) or any(not isinstance(call, dict) for call in calls):
            return None, "invalid"
        if not message.get("reasoning_content") and isinstance(message.get("reasoning"), str):
            message["reasoning_content"] = message["reasoning"]
        message["sft_target"] = (
            role == "assistant" and bool(message["content"].strip() or calls or message.get("reasoning_content"))
            and all(message.get(field) not in (False, 0) for field in ("loss", "loss_mask", "loss_weight", "trainable"))
        )
        messages.append(message)
        tool_calls += len(calls)
    if not messages or messages[-1]["role"] != "assistant" or not any(m["role"] == "user" for m in messages):
        return None, "invalid"

    tools = row.get("tools") or None
    if tools is not None and (not isinstance(tools, list) or any(not isinstance(tool, dict) for tool in tools)):
        return None, "invalid"
    if group == "agent" and (not tools or tool_calls > MAX_AGENT_CALLS or not messages[-1]["content"].strip()
                             or (tool_calls and not any(m["role"] == "tool" for m in messages))):
        return None, "agent_filter"
    encoded = tokenizer.apply_chat_template(messages, tools=tools, chat_template=template,
                                            return_dict=True, tokenize=True,
                                            return_assistant_tokens_mask=True, truncation=False)
    ids, mask = encoded["input_ids"], encoded["assistant_masks"]
    truncated = len(ids) > seq_length
    ids, mask = ids[:seq_length], mask[:seq_length]
    targets = sum(mask[1:])
    if not targets:
        return None, "no_targets_after_truncation" if truncated else "no_targets"
    return (ids, mask, targets, truncated), None


def interleave_identity(path, dtype, width, corpus_rows, identity_rows):
    """Spread the appended identity rows across the actual corpus size."""
    row_bytes = np.dtype(dtype).itemsize * width
    with path.open("r+b") as file:
        file.seek(corpus_rows * row_bytes)
        identity = file.read(identity_rows * row_bytes)
        end = corpus_rows
        for index in range(identity_rows - 1, -1, -1):
            position = (2 * index + 1) * corpus_rows // (2 * identity_rows)
            while end > position:
                start = max(position, end - 1024)
                file.seek(start * row_bytes)
                block = file.read((end - start) * row_bytes)
                file.seek((start + index + 1) * row_bytes)
                file.write(block)
                end = start
            file.seek((position + index) * row_bytes)
            file.write(identity[index * row_bytes:(index + 1) * row_bytes])


def pad_sft_row(item, seq_length, pad_token_id):
    ids, mask, _, _ = item
    input_row = np.full(seq_length, pad_token_id, dtype="<u2")
    mask_row = np.zeros(seq_length, dtype="u1")
    input_row[:len(ids)] = ids
    mask_row[:len(ids)] = mask
    mask_row[0] = 0
    return input_row, mask_row


def record_sft_row(stats, item):
    ids, _, target_count, truncated = item
    stats["rows"] += 1
    stats["input_tokens"] += len(ids)
    stats["supervised_tokens"] += target_count
    stats["truncated_rows"] += int(truncated)
    stats["rows_over_8k"] += int(len(ids) > 8192)
    stats["max_input_length"] = max(stats["max_input_length"], len(ids))


def prepare_sft_sources(data_dir, tokenizer, template, tokens, seq_length, seed):
    if seq_length < 2 or not tokenizer.is_fast or tokenizer.pad_token_id is None:
        raise ValueError("Requires sequence length >= 2 and a fast tokenizer with PAD")
    folders = {"no_think": data_dir / "2605/no_think", "think": data_dir / "2605/think"}
    sources_by_group = {group: sorted(folder.rglob("*.jsonl")) for group, folder in folders.items()}
    agent_dir = data_dir / "2609"
    sources_by_group["agent"] = sorted({
        path for source in AGENT_SOURCES
        for path in (*agent_dir.glob(f"{source}*.jsonl"), *(agent_dir / source).rglob("*.jsonl"))
    })
    missing = [group for group, paths in sources_by_group.items() if not paths]
    if missing:
        raise FileNotFoundError(f"Missing JSONL sources for {', '.join(missing)} under {data_dir}")
    sources = [path for paths in sources_by_group.values() for path in paths]
    source_groups = [group for group, paths in sources_by_group.items() for _ in paths]

    identity_path = data_dir / "self/identity.jsonl"
    identity_lines = [line for line in identity_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    random.Random(seed).shuffle(identity_lines)
    identity = [encode(json.loads(line), tokenizer, template, "self", seq_length) for line in identity_lines]
    if len(identity) != IDENTITY_ROWS or any(reason for _, reason in identity):
        raise ValueError(f"Need {IDENTITY_ROWS} valid self rows that fit --seq-length. Run download_sft.py --state self.")
    identity = [item for item, _ in identity]
    self_tokens = sum(item[2] for item in identity)
    quotas = {group: tokens * percent // 100 for group, percent in MIX.items()}
    quotas["no_think"] += tokens - sum(quotas.values()) - self_tokens
    if min(quotas.values()) <= 0:
        raise ValueError(f"--tokens is too small to fit {self_tokens:,} identity tokens plus the requested mix")
    quotas["self"] = self_tokens
    stats = {group: dict(rows=0, input_tokens=0, supervised_tokens=0, target_tokens=cap,
                         truncated_rows=0, rows_over_8k=0, max_input_length=0, skipped={})
             for group, cap in quotas.items()}

    return sources, source_groups, identity, stats


def select_sft_rows(sources, source_groups, tokenizer, template, seq_length, seed, stats, *, write):
    """Apply the supervised-token quotas without storing token arrays."""
    data_budget = sum(stats[group]["target_tokens"] for group in MIX)
    pending = set(MIX)
    written = 0
    next_log = 1_000_000
    with closing(shuffled_rows(sources, seed)) as shuffled:
        for index, offset, raw in shuffled:
            group = source_groups[index]
            if group not in pending or not raw.strip():
                continue
            try:
                item, reason = encode(json.loads(raw), tokenizer, template, group, seq_length)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{sources[index]} at byte {offset}: {exc}") from exc
            if reason:
                skipped = stats[group]["skipped"]
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            remaining = stats[group]["target_tokens"] - stats[group]["supervised_tokens"]
            if item[2] <= remaining:
                write(index, offset, item)
                record_sft_row(stats[group], item)
                written += item[2]
                if written >= next_log:
                    print(f"[SFT data] {written:,}/{data_budget:,} supervised corpus tokens", flush=True)
                    next_log = (written // 1_000_000 + 1) * 1_000_000
            if item[2] == remaining:
                pending.remove(group)
                if not pending:
                    break
    exhausted = {group: stats[group]["target_tokens"] - stats[group]["supervised_tokens"] for group in sorted(pending)}
    for group, shortfall in exhausted.items():
        print(f"[Exhausted] {group}: {shortfall:,} supervised tokens short; "
              f"skipped={stats[group]['skipped']}", flush=True)
    if not written:
        raise ValueError("No usable corpus rows fit the sequence length and supervised-token budget")
    return exhausted


def compile_sft(data_dir, output_dir, tokenizer, *, tokens, seq_length, seed):
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    if seq_length < 2 or not tokenizer.is_fast or tokenizer.pad_token_id is None:
        raise ValueError("Requires sequence length >= 2 and a fast tokenizer with PAD")
    if len(tokenizer) > 65536 or max(tokenizer.get_vocab().values()) > 65535:
        raise ValueError("Vocabulary does not fit uint16")

    template = assistant_mask_template(tokenizer.get_chat_template())
    sources, source_groups, identity, stats = prepare_sft_sources(
        data_dir, tokenizer, template, tokens, seq_length, seed,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".sft-", dir=output_dir.parent) as directory:
        staging = Path(directory) / "compiled"
        staging.mkdir()
        with (staging / "input_ids.bin").open("wb") as inputs, (staging / "loss_mask.bin").open("wb") as masks:
            def write(item):
                input_row, mask_row = pad_sft_row(item, seq_length, tokenizer.pad_token_id)
                inputs.write(input_row.tobytes())
                masks.write(mask_row.tobytes())

            exhausted = select_sft_rows(
                sources, source_groups, tokenizer, template, seq_length, seed, stats,
                write=lambda _source, _offset, item: write(item),
            )
            for item in identity:
                write(item)
                record_sft_row(stats["self"], item)

        rows = sum(group["rows"] for group in stats.values())
        for filename, dtype in (("input_ids.bin", "<u2"), ("loss_mask.bin", "u1")):
            interleave_identity(staging / filename, dtype, seq_length, rows - len(identity), len(identity))
        supervised_tokens = sum(group["supervised_tokens"] for group in stats.values())
        manifest = dict(
            layout="rainpocket_sft_fixed", sequence_length=seq_length, rows=rows, seed=seed,
            tokenizer_vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id,
            target_supervised_tokens=tokens,
            supervised_tokens=supervised_tokens, supervised_token_shortfall=tokens - supervised_tokens,
            exhausted_groups=exhausted,
            input_tokens=sum(group["input_tokens"] for group in stats.values()),
            truncated_rows=sum(group["truncated_rows"] for group in stats.values()),
            rows_over_8k=sum(group["rows_over_8k"] for group in stats.values()),
            max_input_length=max(group["max_input_length"] for group in stats.values()),
            padded_positions=rows * seq_length - sum(group["input_tokens"] for group in stats.values()),
            files={"input_ids": "input_ids.bin", "loss_mask": "loss_mask.bin"},
            dtypes={"input_ids": "<u2", "loss_mask": "u1"}, groups=stats,
        )
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(output_dir)
    print(f"[Done] {rows:,} rows, {manifest['supervised_tokens']:,} supervised tokens -> {output_dir}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/sft")
    parser.add_argument("--output-dir", type=Path, help="compiled output directory (default data/sft/compiled_4k_trunc at 4096 tokens)")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "minirain_tokenizer")
    parser.add_argument("--tokens", type=token_budget, default=200_000_000, help="supervised token cap (default 200M)")
    parser.add_argument("--seq-length", type=int, default=DEFAULT_SEQUENCE_LENGTH,
                        help="truncate or right-pad to this length (default %(default)s)")
    parser.add_argument("--seed", type=int, default=42, help="document shuffle seed")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = ROOT / "data/sft" / ("compiled_4k_trunc" if args.seq_length == DEFAULT_SEQUENCE_LENGTH else "compiled")
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer.resolve()), use_fast=True, local_files_only=True)
    try:
        compile_sft(args.data_dir, output_dir, tokenizer,
                    tokens=args.tokens, seq_length=args.seq_length, seed=args.seed)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"SFT compile failed: {exc}\n")


if __name__ == "__main__":
    main()
