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
DEFAULT_SEQUENCE_LENGTH = 16_384
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
    if len(ids) > seq_length:
        return None, "too_long"
    targets = sum(mask[1:])
    if not targets:
        return None, "no_targets"
    return (ids, mask, targets), None


def compile_sft(data_dir, output_dir, tokenizer, *, tokens, seq_length, seed):
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    if seq_length < 2 or not tokenizer.is_fast or tokenizer.pad_token_id is None:
        raise ValueError("Requires sequence length >= 2 and a fast tokenizer with PAD")
    if len(tokenizer) > 65536 or max(tokenizer.get_vocab().values()) > 65535:
        raise ValueError("Vocabulary does not fit uint16")

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

    template = assistant_mask_template(tokenizer.get_chat_template())
    identity_path = data_dir / "self/identity.jsonl"
    identity_lines = [line for line in identity_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    random.Random(seed).shuffle(identity_lines)
    identity = [encode(json.loads(line), tokenizer, template, "self", seq_length) for line in identity_lines]
    if len(identity) != IDENTITY_ROWS or any(reason for _, reason in identity):
        raise ValueError(f"Need {IDENTITY_ROWS} valid self rows that fit --seq-length. Run download_sft.py --state self.")
    identity = [item for item, _ in identity]
    self_tokens = sum(item[2] for item in identity)
    quotas = {group: tokens * percent // 100 for group, percent in MIX.items()}
    quotas["no_think"] += tokens - sum(quotas.values())
    quotas["no_think"] -= self_tokens
    if quotas["no_think"] <= 0 or min(quotas.values()) <= 0:
        raise ValueError(f"--tokens is too small to fit {self_tokens:,} identity tokens plus the requested mix")
    quotas["self"] = self_tokens
    data_budget = tokens - self_tokens
    stats = {group: dict(rows=0, input_tokens=0, supervised_tokens=0, target_tokens=cap,
                         rows_over_8k=0, max_input_length=0, skipped={})
             for group, cap in quotas.items()}

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".sft-", dir=output_dir.parent) as directory:
        staging = Path(directory) / "compiled"
        staging.mkdir()
        with (staging / "input_ids.bin").open("wb") as inputs, (staging / "loss_mask.bin").open("wb") as masks:
            def write(group, item):
                ids, mask, target_count = item
                input_row = np.full(seq_length, tokenizer.pad_token_id, dtype="<u2")
                mask_row = np.zeros(seq_length, dtype="u1")
                input_row[:len(ids)] = ids
                mask_row[:len(ids)] = mask
                mask_row[0] = 0
                inputs.write(input_row.tobytes())
                masks.write(mask_row.tobytes())
                stats[group]["rows"] += 1
                stats[group]["input_tokens"] += len(ids)
                stats[group]["supervised_tokens"] += target_count
                stats[group]["rows_over_8k"] += int(len(ids) > 8192)
                stats[group]["max_input_length"] = max(stats[group]["max_input_length"], len(ids))

            pending = set(MIX)
            written = identity_index = 0
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
                    remaining = quotas[group] - stats[group]["supervised_tokens"]
                    if item[2] <= remaining:
                        write(group, item)
                        written += item[2]
                        while (identity_index < len(identity)
                               and 2 * written * len(identity) >= (2 * identity_index + 1) * data_budget):
                            write("self", identity[identity_index])
                            identity_index += 1
                        if written >= next_log:
                            print(f"[Compile] {written:,}/{data_budget:,} supervised corpus tokens", flush=True)
                            next_log = (written // 1_000_000 + 1) * 1_000_000
                    if item[2] >= remaining:
                        pending.remove(group)
                        if not pending:
                            break
            exhausted = {group: quotas[group] - stats[group]["supervised_tokens"] for group in sorted(pending)}
            for group, shortfall in exhausted.items():
                print(f"[Exhausted] {group}: {shortfall:,} supervised tokens short; "
                      f"skipped={stats[group]['skipped']}", flush=True)
            for item in identity[identity_index:]:
                write("self", item)

        rows = sum(group["rows"] for group in stats.values())
        supervised_tokens = sum(group["supervised_tokens"] for group in stats.values())
        manifest = dict(
            layout="rainpocket_sft_fixed", sequence_length=seq_length, rows=rows, seed=seed,
            tokenizer_vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id,
            target_supervised_tokens=tokens,
            supervised_tokens=supervised_tokens, supervised_token_shortfall=tokens - supervised_tokens,
            exhausted_groups=exhausted,
            input_tokens=sum(group["input_tokens"] for group in stats.values()),
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/sft/compiled")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "minirain_tokenizer")
    parser.add_argument("--tokens", type=token_budget, default=200_000_000, help="supervised token cap (default 200M)")
    parser.add_argument("--seq-length", type=int, default=DEFAULT_SEQUENCE_LENGTH,
                        help="fixed padded sequence length (default 16384)")
    parser.add_argument("--seed", type=int, default=42, help="document shuffle seed")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer.resolve()), use_fast=True, local_files_only=True)
    try:
        compile_sft(args.data_dir, args.output_dir, tokenizer,
                    tokens=args.tokens, seq_length=args.seq_length, seed=args.seed)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"SFT compile failed: {exc}\n")


if __name__ == "__main__":
    main()
