"""Evaluate MiniRain pretraining checkpoints on fixed Chinese continuations."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer

from cli.eval import build_model, find_weight, read_checkpoint
from project import TOKENIZER_PATH


def read_cases(path):
    cases = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        fields = ('id', 'prefix', 'reference', 'repeat')
        if not isinstance(case, dict) or any(
            not isinstance(case.get(key), str) or not case[key].strip() for key in fields
        ):
            raise ValueError(f'{path}:{line_number} requires nonempty strings: {fields}')
        if case['id'] in seen or case['reference'] == case['repeat']:
            raise ValueError(f'{path}:{line_number} has a duplicate id or identical candidates')
        seen.add(case['id'])
        cases.append(case)
    if not cases:
        raise ValueError(f'No benchmark cases in {path}')
    return cases


def first_loop(tokens):
    """Find the earliest run of three identical adjacent spans, 4–32 tokens each."""
    for start in range(len(tokens)):
        for width in range(4, min(32, (len(tokens) - start) // 3) + 1):
            unit = tokens[start:start + width]
            if unit == tokens[start + width:start + 2 * width] == tokens[start + 2 * width:start + 3 * width]:
                return {'start_token': start + 1, 'span_tokens': width}
    return None


def score_suffix(model, prefix, suffix, device):
    ids = torch.tensor([prefix + suffix], device=device)
    labels = ids.clone()
    labels[:, :len(prefix)] = -100
    # MiniRain shifts labels internally. The last prefix logit scores suffix[0].
    mean_nll = model(ids, labels=labels, logits_to_keep=1, use_cache=False).loss.item()
    if not math.isfinite(mean_nll):
        raise FloatingPointError('Nonfinite continuation loss')
    return {'tokens': len(suffix), 'mean_nll': mean_nll}


@torch.inference_mode()
def evaluate(model, tokenizer, cases, device, max_new_tokens):
    results = []
    for case in cases:
        # Encode candidates separately so both get exactly the same prefix IDs.
        prefix = [tokenizer.bos_token_id] + tokenizer.encode(case['prefix'], add_special_tokens=False)
        reference = tokenizer.encode(case['reference'], add_special_tokens=False)
        repeat = tokenizer.encode(case['repeat'], add_special_tokens=False)
        if not reference or not repeat or reference == repeat:
            raise ValueError(f'{case["id"]}: candidates must encode to distinct, nonempty token lists')
        max_length = len(prefix) + max(len(reference), len(repeat), max_new_tokens)
        if max_length > model.config.max_position_embeddings:
            raise ValueError(f'{case["id"]}: {max_length} tokens exceed the model context')
        good = score_suffix(model, prefix, reference, device)
        bad = score_suffix(model, prefix, repeat, device)
        ids = torch.tensor([prefix], device=device)
        generated = model.generate_hf(ids, attention_mask=torch.ones_like(ids),
                                      max_new_tokens=max_new_tokens, do_sample=False,
                                      eos_token_id=tokenizer.eos_token_id, use_cache=True)
        tokens = generated[0, len(prefix):].tolist()
        ended = bool(tokens and tokens[-1] == tokenizer.eos_token_id)
        if ended:
            tokens.pop()
        results.append(dict(case, reference_score=good, repeat_score=bad,
                            reference_margin=bad['mean_nll'] - good['mean_nll'],
                            generated_text=tokenizer.decode(tokens, skip_special_tokens=True),
                            generated_tokens=len(tokens), ended_with_eos=ended, loop=first_loop(tokens)))

    count = len(results)
    total_tokens = sum(x['reference_score']['tokens'] for x in results)
    total_nll = sum(x['reference_score']['mean_nll'] * x['reference_score']['tokens'] for x in results)
    mean_nll = total_nll / total_tokens
    summary = dict(cases=count, reference_tokens=total_tokens, reference_mean_nll=mean_nll,
                   reference_ppl=math.exp(mean_nll),
                   preferred_reference=sum(x['reference_margin'] > 0 for x in results),
                   ties=sum(x['reference_margin'] == 0 for x in results),
                   mean_reference_margin=sum(x['reference_margin'] for x in results) / count,
                   loop_cases=sum(x['loop'] is not None for x in results),
                   eos_cases=sum(x['ended_with_eos'] for x in results))
    return summary, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', required=True, nargs='+', type=Path, help='Checkpoint files or experiment directories')
    parser.add_argument('--cases', type=Path, default=Path(__file__).with_name('zh_continuation.jsonl'))
    parser.add_argument('--tokenizer', default=TOKENIZER_PATH)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', choices=['bf16', 'fp16', 'fp32'])
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--output', type=Path, help='Write summary and per-case results as JSON')
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error('--max-new-tokens must be positive')
    cases = read_cases(args.cases)
    device = torch.device(args.device)
    dtype = args.dtype or ('bf16' if device.type == 'cuda' else 'fp32')
    torch_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    report = dict(cases_file=str(args.cases.resolve()),
                  cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                  tokenizer=args.tokenizer, device=str(device), dtype=dtype,
                  decode='greedy', max_new_tokens=args.max_new_tokens, checkpoints=[])
    for path in args.ckpt:
        checkpoint = find_weight(path).resolve()
        config, state = read_checkpoint(checkpoint, None)
        model = build_model(config, state, device, torch_dtype)
        del state
        if len(tokenizer) != model.config.vocab_size or any(
            getattr(tokenizer, key) is None or getattr(tokenizer, key) != getattr(model.config, key)
            for key in ('bos_token_id', 'eos_token_id')
        ):
            raise ValueError(f'{checkpoint}: tokenizer vocab/BOS/EOS do not match the model')
        summary, details = evaluate(model, tokenizer, cases, device, args.max_new_tokens)
        report['checkpoints'].append(dict(checkpoint=str(checkpoint), summary=summary, cases=details))
        print(f'[{checkpoint}] reference PPL={summary["reference_ppl"]:.3f} | '
              f'reference preferred={summary["preferred_reference"]}/{len(cases)} | '
              f'loops={summary["loop_cases"]}/{len(cases)} | EOS={summary["eos_cases"]}/{len(cases)}')
        if args.output is None:
            for case in details:
                print(f'  {case["id"]} | margin={case["reference_margin"]:+.3f} | loop={case["loop"]}')
                print(f'    {case["generated_text"]!r}')
        del model

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(f'[Report] {args.output.resolve()}')


if __name__ == '__main__':
    main()
