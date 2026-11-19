"""Run lm-eval tasks directly on MiniRain pretraining checkpoints."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval.utils import handle_non_serializable, make_table
from transformers import AutoTokenizer

from cli.eval import build_model, find_weight, read_checkpoint
from project import TOKENIZER_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', required=True, nargs='+', type=Path)
    parser.add_argument('--tasks', required=True, nargs='+', help='lm-eval task or group names separated by spaces')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--tokenizer', default=TOKENIZER_PATH)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-fewshot', type=int, default=0)
    parser.add_argument('--limit', type=int, help='Evaluate at most N samples per task for a smoke test')
    parser.add_argument('--include-path', type=Path, help='Additional lm-eval task definitions')
    parser.add_argument('--output', type=Path, help='Save a JSON report')
    parser.add_argument('--log-samples', action='store_true', help='Include per-question results in the JSON report')
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_fewshot < 0 or (args.limit is not None and args.limit < 1):
        parser.error('--batch-size and --limit must be positive, --num-fewshot must be nonnegative')
    if args.log_samples and args.output is None:
        parser.error('--log-samples requires --output')

    manager = TaskManager(include_path=str(args.include_path) if args.include_path else None)
    checkpoint_paths = [find_weight(path).resolve() for path in args.ckpt]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    report = dict(tasks=args.tasks, num_fewshot=args.num_fewshot, limit=args.limit,
                  tokenizer=args.tokenizer, device=str(device), dtype=str(dtype), checkpoints=[])

    for checkpoint in checkpoint_paths:
        config, state = read_checkpoint(checkpoint, None)
        model = build_model(config, state, device, dtype)
        del state
        if len(tokenizer) != model.config.vocab_size or any(
            getattr(tokenizer, field) != getattr(model.config, field)
            for field in ('bos_token_id', 'eos_token_id')
        ):
            raise ValueError(f'{checkpoint}: tokenizer vocab or special tokens do not match the model')
        harness = HFLM(pretrained=model, tokenizer=tokenizer, backend='causal',
                       prefix_token_id=model.config.bos_token_id, batch_size=args.batch_size,
                       max_length=model.config.max_position_embeddings)
        results = simple_evaluate(model=harness, tasks=args.tasks, task_manager=manager,
                                  num_fewshot=args.num_fewshot, limit=args.limit,
                                  log_samples=args.log_samples)
        print(f'[{checkpoint}]\n{make_table(results)}', flush=True)
        if results.get('groups'):
            print(make_table(results, column='groups'), flush=True)
        if args.output:
            entry = dict(checkpoint=str(checkpoint), results=results['results'],
                         n_samples=results['n-samples'], versions=results['versions'])
            if results.get('groups'):
                entry['groups'] = results['groups']
            if args.log_samples:
                entry['samples'] = results['samples']
            report['checkpoints'].append(entry)
        del harness, model, results

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                          default=handle_non_serializable) + '\n', encoding='utf-8')
        print(f'[Report] {args.output.resolve()}')


if __name__ == '__main__':
    main()
