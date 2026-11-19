"""Evaluate MiniRain checkpoints on CLUE ChID with lm-eval."""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from datasets import Dataset, DatasetDict
from lm_eval import simple_evaluate
from lm_eval.api.task import ConfigurableTask
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from transformers import AutoTokenizer

from cli.eval import build_model, find_weight, read_checkpoint
from project import TOKENIZER_PATH

BLANK = re.compile(r'#idiom\d+#')


class ChIDModel(HFLM):
    def _encode_pair(self, context, continuation):
        # Chinese text can merge across the blank boundary under BPE. Encode
        # each side separately so every candidate conditions on the same IDs.
        prefix = [self.prefix_token_id] + self.tok_encode(context, add_special_tokens=False)
        suffix = self.tok_encode(continuation, add_special_tokens=False)
        return prefix, suffix


def load_chid(data_dir, **_):
    """Convert CLUE passages to one multiple-choice document per blank."""
    data_dir = Path(data_dir)
    answers = json.loads((data_dir / 'dev_answer.json').read_text(encoding='utf-8-sig'))
    documents = []
    for line in (data_dir / 'dev.json').read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for passage in row['content']:
            for blank in BLANK.finditer(passage):
                tag = blank.group()
                target = answers[tag]
                if not isinstance(target, int) or not 0 <= target < len(row['candidates']):
                    raise ValueError(f'{tag}: invalid candidate index {target!r}')
                left = BLANK.sub('〔空缺〕', passage[:blank.start()])
                right = BLANK.sub('〔空缺〕', passage[blank.end():])
                documents.append(dict(id=tag, prefix=left,
                                      choices=[idiom + right for idiom in row['candidates']],
                                      answer=target))
    if not documents:
        raise ValueError(f'No ChID blanks found in {data_dir / "dev.json"}')
    return DatasetDict(validation=Dataset.from_list(documents))


def make_task(data_dir):
    return ConfigurableTask(config=dict(
        task='minirain_chid_dev', custom_dataset=load_chid,
        dataset_kwargs=dict(data_dir=str(data_dir)), validation_split='validation',
        output_type='multiple_choice', doc_to_text='prefix', doc_to_choice='choices',
        doc_to_target='answer', target_delimiter='', num_fewshot=0,
        metric_list=[dict(metric='acc', aggregation='mean', higher_is_better=True)],
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', required=True, nargs='+', type=Path)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/chid')
    parser.add_argument('--tokenizer', default=TOKENIZER_PATH)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--limit', type=int, help='Evaluate only N blanks for a smoke test')
    parser.add_argument('--output', type=Path, help='Save metrics and per-question results to JSON')
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error('--batch-size and --limit must be positive')
    for filename in ('dev.json', 'dev_answer.json'):
        if not (args.data_dir / filename).is_file():
            parser.error(f'Missing {args.data_dir / filename}; see bench/README.md for the CLUE ChID download')

    task = make_task(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    reports = []
    for path in args.ckpt:
        checkpoint = find_weight(path).resolve()
        config, state = read_checkpoint(checkpoint, None)
        model = build_model(config, state, device, dtype)
        del state
        if len(tokenizer) != model.config.vocab_size or tokenizer.bos_token_id != model.config.bos_token_id:
            raise ValueError(f'{checkpoint}: tokenizer vocab or BOS does not match the model')
        harness = ChIDModel(pretrained=model, tokenizer=tokenizer, backend='causal',
                            batch_size=args.batch_size, prefix_token_id=tokenizer.bos_token_id,
                            max_length=model.config.max_position_embeddings)
        results = simple_evaluate(model=harness, tasks=[task], limit=args.limit,
                                  num_fewshot=0, log_samples=args.output is not None,
                                  bootstrap_iters=0)
        print(f'[{checkpoint}]\n{make_table(results)}')
        if args.output is not None:
            reports.append(dict(checkpoint=str(checkpoint), results=results['results'],
                                n_samples=results['n-samples'], samples=results['samples']))
        del harness, model

    if args.output is not None:
        report = dict(dataset=str(args.data_dir.resolve()), limit=args.limit, num_fewshot=0,
                      scoring='conditional loglikelihood of candidate plus right context', checkpoints=reports)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
