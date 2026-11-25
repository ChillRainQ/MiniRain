"""Evaluate MiniRain checkpoints on CLUE ChID with lm-eval."""

import argparse
import ast
import json
import re
import sys
from functools import partial
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


def load_original_test(path, **_):
    """Fill each original-ChID blank independently and remove other blanks."""
    documents = []
    with Path(path).open(encoding='utf-8-sig') as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = ast.literal_eval(line)
            parts = row['content'].split('#idiom#')
            answers, candidates = row['groundTruth'], row['candidates']
            if len(parts) - 1 != len(answers) or len(answers) != len(candidates):
                raise ValueError(f'{path}:{line_number}: inconsistent blank, answer and candidate counts')
            for index, (answer, choices) in enumerate(zip(answers, candidates)):
                left = ''.join(parts[:index + 1])
                right = ''.join(parts[index + 1:])
                documents.append(dict(id=f'{line_number}:{index}', prefix='',
                                      choices=[left + idiom + right for idiom in choices],
                                      answer=choices.index(answer)))
    if not documents:
        raise ValueError(f'No ChID blanks found in {path}')
    return DatasetDict(test=Dataset.from_list(documents))


def score_passage_perplexity(doc, results, *, tokenizer):
    scores = []
    for passage, (logprob, _) in zip(doc['choices'], results, strict=True):
        tokens = tokenizer.encode(passage, add_special_tokens=False)
        # Match lm-eval's empty-context path if the passage already begins with BOS.
        length = len(tokens) - int(tokens[0] == tokenizer.bos_token_id)
        scores.append(logprob / length)
    return {'acc': int(max(range(len(scores)), key=scores.__getitem__) == doc['answer'])}


def make_task(data_dir, original_test=None, tokenizer=None):
    config = dict(task='minirain_chid_dev', custom_dataset=load_chid,
                  dataset_kwargs=dict(data_dir=str(data_dir)), validation_split='validation')
    if original_test is not None:
        config = dict(task='minirain_chid_original_test', custom_dataset=load_original_test,
                      dataset_kwargs=dict(path=str(original_test)), test_split='test',
                      process_results=partial(score_passage_perplexity, tokenizer=tokenizer))
    return ConfigurableTask(config=dict(
        **config,
        output_type='multiple_choice', doc_to_text='prefix', doc_to_choice='choices',
        doc_to_target='answer', target_delimiter='', num_fewshot=0,
        metric_list=[dict(metric='acc', aggregation='mean', higher_is_better=True)],
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', required=True, nargs='+', type=Path)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/chid')
    parser.add_argument('--original-test', type=Path, help='Original ChID test_data.txt with embedded answers')
    parser.add_argument('--tokenizer', default=TOKENIZER_PATH)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--limit', type=int, help='Evaluate only N blanks for a smoke test')
    parser.add_argument('--output', type=Path, help='Save metrics and per-question results to JSON')
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error('--batch-size and --limit must be positive')
    data_paths = ([args.original_test] if args.original_test is not None else
                  [args.data_dir / filename for filename in ('dev.json', 'dev_answer.json')])
    for path in data_paths:
        if not path.is_file():
            parser.error(f'Missing {path}; see bench/README.md for the ChID download')

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    task = make_task(args.data_dir, args.original_test, tokenizer)
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
        dataset_path = args.original_test if args.original_test is not None else args.data_dir
        scoring = ('per-token loglikelihood of the filled passage' if args.original_test is not None else
                   'conditional loglikelihood of candidate plus right context')
        report = dict(dataset=str(dataset_path.resolve()), limit=args.limit, num_fewshot=0,
                      scoring=scoring, checkpoints=reports)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
