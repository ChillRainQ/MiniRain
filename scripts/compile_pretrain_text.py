#!/usr/bin/env python3
"""Compile every JSONL text into one uint16 BIN / uint64 IDX pair.

No stage quotas. Split long documents without discarding tails. Each record
has BOS/EOS and at most --max-length tokens; padding is not stored.
"""
from array import array
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'true')
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / 'data/ultradata'
DEFAULT_TOKENIZER = ROOT / 'minirain_tokenizer'
DEFAULT_OUTPUT_DIR = ROOT / 'data/pretrain_all'
OUTPUT_NAMES = ('pretrain_text.bin', 'pretrain_text.idx', 'manifest.json')


def discover_sources(data_dir):
    data_dir = Path(data_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    sources = sorted(p for p in data_dir.rglob('*.jsonl') if p.is_file())
    if not sources:
        raise FileNotFoundError(f'No JSONL files under {data_dir}')
    return sources


class BinWriter:
    def __init__(self, output_dir, flush_tokens):
        self.bin_tmp = output_dir / 'pretrain_text.bin.tmp'
        self.idx_tmp = output_dir / 'pretrain_text.idx.tmp'
        self.bin_file = self.bin_tmp.open('xb')
        try:
            self.idx_file = self.idx_tmp.open('xb')
        except BaseException:
            self.bin_file.close()
            self.bin_tmp.unlink()
            raise
        self.tokens = array('H')
        self.offsets = array('Q', [0])
        self.flush_tokens = flush_tokens
        self.records = self.stored_tokens = 0

    def add(self, body, bos, eos):
        if body and (min(body) < 0 or max(body) > 65535):
            raise ValueError('Token IDs exceed uint16 capacity')
        self.tokens.extend([bos, *body, eos])
        self.stored_tokens += len(body) + 2
        self.records += 1
        self.offsets.append(self.stored_tokens)
        if len(self.tokens) >= self.flush_tokens:
            self.tokens.tofile(self.bin_file)
            self.tokens = array('H')
        if len(self.offsets) >= 100_000:
            self.offsets.tofile(self.idx_file)
            self.offsets = array('Q')

    def finalize(self):
        self.tokens.tofile(self.bin_file)
        self.offsets.tofile(self.idx_file)
        for handle in (self.bin_file, self.idx_file):
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()

    def abort(self):
        for handle in (self.bin_file, self.idx_file):
            handle.close()
        for path in (self.bin_tmp, self.idx_tmp):
            path.unlink(missing_ok=True)


def compile_corpus(data_dir, output_dir, tokenizer, *, tokenizer_dir=None,
                   max_length=8192, batch_size=128, flush_tokens=1_000_000,
                   text_field='text', skip_invalid=False, overwrite=False,
                   log_every=100_000):
    if max_length < 3 or min(batch_size, flush_tokens, log_every) <= 0:
        raise ValueError('max_length must be >= 3; batch/flush/log sizes must be positive')
    if sys.byteorder != 'little' or array('H').itemsize != 2 or array('Q').itemsize != 8:
        raise RuntimeError('Requires little-endian uint16 / uint64 arrays')
    special = {f'{name}_token_id': getattr(tokenizer, f'{name}_token_id')
               for name in ('bos', 'eos', 'pad')}
    if any(value is None or not 0 <= value <= 65535 for value in special.values()):
        raise ValueError('BOS/EOS/PAD IDs must exist and fit uint16')
    vocab_ids = tokenizer.get_vocab().values()
    if not vocab_ids or min(vocab_ids) < 0 or max(vocab_ids) > 65535:
        raise ValueError('Tokenizer vocabulary exceeds uint16 capacity')
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    sources = discover_sources(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in OUTPUT_NAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f'Output already exists; use --overwrite: {existing}')
    temporary = [output_dir / (name + '.tmp') for name in OUTPUT_NAMES]
    if any(path.exists() for path in temporary):
        raise FileExistsError('Compiler temporary files exist; inspect them before retrying')
    writer = BinWriter(output_dir, flush_tokens)
    start = time.monotonic()
    stats = {}
    total_documents = total_text_tokens = 0
    next_log = log_every
    max_body = max_length - 2
    try:
        for file_index, source in enumerate(sources, 1):
            name = source.relative_to(data_dir).as_posix()
            print(f'[{file_index}/{len(sources)}] {name}', flush=True)
            current = dict(lines_read=0, documents=0, records=0, text_tokens=0,
                           split_documents=0, empty_text=0, blank_lines=0, invalid_records=0)
            batch = []

            def process_batch():
                nonlocal total_documents, total_text_tokens, next_log
                if not batch:
                    return
                encoded = tokenizer(batch, add_special_tokens=False, truncation=False,
                                    padding=False, return_attention_mask=False,
                                    return_token_type_ids=False)['input_ids']
                for body in encoded:
                    chunks = max(1, (len(body) + max_body - 1) // max_body)
                    for chunk in range(chunks):
                        writer.add(body[chunk*max_body:(chunk+1)*max_body],
                                   special['bos_token_id'], special['eos_token_id'])
                    current['documents'] += 1
                    current['records'] += chunks
                    current['text_tokens'] += len(body)
                    current['split_documents'] += int(chunks > 1)
                    total_documents += 1
                    total_text_tokens += len(body)
                batch.clear()
                if total_documents >= next_log:
                    elapsed = time.monotonic() - start
                    print(f'  documents={total_documents:,} records={writer.records:,} '
                          f'text_tokens={total_text_tokens:,} '
                          f'{total_text_tokens/max(elapsed, 1e-6):,.0f} tok/s '
                          f'elapsed={elapsed/60:.1f} min', flush=True)
                    next_log = (total_documents // log_every + 1) * log_every

            with source.open('r', encoding='utf-8-sig') as handle:
                for line_no, line in enumerate(handle, 1):
                    current['lines_read'] += 1
                    if not line.strip():
                        current['blank_lines'] += 1
                        continue
                    try:
                        obj = json.loads(line)
                        if not isinstance(obj, dict) or not isinstance(obj.get(text_field), str):
                            raise ValueError(f'Expected string field {text_field!r}')
                    except ValueError as exc:
                        current['invalid_records'] += 1
                        if not skip_invalid:
                            raise ValueError(f'{source}:{line_no}: {exc}') from exc
                        continue
                    text = obj[text_field]
                    if not text.strip():
                        current['empty_text'] += 1
                        continue
                    batch.append(text)
                    if len(batch) >= batch_size:
                        process_batch()
            process_batch()
            stats[name] = current
            print(f'  documents={current["documents"]:,} records={current["records"]:,} '
                  f'text_tokens={current["text_tokens"]:,} '
                  f'invalid={current["invalid_records"]:,} empty={current["empty_text"]:,}', flush=True)
        if writer.records == 0:
            raise ValueError('No nonempty text records found; no dataset published')
        writer.finalize()
        manifest = dict(
            project='MiniRain', layout='unified_text',
            updated_at=datetime.now(timezone.utc).isoformat(),
            data_dir=str(data_dir), tokenizer_dir=str(tokenizer_dir) if tokenizer_dir else None,
            vocab_size=len(tokenizer), max_length=max_length, max_body_tokens=max_body,
            special_tokens=special,
            format=dict(bin='uint16 little-endian', idx='uint64 little-endian',
                        offset_count='records + 1', offset_unit='tokens', padding='not stored',
                        document_concatenation=False, long_documents='split_without_truncation'),
            dataset=dict(bin=OUTPUT_NAMES[0], idx=OUTPUT_NAMES[1], documents=total_documents,
                         records=writer.records, actual_text_tokens=total_text_tokens,
                         stored_tokens=writer.stored_tokens, bin_bytes=writer.stored_tokens*2,
                         idx_bytes=(writer.records+1)*8),
            text_field=text_field, skip_invalid=skip_invalid, sources=stats)
        if writer.bin_tmp.stat().st_size != writer.stored_tokens*2 or writer.idx_tmp.stat().st_size != (writer.records+1)*8:
            raise RuntimeError('BIN/IDX size validation failed')
        with temporary[2].open('x', encoding='utf-8') as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        # Finish processing before publication; publish manifest last.
        for temporary_path, name in zip(temporary, OUTPUT_NAMES):
            os.replace(temporary_path, output_dir / name)
    except BaseException:
        writer.abort()
        temporary[2].unlink(missing_ok=True)
        raise
    print(f'[DONE] documents={total_documents:,} records={writer.records:,} '
          f'text_tokens={total_text_tokens:,} stored_tokens={writer.stored_tokens:,}\n'
          f'BIN: {output_dir / OUTPUT_NAMES[0]}\nIDX: {output_dir / OUTPUT_NAMES[1]}', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--tokenizer', type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--max-length', type=int, default=8192, help='Maximum stored length including BOS/EOS')
    parser.add_argument('--batch-size', type=int, default=128, help='Tokenizer batch size')
    parser.add_argument('--flush-tokens', type=int, default=1_000_000)
    parser.add_argument('--log-every', type=int, default=100_000, help='Progress interval in source documents')
    parser.add_argument('--text-field', default='text')
    parser.add_argument('--skip-invalid', action='store_true', help='Skip malformed/non-text rows and count them')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='List sources without tokenization or writes')
    args = parser.parse_args()
    if args.max_length < 3 or min(args.batch_size, args.flush_tokens, args.log_every) <= 0:
        parser.error('max-length must be >= 3; batch/flush/log sizes must be positive')
    sources = discover_sources(args.data_dir)
    print(f'Source files: {len(sources)}; output: {args.output_dir.resolve()}', flush=True)
    if args.dry_run:
        for source in sources:
            print(source)
        return
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer.resolve()),
                                              use_fast=True, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError('Fast tokenizer required')
    compile_corpus(args.data_dir, args.output_dir, tokenizer,
                   tokenizer_dir=args.tokenizer.resolve(), max_length=args.max_length,
                   batch_size=args.batch_size, flush_tokens=args.flush_tokens,
                   text_field=args.text_field, skip_invalid=args.skip_invalid,
                   overwrite=args.overwrite, log_every=args.log_every)


if __name__ == '__main__':
    main()
