#!/usr/bin/env python3
"""Compile P2 bilingual image-caption pairs and produce exact tokenizer/bucket statistics.

Training-ready metadata/index, not a replacement for the P2 multimodal collator.
Image pixels stay in source files; text token IDs are stored in a packed uint32 binary.

Dependencies: transformers, pillow, numpy, tqdm (optional).
"""
import argparse
import json
import math
import os
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoTokenizer

BUCKETS = [(256, 256), (384, 384), (512, 384), (384, 512)]  # (width, height)
CJK = re.compile(r'[\u3400-\u9fff]')
LATIN = re.compile(r'[A-Za-z]')


def language(text):
    zh = len(CJK.findall(text))
    en = len(LATIN.findall(text))
    if zh and en:
        # A few English letters in a Chinese caption should not classify all as mixed.
        if zh >= en * 2:
            return 'zh_dominant'
        if en >= zh * 2:
            return 'en_dominant'
        return 'mixed'
    if zh:
        return 'zh_dominant'
    if en:
        return 'en_dominant'
    return 'other'


def choose_bucket(w, h):
    # Compare geometry and area in log-space; preserve horizontal/vertical orientation.
    a = w / h
    area = w * h
    return min(BUCKETS, key=lambda b: (
        2.0 * abs(math.log((b[0] / b[1]) / a))
        + abs(math.log((b[0] * b[1]) / area))
    ))


def image_size(path):
    with Image.open(path) as im:
        w, h = im.size
        im.verify()
    if w < 32 or h < 32:
        raise ValueError('image too small')
    return w, h


def flush(items, tokenizer, fp, out, stats, by_source, by_lang, buckets):
    if not items:
        return
    enc = tokenizer([it['caption'] for it in items], add_special_tokens=False,
                    padding=False, truncation=False)['input_ids']
    for it, ids in zip(items, enc):
        if not ids:
            stats['empty_tokenization'] += 1
            continue
        offset = fp.tell() // 4
        np.asarray(ids, dtype='<u4').tofile(fp)
        bucket = it['bucket']
        vis = math.ceil(bucket[0] / 16) * math.ceil(bucket[1] / 16)
        length = len(ids)
        rec = {
            'image': it['image'],
            'source': it['source'],
            'split': it['split'],
            'language': it['lang'],
            'original_size': [it['w'], it['h']],
            'bucket_size': [bucket[0], bucket[1]],
            'vision_tokens': vis,
            'text_offset_u32': offset,
            'text_tokens': length,
            'caption': it['caption'] if it['keep_caption'] else None,
        }
        if not it['keep_caption']:
            del rec['caption']
        out.write(json.dumps(rec, ensure_ascii=False) + '\n')
        stats['pairs'] += 1
        stats['vision_tokens'] += vis
        stats['text_tokens'] += length
        stats['total_tokens_no_special'] += vis + length
        stats['text_chars'] += len(it['caption'])
        stats['text_min'] = min(stats['text_min'], length)
        stats['text_max'] = max(stats['text_max'], length)
        for d in (by_source[it['source']], by_lang[it['lang']]):
            d['pairs'] += 1
            d['vision_tokens'] += vis
            d['text_tokens'] += length
        buckets[f'{bucket[0]}x{bucket[1]}']['pairs'] += 1
        buckets[f'{bucket[0]}x{bucket[1]}']['vision_tokens'] += vis
    items.clear()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('data/image_text'))
    p.add_argument('--tokenizer', required=True, help='P1-Long tokenizer directory or HF repo')
    p.add_argument('--out', type=Path, default=Path('data/p2_compiled'))
    p.add_argument('--files', nargs='+', default=[
        'pairs_allava_parquet.jsonl', 'pairs_flickr.jsonl',
        'pairs_muge.jsonl', 'pairs_cococn.jsonl'])
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--max-items', type=int, default=0, help='per file, for smoke tests')
    p.add_argument('--target-tokens', type=int, default=1_500_000_000)
    p.add_argument('--keep-caption', action='store_true')
    p.add_argument('--allow-missing', action='store_true', help='skip missing images (count errors)')
    args = p.parse_args()
    if args.batch < 1:
        p.error('--batch must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, trust_remote_code=False)
    report = {'tokenizer': str(args.tokenizer), 'root': str(args.root.resolve()),
              'target_tokens': args.target_tokens, 'buckets': BUCKETS,
              'special_tokens_excluded': True, 'vision_stride': 16,
              'splits': {}}
    for file in args.files:
        path = args.root / file
        if not path.exists():
            raise FileNotFoundError(path)
        # Explicitly isolate existing validation data from training.
        handles = {}
        accum = {}
        source = defaultdict(Counter)
        langs = defaultdict(Counter)
        bucket_stats = defaultdict(Counter)
        stats = Counter(pairs=0, vision_tokens=0, text_tokens=0,
                        total_tokens_no_special=0, text_chars=0,
                        empty_caption=0, empty_tokenization=0,
                        invalid_json=0, missing_image=0, bad_image=0)
        stats['text_min'] = 10**12
        stem = path.stem
        try:
            with path.open('r', encoding='utf-8') as f:
                for ix, line in enumerate(f):
                    if args.max_items and ix >= args.max_items:
                        break
                    try:
                        d = json.loads(line)
                        image = str(d['image'])
                        caption = d.get('caption') or d.get('text') or ''
                        if not isinstance(caption, str):
                            caption = str(caption)
                        caption = caption.strip()
                        if not caption:
                            stats['empty_caption'] += 1
                            continue
                        im_path = Path(image)
                        if not im_path.is_absolute():
                            im_path = args.root / im_path
                        if not im_path.is_file():
                            stats['missing_image'] += 1
                            if not args.allow_missing:
                                raise FileNotFoundError(im_path)
                            continue
                        try:
                            w, h = image_size(im_path)
                        except Exception:
                            stats['bad_image'] += 1
                            continue
                        # Keep separate bin / manifest per original split.
                        split = str(d.get('split', 'train')).lower()
                        if split in ('val', 'validation', 'valid', 'dev'):
                            split = 'valid'
                        elif split != 'test':
                            split = 'train'
                        key = f'{stem}.{split}'
                        if key not in handles:
                            handles[key] = (
                                (args.out / (key + '.u32.bin')).open('wb'),
                                (args.out / (key + '.records.jsonl')).open('w', encoding='utf-8'))
                            accum[key] = []
                        buf = accum[key]
                        buf.append(dict(image=str(im_path.resolve()), caption=caption,
                                        source=str(d.get('source', stem)), split=split,
                                        lang=language(caption), w=w, h=h,
                                        bucket=choose_bucket(w, h), keep_caption=args.keep_caption))
                        if len(buf) >= args.batch:
                            flush(buf, tokenizer, *handles[key], stats, source, langs, bucket_stats)
                    except (ValueError, KeyError, json.JSONDecodeError) as e:
                        stats['invalid_json'] += 1
                        continue
                    if (ix + 1) % 100_000 == 0:
                        print(f'[{file}] processed={ix+1:,} compiled={stats["pairs"]:,}', flush=True)
            for key, buf in accum.items():
                flush(buf, tokenizer, *handles[key], stats, source, langs, bucket_stats)
        finally:
            for b, j in handles.values():
                b.close()
                j.close()
        if not stats['pairs']:
            stats['text_min'] = 0
        report['splits'][stem] = {
            'totals': dict(stats),
            'by_source': {k: dict(v) for k, v in source.items()},
            'by_language': {k: dict(v) for k, v in langs.items()},
            'by_bucket': {k: dict(v) for k, v in bucket_stats.items()},
        }
        print(f'[{stem}] pairs={stats["pairs"]:,}, visual={stats["vision_tokens"]:,}, '
              f'text={stats["text_tokens"]:,}, missing={stats["missing_image"]:,}')
    train = Counter()
    val = Counter()
    by_language = defaultdict(Counter)
    by_bucket = defaultdict(Counter)
    for group in report['splits'].values():
        # Split stats are aggregated below by records, since a file can mix splits.
        for lang, cnt in group['by_language'].items():
            by_language[lang].update(cnt)
        for buck, cnt in group['by_bucket'].items():
            by_bucket[buck].update(cnt)
    # Derive exact split counts from manifest files (no tokenizer rerun).
    split_stats = defaultdict(Counter)
    for manifest in args.out.glob('*.records.jsonl'):
        if not any(manifest.name.startswith(Path(x).stem + '.') for x in args.files):
            continue
        with manifest.open(encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                c = split_stats[d['split']]
                c['pairs'] += 1
                c['vision_tokens'] += d['vision_tokens']
                c['text_tokens'] += d['text_tokens']
    report['by_split'] = {k: dict(v) for k, v in split_stats.items()}
    report['by_language'] = {k: dict(v) for k, v in by_language.items()}
    report['by_bucket'] = {k: dict(v) for k, v in by_bucket.items()}
    # Budget is a train-only estimate; the P2 loader should count specials separately.
    train_tokens = (split_stats['train']['vision_tokens'] + split_stats['train']['text_tokens'])
    report['train_tokens_per_epoch_no_special'] = train_tokens
    report['train_epochs_to_target_estimate'] = round(args.target_tokens / train_tokens, 4) if train_tokens else None
    target = args.out / 'stats.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nTRAIN per epoch (no specials): {train_tokens:,} tokens')
    print(f'Target {args.target_tokens:,}: {report["train_epochs_to_target_estimate"]} epochs')
    print(f'Stats: {target}')
    print('NOTE: records are source-specific; P2 loader still needs to insert image/special embeddings, modality mask, labels.')


if __name__ == '__main__':
    main()
