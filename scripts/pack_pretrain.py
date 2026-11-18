#!/usr/bin/env python3
"""Pack unified MiniRain train/validation BINs separately into 8K blocks.

Preserves the original token stream byte for byte, including BOS/EOS.
Blocks may cross document boundaries; no tokenization or disk padding.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def inspect_source(src_bin, src_idx):
    src_bin, src_idx = Path(src_bin), Path(src_idx)
    nbytes, ibytes = src_bin.stat().st_size, src_idx.stat().st_size
    if nbytes < 4 or nbytes % 2 or ibytes < 16 or ibytes % 8:
        raise ValueError(f'Invalid uint16 BIN / uint64 IDX sizes: {src_bin}, {src_idx}')
    offsets = np.memmap(src_idx, dtype='<u8', mode='r')
    try:
        if int(offsets[0]) != 0 or int(offsets[-1]) != nbytes // 2:
            raise ValueError('Source BIN/IDX token offsets do not match')
        for start in range(0, len(offsets)-1, 1_000_000):
            stop = min(start+1_000_000, len(offsets)-1)
            left, right = offsets[start:stop], offsets[start+1:stop+1]
            if np.any(right <= left) or np.any(right-left < 2):
                raise ValueError('Source IDX must contain increasing records of at least two tokens')
        return dict(records=len(offsets)-1, stored_tokens=nbytes//2)
    finally:
        offsets._mmap.close()


def pack(src_bin, src_idx, dst_bin, dst_idx, seq_len=8192, *, overwrite=False,
         publish=True, source_info=None):
    """Copy tokens with bounded memory; write block offsets in vectorized batches."""
    if seq_len < 3:
        raise ValueError('seq_len must be >= 3')
    src_bin, src_idx, dst_bin, dst_idx = map(Path, (src_bin,src_idx,dst_bin,dst_idx))
    if dst_bin.resolve() == dst_idx.resolve():
        raise ValueError('BIN and IDX output paths must be different')
    for target in (dst_bin,dst_idx):
        if target.resolve() in (src_bin.resolve(),src_idx.resolve()):
            raise ValueError('Refusing to overwrite source data')
        if target.exists() and not overwrite:
            raise FileExistsError(target)
        target.parent.mkdir(parents=True,exist_ok=True)
    info = inspect_source(src_bin,src_idx) if source_info is None else source_info
    total = info['stored_tokens']
    blocks = math.ceil(total/seq_len)
    rebalance = total % seq_len == 1
    temporary = [Path(str(path)+'.tmp') for path in (dst_bin,dst_idx)]
    if any(path.exists() for path in temporary):
        raise FileExistsError('Packing temporary files exist; inspect them before retrying')
    owned = []
    started = time.monotonic()
    try:
        with temporary[0].open('xb') as output:
            owned.append(temporary[0])
            with src_bin.open('rb') as source:
                copied = 0
                next_log = 1 << 30
                while chunk := source.read(16 << 20):
                    output.write(chunk)
                    copied += len(chunk)
                    if copied >= next_log:
                        elapsed = time.monotonic()-started
                        print(f'  {dst_bin.name}: {copied/(total*2):.1%} '
                              f'{copied/2**30:.2f}/{total*2/2**30:.2f} GiB '
                              f'{copied/2**20/max(elapsed,1e-6):.0f} MiB/s',flush=True)
                        next_log = copied + (1 << 30)
            output.flush()
            os.fsync(output.fileno())
        with temporary[1].open('xb') as output:
            owned.append(temporary[1])
            for start in range(0, blocks+1, 1_000_000):
                stop = min(start+1_000_000, blocks+1)
                offsets = np.arange(start,stop,dtype='<u8') * seq_len
                np.minimum(offsets,total,out=offsets)
                if rebalance and start <= blocks-1 < stop:
                    # Keep the final singleton without dropping or duplicating it.
                    # The penultimate block has L-1 tokens, the final block has 2.
                    offsets[blocks-1-start] = total-2
                offsets.tofile(output)
            output.flush()
            os.fsync(output.fileno())
        if temporary[0].stat().st_size != total*2 or temporary[1].stat().st_size != (blocks+1)*8:
            raise RuntimeError('Packed output sizes do not match source token count')
        if publish:
            for temporary_path,target in zip(temporary,(dst_bin,dst_idx)):
                os.replace(temporary_path,target)
    except BaseException:
        for path in owned:
            path.unlink(missing_ok=True)
        raise
    return dict(bin=dst_bin.name,idx=dst_idx.name,source_records=info['records'],
        source_tokens=total,records=blocks,stored_tokens=total,
        packed_blocks=blocks,packed_tokens=total,sequence_length=seq_len,
        last_block_length=2 if rebalance else (total-1)%seq_len+1,
        single_token_tail_rebalanced=rebalance,bin_bytes=total*2,idx_bytes=(blocks+1)*8,
        document_boundaries='BOS/EOS preserved',cross_document_attention=True)


def pack_corpus(data_dir, output_dir=None, seq_len=8192, *, overwrite=False):
    data_dir = Path(data_dir).resolve()
    output_dir = (data_dir/'packed' if output_dir is None else Path(output_dir).resolve())
    if output_dir == data_dir:
        raise ValueError('Use a separate output directory to preserve the source manifest')
    if seq_len < 3:
        raise ValueError('seq_len must be >= 3')
    source_manifest = data_dir/'manifest.json'
    original = json.loads(source_manifest.read_text(encoding='utf-8'))
    if original.get('layout') != 'unified_text':
        raise ValueError('Expected the unified text compiler manifest, not a stage/packed manifest')
    plans = []
    inputs = set()
    for split in ('dataset','validation'):
        if split not in original:
            continue
        meta = original[split]
        names = [meta['bin'],meta['idx']]
        if any(Path(name).name != name or name in ('.','..') for name in names):
            raise ValueError('Source manifest must name files within data-dir')
        src_bin,src_idx = (data_dir/name for name in names)
        if src_bin.resolve() in inputs or src_idx.resolve() in inputs:
            raise ValueError('Train and validation must use distinct source files')
        inputs.update((src_bin.resolve(),src_idx.resolve()))
        info = inspect_source(src_bin,src_idx)
        if info != dict(records=meta['records'],stored_tokens=meta['stored_tokens']):
            raise ValueError(f'{split} source counts do not match manifest')
        prefix = src_bin.stem+'_packed'
        plans.append((split,src_bin,src_idx,output_dir/f'{prefix}.bin',output_dir/f'{prefix}.idx',info))
    output_dir.mkdir(parents=True,exist_ok=True)
    targets = [output_dir/'manifest.json'] + [p for plan in plans for p in plan[3:5]]
    if len(set(path.resolve() for path in targets)) != len(targets):
        raise ValueError('Packed output names collide')
    if any(path.resolve() in inputs for path in targets):
        raise ValueError('Packed outputs cannot overwrite source data')
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f'Packed output exists; use --overwrite: {existing}')
    temporary = [Path(str(path)+'.tmp') for path in targets]
    if any(path.exists() for path in temporary):
        raise FileExistsError('Packing temporary files exist; inspect them before retrying')
    # Do not leave an old validation set behind when rebuilding without one.
    if 'validation' not in original and (output_dir/'manifest.json').exists():
        old = json.loads((output_dir/'manifest.json').read_text(encoding='utf-8'))
        if 'validation' in old:
            raise ValueError('Use a new output directory when removing validation')
    manifest = {key:original[key] for key in ('project','tokenizer_dir','vocab_size','special_tokens','split') if key in original}
    manifest.update(layout='unified_text_packed',max_length=seq_len,
        updated_at=datetime.now(timezone.utc).isoformat(),source_dir=str(data_dir),
        source_manifest_sha256=hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        format=dict(bin='uint16 little-endian',idx='uint64 little-endian',
                    offset_unit='tokens',padding='not stored',cross_document_attention=True,
                    token_stream='unchanged',state_reset='at packed block boundaries'))
    try:
        for split,src_bin,src_idx,dst_bin,dst_idx,info in plans:
            print(f'[PACK] {split}: {info["records"]:,} source records; {info["stored_tokens"]:,} tokens',flush=True)
            result = pack(src_bin,src_idx,dst_bin,dst_idx,seq_len,overwrite=overwrite,publish=False,source_info=info)
            result['source_documents'] = original[split].get('documents',info['records'])
            result['actual_text_tokens'] = original[split].get('actual_text_tokens')
            manifest[split] = result
            print(f'  -> {result["packed_blocks"]:,} packed blocks; tail={result["last_block_length"]}',flush=True)
        with temporary[0].open('x',encoding='utf-8') as output:
            json.dump(manifest,output,ensure_ascii=False,indent=2)
            output.flush()
            os.fsync(output.fileno())
        # Both splits complete before publication; the shared manifest goes last.
        for target in targets[1:]+targets[:1]:
            os.replace(Path(str(target)+'.tmp'),target)
    except BaseException:
        for path in temporary:
            path.unlink(missing_ok=True)
        raise
    print(f'[DONE] {output_dir}',flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/pretrain_all')
    parser.add_argument('--output-dir',type=Path,default=None,help='Default: <data-dir>/packed')
    parser.add_argument('--seq-len',type=int,default=8192)
    parser.add_argument('--overwrite',action='store_true')
    args = parser.parse_args()
    if args.seq_len < 3:
        parser.error('seq-len must be >= 3')
    pack_corpus(args.data_dir,args.output_dir,args.seq_len,overwrite=args.overwrite)


if __name__ == '__main__':
    main()
