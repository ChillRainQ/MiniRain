#!/usr/bin/env python3
"""Extend existing UltraData-Code Python and Shell downloads via HF-Mirror.

Default final targets (decimal GB): Python 7.2 GB and sh 2.4 GB,
each split 50/50 between Code-L2 and Code-L3.

This script ONLY handles code_l2_py, code_l3_py, code_l2_sh, code_l3_sh.
It leaves all other JSONL/state files and manifest.json untouched.

Example:
    python extend_ultradata_py_sh.py --out /path/to/existing/ultradata
    python extend_ultradata_py_sh.py --out /path/to/existing/ultradata --dry-run
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '30')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')

import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass

GB = 1_000_000_000
REPO = 'openbmb/UltraData-Code'
REVISION = '8ce985cfd4522832c8b986e05968c87fffa0c66a'

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "ultradata"

@dataclass(frozen=True)
class Job:
    name: str
    config: str
    split: str
    target: int


def atomic_json(path: Path, data: dict):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def text_of(value):
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def extract_text(row, job):
    # Keep the same extraction priority as the original downloader.
    if job.config == 'UltraData-Code-L3':
        for key in ('full_content', 'content', 'raw_content'):
            value = text_of(row.get(key))
            if value.strip():
                return value
        parts = []
        for key in ('task', 'analysis', 'solution', 'test'):
            value = text_of(row.get(key))
            if value.strip():
                parts.append(f'## {key.upper()}\n{value}')
        if parts:
            return '\n\n'.join(parts)
    for key in ('content', 'text', 'full_content', 'raw_content', 'body',
                'messages', 'conversations', 'conversation', 'question',
                'problem', 'analysis', 'solution', 'answer', 'response'):
        value = text_of(row.get(key))
        if value.strip():
            return value
    return json.dumps(row, ensure_ascii=False, default=str) if row else ''


def encode_row(row, job, min_chars):
    text = extract_text(row, job)
    if len(text.strip()) < min_chars:
        return None
    keys = ('uuid', 'source', 'url', 'score', 'quality_score', 'algo_rel_score',
            'repo_name', 'relative_path', 'category', 'content_format',
            'full_content_format', 'style', 'uid')
    obj = {
        'text': text, 'dataset': REPO, 'config': job.config,
        'split': job.split, 'language': job.split,
        'metadata': {k: row[k] for k in keys if row.get(k) is not None},
    }
    return (json.dumps(obj, ensure_ascii=False, default=str) + '\n').encode('utf-8')


def signature(job):
    return {
        'repo': REPO, 'config': job.config, 'split': job.split,
        'revision': REVISION, 'stream_order': 'sequential-v1',
    }


def prepare(job, directory):
    data_path = directory / f'{job.name}.jsonl'
    state_path = directory / f'{job.name}.state.json'
    if data_path.exists() != state_path.exists():
        raise RuntimeError(f'{job.name}: JSONL/state 文件缺失一个，拒绝覆盖。')
    if not data_path.exists():
        return data_path, state_path, {
            **signature(job), 'source_rows': 0, 'records': 0,
            'bytes': 0, 'target_bytes': job.target,
            'exhausted': False, 'limit_reached': False,
        }, False
    with state_path.open('r', encoding='utf-8') as f:
        state = json.load(f)
    for key, expected in signature(job).items():
        if state.get(key) != expected:
            raise RuntimeError(
                f'{job.name}: state 的 {key} 与本脚本不一致，拒绝续传。'
                f'（当前={state.get(key)!r}，要求={expected!r}）'
            )
    for key in ('source_rows', 'records', 'bytes'):
        if key not in state or int(state[key]) < 0:
            raise RuntimeError(f'{job.name}: 无效状态字段 {key}')
    size = data_path.stat().st_size
    if size < int(state['bytes']):
        raise RuntimeError(f'{job.name}: JSONL 比 checkpoint 小，拒绝续传。')
    if int(state['bytes']) > job.target:
        raise RuntimeError(f'{job.name}: 已有容量超过新目标，不自动截断。')
    return data_path, state_path, state, True


def download(job, out, args):
    # Import only when a real network download is needed.
    data_path, state_path, state, exists = prepare(job, out)
    source_rows = int(state['source_rows'])
    records = int(state['records'])
    bytes_done = int(state['bytes'])
    prior_target = int(state.get('target_bytes', job.target))
    exhausted = bool(state.get('exhausted', False))
    limit_reached = bool(state.get('limit_reached', False))
    extra = (data_path.stat().st_size - bytes_done) if exists else 0
    print(f'[PLAN] {job.name:12s} {bytes_done/GB:.3f}/{job.target/GB:.3f} GB'
          f'  remaining={max(0,job.target-bytes_done)/GB:.3f} GB'
          f'  rows={source_rows:,}', flush=True)
    if args.dry_run:
        if extra:
            print(f'       尚未写入 checkpoint 的尾部数据：{extra} bytes（正式运行将回滚该部分）')
        return {'name': job.name, 'target_bytes': job.target, 'downloaded_bytes': bytes_done,
                'records': records, 'source_rows': source_rows, 'exhausted': exhausted}

    if not exists:
        with data_path.open('xb'):
            pass
        atomic_json(state_path, state)
        print(f'[NEW] {job.name}', flush=True)
    elif extra:
        # Checkpoint owns the committed byte boundary.
        with data_path.open('r+b') as f:
            f.truncate(bytes_done)
        print(f'[RECOVER] {job.name}: 丢弃 {extra} 字节未提交尾部', flush=True)

    if exhausted:
        print(f'[WARN] {job.name}: 之前已标记数据源耗尽，无法继续。', flush=True)
        return {'name': job.name, 'target_bytes': job.target, 'downloaded_bytes': bytes_done,
                'records': records, 'source_rows': source_rows, 'exhausted': True}
    if bytes_done >= job.target or (limit_reached and prior_target == job.target):
        print(f'[SKIP] {job.name}: 已达到本次配额', flush=True)
        return {'name': job.name, 'target_bytes': job.target, 'downloaded_bytes': bytes_done,
                'records': records, 'source_rows': source_rows, 'exhausted': False}

    from datasets import load_dataset
    print(f'[LOAD] {job.name}: {REPO}/{job.config}/{job.split}', flush=True)
    ds = load_dataset(REPO, name=job.config, split=job.split, streaming=True,
                      revision=REVISION, token=False)
    print(f'[LOAD] features={ds.features}', flush=True)
    if source_rows:
        print(f'[RESUME] skip {source_rows:,} source rows', flush=True)
        ds = ds.skip(source_rows)

    started = time.monotonic()
    start_bytes = bytes_done
    last_progress = started
    stop_by_quota = False
    end_of_source = False

    def checkpoint(f, *, limit=False, exhausted=False):
        f.flush()
        os.fsync(f.fileno())
        atomic_json(state_path, {
            **signature(job), 'source_rows': source_rows, 'records': records,
            'bytes': bytes_done, 'target_bytes': job.target,
            'exhausted': exhausted, 'limit_reached': limit,
        })

    with data_path.open('r+b') as f:
        f.seek(bytes_done)
        try:
            for row in ds:
                raw = encode_row(row, job, args.min_chars)
                if raw is not None:
                    if bytes_done + len(raw) > job.target:
                        stop_by_quota = True
                        break  # Don't increment source_rows: retry this row if quota grows.
                    f.write(raw)
                    bytes_done += len(raw)
                    records += 1
                source_rows += 1
                if source_rows % args.save_every_rows == 0:
                    checkpoint(f)
                now = time.monotonic()
                if now - last_progress >= 5:
                    speed = (bytes_done - start_bytes) / max(now - started, .001) / 1048576
                    print(f'[PROGRESS] {job.name}: {bytes_done/GB:.3f}/{job.target/GB:.3f} GB'
                          f' | {records:,} records | {speed:.2f} MiB/s', flush=True)
                    last_progress = now
                if bytes_done >= job.target:
                    stop_by_quota = True
                    break
            else:
                end_of_source = True
        except BaseException:
            checkpoint(f)
            raise
        checkpoint(f, limit=stop_by_quota, exhausted=end_of_source)

    print(f'[DONE] {job.name}: {bytes_done/GB:.3f}/{job.target/GB:.3f} GB', flush=True)
    if end_of_source and bytes_done < job.target:
        print(f'[WARN] {job.name}: 数据源已耗尽，未能达到指定容量', flush=True)
    return {'name': job.name, 'target_bytes': job.target, 'downloaded_bytes': bytes_done,
            'records': records, 'source_rows': source_rows, 'exhausted': end_of_source}


def main():

    parser = argparse.ArgumentParser(description='Only extend Python and Shell UltraData-Code parts')
    parser.add_argument('--out', type=Path, required=True,
                        help='Existing directory containing code_l2_py.jsonl and *.state.json')
    parser.add_argument('--python-gb', type=float, default=7.2,
                        help='Final total Python code size, L2+L3, decimal GB (default 7.2)')
    parser.add_argument('--shell-gb', type=float, default=2.4,
                        help='Final total sh code size, L2+L3, decimal GB (default 2.4)')
    parser.add_argument('--save-every-rows', type=int, default=1000)
    parser.add_argument('--min-chars', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    output_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else DEFAULT_OUTPUT_DIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.python_gb < 0 or args.shell_gb < 0 or args.save_every_rows <= 0 or args.min_chars < 0:
        parser.error('容量和参数不得为负，save-every-rows 必须大于零')
    out = args.out.expanduser().resolve()
    if not out.is_dir():
        parser.error(f'目录不存在: {out}；请填写原有数据的实际目录')
    py_bytes = round(args.python_gb * GB)
    sh_bytes = round(args.shell_gb * GB)
    jobs = [
        Job('code_l2_py', 'UltraData-Code-L2', 'py', py_bytes // 2),
        Job('code_l3_py', 'UltraData-Code-L3', 'py', py_bytes - py_bytes // 2),
        Job('code_l2_sh', 'UltraData-Code-L2', 'sh', sh_bytes // 2),
        Job('code_l3_sh', 'UltraData-Code-L3', 'sh', sh_bytes - sh_bytes // 2),
    ]
    print(f'[MIRROR] {os.environ["HF_ENDPOINT"]}', flush=True)
    print(f'[OUTPUT] {out}', flush=True)
    print(f'[TARGET] Python={args.python_gb:.3f} GB, sh={args.shell_gb:.3f} GB', flush=True)
    results = []
    for job in jobs:
        if job.target == 0:
            continue
        result = download(job, out, args)
        results.append(result)
        if not args.dry_run:
            atomic_json(out / 'manifest_extend_py_sh.json', {
                'source': os.environ['HF_ENDPOINT'],
                'description': 'Only Python and Shell code extensions; existing manifest.json unchanged',
                'unit': 'decimal GB',
                'revisions': {REPO: REVISION},
                'results': results,
                'total_target_bytes': py_bytes + sh_bytes,
                'total_saved_bytes': sum(r['downloaded_bytes'] for r in results),
                'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            })
    print('[FINISHED] All selected code jobs checked.', flush=True)


if __name__ == '__main__':
    main()
