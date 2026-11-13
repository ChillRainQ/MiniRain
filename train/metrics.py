"""Local, dependency-free training history and offline reports."""
import csv
import html
import json
import math
import os
import time
from pathlib import Path


class TrainingMetrics:
    FIELDS = ('step', 'epoch', 'next_batch', 'loss', 'lr', 'muon_lr',
              'grad_norm', 'grad_scale', 'tokens_seen', 'step_tokens',
              'tokens_per_sec', 'step_seconds', 'elapsed_seconds', 'eta_seconds',
              'cuda_allocated_mb_rank0', 'cuda_peak_mb_rank0')

    def __init__(self, output_dir, *, resume_step=None, metadata=None, config_dir=None):
        self.directory = Path(output_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'train.jsonl'
        records = []
        if self.path.exists():
            for line in self.path.read_text(encoding='utf8').splitlines():
                try:
                    record = json.loads(line)
                    if resume_step is not None and int(record['step']) <= resume_step:
                        records.append(record)
                except (ValueError, KeyError, TypeError):
                    # A crash may leave a partially written final line.
                    continue
            backup = self.path.with_name(f'train.before-resume-{time.time_ns()}.jsonl')
            os.replace(self.path, backup)
        self._atomic_text('train.jsonl', ''.join(json.dumps(r, allow_nan=False) + '\n'
                                                for r in records))
        self.elapsed_before = records[-1].get('elapsed_seconds', 0.) if records else 0.
        self.last_step = records[-1]['step'] if records else 0
        self.stream = self.path.open('a', encoding='utf8', buffering=1)
        if metadata is not None:
            directory = Path(config_dir) if config_dir is not None else self.directory
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / 'run_config.json'
            temporary = target.with_name(target.name + '.tmp')
            temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
                                            allow_nan=False), encoding='utf8')
            os.replace(temporary, target)
        self.export(status='running')

    def _atomic_text(self, filename, text):
        target = self.directory / filename
        temporary = target.with_name(target.name + '.tmp')
        temporary.write_text(text, encoding='utf8')
        os.replace(temporary, target)

    def log(self, record):
        if record['step'] <= self.last_step:
            raise ValueError('Metric steps must increase monotonically')
        self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        self.last_step = record['step']

    def export(self, *, status='running'):
        self.stream.flush()
        records = [json.loads(line) for line in self.path.read_text(encoding='utf8').splitlines()
                   if line.strip()]
        csv_path = self.directory / 'metrics.csv'
        with csv_path.with_suffix('.csv.tmp').open('w', newline='', encoding='utf8') as stream:
            writer = csv.DictWriter(stream, fieldnames=self.FIELDS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(records)
        os.replace(csv_path.with_suffix('.csv.tmp'), csv_path)
        summary = {'status': status, 'logged_steps': len(records),
                   'last': records[-1] if records else None,
                   'best_loss': min((r['loss'] for r in records), default=None)}
        self._atomic_text('summary.json', json.dumps(summary, ensure_ascii=False, indent=2,
                                                    allow_nan=False))
        charts = []
        for key in ('loss', 'lr', 'muon_lr', 'grad_norm', 'tokens_per_sec',
                    'step_seconds', 'tokens_seen', 'cuda_peak_mb_rank0'):
            points = [(r['step'], r[key]) for r in records
                      if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]
            if not points:
                continue
            # Bound HTML size while retaining the first and latest points.
            stride = max(1, math.ceil(len(points) / 1500))
            sampled = points[::stride]
            if sampled[-1] != points[-1]:
                sampled.append(points[-1])
            xmin, xmax = points[0][0], points[-1][0]
            ymin, ymax = min(y for _, y in points), max(y for _, y in points)
            coords = ' '.join(f'{55 + 620*(x-xmin)/max(1,xmax-xmin):.2f},'
                              f'{185 - 155*(y-ymin)/(ymax-ymin or 1):.2f}' for x, y in sampled)
            charts.append(f'<section><h2>{key}</h2><svg viewBox="0 0 720 225" role="img" '
                          f'aria-label="{key} versus step"><path d="M55 20V185H685" '
                          f'fill="none" stroke="#999"/><polyline points="{coords}" '
                          f'fill="none" stroke="#2474db" stroke-width="2"/>'
                          f'<text x="0" y="28">{ymax:.4g}</text>'
                          f'<text x="0" y="185">{ymin:.4g}</text>'
                          f'<text x="55" y="210">{xmin}</text>'
                          f'<text x="610" y="210">{xmax} step</text></svg></section>')
        self._atomic_text('training_report.html', '<!doctype html><meta charset="utf-8">'
                          '<title>Training report</title><style>body{font:15px system-ui;'
                          'max-width:1100px;margin:30px auto;padding:0 20px;background:#f5f7fa}'
                          'section{background:white;padding:16px;margin:16px 0;border-radius:12px}'
                          'svg{width:100%}text{font-size:12px}pre{white-space:pre-wrap}</style>'
                          '<h1>Training report</h1><p>Offline report · refreshes at checkpoints. '
                          'Loss is token-weighted per optimizer step; CUDA memory is rank 0 only. '
                          'ETA excludes checkpoint overhead.</p><pre>' +
                          html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) +
                          '</pre>' + ''.join(charts))

    def close(self):
        self.stream.close()
