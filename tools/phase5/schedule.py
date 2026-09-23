#!/usr/bin/env python3
"""Build the pinned, alternating 10,000-job KWS/VWW switch stress plan.

This plans jobs only. It does not execute the FPGA or establish model accuracy.
Full-set accuracy must be evaluated separately on every pinned accuracy item.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 0x20_0005
JOBS_PER_MODEL = 5000


def load_split(path):
    raw = path.read_bytes()
    manifest = json.loads(raw)
    accuracy = manifest['splits']['accuracy']
    calibration = manifest['splits']['calibration']
    records = accuracy['records']
    if len(records) != accuracy['count'] or not records:
        raise ValueError(f'{path}: accuracy count differs from records')
    ids = [record['id'] for record in records]
    cal_ids = {record['id'] for record in calibration['records']}
    cal_features = {record['feature_sha256'] for record in calibration['records']}
    features = {record['feature_sha256'] for record in records}
    if len(ids) != len(set(ids)) or set(ids) & cal_ids or features & cal_features:
        raise ValueError(f'{path}: duplicate IDs or calibration/evaluation overlap')
    if len(cal_ids) != calibration['count']:
        raise ValueError(f'{path}: calibration count differs from unique records')
    if not all(len(record['feature_sha256']) == 64 for record in records):
        raise ValueError(f'{path}: missing pinned feature hashes')
    return manifest, hashlib.sha256(raw).hexdigest()


def draw_indices(count, jobs, seed):
    if count <= 0 or jobs <= 0:
        raise ValueError('positive sample and job counts required')
    rng = random.Random(seed)
    indices = []
    while len(indices) < jobs:
        block = list(range(count))
        rng.shuffle(block)
        indices.extend(block[:jobs - len(indices)])
    return indices


def build_jobs(splits, jobs_per_model=JOBS_PER_MODEL, seed=SEED):
    if set(splits) != {'kws', 'vww'}:
        raise ValueError('both pinned primary workloads required')
    selections = {
        name: draw_indices(len(splits[name]['splits']['accuracy']['records']),
                           jobs_per_model, seed + offset)
        for offset, name in enumerate(('kws', 'vww'))
    }
    for pair in range(jobs_per_model):
        for name in ('kws', 'vww'):
            index = selections[name][pair]
            record = splits[name]['splits']['accuracy']['records'][index]
            yield {
                'job': 2 * pair + (name == 'vww'),
                'workload': name,
                'sample_index': index,
                'sample_id': record['id'],
                'feature_sha256': record['feature_sha256'],
            }


def create_plan(manifest_dir=ROOT / 'benchmarks/manifests', output=None):
    splits = {}
    hashes = {}
    for name in ('kws', 'vww'):
        splits[name], hashes[name] = load_split(manifest_dir / f'{name}.data.json')
    jobs = list(build_jobs(splits))
    encoded = b''.join((json.dumps(job, sort_keys=True, separators=(',', ':')) + '\n').encode()
                       for job in jobs)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(encoded)
    counts = {name: splits[name]['splits']['accuracy']['count'] for name in splits}
    return {
        'schema': 1,
        'scope': 'job plan only; no accelerator execution',
        'seed': SEED,
        'jobs': len(jobs),
        'jobs_per_workload': JOBS_PER_MODEL,
        'ordering': 'strictly alternating kws,vww',
        'accuracy_split_counts': counts,
        'unique_stress_samples': {
            name: len({job['sample_id'] for job in jobs if job['workload'] == name})
            for name in splits
        },
        'data_manifest_sha256': hashes,
        'accuracy_split_npz_sha256': {
            name: splits[name]['splits']['accuracy']['npz_sha256'] for name in splits
        },
        'jsonl_sha256': hashlib.sha256(encoded).hexdigest(),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='write the full JSONL job plan')
    parser.add_argument('--summary', type=Path, help='write a compact pinned summary')
    args = parser.parse_args()
    report = create_plan(output=args.output)
    rendered = json.dumps(report, indent=2) + '\n'
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(rendered)
    print(rendered, end='')
