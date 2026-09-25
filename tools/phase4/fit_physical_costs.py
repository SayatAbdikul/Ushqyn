#!/usr/bin/env python3
"""Fit an explicitly held-out first physical engine/DMA latency model."""

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/'docs/research/evidence/phase4'
COMMON = {'conv', 'depthwise', 'relu'}
OP_KIND = {4: 'conv', 5: 'maxpool', 6: 'depthwise',
           7: 'avgpool', 8: 'clip'}
NUMERIC = ('useful_macs', 'read_bytes', 'write_bytes')
SCALE = np.array([1_000_000, 1_000_000, 100_000], float)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def engine_rows(label):
    rows = []
    for name in ('kws', 'vww'):
        report = json.loads((EVIDENCE/f'physical-host-{label}-{name}.json').read_text())
        fixture = json.loads((ROOT/f'work/phase4/rtl-{name}/manifest.json').read_text())
        for record, layer in zip(report['nodes'], fixture['layers']):
            if not record['tiles']:
                continue
            c = record['engine_counters']
            rows.append({'id': f'{name}:{record["node"]}',
                         'kind': record['kind'], 'tiles': record['tiles'],
                         'useful_macs': c['useful_macs'],
                         'read_bytes': c['read_bytes'],
                         'write_bytes': c['write_bytes'],
                         'output_bytes': layer['output_bytes'],
                         'cycles': c['elapsed']})
    directed = json.loads((EVIDENCE/f'physical-kernel-profiles-{label}.json').read_text())
    for case in directed['cases']:
        c = case['runs'][0]
        if any(run['elapsed'] != c['elapsed'] for run in case['runs']):
            raise ValueError(f'directed kernel cycles varied: {case["label"]}')
        descriptor = case['descriptor']
        rows.append({'id': f'directed:{case["label"]}',
                     'kind': OP_KIND[descriptor['opcode']], 'tiles': 1,
                     'useful_macs': c['useful_macs'],
                     'read_bytes': c['read_bytes'],
                     'write_bytes': c['write_bytes'],
                     'output_bytes': descriptor['outputs'],
                     'cycles': c['elapsed']})
    return rows


def metrics(rows):
    if not rows:
        return None
    errors = [abs(r['predicted_cycles']-r['cycles'])/r['cycles'] for r in rows]
    return {'samples': len(rows),
            'median_absolute_percent_error': 100*statistics.median(errors),
            'mean_absolute_percent_error': 100*statistics.mean(errors),
            'worst_absolute_percent_error': 100*max(errors)}


def nonnegative_lstsq(x, y):
    """Exact active-set search for this small, fixed feature family."""
    if x.shape[1] > 14:
        raise ValueError('too many cost features for exhaustive NNLS')
    best = None
    best_loss = math.inf
    for mask in range(1 << x.shape[1]):
        active = [j for j in range(x.shape[1]) if mask & (1 << j)]
        coefficients = np.zeros(x.shape[1])
        if active:
            solution = np.linalg.lstsq(x[:, active], y, rcond=None)[0]
            if np.any(solution < -1e-8):
                continue
            coefficients[active] = np.maximum(solution, 0)
        loss = float(np.sum((x@coefficients-y)**2))
        if loss < best_loss:
            best_loss, best = loss, coefficients
    return best


def fit_engine(rows):
    kinds = sorted({r['kind'] for r in rows})
    for row in rows:
        hashed = hashlib.sha256(row['id'].encode()).digest()[0]
        row['split'] = ('validation' if row['kind'] in COMMON and
                        hashed % 5 == 0 else 'fit')
    fit = [r for r in rows if r['split'] == 'fit']
    validation = [r for r in rows if r['split'] == 'validation']
    if not validation or any(not any(r['kind'] == kind for r in fit)
                             for kind in kinds):
        raise ValueError('invalid engine fit/validation split')

    def features(row):
        one_hot = [float(row['kind'] == kind)*row['output_bytes']/10_000
                   for kind in kinds]
        numeric = np.array([row[name] for name in NUMERIC], float)/SCALE
        return one_hot + numeric.tolist()

    x = np.array([features(r) for r in fit], float)
    y = np.array([r['cycles'] for r in fit], float)
    weights = nonnegative_lstsq(x, y)
    for row in rows:
        row['predicted_cycles'] = max(1.0, float(np.dot(features(row), weights)))
    cross_model = {}
    for held_name in ('kws', 'vww'):
        source = [r for r in rows if not r['id'].startswith(held_name+':')]
        target = [r for r in rows if r['id'].startswith(held_name+':')]
        other_weights = nonnegative_lstsq(
            np.array([features(r) for r in source], float),
            np.array([r['cycles'] for r in source], float))
        predictions = [{**r, 'predicted_cycles': max(
            1.0, float(np.dot(features(r), other_weights)))} for r in target]
        cross_model[held_name] = {'validation_error': metrics(predictions),
                                  'validation_predictions': predictions}
    return {'feature_names': [f'kind_output_bytes:{k}' for k in kinds]+list(NUMERIC),
            'numeric_scales': dict(zip(NUMERIC, SCALE.tolist())),
            'coefficients': weights.tolist(),
            'coefficient_constraint': 'all coefficients nonnegative',
            'fit_error': metrics(fit), 'validation_error': metrics(validation),
            'validation_predictions': [r for r in validation],
            'cross_model_validation': cross_model}


def fit_dma(records):
    rows = []
    held_lengths = {7, 9, 65, 256, 4096}
    for record in records:
        row = {'id': f'{record["direction"]}:{record["length_bytes"]}:'
                     f'{record["external_offset"]}:{record["repeat"]}',
               'direction': record['direction'],
               'length_bytes': record['length_bytes'],
               'external_offset': record['external_offset'],
               'beats': math.ceil(record['length_bytes']/8),
               'cycles': record['dma_cycles'],
               'split': 'validation' if record['length_bytes'] in held_lengths
                       else 'fit'}
        rows.append(row)
    models = {}
    for direction in ('to_sram', 'from_sram'):
        fit = [r for r in rows if r['direction'] == direction and r['split'] == 'fit']
        hold = [r for r in rows if r['direction'] == direction and r['split'] == 'validation']
        x = np.array([[1, r['beats']] for r in fit], float)
        y = np.array([r['cycles'] for r in fit], float)
        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
        for row in fit + hold:
            row['predicted_cycles'] = max(1.0, float(
                coefficients[0] + coefficients[1]*row['beats']))
        models[direction] = {
            'intercept_cycles': float(coefficients[0]),
            'cycles_per_8_byte_beat': float(coefficients[1]),
            'fit_error': metrics(fit), 'validation_error': metrics(hold),
            'validation_predictions': hold}
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', choices=('measured', 'overlap'),
                        default='measured')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = EVIDENCE/('physical-cost-model.json' if
                                args.label == 'measured' else
                                'physical-cost-model-overlap.json')
    engine = engine_rows(args.label)
    profile_path = EVIDENCE/f'physical-dma-profile-{args.label}.json'
    profile = json.loads(profile_path.read_text())
    route_name = ('physical-tiled-host-route.json' if args.label == 'measured'
                  else 'physical-tiled-host-overlap-route.json')
    route = json.loads((EVIDENCE/route_name).read_text())
    if (profile['bitstream_sha256'] != route['bitstream_sha256'] or
            any(json.loads((EVIDENCE/f'physical-host-{args.label}-{name}.json').read_text())
                ['physical_board'] is not True for name in ('kws', 'vww')) or
            json.loads((EVIDENCE/f'physical-kernel-profiles-{args.label}.json')
                       .read_text())['bitstream_sha256'] != route['bitstream_sha256']):
        raise ValueError('mixed bitstream or nonphysical input evidence')
    result = {
        'status': 'initial-physical-cost-fit',
        'bitstream_sha256': route['bitstream_sha256'],
        'split_policy': ('engine: SHA256(model:node) first byte modulo 5, '
                         'common kinds only; DMA: hold out transfer lengths '
                         '7, 9, 65, 256, 4096 bytes'),
        'source_sha256': {
            name: sha(EVIDENCE/name) for name in (
                f'physical-host-{args.label}-kws.json',
                f'physical-host-{args.label}-vww.json',
                f'physical-kernel-profiles-{args.label}.json',
                f'physical-dma-profile-{args.label}.json')},
        'engine_samples_by_kind': dict(Counter(r['kind'] for r in engine)),
        'engine': fit_engine(engine),
        'dma': fit_dma(profile['records']),
        'limits': ['engine features include device SRAM-traffic/MAC counters; '
                   'a compiler-only predictor still needs static feature estimates',
                   'rare kernel kinds lack independent validation samples',
                   'no measured board energy or temperature sweep'],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'engine_validation': result['engine']['validation_error'],
                      'engine_cross_model': {k:v['validation_error'] for k,v in
                                             result['engine']['cross_model_validation'].items()},
                      'dma_validation': {k:v['validation_error']
                                         for k,v in result['dma'].items()}},
                     sort_keys=True))


if __name__ == '__main__':
    main()
