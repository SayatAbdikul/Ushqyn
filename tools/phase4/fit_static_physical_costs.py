#!/usr/bin/env python3
"""Fit a first latency model using only compiler-visible tile geometry."""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/'docs/research/evidence/phase4'
sys.path.insert(0, str(ROOT/'compiler'))
sys.path.insert(0, str(ROOT/'tools/phase4'))
from hardware_v2 import Descriptor
from fit_physical_costs import metrics, nonnegative_lstsq, sha

COMMON = {'conv', 'depthwise', 'relu'}


def static_counts(descriptors, regions):
    macs = 0
    pool_work = 0
    for d in descriptors:
        if d.opcode in (1, 4, 6):
            macs += d.count * d.outputs
        elif d.opcode in (5, 7):
            pool_work += d.outputs*d.kernel_h*d.kernel_w
    return {
        'static_macs': macs,
        'static_input_bytes': sum(r.get('input', {}).get('bytes', 0)
                                  for r in regions),
        'static_weight_bytes': sum(r.get('weight', {}).get('bytes', 0)
                                   for r in regions),
        'static_pool_work': pool_work,
    }


def rows():
    result = []
    for name in ('kws', 'vww'):
        report_path = EVIDENCE/f'physical-host-overlap-{name}.json'
        report = json.loads(report_path.read_text())
        plan_path = ROOT/f'work/phase4/{name}-tiled-plan.json'
        plan = json.loads(plan_path.read_text())
        for node, layer in zip(report['nodes'], plan['layers']):
            if not layer['tiles']:
                continue
            descriptors = [Descriptor.decode(bytes.fromhex(t['descriptor_hex']))
                           for t in layer['tiles']]
            regions = [t['regions'] for t in layer['tiles']]
            result.append({
                'id': f'{name}:{node["node"]}', 'kind': node['kind'],
                'tiles': len(descriptors),
                'output_bytes': sum(d.outputs for d in descriptors),
                'cycles': node['engine_counters']['elapsed'],
                **static_counts(descriptors, regions),
            })
    directed = json.loads((EVIDENCE/'physical-kernel-profiles-overlap.json').read_text())
    kind_by_opcode = {1:'gemm', 4:'conv', 5:'maxpool', 6:'depthwise',
                      7:'avgpool', 8:'clip'}
    for case in directed['cases']:
        d = Descriptor(**case['descriptor'])
        region = {
            'input': {'bytes': (d.input_h*d.input_w*d.input_c if
                                d.opcode in (4,5,6,7) else d.count)},
            'weight': {'bytes': (d.row_stride*d.output_c if
                                 d.opcode in (1,4,6) else 0)},
        }
        result.append({
            'id': 'directed:'+case['label'],
            'kind': kind_by_opcode[d.opcode], 'tiles': 1,
            'output_bytes': d.outputs,
            'cycles': case['runs'][0]['elapsed'],
            **static_counts([d], [region]),
        })
    return result


def fit(records):
    kinds = sorted({r['kind'] for r in records})
    for row in records:
        digest = hashlib.sha256(row['id'].encode()).digest()[0]
        row['split'] = ('validation' if row['kind'] in COMMON and
                        digest % 5 == 0 else 'fit')

    def feature(row):
        return ([row['tiles']] +
                [float(row['kind'] == kind)*row['output_bytes']/10000
                 for kind in kinds] +
                [row['static_macs']/1e6,
                 row['static_input_bytes']/1e4,
                 row['static_weight_bytes']/1e4,
                 row['static_pool_work']/1e5])

    names = ['tile_count']+[f'{k}:output_bytes/10000' for k in kinds]+[
        'macs/1e6', 'input_bytes/1e4', 'weight_bytes/1e4', 'pool_work/1e5']
    if len(names)>14:
        raise ValueError('feature family too large')

    def estimate(train, targets):
        x = np.array([feature(r) for r in train], float)
        y = np.array([r['cycles'] for r in train], float)
        weights = nonnegative_lstsq(x, y)
        predicted = [{**r, 'predicted_cycles': max(1., float(
            np.dot(feature(r), weights)))} for r in targets]
        return weights, predicted

    trained = [r for r in records if r['split']=='fit']
    validation = [r for r in records if r['split']=='validation']
    weights, train_predictions = estimate(trained, trained)
    _, hold_predictions = estimate(trained, validation)
    cross = {}
    for held in ('kws','vww'):
        _, predictions = estimate(
            [r for r in records if not r['id'].startswith(held+':')],
            [r for r in records if r['id'].startswith(held+':')])
        cross[held] = {'error': metrics(predictions),
                       'predictions': predictions}
    return {
        'feature_names': names, 'coefficients': weights.tolist(),
        'fit_error': metrics(train_predictions),
        'validation_error': metrics(hold_predictions),
        'validation_predictions': hold_predictions,
        'cross_model_validation': cross,
    }


def main():
    records = rows()
    route = json.loads((EVIDENCE/'physical-tiled-host-overlap-route.json').read_text())
    result = {
        'status': 'exploratory-compiler-static-latency-fit',
        'bitstream_sha256': route['bitstream_sha256'],
        'split_policy': ('SHA256(model:node) first byte modulo 5; common '
                         'kernel kinds only; explicit whole-model holdout'),
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (
            EVIDENCE/'physical-host-overlap-kws.json',
            EVIDENCE/'physical-host-overlap-vww.json',
            EVIDENCE/'physical-kernel-profiles-overlap.json',
            ROOT/'work/phase4/kws-tiled-plan.json',
            ROOT/'work/phase4/vww-tiled-plan.json')},
        'sample_count_by_kind': dict(Counter(r['kind'] for r in records)),
        'engine': fit(records),
        'limits': ['features are compiler-visible but coverage and cross-model '
                   'error must be judged before scheduler use',
                   'no measured energy or temperature dependence'],
    }
    output = EVIDENCE/'physical-static-cost-model.json'
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'validation': result['engine']['validation_error'],
                      'cross_model': {k:v['error'] for k,v in
                                      result['engine']['cross_model_validation'].items()}}))


if __name__ == '__main__':
    main()
