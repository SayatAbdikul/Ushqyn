#!/usr/bin/env python3
"""Reproduce isolated P6 software experiments; never opens a hardware device."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import math
import sys
import time

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'compiler'))
from integer_reference import evaluate
from static_pipeline import compile_static
from scheduler.abi_verify import replay
from scheduler.current_abi import CostModel, optimize
from scheduler.examples import disjoint_dma, quantized_chain
from scheduler.search import exact_search
from scheduler.spatial import execute_segment, supported_segments


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_model(name):
    paths = {k: ROOT/v for k, v in {
        'source_onnx_sha256': f'work/phase4/{name}-logits.onnx',
        'calibration_sha256': f'work/phase4/{name}-calibration-rebased.json',
        'input_sha256': f'work/phase4/rtl-{name}/input.bin',
        'expected_npz_sha256': f'work/phase4/rtl-{name}/expected.npz'}.items()}
    manifest_path = ROOT/f'work/phase4/rtl-{name}/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for key, path in paths.items():
        if digest(path.read_bytes()) != manifest[key]:
            raise ValueError(f'changed pinned {name} {key}')
    program = compile_static(onnx.load(paths['source_onnx_sha256']), json.loads(paths['calibration_sha256'].read_text()))
    data = np.frombuffer(paths['input_sha256'].read_bytes(), np.int8)
    if program.layers[0].op == 'Transpose':
        first = program.layers[0]
        data = data.reshape(program.tensors[first.output].shape).transpose(np.argsort(first.attributes['perm']))
    else:
        data = data.reshape(program.tensors[program.inputs[0]].shape)
    return program, data, manifest, {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in (*paths.values(), manifest_path)}


def run(output, models):
    output.mkdir(parents=True, exist_ok=True)
    cost_path = ROOT/'docs/research/evidence/phase4/physical-sequence-costs.json'
    model_costs = CostModel(json.loads(cost_path.read_text()))
    report = {'schema': 1, 'status': 'running', 'phase6_gate_complete': False,
              'scope': 'boardless semantics, command legality and restricted-catalogue optimization; no physical improvement claim',
              'host': {'system': platform.platform(), 'processor': platform.machine(), 'python': sys.version.split()[0]},
              'search_settings': {'beam_width': 8, 'max_expansions': 100000, 'seconds': 60,
                                  'tie_break': 'lexicographic candidate IDs', 'random_input_seed': 601},
              'cost_evidence_sha256': digest(cost_path.read_bytes()),
              'cost_calibration_bitstream_sha256': model_costs.record['bitstream_sha256'],
              'synthetic_exact': {}, 'models': {}}
    for name, factory in (('quantized_chain', quantized_chain), ('disjoint_dma', disjoint_dma)):
        problem, _ = factory()
        result = exact_search(problem, 12, max_expansions=1000000)
        if not result['optimal']:
            raise ValueError(f'exact fixture did not complete: {name}')
        result['certificate'] = asdict(result['certificate'])
        report['synthetic_exact'][name] = {'problem': asdict(problem), 'search': result}
    for name in models:
        print(f'{name}: compile, cost and optimize', flush=True)
        program, x, manifest, sources = load_model(name)
        began = time.monotonic()
        artifacts, optimization = optimize(program, model_costs)
        construction_seconds = time.monotonic()-began
        oracles = []
        for label, value in (('pinned-fixture', x), ('seeded-int8-stress', np.random.default_rng(601).integers(-128, 128, x.shape, dtype=np.int8))):
            oracle = evaluate(program, {program.inputs[0]: value})
            if label == 'pinned-fixture':
                for row in manifest['layers']:
                    if digest(oracle[program.layers[row['index']].output].tobytes()) != row['output_sha256']:
                        raise ValueError(f'{name} oracle changed at layer {row["index"]}')
            oracles.append((label, value, oracle))
        row = {'source_sha256': sources, 'candidate_construction_and_search_seconds': construction_seconds,
               'optimization': optimization, 'candidates': {}, 'spatial': []}
        for policy, artifact in artifacts.items():
            stem = f'{name}-{policy}'
            records = {}
            for suffix, data in (('commands.bin', artifact['commands']), ('payload.bin', artifact['payload']),
                                 ('schedule.json', (json.dumps(artifact['schedule'], sort_keys=True, indent=2)+'\n').encode())):
                path = output/f'{stem}.{suffix}'; path.write_bytes(data)
                records[suffix] = {'file': path.name, 'bytes': len(data), 'sha256': digest(data)}
            checks = {label: replay(program, artifact['commands'], artifact['payload'],
                                    {program.inputs[0]: value}, oracle=oracle) for label, value, oracle in oracles}
            row['candidates'][policy] = {'artifacts': records, 'verification': checks,
                                         'components': artifact['components'], 'path': artifact['path'],
                                         'max_live_scratch_bytes': max(t['scratch_bytes'] for t in artifact['schedule']['tiles']),
                                         'prefetch_bytes': artifact['schedule']['prefetch_payload_bytes'],
                                         'physical_status': 'not_run'}
        print(f'{name}: command replay passed; checking spatial segments', flush=True)
        for start, stop in supported_segments(program):
            for label, _, oracle in oracles:
                for tile in ((4, 4), (8, 8)):
                    for cache in (False, True):
                        y, counts = execute_segment(program, start, stop, oracle[program.layers[start].inputs[0]],
                                                    tile=tile, cache=cache, reduction_chunk=32)
                        expected = oracle[program.layers[stop-1].output]
                        if not np.array_equal(y, expected):
                            raise ValueError(f'{name} segment {start}:{stop} differs from integer oracle')
                        row['spatial'].append({'start': start, 'stop': stop, 'sample': label,
                                               'output_sha256': digest(y.tobytes()), 'mismatches': 0, **counts})
        row['spatial_covered_layers'] = sorted({i for a,b in supported_segments(program) for i in range(a,b)})
        row['useful_conv_macs_in_spatial_scope'] = sum(math.prod(program.tensors[l.output].shape)*math.prod(l.parameters['weight'].shape[1:])
                                                      for i, l in enumerate(program.layers)
                                                      if i in row['spatial_covered_layers'] and l.op == 'Conv')
        row['materialized_activation_bytes_in_spatial_scope'] = sum(math.prod(program.tensors[l.inputs[0]].shape)+math.prod(program.tensors[l.output].shape)
                                                                    for i, l in enumerate(program.layers) if i in row['spatial_covered_layers'])
        row['spatial_barrier_layers'] = [dict(index=i, op=l.op) for i,l in enumerate(program.layers)
                                         if i not in row['spatial_covered_layers']]
        report['models'][name] = row
        (output/'report.partial.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
        print(f'{name}: {len(artifacts)} candidate replays × 2 inputs and {len(row["spatial"])} segment checks passed', flush=True)
    files = list((ROOT/'compiler/scheduler').glob('*.py')) + list((ROOT/'compiler').glob('test_scheduler_*.py')) + list((ROOT/'tools/phase6').glob('*.py'))
    report['source_sha256'] = {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in sorted(files)}
    report['status'] = 'passed'
    (output/'report.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'work/phase6/boardless')
    parser.add_argument('--models', nargs='+', choices=('kws', 'vww'), default=['kws', 'vww'])
    args = parser.parse_args()
    run(args.output, args.models)
