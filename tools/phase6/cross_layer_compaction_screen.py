#!/usr/bin/env python3
"""Measure how exact channel compaction changes cross-layer tile opportunities.

Software tile execution is exact. The conservative byte count is only a
necessary screening step; this experiment never emits a fused FPGA command.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'compiler'), str(ROOT/'tools/phase6')]

from channel_compaction import compact_channels, check_oracles
from cross_layer_screen import (bridge_transfers, candidate_scratch_estimate,
                                source_schedule, sha, CALIBRATION)
from followup_graph import group_channels
from run_boardless import load_model
from scheduler.current_abi import CostModel
from scheduler.spatial import execute_segment

OUT = ROOT/'work/phase6/cross-layer-compaction-v1'


def compact_schedule(name):
    return ROOT/f'work/phase6/channel-compaction-v1/fixtures/{name}-pinned-compacted-timed/schedule.json'


def pairs_for(name, program, oracles, path, cost, selected_firsts):
    schedule = json.loads(path.read_text())
    result = {}
    for first in range(3, len(program.layers)-3, 4):
        layers = program.layers[first:first+4]
        if ([layer.op for layer in layers] != ['Conv', 'Relu', 'Conv', 'Relu'] or
                layers[0].attributes.get('group', 1) == 1 or
                layers[2].attributes.get('group', 1) != 1):
            continue
        transfers = bridge_transfers(schedule, first)
        row = {'layers': [first, first+3],
               'bridge_bytes': sum(t['bytes'] for t in transfers),
               'bridge_dma_fit_cycles': sum(cost.dma(t) for t in transfers),
               'candidates': []}
        # Use the same candidates for both graphs, including blocks whose
        # materialization traffic disappears after compaction.
        if first in selected_firsts:
            source_bytes = math.prod(program.tensors[layers[0].inputs[0]].shape)
            for tile in ((8, 8), (16, 16)):
                for cache in (False, True):
                    outputs = []
                    for oracle in oracles:
                        got, counts = execute_segment(
                            program, first, first+4, oracle[layers[0].inputs[0]],
                            tile=tile, cache=cache)
                        expected = oracle[layers[-1].output]
                        if not np.array_equal(got, expected):
                            raise AssertionError(f'{name} layers {first}-{first+3} not exact')
                        outputs.append(hashlib.sha256(got.tobytes()).hexdigest())
                    estimate = candidate_scratch_estimate(program, first, first+4, counts)
                    row['candidates'].append({
                        'tile': list(tile), 'cache': cache,
                        'exact_samples': len(oracles), 'output_sha256': outputs,
                        'source_requested_bytes': counts['source_requested_bytes'],
                        'whole_source_bytes': source_bytes,
                        'semantic_macs': sum(v['macs'] for v in counts['layers'].values()),
                        'output_tiles': counts['output_tiles'],
                        'conservative_buffer_count_bytes': estimate,
                        'buffer_count_fits_32k': estimate <= 32768,
                        'real_port_and_address_certificate': False,
                    })
        result[str(first)] = row
    return {'schedule_sha256': sha(path), 'pairs': result,
            'bridge_bytes_total': sum(row['bridge_bytes'] for row in result.values()),
            'bridge_dma_fit_cycles_total': sum(row['bridge_dma_fit_cycles'] for row in result.values())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cost = CostModel(json.loads(CALIBRATION.read_text()))
    report = {'schema': 1, 'status': 'running', 'physical_board': False,
              'hardware_executable': False,
              'scope': ('Exact software DW/activation/PW/activation tiles on grouped and '
                        'compacted frozen graphs. SRAM byte counts do not certify address '
                        'placement, one-port timing, or missing row-gather lowering.'),
              'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
                                (CALIBRATION, ROOT/'tools/phase6/channel_compaction.py',
                                 ROOT/'tools/phase6/cross_layer_screen.py',
                                 ROOT/'compiler/scheduler/spatial.py')},
              'models': {}}
    for name in ('kws', 'vww'):
        original, x, _, sources = load_model(name)
        grouped, group_maps, _ = group_channels(original)
        compacted, kept, _, _ = compact_channels(grouped)
        maps = {key: tuple(group_maps[key][j] for j in selected)
                for key, selected in kept.items()}
        values = (x, np.random.default_rng(6157).integers(-128, 128, x.shape, dtype=np.int8))
        grouped_oracles = []
        compact_oracles = []
        from integer_reference import evaluate
        for value in values:
            grouped_oracles.append(evaluate(grouped, {grouped.inputs[0]: value}))
            compact_oracles.append(check_oracles(original, compacted, maps, value))
        paths = {'grouped': source_schedule(name), 'compacted': compact_schedule(name)}
        selected_firsts = {3} if name == 'kws' else set()
        for path in paths.values():
            schedule = json.loads(path.read_text())
            selected_firsts.update(first for first in range(3, len(original.layers)-3, 4)
                                  if bridge_transfers(schedule, first))
        graph_results = {
            'grouped': pairs_for(name, grouped, grouped_oracles, paths['grouped'], cost, selected_firsts),
            'compacted': pairs_for(name, compacted, compact_oracles, paths['compacted'], cost, selected_firsts),
        }
        newly_fitting = []
        for first, old in graph_results['grouped']['pairs'].items():
            new = graph_results['compacted']['pairs'].get(first)
            if not new:
                raise AssertionError(f'{name}: compaction changed eligible block {first}')
            if len(old['candidates']) != len(new['candidates']):
                raise AssertionError(f'{name}: unequal candidate matrix at block {first}')
            if not old['candidates']:
                continue
            for before, after in zip(old['candidates'], new['candidates']):
                assert (before['tile'], before['cache']) == (after['tile'], after['cache'])
                if not before['buffer_count_fits_32k'] and after['buffer_count_fits_32k']:
                    newly_fitting.append({'first_layer': int(first), 'tile': after['tile'],
                                          'cache': after['cache'],
                                          'old_buffer_bytes': before['conservative_buffer_count_bytes'],
                                          'new_buffer_bytes': after['conservative_buffer_count_bytes']})
        report['models'][name] = {'model_sources_sha256': sources,
                                  'shared_candidate_first_layers': sorted(selected_firsts),
                                  'candidate_scope': 'same 8x8/16x16, cache off/on matrix for blocks with bridge traffic in either graph, plus audio block 3',
                                  'graphs': graph_results,
                                  'newly_fitting_byte_count_cases': newly_fitting}
    report['status'] = 'passed-software-screen'
    path = OUT/'report.json'
    path.write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
    print(json.dumps({name: {
        'bridge_bytes': {label: value['bridge_bytes_total']
                         for label, value in record['graphs'].items()},
        'newly_fitting_byte_count_cases': len(record['newly_fitting_byte_count_cases'])}
        for name, record in report['models'].items()}, indent=2))
    print(path)


if __name__ == '__main__':
    main()
