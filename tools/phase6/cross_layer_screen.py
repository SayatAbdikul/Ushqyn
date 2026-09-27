#!/usr/bin/env python3
"""Bounded, exact semantic screen for a DW -> activation -> PW -> activation tile.

This deliberately does not claim an executable FPGA schedule. The current ABI
cannot gather the strided source rows of a spatial tile into the scratchpad.
The traffic calculation only estimates the opportunity for a future backend.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'compiler'))
sys.path.insert(0, str(ROOT / 'tools' / 'phase6'))

from integer_reference import evaluate
from run_boardless import load_model
from scheduler.current_abi import CostModel
from scheduler.spatial import execute_segment

OUT = ROOT / 'work/phase6/cross-layer-screen-v1'
CALIBRATION = ROOT / 'docs/research/evidence/phase4/physical-sequence-costs.json'
PHYSICAL = ROOT / 'work/phase6/fused-activation-uart256-24-screen-v1/report.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_schedule(name):
    # This is the pre-epilogue schedule. The final activation lowering retains
    # its store DMA even when the separate activation RUN disappears, whereas
    # the final schedule's shortened stage list does not describe that DMA.
    return ROOT / f'work/phase6/followup_graph/fixtures/{name}-pinned-grouped-timed/schedule.json'


def bridge_transfers(schedule, first):
    stores = [s['store'] for s in schedule['stages'] if s['layer'] == first + 1 and s['store']]
    loads = [t for s in schedule['stages'] if s['layer'] == first + 2
             for t in s['loads'] if t['role'] == 'input']
    return stores + loads


def candidate_scratch_estimate(program, first, stop, counts):
    """Conservative buffer-count screen, not an address/port certificate."""
    rows = program.layers[first:stop]
    max_source_rectangle = max(v['max_input_rectangle_bytes'] for v in counts['layers'].values())
    max_output_tile = max(
        math.prod(program.tensors[l.output].shape[:2]) *
        min(counts['tile'][0], program.tensors[l.output].shape[2]) *
        min(counts['tile'][1], program.tensors[l.output].shape[3])
        for l in rows)
    parameter_bytes = sum(
        a.nbytes for l in rows for a in l.parameters.values()
        if isinstance(a, np.ndarray))
    # One source rectangle, two INT8 tile buffers, cached rectangles, two
    # descriptors, and all block parameters. Partial sums remain in registers.
    return (max_source_rectangle + 2 * max_output_tile +
            counts['peak_retained_cache_bytes'] + parameter_bytes + 256)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    calibration = CostModel(json.loads(CALIBRATION.read_text()))
    physical = json.loads(PHYSICAL.read_text())
    assert physical['status'] == 'passed-short-screen' and physical['completed'] == 10
    report = {
        'schema': 1,
        'status': 'screened-boardless',
        'physical_board': False,
        'hardware_executable': False,
        'scope': ('Exact software INT8 segment outputs, source-work counters, '
                  'conservative byte-capacity estimate, and historical DMA-fit '
                  'traffic opportunity. No FPGA latency, route, or placement claim.'),
        'sources': {str(p.relative_to(ROOT)): sha(p)
                    for p in (CALIBRATION, PHYSICAL, ROOT/'compiler/scheduler/spatial.py')},
        'current_device_cycles': {name: physical['summary'][name]['median_cycles']
                                  for name in ('kws', 'vww')},
        'models': {},
    }
    bridge_fit = {}
    for name in ('kws', 'vww'):
        program, x, _, model_pins = load_model(name)
        schedule_path = source_schedule(name)
        schedule = json.loads(schedule_path.read_text())
        pinned = evaluate(program, {program.inputs[0]: x})
        stress = np.random.default_rng(611).integers(-128, 128, x.shape, dtype=np.int8)
        stress_oracle = evaluate(program, {program.inputs[0]: stress})
        pairs = []
        total_bridge_cycles = 0
        for first in range(3, len(program.layers)-3, 4):
            layers = program.layers[first:first+4]
            if ([l.op for l in layers] != ['Conv', 'Relu', 'Conv', 'Relu'] or
                    layers[0].attributes.get('group', 1) == 1 or
                    layers[2].attributes.get('group', 1) != 1):
                continue
            transfers = bridge_transfers(schedule, first)
            bridge_bytes = sum(t['bytes'] for t in transfers)
            fitted_cycles = sum(calibration.dma(t) for t in transfers)
            total_bridge_cycles += fitted_cycles
            row = {'first_layer': first, 'last_layer': first+3,
                   'bridge_transfer_count': len(transfers),
                   'materialized_bridge_bytes': bridge_bytes,
                   'historically_fitted_bridge_cycles': fitted_cycles,
                   'tile_candidates': []}
            # All positive-traffic blocks and one zero-traffic audio block:
            # enough to test each workload without re-running old 152-case sweeps.
            if bridge_bytes or (name == 'kws' and first == 3):
                full_source = math.prod(program.tensors[layers[0].inputs[0]].shape)
                for tile in ((8, 8), (16, 16)):
                    for cache in (False, True):
                        inputs = ((x, pinned), (stress, stress_oracle))
                        samples = []
                        for _, oracle in inputs:
                            y, counts = execute_segment(
                                program, first, first+4,
                                oracle[layers[0].inputs[0]], tile=tile, cache=cache)
                            expected = oracle[layers[-1].output]
                            if not np.array_equal(y, expected):
                                raise AssertionError(f'{name} {first} {tile} cache={cache} mismatch')
                            samples.append(hashlib.sha256(y.tobytes()).hexdigest())
                        scratch = candidate_scratch_estimate(program, first, first+4, counts)
                        row['tile_candidates'].append({
                            'tile': list(tile), 'cache': cache,
                            'exact_samples': 2, 'output_sha256': samples,
                            'source_requested_bytes': counts['source_requested_bytes'],
                            'whole_source_bytes': full_source,
                            'duplicate_source_requests_bytes': max(0, counts['source_requested_bytes']-full_source),
                            'semantic_macs': sum(v['macs'] for v in counts['layers'].values()),
                            'output_tiles': counts['output_tiles'],
                            'peak_retained_cache_bytes': counts['peak_retained_cache_bytes'],
                            'conservative_buffer_count_bytes': scratch,
                            'buffer_count_fits_32k': scratch <= 32768,
                            'port_address_and_gather_certificate': False,
                        })
            pairs.append(row)
        bridge_fit[name] = total_bridge_cycles
        report['models'][name] = {
            'model_sources_sha256': model_pins,
            'schedule_sha256': sha(schedule_path),
            'bridge_bytes_total': sum(r['materialized_bridge_bytes'] for r in pairs),
            'historically_fitted_bridge_cycles_total': total_bridge_cycles,
            'pairs': pairs,
        }
    # Deliberately generous traffic-only calculation: it removes every selected
    # materialization transfer while charging zero for tile gathers, additional
    # engine commands, halo duplication, altered SRAM contention, or a new RTL.
    # The DMA fit came from an older board image, so this is a sensitivity
    # estimate, not a certified bound on the selected image's latency.
    speedups = {}
    for name in ('kws', 'vww'):
        baseline = report['current_device_cycles'][name]
        saved = bridge_fit[name]
        assert 0 <= saved < baseline
        speedups[name] = baseline / (baseline - saved)
    report['optimistic_traffic_only'] = {
        'assumptions': 'All fitted bridge DMA cycles removed at zero implementation cost',
        'device_speedup': speedups,
        'geometric_mean_speedup': math.sqrt(speedups['kws']*speedups['vww']),
        'traffic_only_sensitivity_reaches_1p03': math.sqrt(speedups['kws']*speedups['vww']) >= 1.03,
        'fit_bitstream_sha256': calibration.record['bitstream_sha256'],
        'selected_bitstream_sha256': json.loads((ROOT/'work/phase6/fused-activation-uart256-24-screen-v1/plan.json').read_text())['image']['sha256'],
    }
    path = OUT/'report.json'
    path.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'status': report['status'],
                      'bridge_bytes': {k: report['models'][k]['bridge_bytes_total'] for k in report['models']},
                      'traffic_only_geomean': report['optimistic_traffic_only']['geometric_mean_speedup'],
                      'report': str(path)}, indent=2))


if __name__ == '__main__':
    run()
