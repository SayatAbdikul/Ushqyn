#!/usr/bin/env python3
"""Test an address-aware cost hypothesis within one fixed DeFiNES policy family.

All predictions are relative to a real full-model VWW anchor. Only the per-RUN
constant is calibrated, on two previously measured heights; held-out heights
are then measured with the unchanged native executable. This is not a general
cycle-accurate model, a board benchmark, or a new scheduling mechanism.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'compiler'), str(ROOT / 'tools/phase6')]
from matched_defines_baseline import (CMD, NATIVE, check_oracles, compile_config,
    compose, decode_fused, fixture, graph_identity, matched_model, native_measure,
    proxy, save, sha)
from matched_b1b2_generic import native_identity

BASE = ROOT / 'work/phase6/hypothesis-cost-v1'
FROZEN = ROOT / 'work/phase6/matched-baselines-v1'
HEIGHTS = (24, 16, 12, 8, 6, 4, 1)
TRAIN = (24, 12)
HELD_OUT = (16, 8, 4, 1)


def read_evidence(path):
    path = Path(path)
    return json.loads(path.read_text()), dict(file=str(path.relative_to(ROOT)), sha256=sha(path))


def validate_fixture(row):
    directory = ROOT / row['directory']
    for name, digest in row['files'].items():
        if sha(directory / name) != digest:
            raise ValueError(f'fixture changed: {directory / name}')
    if sorted(n['stall_seed'] for n in row['native']) != [0, 6063]:
        raise ValueError('fixture must have both distinct stall seeds')
    for native in row['native']:
        if (native['status'] != 'passed' or native['executable_sha256'] != sha(NATIVE)
                or native['fixture_files'] != row['files']):
            raise ValueError('unbound native evidence')
    if row['replay']['status'] != 'passed':
        raise ValueError('replay incomplete')


def scalar_line_trace(d):
    """Functional cache trace for the existing small general-CONV broadcast path.

    The implementation follows engine.sv line_index/gather_limit and paired
    output-channel traversal. It predicts hits/misses, not clock timings. No
    SRAM data values, calibration observations or native traces are consulted.
    """
    if not (d.opcode == 4 and d.kernel_h == d.kernel_w == 3 and
            d.output_c > 1 and d.count <= 128 and
            d.count == d.input_c * d.kernel_h * d.kernel_w):
        raise ValueError('unsupported descriptor for scalar cache hypothesis')
    oh = (d.input_h + d.pad_top + d.pad_bottom - d.kernel_h) // d.stride_h + 1
    ow = (d.input_w + d.pad_left + d.pad_right - d.kernel_w) // d.stride_w + 1
    if oh * ow * d.output_c != d.outputs:
        raise ValueError('descriptor geometry inconsistent')
    cache = {}
    f = Counter()
    for oc in range(0, d.output_c, 2):
        # Odd output channels consume the even channel's activation_tile.
        for oy in range(oh):
            for ox in range(ow):
                col = 0
                while col < d.count:
                    ic = col // (d.kernel_h * d.kernel_w)
                    ky = (col // d.kernel_w) % d.kernel_h
                    kx = col % d.kernel_w
                    y = oy * d.stride_h + ky - d.pad_top
                    x = ox * d.stride_w + kx - d.pad_left
                    address = d.input + ic * d.input_h * d.input_w + y * d.input_w + x
                    if y < 0 or x < 0 or y >= d.input_h or x >= d.input_w:
                        f['padding_steps'] += 1
                        col += 1
                        continue
                    take = min(d.kernel_w-kx, 8-address % 8, 8-col % 8,
                               d.count-col, d.input_w-x)
                    index = (y % 4)*8 + (address // 8) % 8
                    tag = address // 8
                    f['gather_steps'] += 1
                    if cache.get(index) != tag:
                        cache[index] = tag
                        f['line_misses'] += 1
                    else:
                        f['line_hits'] += 1
                    col += take
    return dict(f)


def features(code, record):
    f = Counter()
    descs = []
    for run in record['runs']:
        d, fused = decode_fused(bytes.fromhex(run['descriptor_hex']))
        if not fused:
            raise ValueError('expected common exact fused epilogue')
        descs.append(d)
        f.update(scalar_line_trace(d))
        f['runs'] += 1
        f['macs'] += d.outputs*d.count
    for i in range(0, len(code), 16):
        op, flags, _, ext, sram, length = CMD.unpack_from(code, i)
        if op:
            f['commands_without_halt'] += 1
        if op == 1:
            f['dma_commands'] += 1
            f['dma_beats'] += (length+7)//8
            f['dma_bytes'] += length
    return dict(f), descs


def instruction_score(f, per_run):
    # Two extra fixed SRAM states on a line miss; 3 states per ideal DMA beat
    # plus one DMA boundary cycle; four sequencer states per command. The
    # per-RUN fixed engine term is explicitly calibrated below, not inferred.
    return (2*f['line_misses'] + per_run*f['runs'] + 3*f['dma_beats'] +
            f['dma_commands'] + 4*f['commands_without_halt'])


def native_at(row, seed):
    return next(n for n in row['native'] if n['stall_seed'] == seed)


def run(output=BASE):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    final, final_pin = read_evidence(FROZEN/'b3-final/report.json')
    search, search_pin = read_evidence(FROZEN/'b3-final-vww/report.json')
    weights, weights_pin = read_evidence(FROZEN/'b3-weights-vww/report.json')
    catalogue, catalogue_pin = read_evidence(ROOT/search['models']['vww']['catalogue_file'])
    if any(r['status'] != 'passed' for r in (final, search, weights)):
        raise ValueError('incomplete source evidence')
    pins = dict(final['compiler_sources'])
    pins[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    for name, digest in pins.items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'execution source changed: {name}')
    original, pinned, program, maps, provenance = matched_model('vww')
    graph = graph_identity(program)
    if graph != search['models']['vww']['graph_sha256']:
        raise ValueError('graph differs from source search')
    oracle = check_oracles(original, program, maps, pinned)
    base_candidate = next(c for c in search['models']['vww']['candidates'] if c['rank'] == 0)
    path = copy.deepcopy(base_candidate['configuration_path'])
    if path[0] != dict(h=24, w=48, kind='rectangle', mode=1, prefetch=True,
                       retain=True, start=1, stop=3):
        raise ValueError('unexpected fixed policy path')
    tail = dict(configuration=dict(kind='fallback', start=path[-1]['stop'], stop=len(program.layers)))
    reused = {}
    for pair in weights['models']['vww']['pairs']:
        p = pair['configurations']
        if p[1:] != path[1:]:
            continue
        first = {k: v for k, v in p[0].items() if k != 'resident_weights'}
        if {k: v for k, v in first.items() if k != 'h'} != {k: v for k, v in path[0].items() if k != 'h'}:
            continue
        if first['h'] in (24, 12, 6):
            validate_fixture(pair['control'])
            reused[first['h']] = copy.deepcopy(pair['control'])
    if set(reused) != {24, 12, 6}:
        raise ValueError('expected exact existing calibration/control fixtures')
    variants = {}
    for height in HEIGHTS:
        cfg = dict(path[0], h=height)
        code, payload, record = compile_config(program, cfg)
        f, descs = features(code, record)
        matched = next((c for c in catalogue['candidates'] if c['configuration'] == cfg), None)
        if matched is not None:
            if (matched['status'] != 'executable' or matched['code_sha256'] != sha(code)
                    or matched['payload_sha256'] != sha(payload)):
                raise ValueError('catalogue geometry changed')
        variants[height] = dict(configuration=cfg, features=f, structural_proxy=proxy(code, record),
            catalogue_id=None if matched is None else matched['id'],
            source_geometry='published catalogue grid' if matched else 'existing resident-control half-height screen',
            descriptors=[asdict(d) for d in descs])
    a, b = (variants[h]['features'] for h in TRAIN)
    measured_engine_delta = native_at(reused[12], 0)['engine_cycles'] - native_at(reused[24], 0)['engine_cycles']
    per_run = (measured_engine_delta - 2*(b['line_misses']-a['line_misses']))/(b['runs']-a['runs'])
    if per_run != int(per_run) or per_run < 0:
        raise ValueError('invalid calibrated run cost')
    per_run = int(per_run)
    baseline_cost = instruction_score(a, per_run)
    baseline_cycles = native_at(reused[24], 0)['elapsed_cycles']
    for height, item in variants.items():
        item['predicted_fixed_cycles'] = baseline_cycles + instruction_score(item['features'], per_run)-baseline_cost
    # Commit predictions and scope before consulting new held-out native timings.
    report = dict(schema=1, status='predictions-frozen', physical_board=False,
        graph_sha256=graph, provenance=provenance, compiler_sources=pins, native=native_identity(),
        evidence=[final_pin, search_pin, weights_pin, catalogue_pin],
        scope='VWW first Conv+activation stack height; fixed full-model path, exact graph/arithmetic, mode1, allocator, retention, prefetch, RTL and ABI',
        experiment=dict(heights=list(HEIGHTS), calibration_heights=list(TRAIN), new_holdout_heights=list(HELD_OUT),
                        already_observed_check_height=6, stall_seeds=[0,6063],
                        cost_fit='one per-RUN constant from existing h24/h12 fixed-seed engine counters; no holdout fitting',
                        per_run_cycles=per_run, cache_miss_penalty=2,
                        score='2*line_misses + per_run*runs + 3*dma_beats + dma_commands + 4*commands_without_halt',
                        prediction_scope='relative fixed native memory timing in this descriptor family only'),
        variants=variants, claim='cost-model falsification within existing DeFiNES policy; not a novel scheduling mechanism')
    prediction_file = output/'predictions.json'
    save(prediction_file, report)
    prediction_hash = sha(prediction_file)
    save(output/'report.json', report)
    for height, item in variants.items():
        chosen = [dict(configuration=cfg) for cfg in [dict(path[0], h=height)]+path[1:]]
        cdir = output/'components'/f'h{height:02d}'
        code, payload, record = compose(program, chosen, tail, oracle, cdir)
        if height in reused:
            row = reused[height]
            if sha(code) != row['files']['commands.bin'] or sha(payload) != row['files']['payload.bin']:
                raise ValueError('existing control is not byte-identical to the fixed path')
            item['fixture'] = row
            item['execution'] = 'reused exact whole-model native evidence; code/payload recompiled and compared'
        else:
            directory = output/'fixtures'/f'h{height:02d}-pinned'
            files = fixture(directory, program, oracle, code, payload, record)
            natives = [native_measure(directory, seed, output/f'h{height:02d}-s{seed}.json') for seed in (0,6063)]
            item['fixture'] = dict(directory=str(directory.relative_to(ROOT)), files=files, native=natives, replay=record['replay'])
            item['execution'] = 'new exact whole-model native runs after predictions frozen'
        validate_fixture(item['fixture'])
        actual = native_at(item['fixture'], 0)['elapsed_cycles']
        item['fixed_prediction_error_cycles'] = item['predicted_fixed_cycles']-actual
        item['native_geomean_cycles'] = math.sqrt(math.prod(n['elapsed_cycles'] for n in item['fixture']['native']))
        print('height', height, 'predicted', item['predicted_fixed_cycles'], 'actual', actual,
              'error', item['fixed_prediction_error_cycles'], flush=True)
        save(output/'report.json', report)
    # Uniform eight-byte rebasing rotates lower cache-index bits. It is a trace
    # negative control, not a claim these mutated descriptors are valid layouts.
    code, payload, record = compile_config(program, path[0])
    _, descriptors = features(code, record)
    shifts = []
    original_trace = variants[24]['features']
    for delta in range(0, 64, 8):
        aggregate = Counter()
        for d in descriptors:
            aggregate.update(scalar_line_trace(replace(d, input=d.input+delta)))
        invariant = all(aggregate[k] == original_trace[k] for k in ('line_misses','line_hits','gather_steps','padding_steps'))
        if not invariant:
            raise ValueError('uniform cache-set permutation hypothesis failed')
        shifts.append(dict(base_shift_bytes=delta, counters=dict(aggregate), invariant=invariant))
    report['uniform_rebase_negative_control'] = dict(status='passed', scope='functional cache trace only', shifts=shifts,
        reason='word-aligned uniform base shift bijectively permutes the low three set bits; full tags preserve equality')
    proxy_winner = min(variants, key=lambda h: (variants[h]['structural_proxy'],h))
    instruction_winner = min(variants, key=lambda h: (variants[h]['predicted_fixed_cycles'],h))
    actual_winner = min(variants, key=lambda h: (native_at(variants[h]['fixture'],0)['elapsed_cycles'],h))
    comparisons = {}
    for seed in (0,6063):
        values = {h:native_at(item['fixture'],seed)['elapsed_cycles'] for h,item in variants.items()}
        winner = min(values, key=values.get)
        comparisons[seed] = dict(winner_height=winner,winner_cycles=values[winner],
            proxy_selected_height=proxy_winner,proxy_selected_cycles=values[proxy_winner],
            proxy_regret_cycles=values[proxy_winner]-values[winner],
            proxy_regret_fraction=values[proxy_winner]/values[winner]-1,
            instruction_selected_height=instruction_winner,instruction_regret_cycles=values[instruction_winner]-values[winner])
    report['result'] = dict(structural_proxy_winner=proxy_winner,instruction_cost_winner=instruction_winner,
        native_fixed_winner=actual_winner,comparisons=comparisons,
        new_holdout_fixed_errors={h:variants[h]['fixed_prediction_error_cycles'] for h in HELD_OUT},
        all_equal_macs=len({v['features']['macs'] for v in variants.values()})==1,
        strict_proxy_inversion=variants[24]['structural_proxy']<variants[12]['structural_proxy'] and
            native_at(variants[24]['fixture'],0)['elapsed_cycles']>native_at(variants[12]['fixture'],0)['elapsed_cycles'],
        interpretation='address-sensitive line-cache cost corrects a real structural-proxy inversion inside the existing mode1 schedule space',
        novelty_status='not established: standard cache-aware cost modelling; no policy/architecture mechanism added',
        limits=['one changed stack in one model; calibration uses two existing geometries',
                'random-stall and physical-memory predictions are not provided',
                'bounded height grid and fixed path; no global optimum claim',
                'native exact logits on pinned input; not full accuracy or board verification'])
    for name,digest in pins.items():
        if sha(ROOT/name)!=digest:
            raise ValueError(f'source changed during experiment: {name}')
    if sha(prediction_file)!=prediction_hash:
        raise ValueError('frozen predictions changed')
    for item in variants.values():validate_fixture(item['fixture'])
    report.update(status='passed',seconds=time.monotonic()-start,predictions=dict(file=str(prediction_file.relative_to(ROOT)),sha256=prediction_hash))
    save(output/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE)
    args=parser.parse_args()
    run(args.output)
