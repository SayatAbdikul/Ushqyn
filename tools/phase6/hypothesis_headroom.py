#!/usr/bin/env python3
"""Summarize sealed Task 1 evidence and a fixed-engine occupancy counterfactual.

This does not extrapolate native RAM timing to physical SDRAM, or claim that
engine-busy occupancy is a universal lower bound for a different schedule.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/research/evidence/phase6/hypothesis-v2'
IMAGE = '3e943639382e4efea62d4d2072c016fa40eacc7a97ef6bad7f5bfdc69d8dafce'
NATIVE = 'b97fbf944827a5345670f3603e7f10368a897e6bcb19d16cc56905d8e06d16ec'
CAMPAIGNS = {'board': 30, 'combined-finalists-board-x-after-reconnect': 45}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + '\n')


def verify_campaign(directory, count):
    report = read(directory / 'report.json')
    seal = read(directory / 'seal.json')
    for name in ('plan', 'records', 'report'):
        filename = name + ('.jsonl' if name == 'records' else '.json')
        if sha(directory / filename) != seal[name + '_sha256']:
            raise ValueError('campaign seal mismatch: ' + str(directory / filename))
    for name in ('plan', 'records'):
        if report[name + '_sha256'] != seal[name + '_sha256']:
            raise ValueError('report/seal mismatch')
    log_key = 'program_log_sha256'
    if log_key in report and sha(directory / 'program.log') != report[log_key]:
        raise ValueError('program log mismatch')
    if (not report['status'].startswith('passed') or not report['physical_board']
            or report['completed'] != count or report['planned'] != count
            or report['output_mismatches']):
        raise ValueError('incomplete or failing board campaign')
    records = [json.loads(line) for line in (directory / 'records.jsonl').read_text().splitlines()]
    if len(records) != count:
        raise ValueError('missing board records')
    for record in records:
        raw = {k: v for k, v in record.items() if k != 'record_sha256'}
        signature = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if signature != record['record_sha256']:
            raise ValueError('record digest mismatch')
        if (record['bitstream_sha256'] != IMAGE or record['core_clock_hz'] != 27_000_000
                or record['expected_hex'] != record['output_hex']):
            raise ValueError('hardware identity or exact output mismatch')
    return report, records


def occupancy(records):
    rows = []
    for r in records:
        elapsed, engine, dma, overlap = [r[k] for k in
            ('elapsed_cycles', 'engine_cycles', 'dma_cycles', 'overlap_cycles')]
        idle = elapsed - (engine + dma - overlap)
        if not (0 <= overlap <= min(engine, dma) and idle >= 0 and engine > 0):
            raise ValueError('invalid counter partition')
        rows.append(dict(elapsed_cycles=elapsed, engine_busy_cycles=engine,
            dma_without_engine_cycles=dma-overlap, neither_busy_cycles=idle,
            counterfactual_latency_reduction=1-engine/elapsed))
    return dict(records=rows, median={k: statistics.median(r[k] for r in rows) for k in rows[0]})


def run(task1_root=None, output=OUT):
    output = Path(output).resolve()
    if task1_root is not None:
        source = Path(task1_root) / 'docs/research/evidence/phase6/matched-baselines-v1'
        for name, count in CAMPAIGNS.items():
            verify_campaign(source / name, count)
            for filename in ('plan.json', 'report.json', 'records.jsonl', 'seal.json', 'program.log'):
                destination = output / 'task1' / name / filename
                data = (source / name / filename).read_bytes()
                if destination.exists() and destination.read_bytes() != data:
                    raise ValueError('refusing to replace imported evidence')
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
    imported = {}
    for name, count in CAMPAIGNS.items():
        imported[name] = verify_campaign(output / 'task1' / name, count)
    board, records = imported['board']
    physical = {m: occupancy([r for r in records if r['kind']=='timed' and r['policy']=='B4' and r['model']==m])
                for m in ('kws', 'vww')}
    local_path = ROOT / 'work/phase6/matched-baselines-v1/b3-final/report.json'
    local = read(local_path)
    if local['status'] != 'passed' or local['native_sha256'] != NATIVE:
        raise ValueError('missing completed generic B3 native result')
    native = {}
    for m, selected in local['selected'].items():
        fixture = selected['pinned']
        for name, digest in fixture['files'].items():
            if sha(ROOT / fixture['directory'] / name) != digest:
                raise ValueError('native fixture changed')
        rows = fixture['native']
        if ({r['stall_seed'] for r in rows} != {0, 6063}
                or any(r['status']!='passed' or r['executable_sha256']!=NATIVE
                       or r['physical_board'] or r['fixture_files']!=fixture['files'] for r in rows)):
            raise ValueError('unmatched native observations')
        native[m] = dict(directory=fixture['directory'], files=fixture['files'],
                        graph_sha256=local['models'][m]['graph_sha256'], seeds={str(r['stall_seed']):occupancy([r]) for r in rows})
    contract = dict(scope='Conditional occupancy counterfactual: erase all cycles outside engine busy, holding the observed engine-busy count fixed.',
        limitation='Engine busy includes stalls, dispatch, copy and quantization. New kernels, tiling, layout or contention changes can change this count. These are not general accelerator lower bounds or predicted attainable speedups.',
        counter_source='rtl/v2/tile_sequencer.sv',
        counter_source_sha256=sha(ROOT / 'rtl/v2/tile_sequencer.sv'),
        timing_boundary='sequencer launch to HALT, including scheduled DMA, excluding host UART load/readback')
    fixed = 1 - math.sqrt(math.prod(physical[m]['median']['engine_busy_cycles'] / physical[m]['median']['elapsed_cycles'] for m in physical))
    native_fixed = {str(seed):1-math.sqrt(math.prod(native[m]['seeds'][str(seed)]['median']['engine_busy_cycles']/native[m]['seeds'][str(seed)]['median']['elapsed_cycles'] for m in native)) for seed in (0,6063)}
    report = dict(schema=1, status='passed-analysis', new_board_runs=0, physical_contract=contract,
        bitstream_sha256=IMAGE, native_executable_sha256=NATIVE, physical_current=physical,
        native_generic_B3=native, physical_fixed_engine_geomean_latency_reduction=fixed,
        native_fixed_engine_geomean_latency_reduction=native_fixed,
        task1_primary_comparisons=board['summary']['comparisons'],
        task1_finalist_summary=imported['combined-finalists-board-x-after-reconnect'][0]['summary'],
        gate=dict(geomean_latency_ratio_at_most=.85, neither_workload_latency_ratio_above=1.05,
            equivalent_geomean_speedup_at_least=1/.85,
            vww_reduction_required_if_kws_unchanged=1-.85**2,
            statement='This 15% project go/no-go criterion is not a publication acceptance rule.'),
        evidence={str(p.relative_to(ROOT)):sha(p) for p in sorted((output/'task1').rglob('*')) if p.is_file()},
        source_sha256=sha(Path(__file__)),
        local_B3_report=dict(path=str(local_path.relative_to(ROOT)), sha256=sha(local_path)),
        limitations=['Imported Task 1 baseline exploration is bounded, not an exhaustive DeFiNES reproduction.',
            'Local generic B3 results are native RTL only; their graph transforms and programs differ from imported Task 1. Do not directly compare native cycles with board cycles.',
            'Imported run hashes validate records; full original fixture/source/bitstream closure remains in Task 1 archive.'])
    save(output/'headroom.json', report)
    print(json.dumps(dict(status=report['status'], physical_fixed_engine_reduction=fixed,
        native_fixed_engine_reduction=native_fixed, target_vww_reduction_if_kws_unchanged=1-.85**2), indent=2))
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task1-root', type=Path)
    parser.add_argument('--output', type=Path, default=OUT)
    args=parser.parse_args()
    run(args.task1_root, args.output)
