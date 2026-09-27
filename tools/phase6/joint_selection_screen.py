#!/usr/bin/env python3
"""Same-executable 2x2 compaction/fused-prefetch screen on pinned KWS and VWW.

All four cells use the frozen fused engine native executable. The experiment
enumerates a restricted compiler catalogue, not the Phase 6 joint physical
bank-placement/search mechanism. No board is opened or programmed.
The second factor combines activation fusion with safe-tail prefetch; it is
not an isolated activation-fusion ablation.
"""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXECUTABLE = ROOT / 'work/phase6/experiments-v1/fused-activation-v1/native/Vv2_tiled_host_bridge'
EXPECTED_EXECUTABLE_SHA256 = '58129ac01629b52418f77f06073f5c6cf109ab006cbd4116fa0054dd8bffcaad'
DEFAULT_OUTPUT = ROOT / 'work/phase6/joint-selection-v1'
SEEDS = (0, 6063)

# Cell bits are (channel compaction, exact activation fusion + tail prefetch).
FACTOR_DEFINITIONS = {
    'compaction': 'exact internal channel compaction',
    'fusion': 'exact activation epilogue fusion plus safe-tail parameter prefetch',
}
SOURCES = {
    '00': ('work/phase6/followup_graph', 'fixtures.json', 'label', '-pinned-grouped-timed'),
    '01': ('work/phase6/experiments-v1/fused-activation-v1', 'fixtures.json', 'name', '-pinned-grouped-fused-timed'),
    '10': ('work/phase6/channel-compaction-v1', 'report.json', 'label', '-pinned-compacted-timed'),
    '11': ('work/phase6/channel-compaction-v1/fused', 'report.json', 'label', '-pinned-compacted-fused-timed'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, record):
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')


def prepare():
    if sha(EXECUTABLE) != EXPECTED_EXECUTABLE_SHA256:
        raise ValueError('frozen fused RTL executable changed')
    original_native = json.loads((EXECUTABLE.parent / 'report.json').read_text())
    if original_native['status'] != 'passed' or original_native['executable_sha256'] != sha(EXECUTABLE):
        raise ValueError('selected native RTL evidence changed')
    compacted_fused = json.loads((ROOT / SOURCES['11'][0] / 'report.json').read_text())
    compacted_plain_path = ROOT / SOURCES['10'][0] / 'report.json'
    if (compacted_fused['status'] != 'passed' or
            compacted_fused['source_report_sha256'] != sha(compacted_plain_path) or
            compacted_fused['executable_sha256'] != sha(EXECUTABLE)):
        raise ValueError('compacted fused generation/native evidence changed')

    plan = {'schema': 1, 'physical_board': False, 'clock_hz': 24_000_000,
            'executable': str(EXECUTABLE.relative_to(ROOT)),
            'executable_sha256': sha(EXECUTABLE), 'seeds': list(SEEDS),
            'scope': 'restricted 2x2 exact channel compaction and activation-fusion-plus-safe-tail-prefetch catalogue on one native RTL executable; no isolated fusion ablation, no full cross-layer/physical-bank search, no G6 or novelty claim',
            'factor_definitions': FACTOR_DEFINITIONS,
            'fixtures': {}, 'source_manifests': {}}
    for cell, (prefix, manifest_name, label_key, suffix) in SOURCES.items():
        root = ROOT / prefix
        manifest_path = root / manifest_name
        manifest = json.loads(manifest_path.read_text())
        expected_status = 'passed' if cell == '11' else 'passed-replay'
        if manifest['status'] != expected_status:
            raise ValueError(f'{cell}: source manifest is incomplete')
        plan['source_manifests'][cell] = {'path': str(manifest_path.relative_to(ROOT)),
                                          'sha256': sha(manifest_path)}
        rows = manifest['results'] if cell == '11' else manifest['fixtures']
        for model in ('kws', 'vww'):
            name = model + suffix
            selected = [row for row in rows if row[label_key] == name]
            if len(selected) != 1:
                raise ValueError(f'{cell}/{model}: expected exactly one fixture')
            row = selected[0]
            if row['model'] != model or row['sample'] != 'pinned' or row['verification']['status'] != 'passed':
                raise ValueError(f'{cell}/{model}: incomplete oracle replay')
            if cell == '11' and (row['lifetimes']['status'] != 'passed' or
                                  [(item['seed'], item['status']) for item in row['native']] !=
                                  [(0, 'passed'), (6063, 'passed')]):
                raise ValueError(f'{cell}/{model}: compacted fusion evidence incomplete')
            fixture = root / 'fixtures' / name
            for filename, digest in row['files'].items():
                if sha(fixture / filename) != digest:
                    raise ValueError(f'{cell}/{model}: changed fixture file {filename}')
            schedule = json.loads((fixture / 'schedule.json').read_text())
            if (sha(fixture / 'commands.bin') != schedule['program_sha256'] or
                    sha(fixture / 'payload.bin') != schedule['image_sha256'] or
                    (fixture / 'commands.bin').stat().st_size != 16*schedule['command_count'] or
                    'output.bin' not in (fixture / 'checks.txt').read_text()):
                raise ValueError(f'{cell}/{model}: incomplete executable fixture identity')
            plan['fixtures'][model + ':' + cell] = {
                'directory': str(fixture.relative_to(ROOT)), 'files': row['files'],
                'command_count': schedule['command_count'],
                'output_sha256': sha(fixture / 'output.bin'),
                'input_sha256': sha(fixture / 'input.bin')}
    for model in ('kws', 'vww'):
        rows = [plan['fixtures'][model + ':' + cell] for cell in SOURCES]
        if len({row['input_sha256'] for row in rows}) != 1 or len({row['output_sha256'] for row in rows}) != 1:
            raise ValueError(f'{model}: 2x2 cells changed the public input or exact logits')
    return plan


def greedy(values, order):
    bits = ['0', '0']
    for factor in order:
        index = 0 if factor == 'compaction' else 1
        candidate = bits.copy()
        candidate[index] = '1'
        if values[''.join(candidate)] < values[''.join(bits)]:
            bits = candidate
    return ''.join(bits)


def analyze(plan, measured):
    by_model = {}
    for model in ('kws', 'vww'):
        records = {}
        for seed in SEEDS:
            values = {cell: measured[f'{model}:{cell}:{seed}']['elapsed_cycles'] for cell in SOURCES}
            base, compact, fused, joint = (values[c] for c in ('00', '10', '01', '11'))
            best = min(values, key=lambda cell: (values[cell], cell))
            independent = str(int(compact < base)) + str(int(fused < base))
            cf = greedy(values, ('compaction', 'fusion'))
            fc = greedy(values, ('fusion', 'compaction'))
            records[str(seed)] = {
                'cycles': values, 'best_cell': best,
                'best_speedup_vs_00': base/values[best],
                'compaction_alone_saved_cycles': base-compact,
                'fusion_alone_saved_cycles': base-fused,
                'joint_saved_cycles': base-joint,
                'interaction_extra_saved_cycles': compact+fused-base-joint,
                'interaction_fraction_of_reference': (compact+fused-base-joint)/base,
                'independent_one_pass_choice': independent,
                'independent_one_pass_regret_cycles': values[independent]-values[best],
                'sequential_compaction_then_fusion': cf,
                'sequential_compaction_then_fusion_regret_cycles': values[cf]-values[best],
                'sequential_fusion_then_compaction': fc,
                'sequential_fusion_then_compaction_regret_cycles': values[fc]-values[best],
            }
        by_model[model] = records
    shared_policy = {}
    for seed in SEEDS:
        scores = {}
        for cell in SOURCES:
            speedups = [by_model[model][str(seed)]['cycles']['00'] /
                        by_model[model][str(seed)]['cycles'][cell]
                        for model in ('kws', 'vww')]
            scores[cell] = math.prod(speedups)**0.5
        winner = max(scores, key=lambda cell: (scores[cell], cell))
        shared_policy[str(seed)] = {'geomean_speedup_vs_00': scores,
                                    'best_uniform_cell': winner}
    return {'factor_definitions': FACTOR_DEFINITIONS,
            'interpretation': 'All fusion-labelled effects include safe-tail prefetch. Factor interactions do not establish a need for joint search or novelty.',
            'models': by_model, 'shared_policy': shared_policy}


def run(plan, output):
    if output.exists():
        raise FileExistsError(f'preserve prior evidence; use a new --output: {output}')
    output.mkdir(parents=True)
    (output / 'runs').mkdir()
    save(output / 'plan.json', plan)
    report = {'schema': 1, 'status': 'running', 'physical_board': False,
              'plan_sha256': sha(output / 'plan.json'), 'runs': {},
              'scope': plan['scope']}
    save(output / 'report.partial.json', report)
    for model in ('kws', 'vww'):
        for cell in SOURCES:
            fixture = ROOT / plan['fixtures'][model + ':' + cell]['directory']
            for seed in SEEDS:
                key = f'{model}:{cell}:{seed}'
                path = output / 'runs' / f'{model}-{cell}-s{seed}.json'
                completed = subprocess.run([str(EXECUTABLE), str(fixture), str(seed), str(path)],
                                           cwd=ROOT, capture_output=True, text=True)
                if completed.returncode:
                    raise RuntimeError(f'{key}: {completed.stdout}\n{completed.stderr}')
                row = json.loads(path.read_text())
                if (row['status'] != 'passed' or row['physical_board'] or
                        row['stall_seed'] != seed or row['tensor_checks'] != 1):
                    raise ValueError(f'{key}: exact output check failed')
                report['runs'][key] = dict(row, report_sha256=sha(path),
                                           fixture_directory=plan['fixtures'][model + ':' + cell]['directory'])
                save(output / 'report.partial.json', report)
    report['analysis'] = analyze(plan, report['runs'])
    report['status'] = 'passed'
    save(output / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='execute all 16 native RTL fixture runs')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plan = prepare()
    if not args.run:
        print(json.dumps({'status': 'ready', 'fixture_count': len(plan['fixtures']),
                          'executable_sha256': plan['executable_sha256']}, sort_keys=True))
    else:
        report = run(plan, args.output)
        print(json.dumps({'status': report['status'],
                          'runs': len(report['runs']),
                          'analysis': report['analysis']}, sort_keys=True))
