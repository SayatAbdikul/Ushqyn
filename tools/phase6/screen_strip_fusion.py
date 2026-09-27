#!/usr/bin/env python3
"""Preflight a VWW strip-fusion full-model screen on the selected 27 MHz image.

The default verifies sources, exact native results, all fixture bytes, the
routed image, and the matched physical baseline. --run programs volatile FPGA
SRAM and performs exactly ten KWS/VWW inferences (one stress, one warmup, three
timed per model); it is deliberately never invoked by preflight.
"""

import argparse
import json
from pathlib import Path

import run_priority as priority
import run_screening as screening
import screen_channel_compaction as compacted
import screen_pool27 as pool27
from variants import ROOT, sha


STRIP = ROOT / 'work/phase6/strip-fusion-vww-v1'
FULL = STRIP / 'full'
FIXTURES = FULL / 'fixtures'
REPORT = FULL / 'report.json'
SOURCE = ROOT / 'tools/phase6/strip_fusion_vww.py'
NATIVE = ROOT / 'work/phase6/pool-timing-v1/native/Vv2_tiled_host_bridge'
BASELINE = ROOT / 'work/phase6/pool-timing-v1/physical-compacted27-v1'
PLAN_DIR = ROOT / 'work/phase6/strip-fusion-board-v1'
VWW = {'pinned': 'vww-pinned-compacted-strip-timed',
       'stress': 'vww-stress-compacted-strip-check'}
REQUIRED_FILES = ('commands.bin', 'payload.bin', 'input.bin',
                  'output.bin', 'checks.txt', 'schedule.json')


def _rows_from_report(report):
    records = report['fixtures']
    if isinstance(records, list):
        records = {row.get('label', row.get('name')): row for row in records}
    if not isinstance(records, dict):
        raise ValueError('full-model report has no fixture records')
    return records


def _baseline(plan):
    old_plan = json.loads((BASELINE / 'plan.json').read_text())
    report = json.loads((BASELINE / 'report.json').read_text())
    seal = json.loads((BASELINE / 'seal.json').read_text())
    rows = priority.read_rows(BASELINE / 'records.jsonl')
    if (report.get('status') != 'passed-short-screen' or
            report.get('physical_board') is not True or
            (report.get('planned'), report.get('completed')) != (10, 10) or
            old_plan['image'] != plan['image'] or
            old_plan['fixture_names'] != plan['fixture_names'] or
            old_plan['fixture_root'] != plan['fixture_root'] or
            report['plan_sha256'] != sha(BASELINE / 'plan.json') or
            report['records_sha256'] != sha(BASELINE / 'records.jsonl') or
            seal['plan_sha256'] != sha(BASELINE / 'plan.json') or
            seal['report_sha256'] != sha(BASELINE / 'report.json') or
            seal['records_sha256'] != sha(BASELINE / 'records.jsonl') or
            len(rows) != 10 or
            any(row['bitstream_sha256'] != plan['image']['sha256'] for row in rows)):
        raise ValueError('matched compacted27 physical baseline is incomplete or changed')
    for model in ('kws', 'vww'):
        selected = [row for row in rows if row['model'] == model]
        if (len(selected) != 5 or
                sum(row['kind'] == 'stress' for row in selected) != 1 or
                sum(row['kind'] == 'warmup' for row in selected) != 1 or
                sum(row['kind'] == 'timed' for row in selected) != 3 or
                report['summary'][model]['median_cycles'] <= 0):
            raise ValueError(f'matched compacted27 {model} controls are incomplete')
    return dict(folder=str(BASELINE.relative_to(ROOT)),
                plan_sha256=sha(BASELINE / 'plan.json'),
                report_sha256=sha(BASELINE / 'report.json'),
                records_sha256=sha(BASELINE / 'records.jsonl'),
                seal_sha256=sha(BASELINE / 'seal.json'),
                summary=report['summary'])


def _fixture(label, raw, model, sample):
    if not isinstance(raw, dict) or raw.get('verification', {}).get('status') != 'passed':
        raise ValueError(f'{label}: no exact replay certificate')
    if model == 'vww':
        native = raw.get('native')
        if (not isinstance(native, list) or
                [(entry.get('stall_seed', entry.get('seed')), entry.get('status'))
                 for entry in native] != [(0, 'passed'), (6063, 'passed')] or
                any(entry.get('tensor_checks', 0) < 1 for entry in native)):
            raise ValueError(f'{label}: missing two exact native RTL cases')
    files = raw.get('files')
    if not isinstance(files, dict) or any(name not in files for name in REQUIRED_FILES):
        raise ValueError(f'{label}: missing fixture file hashes')
    path = FIXTURES / label
    screening.verify_files(path, files)
    schedule = json.loads((path / 'schedule.json').read_text())
    commands = path / 'commands.bin'
    payload = path / 'payload.bin'
    if (sha(commands) != schedule['program_sha256'] or
            sha(payload) != schedule['image_sha256'] or
            commands.stat().st_size != 16 * schedule['command_count'] or
            commands.stat().st_size > 32_768 or
            payload.stat().st_size > 8 * 1024 * 1024 or
            schedule['final_output']['bytes'] != (path / 'output.bin').stat().st_size or
            (path / 'input.bin').stat().st_size != (490 if model == 'kws' else 27_648)):
        raise ValueError(f'{label}: invalid ABI, public input shape, or memory capacity')
    return dict(raw, label=label, model=model, sample=sample)


def prepare():
    # This invokes all 27 MHz route/source/RTL/UART checks already used by the
    # physical compacted27 campaign. Its fixture identities are our control.
    original = pool27.prepare('compacted')
    baseline = _baseline(original)
    report = json.loads(REPORT.read_text())
    source_hashes = report.get('source_sha256', {})
    if (report.get('status') != 'passed-native' or
            report.get('physical_board') is not False or
            not isinstance(source_hashes, dict) or
            source_hashes.get(str(SOURCE.relative_to(ROOT))) != sha(SOURCE) or
            report.get('selected_native_executable_sha256') != sha(NATIVE)):
        raise ValueError('strip-fusion full-model source or native evidence changed')
    if report.get('model') != 'vww' or report.get('layers') != [3, 6]:
        raise ValueError('strip-fusion report does not target VWW layers 3–6')
    records = _rows_from_report(report)
    names = dict(original['fixture_names'])
    names['vww'] = VWW
    selected = {}
    for model in ('kws', 'vww'):
        for sample in ('pinned', 'stress'):
            label = names[model][sample]
            old_label = original['fixture_names'][model][sample]
            old_path = Path(original['fixture_root']) / old_label
            new_path = FIXTURES / label
            if label not in records:
                raise ValueError(f'{label}: absent from full-model report')
            if model == 'kws':
                raw = original['fixtures'][old_label]
                if records[label]['files'] != raw['files']:
                    raise ValueError(f'{label}: KWS report differs from compacted control')
            else:
                raw = records[label]
            selected[label] = _fixture(label, raw, model, sample)
            for filename in ('input.bin', 'output.bin'):
                if (new_path / filename).read_bytes() != (old_path / filename).read_bytes():
                    raise ValueError(f'{label}: {filename} changed against compacted control')
    if (len(selected) != 4 or len(records) < 4 or
            selected[VWW['pinned']]['files']['commands.bin'] ==
            original['fixtures'][original['fixture_names']['vww']['pinned']]['files']['commands.bin']):
        raise ValueError('strip-fusion command stream is absent')
    plan = dict(original)
    plan.update(schema=4, variant='strip-fusion-vww-pool27-compacted-v1',
                fixture_root=str(FIXTURES), fixture_names=names, fixtures=selected,
                strip_full_report_sha256=sha(REPORT), strip_source_sha256=sha(SOURCE),
                strip_board_runner_sha256=sha(Path(__file__)),
                matched_compacted27_baseline=baseline,
                scope='unchanged 27 MHz image; VWW layers 3–6 strip-fusion schedule; '
                      'KWS control; one stress, one warmup and three pinned timings per model')
    dependencies = dict(original['physical_dependencies'])
    for path in (SOURCE, REPORT, STRIP / 'report.json', Path(__file__),
                 BASELINE / 'plan.json', BASELINE / 'report.json',
                 BASELINE / 'records.jsonl', BASELINE / 'seal.json'):
        dependencies[str(path.relative_to(ROOT))] = sha(path)
    plan['physical_dependencies'] = dependencies
    return plan


def compare(output, plan):
    report = json.loads((output / 'report.json').read_text())
    if (report['status'] != 'passed-short-screen' or report['completed'] != 10 or
            report['plan_sha256'] != sha(output / 'plan.json')):
        raise ValueError('strip-fusion physical screen did not complete exactly')
    reference = plan['matched_compacted27_baseline']['summary']
    result = dict(status='passed-matched-short-comparison', physical_board=True,
                  report_sha256=sha(output / 'report.json'),
                  records_sha256=sha(output / 'records.jsonl'),
                  matched_baseline_report_sha256=plan['matched_compacted27_baseline']['report_sha256'],
                  bitstream_sha256=plan['image']['sha256'], models={})
    for model in ('kws', 'vww'):
        old = reference[model]
        new = report['summary'][model]
        result['models'][model] = dict(
            baseline_cycles=old['median_cycles'], candidate_cycles=new['median_cycles'],
            baseline_latency_ms=old['median_device_latency_ms'],
            candidate_latency_ms=new['median_device_latency_ms'],
            latency_speedup=old['median_device_latency_ms'] / new['median_device_latency_ms'],
            baseline_inferences_per_second=old['device_inferences_per_second'],
            candidate_inferences_per_second=new['device_inferences_per_second'])
    screening.save_json(output / 'strip-comparison.json', result)
    screening.save_json(output / 'strip-seal.json', dict(
        runner_sha256=plan['strip_board_runner_sha256'],
        source_sha256=plan['strip_source_sha256'],
        full_report_sha256=plan['strip_full_report_sha256'],
        plan_sha256=sha(output / 'plan.json'),
        report_sha256=sha(output / 'report.json'),
        records_sha256=sha(output / 'records.jsonl'),
        comparison_sha256=sha(output / 'strip-comparison.json')))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--port', default='/dev/cu.usbserial-20250303171')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    plan = prepare()
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    screening.save_json(PLAN_DIR / 'plan.json', plan)
    if not args.run:
        print(json.dumps(dict(status='preflight-passed', planned=10,
                              bitstream_sha256=plan['image']['sha256'],
                              fixture_names=plan['fixture_names'],
                              baseline_report_sha256=plan['matched_compacted27_baseline']['report_sha256'],
                              plan_sha256=sha(PLAN_DIR / 'plan.json')), sort_keys=True))
        return
    if args.output is None:
        parser.error('--run requires a fresh --output directory')
    output = args.output.resolve()
    compacted.run(plan, output, args.port)
    if (sha(Path(__file__)) != plan['strip_board_runner_sha256'] or
            sha(SOURCE) != plan['strip_source_sha256'] or
            sha(REPORT) != plan['strip_full_report_sha256']):
        raise ValueError('strip-fusion source changed during physical run')
    print(json.dumps(compare(output, plan), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
