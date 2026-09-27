#!/usr/bin/env python3
"""Preflight a composable VWW pair3+7+11 screen on the selected 27 MHz FPGA.

The default is read-only except for a frozen plan JSON. --run performs exactly
ten short, exact inferences on the unchanged routed bitstream. It compares the
new VWW schedule with the completed pair3+11 physical screen, while KWS remains
an unchanged control.
"""

import argparse
import json
from pathlib import Path
import statistics

import run_priority as priority
import run_screening as screening
import screen_channel_compaction as compacted
import screen_pair11_fusion as pair11
from variants import ROOT, sha


PAIR7 = ROOT / 'work/phase6/strip-fusion-pair7-v1'
FULL = PAIR7 / 'full'
FIXTURES = FULL / 'fixtures'
REPORT = FULL / 'report.json'
SOURCE = ROOT / 'tools/phase6/strip_fusion_pair7.py'
PAIR11_PHYSICAL = ROOT / 'work/phase6/pair11-fusion-board-v1/physical-short-v1'
PLAN_DIR = ROOT / 'work/phase6/pair7-fusion-board-v1'
VWW = {'pinned': 'vww-pinned-compacted-strip3-7-11-timed',
       'stress': 'vww-stress-compacted-strip3-7-11-check'}
REQUIRED_FILES = ('commands.bin', 'payload.bin', 'input.bin',
                  'output.bin', 'checks.txt', 'schedule.json')


def _fixture(label, raw, model, sample):
    if not isinstance(raw, dict) or raw.get('verification', {}).get('status') != 'passed':
        raise ValueError(f'{label}: no exact full-model replay certificate')
    if model == 'vww':
        native = raw.get('native')
        if (not isinstance(native, list) or
                [(row.get('stall_seed', row.get('seed')), row.get('status'))
                 for row in native] != [(0, 'passed'), (6063, 'passed')] or
                any(row.get('tensor_checks', 0) < 1 for row in native)):
            raise ValueError(f'{label}: missing exact full-model native seed coverage')
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


def _matched_pair11(plan):
    prior = json.loads((PAIR11_PHYSICAL / 'plan.json').read_text())
    report = json.loads((PAIR11_PHYSICAL / 'report.json').read_text())
    seal = json.loads((PAIR11_PHYSICAL / 'seal.json').read_text())
    extra = json.loads((PAIR11_PHYSICAL / 'pair11-seal.json').read_text())
    comparison = json.loads((PAIR11_PHYSICAL / 'pair11-comparison.json').read_text())
    rows = priority.read_rows(PAIR11_PHYSICAL / 'records.jsonl')
    if (report.get('status') != 'passed-short-screen' or
            report.get('physical_board') is not True or
            (report.get('planned'), report.get('completed')) != (10, 10) or
            prior['image'] != plan['image'] or
            prior['fixture_names'] != plan['fixture_names'] or
            prior['fixture_root'] != plan['fixture_root'] or
            report['plan_sha256'] != sha(PAIR11_PHYSICAL / 'plan.json') or
            report['records_sha256'] != sha(PAIR11_PHYSICAL / 'records.jsonl') or
            seal['plan_sha256'] != sha(PAIR11_PHYSICAL / 'plan.json') or
            seal['report_sha256'] != sha(PAIR11_PHYSICAL / 'report.json') or
            seal['records_sha256'] != sha(PAIR11_PHYSICAL / 'records.jsonl') or
            extra['runner_sha256'] != sha(ROOT / 'tools/phase6/screen_pair11_fusion.py') or
            extra['source_sha256'] != plan['pair11_source_sha256'] or
            extra['full_report_sha256'] != plan['pair11_full_report_sha256'] or
            extra['plan_sha256'] != sha(PAIR11_PHYSICAL / 'plan.json') or
            extra['report_sha256'] != sha(PAIR11_PHYSICAL / 'report.json') or
            extra['records_sha256'] != sha(PAIR11_PHYSICAL / 'records.jsonl') or
            extra['comparison_sha256'] != sha(PAIR11_PHYSICAL / 'pair11-comparison.json') or
            comparison.get('status') != 'passed-matched-short-comparison' or
            comparison['report_sha256'] != sha(PAIR11_PHYSICAL / 'report.json') or
            comparison['matched_pair3_report_sha256'] !=
                plan['matched_pair3_baseline']['report_sha256'] or
            len(rows) != 10 or
            any(row['bitstream_sha256'] != plan['image']['sha256'] or
                row['output_hex'] != row['expected_hex'] for row in rows)):
        raise ValueError('matched pair3+11 physical evidence is incomplete or changed')
    for model in ('kws', 'vww'):
        model_rows = [row for row in rows if row['model'] == model]
        timed = [row['elapsed_cycles'] for row in model_rows if row['kind'] == 'timed']
        if (len(model_rows) != 5 or len(timed) != 3 or
                sum(row['kind'] == 'stress' for row in model_rows) != 1 or
                sum(row['kind'] == 'warmup' for row in model_rows) != 1 or
                statistics.median(timed) != report['summary'][model]['median_cycles']):
            raise ValueError(f'pair3+11 {model} physical timing/control changed')
    return dict(folder=str(PAIR11_PHYSICAL.relative_to(ROOT)),
                plan_sha256=sha(PAIR11_PHYSICAL / 'plan.json'),
                report_sha256=sha(PAIR11_PHYSICAL / 'report.json'),
                records_sha256=sha(PAIR11_PHYSICAL / 'records.jsonl'),
                seal_sha256=sha(PAIR11_PHYSICAL / 'seal.json'),
                pair11_seal_sha256=sha(PAIR11_PHYSICAL / 'pair11-seal.json'),
                comparison_sha256=sha(PAIR11_PHYSICAL / 'pair11-comparison.json'),
                summary=report['summary'])


def prepare():
    # Inherit all selected RTL/route/native, compacted graph, pair3 and pair11
    # source checks. The pair3+11 physical baseline isolates pair7's increment.
    original = pair11.prepare()
    baseline = _matched_pair11(original)
    report = json.loads(REPORT.read_text())
    sources = report.get('source_sha256', {})
    if (report.get('status') != 'passed-native' or
            report.get('physical_board') is not False or
            report.get('model') != 'vww' or
            report.get('layers') != [3, 6, 7, 10, 11, 14] or
            not isinstance(sources, dict) or
            sources.get(str(SOURCE.relative_to(ROOT))) != sha(SOURCE) or
            sources.get(str(pair11.SOURCE.relative_to(ROOT))) != sha(pair11.SOURCE) or
            sources.get(str(pair11.pair3.SOURCE.relative_to(ROOT))) !=
                sha(pair11.pair3.SOURCE) or
            report.get('selected_native_executable_sha256') != sha(pair11.pair3.NATIVE)):
        raise ValueError('pair7 full-model source, native, or fused-pair evidence changed')
    screening.verify_files(ROOT, sources)
    records = pair11.pair3._rows_from_report(report)
    names = dict(original['fixture_names'])
    names['vww'] = VWW
    selected = {}
    for model in ('kws', 'vww'):
        for sample in ('pinned', 'stress'):
            label = names[model][sample]
            if label not in records:
                raise ValueError(f'{label}: absent from pair7 full-model report')
            previous = original['fixture_names'][model][sample]
            before = Path(original['fixture_root']) / previous
            now = FIXTURES / label
            if model == 'kws':
                raw = original['fixtures'][previous]
                if records[label]['files'] != raw['files']:
                    raise ValueError(f'{label}: KWS control fixture changed')
            else:
                raw = records[label]
            selected[label] = _fixture(label, raw, model, sample)
            for filename in ('input.bin', 'output.bin'):
                if (now / filename).read_bytes() != (before / filename).read_bytes():
                    raise ValueError(f'{label}: public {filename} changed against pair3+11')
    if (len(selected) != 4 or len(records) < 4 or
            selected[VWW['pinned']]['files']['commands.bin'] ==
                original['fixtures'][original['fixture_names']['vww']['pinned']]['files']['commands.bin']):
        raise ValueError('composable pair7 command stream absent')
    plan = dict(original)
    plan.update(schema=6, variant='strip-fusion-vww-pair3-7-11-pool27-v1',
                fixture_root=str(FIXTURES), fixture_names=names, fixtures=selected,
                pair7_full_report_sha256=sha(REPORT), pair7_source_sha256=sha(SOURCE),
                pair7_board_runner_sha256=sha(Path(__file__)),
                matched_pair11_baseline=baseline,
                scope='unchanged 27 MHz image; composable VWW pair3+7+11 schedule; '
                      'KWS control; one stress, one warmup and three pinned timings per model')
    dependencies = dict(original['physical_dependencies'])
    for path in (SOURCE, REPORT, Path(__file__),
                 *(PAIR11_PHYSICAL / name for name in
                   ('plan.json', 'report.json', 'records.jsonl', 'seal.json',
                    'pair11-comparison.json', 'pair11-seal.json'))):
        dependencies[str(path.relative_to(ROOT))] = sha(path)
    plan['physical_dependencies'] = dependencies
    return plan


def compare(output, plan):
    report = json.loads((output / 'report.json').read_text())
    if (report['status'] != 'passed-short-screen' or report['completed'] != 10 or
            report['plan_sha256'] != sha(output / 'plan.json')):
        raise ValueError('pair7 physical short screen is incomplete')
    reference = plan['matched_pair11_baseline']['summary']
    result = dict(status='passed-matched-short-comparison', physical_board=True,
                  report_sha256=sha(output / 'report.json'),
                  records_sha256=sha(output / 'records.jsonl'),
                  matched_pair11_report_sha256=plan['matched_pair11_baseline']['report_sha256'],
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
    screening.save_json(output / 'pair7-comparison.json', result)
    screening.save_json(output / 'pair7-seal.json', dict(
        runner_sha256=plan['pair7_board_runner_sha256'],
        source_sha256=plan['pair7_source_sha256'],
        full_report_sha256=plan['pair7_full_report_sha256'],
        plan_sha256=sha(output / 'plan.json'),
        report_sha256=sha(output / 'report.json'),
        records_sha256=sha(output / 'records.jsonl'),
        comparison_sha256=sha(output / 'pair7-comparison.json')))
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
                              pair11_report_sha256=plan['matched_pair11_baseline']['report_sha256'],
                              plan_sha256=sha(PLAN_DIR / 'plan.json')), sort_keys=True))
        return
    if args.output is None:
        parser.error('--run requires a fresh --output directory')
    output = args.output.resolve()
    compacted.run(plan, output, args.port)
    if (sha(Path(__file__)) != plan['pair7_board_runner_sha256'] or
            sha(SOURCE) != plan['pair7_source_sha256'] or
            sha(REPORT) != plan['pair7_full_report_sha256']):
        raise ValueError('pair7 source changed during physical run')
    print(json.dumps(compare(output, plan), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
