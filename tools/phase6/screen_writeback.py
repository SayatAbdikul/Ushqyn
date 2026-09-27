#!/usr/bin/env python3
"""Short physical comparison of direct writeback against the screened spatial image.

Read-only preflight by default. --run requires exclusive ownership and successful
engine, full-model RTL and routed timing evidence for this exact candidate.
The paused full-accuracy campaign stays paused. No automatic expanded tests.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import serial
import run_screening as screening
import run_priority as priority
from run_writeback import BASE, ENGINE

ROOT = screening.ROOT
sys.path[:0] = [str(ROOT / p) for p in ('compiler', 'tools/phase4', 'tools/phase2')]
from tiled_host import TiledClient
from hardware_v2 import Descriptor


def prepare():
    old = screening.make_plan()
    evidence = {name: json.loads((BASE / name / 'report.json').read_text())
                for name in ('engine', 'native', 'route')}
    for name in ('engine', 'native'):
        if evidence[name]['status'] != 'passed': raise ValueError(name + ' did not pass')
        screening.verify_files(ROOT, evidence[name]['sources'])
    engine_xml = BASE / 'engine/results.xml'
    cases = ET.parse(engine_xml).findall('.//testcase')
    if (evidence['engine']['results_sha256'] != screening.sha(engine_xml) or len(cases) != 3
            or any(c.find('failure') is not None or c.find('error') is not None for c in cases)):
        raise ValueError('engine regression evidence is incomplete or changed')
    route = evidence['route']
    if (route['status'] != 'passed-route' or route['engine_sha256'] != screening.sha(ENGINE)
            or route['core_clock_mhz'] != 20.25 or route['routed_core_fmax_mhz'] < 20.25
            or route['setup_violated_endpoints'] or route['hold_violated_endpoints']
            or any(r['used'] > r['available'] for r in route['resources'].values())
            or route['build_tcl_sha256'] != screening.sha(ROOT / 'hardware/phase6/build.tcl')):
        raise ValueError('candidate route/timing identity is invalid')
    screening.verify_files(ROOT, {route['bitstream']: route['bitstream_sha256']})
    fixtures = {f['name']: f for f in json.loads((screening.BASE / 'fixtures.json').read_text())['fixtures']
                if '-resident-half-' in f['name']}
    rows = evidence['native']['results']
    if len(rows) != 24: raise ValueError('native coverage incomplete')
    reference = json.loads((screening.BASE / 'native-spatial/report.json').read_text())
    screening.verify_files(ROOT, reference['source_sha256'])
    savings = []
    for name, fixture in fixtures.items():
        directory = screening.BASE / 'fixtures' / name
        screening.verify_files(directory, fixture['files'])
        for seed in (0, 6063):
            group = [r for r in rows if r['fixture'] == name and r['stall_seed'] == seed]
            if (len(group) != 1 or group[0]['status'] != 'passed'
                    or group[0]['fixture_files'] != fixture['files']
                    or group[0]['tensor_checks'] != len((directory / 'checks.txt').read_text().splitlines())):
                raise ValueError('native fixture coverage/identity failed')
        # Serial fixed-RAM cycle savings must equal the removed states exactly.
        if '-serial-' in name:
            stages = json.loads((directory / 'schedule.json').read_text())['stages']
            descriptors = [Descriptor.decode(bytes.fromhex(s['descriptor_hex'])) for s in stages]
            expected = sum(d.outputs * (1 if d.opcode == 3 else 2) for d in descriptors)
            previous = next(r for r in reference['results'] if r['fixture'] == name and r['stall_seed'] == 0)
            current = next(r for r in rows if r['fixture'] == name and r['stall_seed'] == 0)
            if previous['elapsed_cycles'] - current['elapsed_cycles'] != expected:
                raise ValueError('serial cycle delta does not match the removed control states')
            savings.append({'fixture': name, 'saved_cycles': expected})
    paused = ROOT / 'work/phase6/priority-v1'
    report = json.loads((paused / 'report.json').read_text())
    if report['status'] != 'interrupted' or report['output_mismatches']:
        raise ValueError('full-accuracy campaign is not safely interrupted')
    identity, _, initial, bundles, _, locations = priority.prepare()
    if identity != json.loads((paused / 'plan.json').read_text()):
        raise ValueError('paused campaign identities changed')
    if not priority.confirmation_summary(priority.read_rows(paused / 'confirmation.jsonl'), initial)['passed']:
        raise ValueError('paused campaign confirmation invalid')
    for model, bundle in bundles.items():
        priority.check_prefix(priority.read_rows(paused / f'{model}-accuracy.jsonl'), model,
            bundle['source'][5], identity['images']['spatial']['sha256'], bundle['schedule'], locations[model])
    order = ['spatial', 'writeback']; random.Random(20260927).shuffle(order)
    return {'schema': 1, 'core_clock_hz': screening.CLOCK_HZ, 'baud': 750000,
            'images': {'spatial': old['images']['spatial'],
                       'writeback': {'file': route['bitstream'], 'sha256': route['bitstream_sha256']}},
            'fixtures': fixtures, 'timing_order': order, 'serial_cycle_savings': savings,
            'planned': {'correctness': 4, 'warmup': 4, 'timed': 40, 'total': 48},
            'evidence_sha256': {name: screening.sha(BASE / name / 'report.json') for name in evidence},
            'sources': {str(p.relative_to(ROOT)): screening.sha(p) for p in (
                Path(__file__), ROOT / 'tools/phase6/run_screening.py',
                ROOT / 'tools/phase6/run_writeback.py',
                ROOT / 'tools/phase6/run_priority.py', ROOT / 'tools/phase4/tiled_host.py',
                ROOT / 'tools/phase2/host.py', ENGINE)},
            'preserved_accuracy': {p.name: screening.sha(p) for p in sorted(paused.iterdir()) if p.suffix in ('.json', '.jsonl')},
            'scope': 'selected-input correctness and one paired timing session; not full accuracy or G6 closure'}


def summarize(rows):
    if len(rows) != 48: raise ValueError('incomplete screening')
    timings = {}; checks = [r for r in rows if r['kind'] == 'correctness']
    if len(checks) != 4 or len({r['fixture'] for r in checks}) != 4:
        raise ValueError('incomplete correctness coverage')
    for variant in ('spatial', 'writeback'):
        timings[variant] = {}
        for model in ('kws', 'vww'):
            group = [r for r in rows if r['variant'] == variant and r['model'] == model]
            warmups = [r for r in group if r['kind'] == 'warmup']
            timed = [r for r in group if r['kind'] == 'timed']
            if len(warmups) != 1 or sorted(r['repeat'] for r in timed) != list(range(1, 11)):
                raise ValueError('timing coverage incomplete')
            values = [r['elapsed_cycles'] for r in timed]; median = statistics.median(values)
            timings[variant][model] = {'median_cycles': median,
                'median_ms': median / screening.CLOCK_HZ * 1000,
                'range_over_median': (max(values)-min(values))/median}
    comparisons = {m: {'speedup': timings['spatial'][m]['median_cycles']/timings['writeback'][m]['median_cycles'],
                       'latency_reduction': 1-timings['writeback'][m]['median_cycles']/timings['spatial'][m]['median_cycles']}
                   for m in ('kws', 'vww')}
    stable = all(t['range_over_median'] <= .02 for models in timings.values() for t in models.values())
    return {'timings': timings, 'comparisons': comparisons, 'stable': stable,
            'improves_both_models': all(c['speedup'] > 1 for c in comparisons.values()),
            'correctness_tensor_checks': sum(len(r['tensor_checks']) for r in checks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--output', type=Path, default=BASE / 'physical')
    parser.add_argument('--port', default='/dev/cu.usbserial-20250303171')
    args = parser.parse_args(); plan = prepare()
    if not args.run:
        print(json.dumps({'status': 'preflight-passed', 'planned': plan['planned'],
                          'serial_cycle_savings': plan['serial_cycle_savings']})); return
    screening.require_board_free(args.port)
    if args.output.exists(): raise FileExistsError('preserve prior evidence; select a fresh output directory')
    lock = (ROOT / 'work/phase6/physical-board.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    args.output.mkdir(parents=True); screening.save_json(args.output / 'plan.json', plan)
    report = {'status': 'running', 'physical_board': True, 'started_at': screening.timestamp(),
              'pid': os.getpid(), 'completed': 0, 'output_mismatches': 0, 'programming_events': [],
              'plan_sha256': screening.sha(args.output / 'plan.json')}
    def save():
        report['updated_at'] = screening.timestamp(); screening.save_json(args.output / 'report.json', report)
    def interrupted(signum, frame): raise KeyboardInterrupt('interrupted; completed records preserved')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    rows = []; save()
    try:
        with (args.output / 'records.jsonl').open('x') as stream:
            for stage, variant in [('correctness', 'writeback')] + [('timing', v) for v in plan['timing_order']]:
                screening.require_board_free(args.port)
                image = plan['images'][variant]; screening.verify_files(ROOT, {image['file']: image['sha256']})
                log = args.output / f'program-{stage}-{variant}.log'
                report['current'] = {'stage': stage, 'variant': variant, 'operation': 'programming'}; save()
                with log.open('x') as output:
                    subprocess.run(['openFPGALoader', '-b', 'tangnano20k', '--ftdi-serial', '2025030317',
                        '--freq', '2500000', '-m', '-v', str(ROOT / image['file'])],
                        stdout=output, stderr=subprocess.STDOUT, timeout=90, check=True)
                report['programming_events'].append({'stage': stage, 'variant': variant,
                    'at': screening.timestamp(), 'bitstream_sha256': image['sha256'], 'log_sha256': screening.sha(log)})
                save(); time.sleep(1)
                with serial.Serial(args.port, 750000, timeout=5, write_timeout=5) as uart:
                    uart.reset_input_buffer(); client = TiledClient(uart); client.capabilities(); client.wait_idle(30)
                    for model in ('kws', 'vww'):
                        for sample in (('pinned', 'stress') if stage == 'correctness' else ('pinned',)):
                            fixture = f'{model}-{sample}-resident-half-overlap-' + ('check' if stage == 'correctness' else 'timed')
                            directory = screening.BASE / 'fixtures' / fixture
                            report['current'] = {'stage': stage, 'variant': variant, 'fixture': fixture, 'operation': 'load-and-verify'}; save()
                            schedule, loaded = screening.load_fixture(client, directory, plan['fixtures'][fixture])
                            for repeat in range(1 if stage == 'correctness' else 11):
                                row = screening.execute(client, directory, schedule, 30)
                                row.update(variant=variant, model=model, fixture=fixture, repeat=repeat,
                                    kind='correctness' if stage == 'correctness' else ('warmup' if repeat == 0 else 'timed'),
                                    bitstream_sha256=image['sha256'], at=screening.timestamp(), load_seconds=loaded if repeat == 0 else 0)
                                priority.append_row(stream, row); rows.append(row)
                                report['completed'] = len(rows); save()
                            print(f'{screening.timestamp()} {len(rows)}/48: {variant} {fixture}', flush=True)
        report['summary'] = summarize(rows)
        screening.verify_files(ROOT / 'work/phase6/priority-v1', plan['preserved_accuracy'])
        accepted = report['summary']['stable'] and report['summary']['improves_both_models']
        report.update(status='passed-screening' if accepted else 'rejected-performance', finished_at=screening.timestamp(),
                      records_sha256=screening.sha(args.output / 'records.jsonl'))
        save(); print(json.dumps(report['summary']), flush=True)
    except BaseException as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['failure'] = repr(error)
        if isinstance(error, screening.OutputMismatch): report['output_mismatches'] += 1
        save(); raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()


if __name__ == '__main__': main()
