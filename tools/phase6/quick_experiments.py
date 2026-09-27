#!/usr/bin/env python3
"""Ten-inference FPGA screen per image; never starts full accuracy/stability.

Each model: one seeded stress input, one pinned warmup, three pinned timings.
Full intermediate tensors are checked in RTL before this physical final-logit
screen. All uploads are read back. No automatic retry after an ambiguous run.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import serial
import run_screening as screening
import run_priority as priority
from experiments import BASE
from variants import ROOT, check_frozen

sys.path[:0] = [str(ROOT / p) for p in ('tools/phase4', 'tools/phase2')]
from tiled_host import TiledClient


def prepare(labels, schedule_kind='original'):
    check_frozen()
    old = screening.make_plan()
    images = {}
    for label in labels:
        directory = ROOT / 'work/phase6/writeback-v1' if label == 'writeback' else BASE / label
        engine = json.loads((directory / 'engine/report.json').read_text())
        native = json.loads((directory / 'native/report.json').read_text())
        route = json.loads((directory / 'route/report.json').read_text())
        for report in (engine, native):
            if report['status'] != 'passed': raise ValueError(label + ' RTL not passed')
            screening.verify_files(ROOT, report['sources'])
        identity = directory / 'identity.json'
        if identity.exists() and json.loads(identity.read_text())['parameters'].get('SCALAR_LUT'):
            edge = json.loads((directory / 'edges/report.json').read_text())
            edge_xml = directory / 'edges/results.xml'
            edge_cases = ET.parse(edge_xml).findall('.//testcase')
            if (edge['status'] != 'passed' or len(edge_cases) != 1 or screening.sha(edge_xml) != edge['results_sha256']
                    or any(c.find('failure') is not None or c.find('error') is not None for c in edge_cases)):
                raise ValueError('activation exhaustive edge coverage failed')
            screening.verify_files(ROOT, edge['sources'])
        xml = directory / 'engine/results.xml'; cases = ET.parse(xml).findall('.//testcase')
        if (screening.sha(xml) != engine['results_sha256'] or len(cases) != 3
                or any(c.find('failure') is not None or c.find('error') is not None for c in cases)):
            raise ValueError(label + ' engine evidence changed')
        rows = native['results']
        required = {name for name in old['fixtures'] if '-resident-half-overlap-' in name}
        for name in required:
            for seed in (0, 6063):
                group = [r for r in rows if r['fixture'] == name and r['stall_seed'] == seed]
                if len(group) != 1 or group[0]['status'] != 'passed': raise ValueError('native coverage incomplete')
                screening.verify_files(screening.BASE / 'fixtures' / name, group[0]['fixture_files'])
        source_key = next(k for k in native['sources'] if k.endswith('engine.sv'))
        if (route['status'] != 'passed-route' or route['setup_violated_endpoints'] or route['hold_violated_endpoints']
                or route['engine_sha256'] != native['sources'][source_key]
                or route['routed_core_fmax_mhz'] < route['core_clock_mhz']
                or any(r['used'] > r['available'] for r in route['resources'].values())):
            raise ValueError(label + ' route not eligible')
        screening.verify_files(ROOT, {route['bitstream']: route['bitstream_sha256']})
        images[label] = {'file': route['bitstream'], 'sha256': route['bitstream_sha256'],
            'clock_hz': round(route['core_clock_mhz']*1_000_000),
            'evidence': {s: screening.sha(directory / s / 'report.json') for s in ('engine', 'native', 'route')}}
    paused = ROOT / 'work/phase6/priority-v1'
    if json.loads((paused / 'report.json').read_text())['status'] != 'interrupted':
        raise ValueError('full accuracy must remain paused')
    fixtures = old['fixtures']; fixture_root = screening.BASE / 'fixtures'
    names = {m: {s: f'{m}-{s}-resident-half-overlap-'+('timed' if s=='pinned' else 'check')
                 for s in ('pinned','stress')} for m in ('kws','vww')}
    if schedule_kind == 'chain':
        chain = BASE / 'chain'; manifest = json.loads((chain / 'fixtures.json').read_text())
        if manifest['status'] != 'passed-replay': raise ValueError('chain replay incomplete')
        fixtures = {r['name']: r for r in manifest['fixtures']}; fixture_root = chain / 'fixtures'
        names = {m: {s: f'{m}-{s}-chain-'+('timed' if s=='pinned' else 'check')
                     for s in ('pinned','stress')} for m in ('kws','vww')}
        for label, image in images.items():
            route = json.loads((BASE / label / 'route/report.json').read_text())
            native = json.loads((chain / f'native-{route.get("source_label",label)}' / 'report.json').read_text())
            screening.verify_files(ROOT, native['sources'])
            if (native['status'] != 'passed' or native['fixture_manifest_sha256'] != screening.sha(chain / 'fixtures.json')
                    or route['engine_sha256'] not in [v for k,v in native['sources'].items() if k.endswith('engine.sv')]):
                raise ValueError('chain RTL identity failed')
            for name, fixture in fixtures.items():
                for seed in (0,6063):
                    matches = [r for r in native['results'] if r['fixture'] == name and r['stall_seed'] == seed]
                    if len(matches)!=1 or matches[0]['status']!='passed' or matches[0]['fixture_files']!=fixture['files']:
                        raise ValueError('chain RTL coverage failed')
            image['chain_native_sha256'] = screening.sha(chain / f'native-{route.get("source_label",label)}' / 'report.json')
    return {'schema': 1, 'images': images, 'fixtures': fixtures, 'labels': labels,
        'fixture_root': str(fixture_root.relative_to(ROOT)), 'fixture_names': names, 'schedule_kind': schedule_kind,
        'planned': len(labels)*10, 'repeats': 3, 'physical_board': True,
        'scope': 'short final-logit screen after full-model RTL; not full accuracy, stability or release qualification',
        'runner_sha256': screening.sha(Path(__file__)),
        'preserved_accuracy': {p.name: screening.sha(p) for p in paused.iterdir() if p.suffix in ('.json', '.jsonl')}}


def summarize(rows, labels):
    if len(rows) != len(labels)*10: raise ValueError('incomplete quick screen')
    summary = {}
    for label in labels:
        summary[label] = {}
        for model in ('kws', 'vww'):
            group = [r for r in rows if r['variant'] == label and r['model'] == model]
            if sorted(r['kind'] for r in group) != ['stress', 'timed', 'timed', 'timed', 'warmup']:
                raise ValueError('coverage missing or duplicated')
            values = [r['elapsed_cycles'] for r in group if r['kind'] == 'timed']
            if sorted(r['repeat'] for r in group if r['kind'] == 'timed') != [1, 2, 3]:
                raise ValueError('timing repeats invalid')
            if any(r['output_hex'] != r['expected_hex'] for r in group): raise ValueError('output mismatch')
            median = statistics.median(values)
            summary[label][model] = {'median_cycles': median,
                'median_ms': statistics.median(r['device_latency_ms'] for r in group if r['kind'] == 'timed'),
                'range_over_median': (max(values)-min(values))/median}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--schedule', choices=('original','chain'), default='original')
    parser.add_argument('--port', default='/dev/cu.usbserial-20250303171')
    args = parser.parse_args()
    if len(set(args.labels)) != len(args.labels): raise ValueError('duplicate image')
    plan = prepare(args.labels, args.schedule)
    if not args.run:
        print(json.dumps({'status': 'preflight-passed', 'planned': plan['planned'], 'labels': args.labels})); return
    screening.require_board_free(args.port)
    if args.output.exists(): raise FileExistsError('preserve prior evidence; use a fresh directory')
    lock = (ROOT / 'work/phase6/physical-board.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    args.output.mkdir(parents=True); screening.save_json(args.output / 'plan.json', plan)
    report = {'status': 'running', 'started_at': screening.timestamp(), 'pid': os.getpid(),
        'completed': 0, 'planned': plan['planned'], 'output_mismatches': 0,
        'plan_sha256': screening.sha(args.output / 'plan.json'), 'programming': []}
    def save():
        report['updated_at'] = screening.timestamp(); screening.save_json(args.output / 'report.json', report)
    def interrupted(signum, frame): raise KeyboardInterrupt('interrupted; completed records preserved')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    rows = []; save()
    try:
        with (args.output / 'records.jsonl').open('x') as stream:
            for label, image in plan['images'].items():
                screening.require_board_free(args.port)
                screening.verify_files(ROOT, {image['file']: image['sha256']})
                report['current'] = {'variant': label, 'operation': 'programming'}; save()
                log = args.output / f'program-{label}.log'
                with log.open('x') as output:
                    subprocess.run(['openFPGALoader', '-b', 'tangnano20k', '--ftdi-serial', '2025030317',
                        '--freq', '2500000', '-m', '-v', str(ROOT / image['file'])],
                        stdout=output, stderr=subprocess.STDOUT, timeout=90, check=True)
                report['programming'].append({'variant': label, 'at': screening.timestamp(),
                    'bitstream_sha256': image['sha256'], 'log_sha256': screening.sha(log)})
                save(); time.sleep(1)
                with serial.Serial(args.port, 750000, timeout=5, write_timeout=5) as uart:
                    uart.reset_input_buffer(); client = TiledClient(uart); client.capabilities()
                    for model in ('kws', 'vww'):
                        name = plan['fixture_names'][model]['pinned']
                        directory = ROOT / plan['fixture_root'] / name
                        report['current'] = {'variant': label, 'model': model, 'operation': 'load-and-verify'}; save()
                        schedule, loaded = screening.load_fixture(client, directory, plan['fixtures'][name])
                        stress_name = plan['fixture_names'][model]['stress']
                        stress = ROOT / plan['fixture_root'] / stress_name
                        screening.verify_files(stress, plan['fixtures'][stress_name]['files'])
                        report['current']['operation'] = 'short-screen'; save()
                        for case in range(5):
                            source = stress if case == 0 else directory
                            data = (source / 'input.bin').read_bytes(); expected = (source / 'output.bin').read_bytes()
                            row = priority.execute_sample(client, {'schedule': schedule},
                                {'external_base': 0, 'entry': 0}, data, expected, 30)
                            row['device_latency_ms'] = row['elapsed_cycles']/image['clock_hz']*1000
                            row.update(variant=label, model=model, repeat=max(0, case-1),
                                kind='stress' if case == 0 else ('warmup' if case == 1 else 'timed'),
                                input_sha256=priority.digest(data), bitstream_sha256=image['sha256'],
                                at=screening.timestamp(), load_seconds=loaded if case == 0 else 0)
                            priority.append_row(stream, row); rows.append(row); report['completed'] = len(rows); save()
                        print(f'{screening.timestamp()} {report["completed"]}/{plan["planned"]} {label} {model}', flush=True)
        report['summary'] = summarize(rows, args.labels)
        screening.verify_files(ROOT / 'work/phase6/priority-v1', plan['preserved_accuracy'])
        report.update(status='passed-short-screen', finished_at=screening.timestamp(),
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
