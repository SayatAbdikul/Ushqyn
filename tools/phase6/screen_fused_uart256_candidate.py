#!/usr/bin/env python3
"""Short exact board screen for the fused engine with the existing UART256 host.

Preflight reads pinned RTL, native, compiler, Gowin, and bridge evidence.
--run alone programs the volatile bitstream and checks ten KWS/VWW outputs.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
import xml.etree.ElementTree as ET

import serial

import run_priority as priority
import run_screening as screening
import screen_engine_schedule as physical
from uart_burst import BURST_BYTES, BurstTiledClient
from variants import ROOT, check_frozen, sha


BASE = ROOT / 'work/phase6/experiments-v1/fused-activation-v1'
ROUTE = BASE / 'route24/report.json'
PROJECT = BASE / 'route24/phase6_uart_burst/phase6_uart_burst.gprj'
VARIANT = 'fused-activation-uart256-v1'


def _test_report(label, cases):
    folder = BASE / label
    report = json.loads((folder / 'report.json').read_text())
    if report.get('status') != 'passed':
        raise ValueError(f'{label} RTL/integration test did not pass')
    screening.verify_files(ROOT, report['sources'])
    key = str((BASE / 'engine.sv').relative_to(ROOT))
    if report['sources'].get(key) != sha(BASE / 'engine.sv'):
        raise ValueError(f'{label} test used another engine')
    xml = folder / 'results.xml'
    found = ET.parse(xml).findall('.//testcase')
    if (sha(xml) != report['results_sha256'] or len(found) != cases or
            any(case.find('failure') is not None or case.find('error') is not None
                for case in found)):
        raise ValueError(f'{label} test XML changed or failed')
    return sha(folder / 'report.json')


def prepare():
    check_frozen()
    physical.IMAGE_LABEL = BASE.name
    physical.IMAGE_ROOT = BASE
    physical.ROUTE_REPORT = ROUTE
    plan = physical.prepare(BASE, BASE / 'fixtures.json',
                            BASE / 'native/report.json', VARIANT)
    route = json.loads(ROUTE.read_text())
    if (route['status'] != 'passed-route' or route['core_clock_mhz'] != 24 or
            route['engine_sha256'] != sha(BASE / 'engine.sv') or
            route['bitstream_sha256'] != plan['image']['sha256']):
        raise ValueError('fused UART256 routed image identity mismatch')
    for label, count in (('stream-edges', 1), ('scalar-edges', 1),
                         ('integration', 2)):
        plan.setdefault('additional_test_reports', {})[label] = _test_report(label, count)

    # The original route report hashes the generated core, PLL, top and Tcl.
    # Check the Gowin project itself to prove this bitstream also used the
    # audited 256-byte parser and bridge, not the later 512-byte variant.
    project = ET.parse(PROJECT)
    selected = {Path(node.attrib['path']).resolve() for node in
                project.findall('.//File') if node.attrib.get('enable') == '1'}
    must_include = {path.resolve() for path in (
        BASE / 'engine.sv', BASE / 'host24.sv', BASE / 'pll24.v',
        ROOT / 'rtl/phase6/uart_burst_command.sv',
        ROOT / 'rtl/phase6/uart_burst_bridge.sv')}
    if (len(selected) != 18 or not must_include <= selected or
            any('uart_burst_512' in str(path) for path in selected)):
        raise ValueError('Gowin project is not the lean fused UART256 image')
    script = (BASE / 'build24.tcl').read_text()
    if (script.count('rtl/phase6/uart_burst_command.sv') != 1 or
            script.count('rtl/phase6/uart_burst_bridge.sv') != 1 or
            'uart_burst_512' in script):
        raise ValueError('routed Tcl source list is not UART256')
    host = (BASE / 'host24.sv').read_text()
    pll = (BASE / 'pll24.v').read_text()
    if (host.count('24000000') != 3 or
            host.count('BAUD_RATE(750000)') != 2 or
            '.FBDIV_SEL(7), .IDIV_SEL(8)' not in pll):
        raise ValueError('fused UART256 clock/baud configuration mismatch')
    source_paths = selected | {(BASE / 'build24.tcl').resolve()}
    source_hashes = {str(path.relative_to(ROOT)): sha(path)
                     for path in sorted(source_paths)}
    for path, digest in route['sources'].items():
        if source_hashes.get(path) != digest:
            raise ValueError(f'routed core source changed: {path}')
    integration = json.loads((BASE / 'integration/report.json').read_text())
    for name in ('rtl/phase6/uart_burst_command.sv',
                 'rtl/phase6/uart_burst_bridge.sv'):
        if source_hashes.get(name) != integration['sources'].get(name):
            raise ValueError(f'UART256 integration source differs from route: {name}')
    plan.update(schema=2, image_label=BASE.name,
        source_coverage=dict(gowin_project=str(PROJECT.relative_to(ROOT)),
                             gowin_project_sha256=sha(PROJECT),
                             project_files=len(selected),
                             source_sha256=source_hashes),
        protocol_burst_bytes=BURST_BYTES, baud=750000,
        physical_runner_sha256=sha(Path(__file__)),
        physical_dependencies={str(path.relative_to(ROOT)): sha(path) for path in (
            ROOT / 'tools/phase6/uart_burst.py',
            ROOT / 'tools/phase6/screen_engine_schedule.py',
            ROOT / 'tools/phase6/run_screening.py',
            ROOT / 'tools/phase6/run_priority.py',
            ROOT / 'tools/phase4/tiled_host.py',
            ROOT / 'tools/phase2/host.py')})
    if BURST_BYTES != 256 or plan['planned'] != 10:
        raise ValueError('physical screen bound or UART feature size changed')
    return plan


def run(plan, output, port):
    screening.require_board_free(port)
    if output.exists():
        raise FileExistsError('preserve prior evidence; choose a new output')
    lock = (ROOT / 'work/phase6/physical-board.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output.mkdir(parents=True)
    screening.save_json(output / 'plan.json', plan)
    report = dict(schema=2, status='running', physical_board=True,
        started_at=screening.timestamp(), variant=plan['variant'],
        planned=10, completed=0, plan_sha256=sha(output / 'plan.json'))

    def save():
        report['updated_at'] = screening.timestamp()
        screening.save_json(output / 'report.json', report)

    def interrupted(signum, frame):
        raise KeyboardInterrupt('UART256 short screen interrupted; signed records preserved')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    rows = []
    loads = {}
    save()
    try:
        image = plan['image']
        screening.verify_files(ROOT, {image['file']: image['sha256']})
        report['current'] = 'programming'; save()
        with (output / 'program.log').open('x') as log:
            subprocess.run(['openFPGALoader', '-b', 'tangnano20k',
                '--ftdi-serial', '2025030317', '--freq', '2500000', '-m', '-v',
                str(ROOT / image['file'])], stdout=log,
                stderr=subprocess.STDOUT, timeout=90, check=True)
        report['program_log_sha256'] = sha(output / 'program.log')
        time.sleep(1)
        with serial.Serial(port, plan['baud'], timeout=5, write_timeout=5) as uart:
            uart.reset_input_buffer()
            client = BurstTiledClient(uart)
            client.capabilities()  # Read-only FEATURES must advertise 256.
            report['feature_burst_bytes'] = BURST_BYTES
            save()
            with (output / 'records.jsonl').open('x') as stream:
                for model in ('kws', 'vww'):
                    for kind in ('stress', 'pinned'):
                        name = plan['fixture_names'][model][kind]
                        folder = BASE / 'fixtures' / name
                        report['current'] = f'{model}-{kind}-load'; save()
                        schedule, load_seconds = screening.load_fixture(
                            client, folder, plan['fixtures'][name])
                        loads[name] = load_seconds
                        repeats = 1 if kind == 'stress' else 4
                        for case in range(repeats):
                            report['current'] = f'{model}-{kind}-{case}'; save()
                            data = (folder / 'input.bin').read_bytes()
                            expected = (folder / 'output.bin').read_bytes()
                            row = physical.execute_verified_sample(
                                client, schedule, data, expected, 30)
                            row['device_latency_ms'] = (
                                row['elapsed_cycles'] / image['clock_hz'] * 1000)
                            row.update(variant=plan['variant'], model=model,
                                fixture=name, kind=('stress' if kind == 'stress' else
                                    'warmup' if case == 0 else 'timed'),
                                repeat=case if kind == 'stress' else max(0, case-1),
                                input_sha256=priority.digest(data),
                                bitstream_sha256=image['sha256'],
                                burst_bytes=BURST_BYTES,
                                load_seconds=load_seconds if case == 0 else 0,
                                at=screening.timestamp())
                            priority.append_row(stream, row)
                            rows.append(row)
                            report['completed'] = len(rows); save()
                    print(f'{screening.timestamp()} {model}: {len(rows)}/10 exact',
                          flush=True)
        if priority.read_rows(output / 'records.jsonl') != rows or len(rows) != 10:
            raise ValueError('signed physical record round trip or count mismatch')
        summary = {}
        for model in ('kws', 'vww'):
            timed = [row for row in rows if row['model'] == model and row['kind'] == 'timed']
            if len(timed) != 3:
                raise ValueError(f'{model} is missing three timed runs')
            cycles = statistics.median(row['elapsed_cycles'] for row in timed)
            summary[model] = dict(
                median_cycles=cycles,
                median_device_latency_ms=cycles / image['clock_hz'] * 1000,
                device_inferences_per_second=image['clock_hz'] / cycles,
                median_input_upload_seconds=statistics.median(
                    row['input_upload_seconds'] for row in timed),
                median_input_readback_seconds=statistics.median(
                    row['input_readback_seconds'] for row in timed),
                median_wall_seconds=statistics.median(
                    row['wall_seconds'] for row in timed),
                pinned_fixture_load_seconds=loads[plan['fixture_names'][model]['pinned']])
        screening.verify_files(ROOT, plan['dependency_sha256'])
        screening.verify_files(ROOT, plan['physical_dependencies'])
        screening.verify_files(ROOT, plan['source_coverage']['source_sha256'])
        screening.verify_files(ROOT, {image['file']: image['sha256']})
        screening.verify_files(ROOT / 'work/phase6/priority-v1', plan['preserved_accuracy'])
        if (sha(Path(__file__)) != plan['physical_runner_sha256'] or
                sha(BASE / 'fixtures.json') != plan['manifest_sha256'] or
                sha(BASE / 'native/report.json') != plan['native_report_sha256'] or
                sha(PROJECT) != plan['source_coverage']['gowin_project_sha256'] or
                any(sha(BASE / label / 'report.json') != digest
                    for label, digest in plan['additional_test_reports'].items())):
            raise ValueError('UART256 physical evidence identity changed during run')
        report.update(status='passed-short-screen',
            summary=summary, records_sha256=sha(output / 'records.jsonl'),
            finished_at=screening.timestamp())
        save()
        screening.save_json(output / 'seal.json', dict(
            report_sha256=sha(output / 'report.json'),
            plan_sha256=sha(output / 'plan.json'),
            records_sha256=sha(output / 'records.jsonl'),
            runner_sha256=plan['physical_runner_sha256']))
        print(json.dumps(summary, sort_keys=True), flush=True)
    except BaseException as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['failure'] = repr(error)
        if (output / 'records.jsonl').exists():
            report['records_sha256'] = sha(output / 'records.jsonl')
        save()
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--port', default='/dev/cu.usbserial-20250303171')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    plan = prepare()
    if args.run:
        if args.output is None:
            parser.error('--run requires a fresh --output directory')
        run(plan, args.output.resolve(), args.port)
    else:
        print(json.dumps(dict(status='preflight-passed',
            variant=plan['variant'], planned=plan['planned'],
            bitstream_sha256=plan['image']['sha256'],
            protocol_burst_bytes=plan['protocol_burst_bytes']),
            sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
