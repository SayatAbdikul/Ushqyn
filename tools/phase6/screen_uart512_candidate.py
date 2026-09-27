#!/usr/bin/env python3
"""Audit and briefly screen the integrated stream-mask/512-byte UART image.

Preflight is read-only. --run programs the volatile FPGA image and performs
ten exact KWS/VWW inferences, with no retry or automatic expansion.
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
from uart_burst_512 import BURST_BYTES, BurstTiledClient
from variants import ROOT, check_frozen, sha


BASE = ROOT / 'work/phase6/experiments-v1/stream-mask-uart512-v1'
CORE = ROOT / 'work/phase6/experiments-v1/stream-mask-v1'
FIXTURES = ROOT / 'work/phase6/followup_graph_tail_prefetch'
def _report(path, status):
    value = json.loads(path.read_text())
    if value.get('status') != status:
        raise ValueError(f'{path}: expected {status}, got {value.get("status")}')
    return value


def _tests(path, report, count):
    if sha(path) != report['results_sha256']:
        raise ValueError(f'RTL test XML changed: {path}')
    cases = ET.parse(path).findall('.//testcase')
    if len(cases) != count or any(
            case.find('failure') is not None or case.find('error') is not None
            for case in cases):
        raise ValueError(f'RTL tests missing or failed: {path}')


def prepare(route_report):
    """Prove image, bridge, native model, and schedule identity before JTAG."""
    check_frozen()
    route_path = Path(route_report).resolve()
    if BASE not in route_path.parents:
        raise ValueError('route report must belong to the integrated UART512 image')
    identity = json.loads((BASE / 'identity.json').read_text())
    core_identity = json.loads((CORE / 'identity.json').read_text())
    engine = BASE / 'engine.sv'
    engine_sha = sha(engine)
    if (identity['engine_sha256'] != engine_sha or
            identity['parent_sha256'] != sha(CORE / 'engine.sv') or
            core_identity['engine_sha256'] != engine_sha or
            identity['parser_sha256'] != sha(ROOT / 'rtl/phase6/uart_burst_512_command.sv') or
            identity['bridge_sha256'] != sha(ROOT / 'rtl/phase6/uart_burst_512_bridge.sv') or
            identity['top_sha256'] != sha(ROOT / 'hardware/phase6/uart_burst_512_tiled_host.sv') or
            identity['build_template_sha256'] != sha(ROOT / 'hardware/phase6/build_uart_burst_512.tcl') or
            identity['host_client_sha256'] != sha(ROOT / 'tools/phase6/uart_burst_512.py')):
        raise ValueError('integrated engine or UART512 source identity changed')

    # The new output stream and its legacy activation corner cases were tested
    # on the byte-identical parent core. Check both, not just a generic engine
    # test that would miss stream backpressure and flush behavior.
    core_reports = {}
    for label, count in (('engine', 3), ('edges', 1), ('stream-edges', 1)):
        report = _report(CORE / label / 'report.json', 'passed')
        screening.verify_files(ROOT, report['sources'])
        if report['sources'].get(str((CORE / 'engine.sv').relative_to(ROOT))) != engine_sha:
            raise ValueError(f'{label} RTL test used a different engine')
        _tests(CORE / label / 'results.xml', report, count)
        core_reports[label] = sha(CORE / label / 'report.json')
    integration_path = BASE / 'integration/report.json'
    integration = _report(integration_path, 'passed')
    screening.verify_files(ROOT, integration['sources'])
    if integration['engine_sha256'] != engine_sha:
        raise ValueError('512-byte bridge integration used a different engine')
    _tests(BASE / 'integration/results.xml', integration, 2)

    route = _report(route_path, 'passed-route')
    mhz = route['core_clock_mhz']
    if (mhz not in (22.5, 24, 27) or route['engine_sha256'] != engine_sha or
            route['setup_violated_endpoints'] or route['hold_violated_endpoints'] or
            route['routed_core_fmax_mhz'] < mhz or
            any(item['used'] > item['available'] for item in route['resources'].values())):
        raise ValueError('routed UART512 image is not eligible')
    clock_name = str(mhz).replace('.', 'p')
    # JSON clocks may be 24 rather than 24.0; route-inputs names use 24p0.
    if 'p' not in clock_name:
        clock_name += 'p0'
    route_inputs_path = BASE / f'route-inputs-{clock_name}.json'
    route_inputs = _report(route_inputs_path, 'prepared-route-inputs')
    if (route_inputs['gowin_project'] != 'phase6_uart_burst_512' or
            route_inputs['core_clock_mhz'] != mhz or
            route_inputs['engine_sha256'] != engine_sha or
            route_inputs['sources'].get(str(engine.relative_to(ROOT))) != engine_sha):
        raise ValueError('generated route-input identity mismatch')
    screening.verify_files(ROOT, route_inputs['sources'])
    if not route.get('sources'):
        raise ValueError('route report must hash every generated UART512 input')
    screening.verify_files(ROOT, route['sources'])
    coverage = route.get('source_coverage', {})
    project_path = ROOT / coverage.get('gowin_project', '')
    if (coverage.get('status') != 'passed' or
            coverage.get('project_files') != 18 or
            coverage.get('hashed_sources') != len(route['sources']) or
            coverage.get('generated_inputs') != sorted(route_inputs['sources']) or
            BASE not in project_path.resolve().parents or
            sha(project_path) != coverage.get('gowin_project_sha256')):
        raise ValueError('Gowin project source coverage is not proven')
    for file, digest in route_inputs['sources'].items():
        if route['sources'].get(file) != digest:
            raise ValueError(f'routed image omitted or changed UART512 source: {file}')
    for suffix in ('uart_burst_512_command.sv', 'uart_burst_512_bridge.sv'):
        if not any(file.endswith(suffix) for file in route_inputs['sources']):
            raise ValueError(f'UART512 build omitted {suffix}')
    generated_host = BASE / f'host{clock_name}.sv'
    template = (ROOT / 'hardware/phase6/uart_burst_512_tiled_host.sv').read_text()
    if (template.count('20250000') != 3 or
            generated_host.read_text() != template.replace('20250000', str(round(mhz * 1_000_000)))):
        raise ValueError('UART or SDRAM frequency constants do not match route clock')
    bitstream = ROOT / route['bitstream']
    if BASE not in bitstream.resolve().parents or sha(bitstream) != route['bitstream_sha256']:
        raise ValueError('routed UART512 bitstream identity mismatch')

    manifest_path = FIXTURES / 'fixtures.json'
    manifest = _report(manifest_path, 'passed-replay')
    if len(manifest['fixtures']) != 6:
        raise ValueError('expected six grouped tail-prefetch fixtures')
    fixtures = {}
    for row in manifest['fixtures']:
        name = row['name']
        if (name in fixtures or Path(name).name != name or
                row.get('model') not in ('kws', 'vww') or
                row.get('verification', {}).get('status') != 'passed'):
            raise ValueError(f'invalid grouped fixture: {name}')
        folder = FIXTURES / 'fixtures' / name
        screening.verify_files(folder, row['files'])
        schedule = json.loads((folder / 'schedule.json').read_text())
        if (sha(folder / 'commands.bin') != schedule['program_sha256'] or
                sha(folder / 'payload.bin') != schedule['image_sha256'] or
                (folder / 'commands.bin').stat().st_size != schedule['command_count'] * 16):
            raise ValueError(f'invalid grouped schedule: {name}')
        fixtures[name] = row
    selected = physical._fixture_names(fixtures)
    native_path = BASE / 'native-grouped-tail-prefetch/report.json'
    native = _report(native_path, 'passed')
    native_binary = CORE / 'native/Vv2_tiled_host_bridge'
    core_native = _report(CORE / 'native/report.json', 'passed')
    screening.verify_files(ROOT, core_native['sources'])
    if (core_native['executable_sha256'] != sha(native_binary) or
            core_native['sources'].get(str((CORE / 'engine.sv').relative_to(ROOT))) != engine_sha or
            len(core_native['results']) != 12):
        raise ValueError('stream-mask native executable identity mismatch')
    if (native['engine_sha256'] != engine_sha or
            native['fixture_manifest_sha256'] != sha(manifest_path) or
            native['executable_sha256'] != sha(native_binary) or
            len(native['results']) != 12):
        raise ValueError('grouped full-model native evidence identity mismatch')
    for name in fixtures:
        for seed in (0, 6063):
            matches = [row for row in native['results']
                       if row['fixture'] == name and row['seed'] == seed]
            if len(matches) != 1 or matches[0]['tensor_checks'] < 1:
                raise ValueError(f'full-model native coverage missing: {name}/{seed}')
            result_path = native_path.parent / f'{name}-s{seed}.json'
            if (sha(result_path) != matches[0]['result_sha256'] or
                    json.loads(result_path.read_text())['status'] != 'passed'):
                raise ValueError(f'full-model native evidence changed: {name}/{seed}')
    return dict(schema=1, image='stream-mask-uart512-v1',
                core_clock_hz=round(mhz * 1_000_000), baud=750000,
                bitstream=str(bitstream.relative_to(ROOT)),
                bitstream_sha256=route['bitstream_sha256'],
                route_report=str(route_path.relative_to(ROOT)),
                route_report_sha256=sha(route_path),
                route_inputs_file=str(route_inputs_path.relative_to(ROOT)),
                route_inputs_sha256=sha(route_inputs_path),
                route_sources=route['sources'],
                integration_report_sha256=sha(integration_path),
                core_reports_sha256=core_reports,
                fixture_manifest_sha256=sha(manifest_path),
                native_report_sha256=sha(native_path),
                fixtures=fixtures, selected=selected,
                planned=10, repeats=3, burst_bytes=BURST_BYTES,
                runner_sha256=sha(Path(__file__)),
                dependencies={str(path.relative_to(ROOT)): sha(path) for path in (
                    ROOT / 'tools/phase6/uart_burst_512.py',
                    ROOT / 'tools/phase6/screen_engine_schedule.py',
                    ROOT / 'tools/phase6/run_screening.py',
                    ROOT / 'tools/phase6/run_priority.py',
                    ROOT / 'tools/phase4/tiled_host.py',
                    ROOT / 'tools/phase2/host.py')})


def run(plan, output, port):
    screening.require_board_free(port)
    if output.exists():
        raise FileExistsError('preserve previous physical evidence; choose a fresh output')
    lock = (ROOT / 'work/phase6/physical-board.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output.mkdir(parents=True)
    screening.save_json(output / 'plan.json', plan)
    report = dict(schema=1, status='running', physical_board=True,
                  started_at=screening.timestamp(), planned=plan['planned'], completed=0,
                  image=plan['image'], plan_sha256=sha(output / 'plan.json'))

    def save():
        report['updated_at'] = screening.timestamp()
        screening.save_json(output / 'report.json', report)

    def interrupted(signum, frame):
        raise KeyboardInterrupt('physical UART512 screen interrupted; signed records preserved')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    rows = []
    save()
    try:
        with (output / 'records.jsonl').open('x') as stream:
            screening.verify_files(ROOT, {plan['bitstream']: plan['bitstream_sha256']})
            report['current'] = 'programming'; save()
            with (output / 'program.log').open('x') as log:
                subprocess.run(['openFPGALoader', '-b', 'tangnano20k',
                    '--ftdi-serial', '2025030317', '--freq', '2500000', '-m', '-v',
                    str(ROOT / plan['bitstream'])], stdout=log,
                    stderr=subprocess.STDOUT, timeout=90, check=True)
            report['program_log_sha256'] = sha(output / 'program.log')
            time.sleep(1)
            with serial.Serial(port, plan['baud'], timeout=5, write_timeout=5) as uart:
                uart.reset_input_buffer()
                client = BurstTiledClient(uart)
                client.capabilities()  # Includes an exact FEATURES=512 probe.
                report['feature_burst_bytes'] = BURST_BYTES
                pattern = bytes(((index * 73 + 19) & 255) for index in range(8192))
                started = time.monotonic()
                client.write_external(0x700000, pattern)
                upload = time.monotonic() - started
                if client.read_external(0x700000, len(pattern)) != pattern:
                    raise AssertionError('512-byte burst SDRAM readback mismatch')
                report['burst_probe'] = dict(offset=0x700000, bytes=len(pattern),
                    upload_seconds=upload, sha256=priority.digest(pattern),
                    readback_verified=True)
                save()
                for model in ('kws', 'vww'):
                    for kind in ('stress', 'pinned'):
                        name = plan['selected'][model][kind]
                        folder = FIXTURES / 'fixtures' / name
                        report['current'] = f'{model}-{kind}-load'; save()
                        schedule, loaded = screening.load_fixture(client, folder, plan['fixtures'][name])
                        repeats = 1 if kind == 'stress' else 4
                        for case in range(repeats):
                            report['current'] = f'{model}-{kind}-{case}'; save()
                            data = (folder / 'input.bin').read_bytes()
                            expected = (folder / 'output.bin').read_bytes()
                            row = physical.execute_verified_sample(client, schedule, data, expected, 30)
                            row['device_latency_ms'] = row['elapsed_cycles'] / plan['core_clock_hz'] * 1000
                            row.update(image=plan['image'], model=model, fixture=name,
                                kind=('stress' if kind == 'stress' else 'warmup' if case == 0 else 'timed'),
                                repeat=case if kind == 'stress' else max(0, case-1),
                                input_sha256=priority.digest(data),
                                bitstream_sha256=plan['bitstream_sha256'],
                                load_seconds=loaded if case == 0 else 0,
                                burst_bytes=BURST_BYTES, at=screening.timestamp())
                            priority.append_row(stream, row)
                            rows.append(row)
                            report['completed'] = len(rows); save()
                    print(f'{screening.timestamp()} {model}: {report["completed"]}/10 exact', flush=True)
        stored = priority.read_rows(output / 'records.jsonl')
        if stored != rows or len(rows) != plan['planned']:
            raise ValueError('signed physical record count or identity mismatch')
        summary = {}
        for model in ('kws', 'vww'):
            timings = [row['elapsed_cycles'] for row in rows
                       if row['model'] == model and row['kind'] == 'timed']
            if len(timings) != 3:
                raise ValueError(f'missing {model} timing repetitions')
            median = statistics.median(timings)
            summary[model] = dict(timed_cycles=timings, median_cycles=median,
                median_latency_ms=median / plan['core_clock_hz'] * 1000,
                device_inferences_per_second=plan['core_clock_hz'] / median)
        screening.verify_files(ROOT, plan['dependencies'])
        screening.verify_files(ROOT, plan['route_sources'])
        screening.verify_files(ROOT, {plan['bitstream']: plan['bitstream_sha256']})
        if (sha(Path(__file__)) != plan['runner_sha256'] or
                sha(ROOT / plan['route_report']) != plan['route_report_sha256'] or
                sha(ROOT / plan['route_inputs_file']) != plan['route_inputs_sha256'] or
                sha(FIXTURES / 'fixtures.json') != plan['fixture_manifest_sha256'] or
                sha(BASE / 'integration/report.json') != plan['integration_report_sha256'] or
                sha(BASE / 'native-grouped-tail-prefetch/report.json') != plan['native_report_sha256']):
            raise ValueError('physical evidence source identity changed during run')
        report.update(status='passed-short-screen', summary=summary,
            records_sha256=sha(output / 'records.jsonl'), finished_at=screening.timestamp())
        save()
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
    parser.add_argument('--route-report', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--port', default='/dev/cu.usbserial-20250303171')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    plan = prepare(args.route_report)
    if args.run:
        if args.output is None:
            parser.error('--run requires a fresh --output directory')
        run(plan, args.output.resolve(), args.port)
    else:
        print(json.dumps(dict(status='preflight-passed', image=plan['image'],
            core_clock_hz=plan['core_clock_hz'], planned=plan['planned'],
            bitstream_sha256=plan['bitstream_sha256']), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
