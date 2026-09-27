#!/usr/bin/env python3
"""Run the bounded, user-approved Phase 6 physical screening (156 inferences).

The default only verifies artifacts. --run programs volatile FPGA SRAM after
auditing the completed Phase 5 switch campaign. Accuracy queues stay paused.
No retries of programming, launches, or UART failures; no automatic expansion.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'work/phase6/optimization'
VARIANTS = ('baseline', 'c256p1', 'spatial')
SEED = 20260926
CLOCK_HZ = 20_250_000


class OutputMismatch(AssertionError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def verify_files(directory, files):
    for name, digest in files.items():
        path = directory / name
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f'artifact hash mismatch: {path}')


def make_plan():
    """Freeze a subset of the existing audited artifacts, not new models."""
    sys.path.insert(0, str(ROOT / 'tools/phase6'))
    from variants import check_frozen
    check_frozen()
    board = json.loads((BASE / 'board-plan.json').read_text())
    summary = json.loads((BASE / 'summary.json').read_text())
    if (summary['status'] != 'passed-boardless-milestone'
            or sha(BASE / 'summary.json') != board['simulation_summary_sha256']):
        raise ValueError('boardless audit identity mismatch')
    verify_files(ROOT, summary['source_sha256'])
    fixtures = {f['name']: f for f in json.loads(
        (BASE / 'fixtures.json').read_text())['fixtures']}
    selected = {}
    blocks = []
    rng = random.Random(SEED)
    for stage in ('correctness', 'timing'):
        hardware = list(VARIANTS)
        if stage == 'timing':
            rng.shuffle(hardware)
        for variant in hardware:
            cases = []
            for model in ('kws', 'vww'):
                for resident in (False, True):
                    for sample in (('pinned', 'stress') if stage == 'correctness' else ('pinned',)):
                        name = (f'{model}-{sample}-' + ('resident-' if resident else '')
                                + 'half-overlap-' + ('check' if stage == 'correctness' else 'timed'))
                        source = fixtures[name]
                        directory = BASE / 'fixtures' / name
                        verify_files(directory, source['files'])
                        selected[name] = source
                        cases.append(name)
            rng.shuffle(cases)
            blocks.append({'stage': stage, 'variant': variant, 'fixtures': cases})
    images = {v: board['images'][v] for v in VARIANTS}
    for image in images.values():
        verify_files(ROOT, {image['file']: image['sha256']})
        if image.get('routed_core_fmax_mhz', 22.668) <= CLOCK_HZ / 1e6:
            raise ValueError('candidate has insufficient routed clock frequency')
    projections = {(r['variant'], r['fixture']): r['physical_dma_fit_projection_cycles']
                   for r in summary['cost_checks']}
    return {'schema': 1, 'seed': SEED, 'core_clock_hz': CLOCK_HZ,
            'planned': {'correctness': 24, 'warmup': 12, 'timed': 120, 'total': 156},
            'repeats': 10, 'blocks': blocks, 'fixtures': selected, 'images': images,
            'projections': {v: {name: projections[v, name] for name in selected
                                if name.endswith('-timed')} for v in VARIANTS},
            'board_plan_sha256': sha(BASE / 'board-plan.json'),
            'boardless_summary_sha256': sha(BASE / 'summary.json'),
            'fixtures_manifest_sha256': sha(BASE / 'fixtures.json'),
            'runner_sha256': sha(Path(__file__)),
            'protocol_sources': {str(p.relative_to(ROOT)): sha(p) for p in (
                ROOT / 'tools/phase4/tiled_host.py', ROOT / 'tools/phase2/host.py')},
            'authorization': 'User requested short Phase 6 screening after the completed switch test; remaining Phase 5 queues are paused.',
            'limits': ['selected-input screening, not full-dataset accuracy or G6 closure',
                       'no measured power/energy', 'one timing session per hardware version',
                       'no automatic confirmation or expanded campaign'],
            'criteria': {'timing_range_over_median_max': 0.02,
                         'geomean_speedup_min': 1.3, 'per_model_latency_regression_max': 0.05,
                         'prediction_median_abs_error_max': 0.10,
                         'prediction_max_abs_error_max': 0.20}}


def audit_switch():
    sys.path.insert(0, str(ROOT / 'tools/phase5'))
    from audit import inspect_dual_release, inspect_full_campaign
    from schedule import create_plan
    release = inspect_dual_release(ROOT)
    if not inspect_full_campaign(ROOT, 'switch', release, create_plan()):
        raise ValueError('complete, independently audited Phase 5 switch pass required')
    parent = ROOT / 'work/phase5'
    return {name: sha(parent / name) for name in (
        'dual-switch-full-v2.json', 'dual-switch-full-v2.jsonl')}


def require_board_free(port):
    processes = subprocess.run(['/bin/ps', '-axo', 'pid=,command='],
                               capture_output=True, text=True, check=True).stdout
    names = ('run_dual_board.py', 'run_remaining_campaigns.py',
             'run_baselines_after_accuracy.py', 'run_baseline_probe.py')
    for line in processes.splitlines():
        if any(f'tools/phase5/{name}' in line for name in names):
            raise RuntimeError(f'Phase 5 board owner or queue still active: {line.strip()}')
    for device in (port, port.replace('/cu.', '/tty.')):
        result = subprocess.run(['/usr/sbin/lsof', '-t', device],
                                capture_output=True, text=True, timeout=20)
        if result.returncode not in (0, 1) or result.stderr.strip():
            raise RuntimeError(f'could not verify UART ownership: {result.stderr}')
        if result.stdout.strip():
            raise RuntimeError(f'UART already in use by PID(s): {result.stdout.strip()}')


def check_counters(registers, command_count):
    if len(registers) != 32 or registers[1] or registers[4:8] != b'SEQ4':
        raise AssertionError(f'sequencer status/capability failure: {registers.hex()}')
    elapsed, engine, dma, overlap, index, _ = struct.unpack('<6I', registers[8:])
    if index != command_count - 1:
        raise AssertionError(f'incomplete sequence: {index}, expected {command_count - 1}')
    if (min(elapsed, engine, dma) <= 0 or overlap > min(engine, dma)
            or engine + dma - overlap > elapsed):
        raise AssertionError(f'inconsistent counters: {(elapsed, engine, dma, overlap)}')
    return dict(elapsed_cycles=elapsed, engine_cycles=engine, dma_cycles=dma,
                overlap_cycles=overlap, command_index=index,
                device_latency_ms=elapsed / CLOCK_HZ * 1000)


def load_fixture(client, directory, pins):
    from host import RESET
    verify_files(directory, pins['files'])
    commands = (directory / 'commands.bin').read_bytes()
    payload = (directory / 'payload.bin').read_bytes()
    schedule = json.loads((directory / 'schedule.json').read_text())
    if (hashlib.sha256(commands).hexdigest() != schedule['program_sha256']
            or hashlib.sha256(payload).hexdigest() != schedule['image_sha256']
            or len(commands) != schedule['command_count'] * 16):
        raise ValueError('fixture commands, payload or schedule disagree')
    client.wait_idle(30)
    client.exchange(RESET)
    client.write(0x41001c, b'\0\0')
    started = time.monotonic()
    client.write_external(0, payload)
    if client.read_external(0, len(payload)) != payload:
        raise AssertionError('SDRAM payload readback mismatch')
    client.write(0x500000, commands)
    if client.read(0x500000, len(commands)) != commands:
        raise AssertionError('command memory readback mismatch')
    return schedule, time.monotonic() - started


def execute(client, directory, schedule, timeout):
    from host import STATUS, decode_status
    start = time.monotonic()
    data = (directory / 'input.bin').read_bytes()
    client.write_external(0, data)
    loaded = time.monotonic()
    client.write(0x410000, b'\x01')
    deadline = time.monotonic() + timeout
    while True:
        status = decode_status(client.exchange(STATUS))
        if status['error'] or status['protocol_errors']:
            raise AssertionError(f'engine/protocol error: {status}')
        if not status['busy']:
            break
        if time.monotonic() > deadline:
            raise TimeoutError('ambiguous launch; no automatic retry')
    result = check_counters(client.read(0x410000, 32), schedule['command_count'])
    result['launch_poll_seconds'] = time.monotonic() - loaded
    checked = []
    for line in (directory / 'checks.txt').read_text().splitlines():
        address, name = line.split()
        wanted = (directory / name).read_bytes()
        actual = client.read_external(int(address), len(wanted))
        if actual != wanted:
            bad = next((i for i, (a, b) in enumerate(zip(actual, wanted)) if a != b),
                       min(len(actual), len(wanted)))
            raise OutputMismatch(f'{directory.name}/{name}: first mismatch byte {bad}; '
                                 f'actual_sha256={hashlib.sha256(actual).hexdigest()} '
                                 f'expected_sha256={hashlib.sha256(wanted).hexdigest()}')
        checked.append({'file': name, 'bytes': len(wanted),
                        'sha256': hashlib.sha256(actual).hexdigest()})
    result.update(input_upload_seconds=loaded-start,
                  input_run_readback_wall_seconds=time.monotonic()-start,
                  tensor_checks=checked, checked_bytes=sum(x['bytes'] for x in checked))
    return result


def summarize(records, plan):
    groups = {}
    for row in records:
        if row['kind'] == 'timed':
            groups.setdefault((row['variant'], row['fixture']), []).append(row['elapsed_cycles'])
    timings = {}
    for (variant, fixture), cycles in groups.items():
        if len(cycles) != 10:
            raise ValueError('incomplete timing group')
        median = statistics.median(cycles)
        predicted = plan['projections'][variant][fixture]
        timings.setdefault(variant, {})[fixture] = {
            'runs': len(cycles), 'median_cycles': median, 'min_cycles': min(cycles),
            'max_cycles': max(cycles), 'median_ms': median/CLOCK_HZ*1000,
            'range_over_median': (max(cycles)-min(cycles))/median,
            'predicted_cycles': predicted,
            'prediction_error_fraction': (predicted-median)/median}
    comparisons = {}
    for variant in VARIANTS[1:]:
        for resident in (False, True):
            ratios = {}
            for model in ('kws', 'vww'):
                name = f'{model}-pinned-' + ('resident-' if resident else '') + 'half-overlap-timed'
                ratios[model] = timings['baseline'][name]['median_cycles']/timings[variant][name]['median_cycles']
            geometric = math.sqrt(math.prod(ratios.values()))
            comparisons[f'{variant}/' + ('retention' if resident else 'original')] = {
                'speedup_by_model': ratios, 'geomean_speedup': geometric,
                'meets_performance_screen': geometric >= 1.3 and min(ratios.values()) >= 1/1.05}
    errors = [abs(r['prediction_error_fraction']) for cases in timings.values() for r in cases.values()]
    return {'timings': timings, 'matched_schedule_comparisons': comparisons,
            'timing_consistent': all(r['range_over_median'] <= .02 for cases in timings.values() for r in cases.values()),
            'prediction_median_abs_error': statistics.median(errors),
            'prediction_max_abs_error': max(errors),
            'prediction_screen_passed': statistics.median(errors) <= .10 and max(errors) <= .20,
            'independent_confirmation_required': True, 'phase6_gate_complete': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'work/phase6/screening-v1')
    parser.add_argument('--port', default='/dev/cu.usbserial-20250303171')
    parser.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args()
    plan = make_plan()
    switch = audit_switch()
    if not args.run:
        print(json.dumps({'status': 'preflight-passed', 'planned': plan['planned'],
                          'switch_audit_passed': True,
                          'timing_hardware_order': [b['variant'] for b in plan['blocks'] if b['stage']=='timing']}))
        return
    if args.output.exists():
        raise FileExistsError('choose a fresh output directory; never overwrite evidence')
    require_board_free(args.port)
    args.output.mkdir(parents=True)
    lock = (ROOT / 'work/phase6/physical-board.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save_json(args.output / 'plan.json', plan)
    preserved = args.output / 'phase5-switch'; preserved.mkdir()
    for name, digest in switch.items():
        source = ROOT / 'work/phase5' / name
        shutil.copy2(source, preserved / name)
        if sha(preserved / name) != digest:
            raise ValueError('switch preservation failed')
    report = {'schema': 1, 'status': 'running', 'physical_board': True,
              'started_at': timestamp(), 'pid': os.getpid(),
              'plan_sha256': sha(args.output / 'plan.json'), 'switch_evidence': switch,
              'planned': plan['planned'], 'completed': {'correctness': 0, 'warmup': 0, 'timed': 0, 'total': 0},
              'programming_events': [], 'loads': [], 'output_mismatches': 0,
              'phase5_accuracy_queues_paused': True, 'phase6_gate_complete': False,
              'limits': plan['limits']}
    records = []

    def save():
        report['updated_at'] = timestamp()
        save_json(args.output / 'report.json', report)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}; campaign stopped without relaunch')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save()
    sys.path.insert(0, str(ROOT / 'tools/phase4'))
    from tiled_host import TiledClient
    import serial
    try:
        with (args.output / 'records.jsonl').open('x') as stream:
            for number, block in enumerate(plan['blocks']):
                stage, variant = block['stage'], block['variant']
                if stage == 'timing' and report['completed']['correctness'] != 24:
                    raise ValueError('all 24 correctness runs must pass before timing')
                require_board_free(args.port)
                image = plan['images'][variant]
                verify_files(ROOT, {image['file']: image['sha256']})
                report['current'] = {'stage': stage, 'variant': variant, 'operation': 'programming'}; save()
                log = args.output / f'program-{number}-{variant}.log'
                command = ['openFPGALoader', '-b', 'tangnano20k', '--ftdi-serial', '2025030317',
                           '--freq', '2500000', '-m', '-v', str(ROOT / image['file'])]
                print(f'{timestamp()} {stage}: programming {variant}', flush=True)
                with log.open('w') as output:
                    subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                   check=True, timeout=90)
                report['programming_events'].append({'stage': stage, 'variant': variant,
                    'bitstream_sha256': image['sha256'], 'log': log.name,
                    'log_sha256': sha(log), 'at': timestamp()}); save()
                time.sleep(1)
                with serial.Serial(args.port, 750000, timeout=5, write_timeout=5) as uart:
                    uart.reset_input_buffer()
                    client = TiledClient(uart)
                    client.capabilities(); client.wait_idle(30)
                    if client.read(0x410004, 4) != b'SEQ4':
                        raise ValueError('board lacks autonomous sequence capability')
                    for name in block['fixtures']:
                        report['current'] = {'stage': stage, 'variant': variant,
                                             'fixture': name, 'operation': 'load-and-verify'}; save()
                        directory = BASE / 'fixtures' / name
                        schedule, seconds = load_fixture(client, directory, plan['fixtures'][name])
                        report['loads'].append({'stage': stage, 'variant': variant,
                                                'fixture': name, 'seconds': seconds}); save()
                        print(f'{timestamp()} {variant}: {name} uploaded and verified ({seconds:.1f}s)', flush=True)
                        kinds = ['correctness'] if stage == 'correctness' else ['warmup'] + ['timed'] * 10
                        for repeat, kind in enumerate(kinds):
                            report['current'].update(operation='execute', repeat=repeat, kind=kind); save()
                            result = execute(client, directory, schedule, args.timeout)
                            result.update(variant=variant, fixture=name, kind=kind,
                                          repeat=repeat, at=timestamp())
                            stream.write(json.dumps(result, sort_keys=True) + '\n')
                            stream.flush(); os.fsync(stream.fileno())
                            records.append(result)
                            report['completed'][kind] += 1; report['completed']['total'] += 1
                            save()
                            print(f'{timestamp()} {report["completed"]["total"]}/156 '
                                  f'{variant} {kind}: {name} {result["elapsed_cycles"]} cycles, '
                                  f'{len(result["tensor_checks"])} exact tensors', flush=True)
        if report['completed'] != plan['planned']:
            raise ValueError('screening count mismatch')
        report['summary'] = summarize(records, plan)
        report['records_sha256'] = sha(args.output / 'records.jsonl')
        report['status'] = 'passed-screening'
        report['finished_at'] = timestamp(); save()
        print('156-run screening completed; confirmation/expansion has NOT started', flush=True)
    except BaseException as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['failure'] = repr(error)
        if isinstance(error, OutputMismatch):
            report['output_mismatches'] += 1
        path = args.output / 'records.jsonl'
        if path.exists(): report['records_sha256'] = sha(path)
        save()
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()


if __name__ == '__main__':
    main()
