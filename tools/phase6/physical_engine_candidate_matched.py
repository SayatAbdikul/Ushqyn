#!/usr/bin/env python3
"""Optional short RTL screen with the historical per-timing reload protocol.

Each model runs one stress inference, then three independent fixture reloads,
each immediately followed by one warmup and one timed inference. Preparation
is boardless. --run requires an explicit separate invocation and a free board.
"""
import argparse
import fcntl
import json
import math
from pathlib import Path
import signal
import statistics
import subprocess
import time

import physical_engine_candidate as screened

common = screened.common
ROOT = screened.ROOT
BASE = screened.BASE


def groups():
    for model in ('kws', 'vww'):
        yield model, 'stress', 0, ('stress',)
        for repeat in range(3):
            yield model, 'pinned', repeat, ('warmup', 'timed')


def check_protocol(records):
    expected = [(model, sample, repeat, kind, load_id)
        for load_id, (model, sample, repeat, kinds) in enumerate(groups())
        for kind in kinds]
    actual = [(r['model'], r['sample'], r['repeat'], r['kind'], r['load_id']) for r in records]
    if actual != expected:
        raise ValueError('record order does not match reload/warmup/timed protocol')
    for i, row in enumerate(records):
        if row['order_index'] != i:
            raise ValueError('record order index differs')
        if row['kind'] == 'timed':
            previous = records[i-1]
            if (row['load_seconds'] != 0 or previous['kind'] != 'warmup'
                    or previous['load_id'] != row['load_id']):
                raise ValueError('timed record lacks immediate same-load warmup')
        elif row['load_seconds'] <= 0:
            raise ValueError('stress/warmup lacks a fresh fixture reload')


def prepare(route_path):
    plan = screened.prepare(route_path)
    historical_plan = common.read(screened.REFERENCE / 'plan.json')
    historical_report = common.read(screened.REFERENCE / 'report.json')
    if historical_plan['repeats'] != 3:
        raise ValueError('historical repeated-measurement count differs')
    # Pin and inspect the actual historical runner used by the sealed report.
    runner = 'tools/phase6/matched_campaign.py'
    expected_runner = historical_plan['source_sha256'][runner]
    common.verify(ROOT, {runner: expected_runner})
    if plan['source_sha256'].get(runner) != expected_runner:
        raise ValueError('historical protocol source differs from shared runner')
    evidence = {}
    historical_rows = historical_report['records']
    for model in ('kws', 'vww'):
        representative = next(name for name, members in historical_plan['measurement_groups'][model].items()
                              if 'B3-adaptation' in members)
        rows = [r for r in historical_rows if r['model'] == model and r['policy'] == representative]
        expected = [('stress', 0)] + [(kind, repeat) for repeat in range(3) for kind in ('warmup', 'timed')]
        if [(r['kind'], r['repeat']) for r in rows] != expected:
            raise ValueError('historical baseline lacks the required per-timing warmups')
        for i, row in enumerate(rows):
            if row['kind'] == 'timed':
                # Check immediate adjacency in the complete signed record list,
                # not just after filtering out other policies.
                position = historical_rows.index(row)
                if (position == 0 or historical_rows[position-1] != rows[i-1]
                        or row['load_seconds'] != 0):
                    raise ValueError('historical timed sample is not immediately after warmup')
            elif row['load_seconds'] <= 0:
                raise ValueError('historical warmup/stress lacks reload evidence')
        timed = [r['elapsed_cycles'] for r in rows if r['kind'] == 'timed']
        if timed != plan['baseline'][model]['timed_cycles']:
            raise ValueError('historical summary differs from signed baseline timings')
        evidence[model] = dict(measured_policy=representative, timed_cycles=timed,
            immediate_warmup_before_every_timing=True, reloads=4)
    plan.update(planned=14, reloads=8, timed_per_model=3, stress_per_model=1,
        warmup_per_model=3, baseline_protocol=evidence,
        scope='same B3 programs; one stress and three fixture reload/warmup/timed groups per model',
        protocol='model order KWS then VWW; stress first; reload fixture before every warmup; one warmup immediately before each timed sample',
        limitations=['Short separate-session comparison with sealed historical physical baseline; not an interleaved old/new image campaign.',
            'Reload and immediate warmup protocol matches each historical B3 timing; historical VWW rounds also included another policy between groups.',
            'Pinned/stress exactness only, not full-dataset accuracy, endurance or measured power.'])
    path = Path(__file__).resolve()
    plan['source_sha256'][common.relative(path)] = common.sha(path)
    return plan


def run(plan, output, port):
    import serial
    import run_screening as screening
    import run_priority as priority
    from screen_engine_schedule import execute_verified_sample
    from uart_burst import BurstTiledClient
    output = Path(output).resolve()
    screened.audit(plan)
    screening.require_board_free(port)
    if output.exists():
        raise FileExistsError('choose a fresh physical evidence directory')
    locks = []
    old_handlers = {}
    try:
        paths = [(ROOT / 'work/phase6/physical-board.lock', 'a'),
            (Path(plan['original_checkout']) / 'work/phase6/physical-board.lock', 'r')]
        for path, mode in paths:
            if mode == 'r' and not path.exists():
                continue
            if any(Path(lock.name).resolve() == path.resolve() for lock in locks):
                continue
            lock = path.open(mode)
            locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        screening.require_board_free(port)
        screened.audit(plan)
        output.mkdir(parents=True)
        common.save(output / 'plan.json', plan)
        report = dict(status='running', physical_board=True, planned=14, completed=0, records=[],
            protocol=plan['protocol'], started_at=screening.timestamp(),
            plan_sha256=common.sha(output / 'plan.json'))

        def save():
            report['updated_at'] = screening.timestamp()
            common.save(output / 'report.json', report)

        def interrupt(signum, frame):
            raise KeyboardInterrupt('completed signed evidence preserved')

        old_handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
        save()
        try:
            with (output / 'program.log').open('x') as log:
                subprocess.run(['openFPGALoader', '-b', 'tangnano20k', '--ftdi-serial', '2025030317',
                    '--freq', '2500000', '-m', '-v', str(common.local(plan['bitstream']))],
                    stdout=log, stderr=subprocess.STDOUT, timeout=90, check=True)
            report['program_log_sha256'] = common.sha(output / 'program.log')
            save()
            time.sleep(1)
            with serial.Serial(port, plan['baud'], timeout=5, write_timeout=5) as uart, (output / 'records.jsonl').open('x') as stream:
                uart.reset_input_buffer()
                client = BurstTiledClient(uart)
                client.capabilities()
                for load_id, (model, sample, repeat, kinds) in enumerate(groups()):
                    fixture = plan['fixtures'][model][sample]
                    directory = common.local(fixture['directory'])
                    report['current'] = f'{model}/{sample}/{repeat}/load'
                    save()
                    schedule, loaded = screening.load_fixture(client, directory, fixture)
                    for kind in kinds:
                        report['current'] = f'{model}/{kind}/{repeat}'
                        save()
                        row = execute_verified_sample(client, schedule, (directory / 'input.bin').read_bytes(),
                            (directory / 'output.bin').read_bytes(), 30)
                        checks = []
                        if sample == 'stress':
                            for line in (directory / 'checks.txt').read_text().splitlines():
                                address, name = line.split()
                                wanted = (directory / name).read_bytes()
                                actual = client.read_external(int(address), len(wanted))
                                if actual != wanted:
                                    raise screening.OutputMismatch(f'{model}/{name}: stress tensor mismatch')
                                checks.append(dict(file=name, bytes=len(actual), sha256=priority.digest(actual)))
                        row.update(model=model, sample=sample, kind=kind, repeat=repeat, load_id=load_id,
                            order_index=len(report['records']), fixture=fixture['directory'],
                            input_sha256=fixture['files']['input.bin'], fixture_files=fixture['files'],
                            bitstream_sha256=plan['bitstream_sha256'], at=screening.timestamp(),
                            device_latency_ms=row['elapsed_cycles'] / 27_000,
                            load_seconds=loaded if kind != 'timed' else 0, stress_tensor_checks=checks)
                        priority.append_row(stream, row)
                        report['records'].append(row)
                        report['completed'] = len(report['records'])
                        save()
                        print(report['current'], row['elapsed_cycles'], flush=True)
            screened.audit(plan)
            if priority.read_rows(output / 'records.jsonl') != report['records'] or report['completed'] != plan['planned']:
                raise ValueError('signed record completeness mismatch')
            check_protocol(report['records'])
            summary = {}
            for model in ('kws', 'vww'):
                rows = [r for r in report['records'] if r['model'] == model and r['kind'] == 'timed']
                cycles = [r['elapsed_cycles'] for r in rows]
                if len(cycles) != 3:
                    raise ValueError('missing timing records')
                median = statistics.median(cycles)
                baseline = plan['baseline'][model]['median_cycles']
                summary[model] = dict(timed_cycles=cycles, median_cycles=median,
                    median_device_latency_ms=median / 27_000, device_inferences_per_second=27_000_000 / median,
                    baseline_median_cycles=baseline, speedup=baseline / median,
                    latency_reduction_fraction=1-median / baseline,
                    range_over_median=(max(cycles)-min(cycles)) / median,
                    median_engine_cycles=statistics.median(r['engine_cycles'] for r in rows),
                    median_dma_cycles=statistics.median(r['dma_cycles'] for r in rows),
                    median_overlap_cycles=statistics.median(r['overlap_cycles'] for r in rows))
            report.update(status='passed-short-screen', summary=summary, protocol_verified=True,
                geometric_mean_speedup=math.prod(r['speedup'] for r in summary.values()) ** 0.5,
                resources=plan['route']['resources'], routed_core_fmax_mhz=plan['route']['routed_core_fmax_mhz'],
                records_sha256=common.sha(output / 'records.jsonl'), finished_at=screening.timestamp())
            save()
            common.save(output / 'seal.json', {k+'_sha256': common.sha(output / f) for k, f in
                [('plan', 'plan.json'), ('report', 'report.json'), ('records', 'records.jsonl')]})
            return report
        except BaseException as error:
            report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', failure=repr(error))
            if (output / 'records.jsonl').exists():
                report['records_sha256'] = common.sha(output / 'records.jsonl')
            save()
            raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        for lock in reversed(locks):
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route-report', type=Path, default=BASE / 'route27/report.json')
    parser.add_argument('--output', type=Path, default=BASE / 'physical/matched-v1')
    parser.add_argument('--port', default='/dev/cu.usbserial-20250303171')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--archive-only', action='store_true')
    args = parser.parse_args()
    if args.archive_only:
        print(json.dumps(screened.archive(args.output), indent=2))
    else:
        plan = prepare(args.route_report)
        if args.run:
            report = run(plan, args.output, args.port)
            print(json.dumps(dict(summary=report['summary'], archive=screened.archive(args.output)), indent=2))
        else:
            print(json.dumps(dict(status=plan['status'], planned=plan['planned'], reloads=plan['reloads'],
                protocol=plan['protocol'], source_sha256=plan['source_sha256'][common.relative(Path(__file__))],
                bitstream_sha256=plan['bitstream_sha256']), indent=2))
