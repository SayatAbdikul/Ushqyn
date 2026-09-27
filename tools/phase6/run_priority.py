#!/usr/bin/env python3
"""Confirm the screened winner, then evaluate both complete accuracy sets.

--run accesses JTAG/UART. --resume continues checked accuracy prefixes only
after a fully passed confirmation. No automatic retries or additional tests.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import struct
import subprocess
import sys
import time

import numpy as np
import serial

import run_screening as screening

ROOT = screening.ROOT
BASE = screening.BASE
for directory in ('compiler', 'tools/phase4', 'tools/phase5'):
    sys.path.insert(0, str(ROOT / directory))
from tiled_host import TiledClient
from host import RESET, STATUS, decode_status
from probe_primary import prepared
from integer_reference import evaluate
from dual_schedule import combine


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_rows(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        signature = row.pop('record_sha256')
        if digest(json.dumps(row, sort_keys=True, separators=(',', ':')).encode()) != signature:
            raise ValueError(f'corrupted record: {path}')
        rows.append(row)
    return rows


def append_row(stream, row):
    signed = dict(row, record_sha256=digest(json.dumps(row, sort_keys=True, separators=(',', ':')).encode()))
    stream.write(json.dumps(signed, sort_keys=True) + '\n')
    stream.flush(); os.fsync(stream.fileno())


def accepted_screening():
    folder = ROOT / 'work/phase6/screening-v1'
    report = json.loads((folder / 'report.json').read_text())
    plan = json.loads((folder / 'plan.json').read_text())
    if (report['status'] != 'passed-screening' or report['output_mismatches']
            or report['completed'] != {'correctness': 24, 'warmup': 12, 'timed': 120, 'total': 156}
            or screening.sha(folder / 'plan.json') != report['plan_sha256']
            or screening.sha(folder / 'records.jsonl') != report['records_sha256']
            or screening.sha(ROOT / 'tools/phase6/run_screening.py') != plan['runner_sha256']):
        raise ValueError('screening evidence/source identity invalid')
    rows = [json.loads(line) for line in (folder / 'records.jsonl').read_text().splitlines()]
    for variant in screening.VARIANTS:
        for kind, count in (('correctness', 8), ('warmup', 4), ('timed', 40)):
            if sum(r['variant'] == variant and r['kind'] == kind for r in rows) != count:
                raise ValueError('screening coverage incomplete')
    recomputed = screening.summarize(rows, plan)
    if (recomputed != report['summary'] or not recomputed['timing_consistent']
            or not recomputed['matched_schedule_comparisons']['spatial/retention']['meets_performance_screen']):
        raise ValueError('screened winner did not pass performance criteria')
    return report


def prepare():
    plan = screening.make_plan()
    initial = accepted_screening()
    switch = screening.audit_switch()
    bundles = {}
    for model in ('kws', 'vww'):
        source = prepared(model)
        name = f'{model}-pinned-resident-half-overlap-timed'
        directory = BASE / 'fixtures' / name
        screening.verify_files(directory, plan['fixtures'][name]['files'])
        bundles[model] = {'source': source, 'fixture': name,
                          'commands': (directory / 'commands.bin').read_bytes(),
                          'payload': (directory / 'payload.bin').read_bytes(),
                          'schedule': json.loads((directory / 'schedule.json').read_text())}
    combined, locations = combine(bundles['kws']['commands'], bundles['kws']['payload'],
                                  bundles['vww']['commands'], bundles['vww']['payload'])
    rng = random.Random(20260927)
    blocks = []
    for session in (2, 3):
        variants = ['baseline', 'spatial']; rng.shuffle(variants)
        for variant in variants:
            models = ['kws', 'vww']; rng.shuffle(models)
            blocks.append({'session': session, 'variant': variant, 'models': models})
    identity = {'schema': 1, 'screening_report_sha256': screening.sha(ROOT / 'work/phase6/screening-v1/report.json'),
                'screening_plan_sha256': initial['plan_sha256'], 'switch_evidence': switch,
                'images': {v: plan['images'][v] for v in ('baseline', 'spatial')},
                'combined_program_sha256': digest(combined), 'locations': locations,
                'accuracy': {m: {'count': len(b['source'][5]), 'split_sha256': b['source'][6],
                    'fixture': b['fixture'], 'fixture_files': plan['fixtures'][b['fixture']]['files']} for m, b in bundles.items()},
                'source_sha256': {str(path.relative_to(ROOT)): screening.sha(path) for path in (
                    Path(__file__), ROOT / 'tools/phase6/run_screening.py', ROOT / 'compiler/integer_reference.py',
                    ROOT / 'tools/phase5/probe_primary.py', ROOT / 'tools/phase5/dual_schedule.py',
                    ROOT / 'tools/phase4/tiled_host.py', ROOT / 'tools/phase2/host.py')},
                'confirmation_blocks': blocks, 'confirmation_repeats': 10,
                'confirmation_planned': {'timed': 80, 'warmup': 8},
                'core_clock_hz': screening.CLOCK_HZ, 'baud': 750000,
                'scope': 'two additional confirmation sessions, then full KWS/VWW accuracy on spatial hardware with simple SRAM retention',
                'not_claimed': ['Phase 5 original-image accuracy closure', 'G6 closure', 'SOTA', 'measured energy',
                                'new 10000-job alternating stability campaign']}
    return identity, plan, initial, bundles, combined, locations


def check_prefix(rows, model, samples, image_sha, schedule, location):
    for index, row in enumerate(rows):
        if index >= len(samples):
            raise ValueError('accuracy records exceed dataset')
        sample = samples[index]
        logits = np.frombuffer(bytes.fromhex(row['output_hex']), np.int8)
        if (row['sample_index'] != index or row['workload'] != model or row['sample_id'] != sample['id']
                or row['feature_sha256'] != sample['feature_sha256'] or row['label'] != sample['label']
                or row['bitstream_sha256'] != image_sha or row['oracle'] != 'independent-centered-integer'
                or row['output_hex'] != row['expected_hex'] or len(logits) != schedule['final_output']['bytes']
                or row['prediction'] != int(np.argmax(logits)) or row['elapsed_cycles'] <= 0
                or row['command_index'] != location['entry'] + schedule['command_count'] - 1):
            raise ValueError(f'invalid accuracy prefix at {model} sample {index}')
        e, c, d, o = (row[k] for k in ('elapsed_cycles', 'engine_cycles', 'dma_cycles', 'overlap_cycles'))
        if min(e, c, d) <= 0 or o < 0 or o > min(c, d) or c + d - o > e:
            raise ValueError('invalid prefix counters')


def confirmation_summary(rows, screening_report):
    medians = {'1': {v: {m: screening_report['summary']['timings'][v][
        f'{m}-pinned-resident-half-overlap-timed']['median_cycles'] for m in ('kws', 'vww')}
        for v in ('baseline', 'spatial')}}
    if len(rows) != 88:
        raise ValueError('confirmation requires exactly 88 records')
    spreads = []
    for session in (2, 3):
        medians[str(session)] = {}
        for variant in ('baseline', 'spatial'):
            medians[str(session)][variant] = {}
            for model in ('kws', 'vww'):
                group = [r for r in rows if r['session'] == session and r['variant'] == variant and r['model'] == model]
                if (sum(r['kind'] == 'warmup' for r in group) != 1
                        or sorted(r['repeat'] for r in group if r['kind'] == 'timed') != list(range(1, 11))):
                    raise ValueError('confirmation group missing or duplicated repetitions')
                values = [r['elapsed_cycles'] for r in group if r['kind'] == 'timed']
                median = statistics.median(values)
                medians[str(session)][variant][model] = median
                spreads.append((max(values) - min(values)) / median)
    comparisons = {}
    for session, variants in medians.items():
        ratios = {m: variants['baseline'][m] / variants['spatial'][m] for m in ('kws', 'vww')}
        comparisons[session] = {'speedup': ratios, 'geomean_speedup': math.sqrt(math.prod(ratios.values()))}
    session_spreads = {f'{v}/{m}': (max(values)-min(values))/statistics.median(values)
                      for v in ('baseline', 'spatial') for m in ('kws', 'vww')
                      for values in [[medians[str(s)][v][m] for s in (1, 2, 3)]]}
    passed = (max(spreads) <= .02 and max(session_spreads.values()) <= .03
              and all(c['geomean_speedup'] >= 1.3 and min(c['speedup'].values()) >= 1/1.05
                      for c in comparisons.values()))
    return {'passed': passed, 'session_median_cycles': medians, 'comparisons': comparisons,
            'max_within_configuration_spread': max(spreads), 'session_spreads': session_spreads}


def sample_reference(bundle, index):
    program, _, _, _, features, samples, _ = bundle['source']
    feature = np.asarray(features[index]); sample = samples[index]
    if digest(feature.tobytes()) != sample['feature_sha256']:
        raise ValueError('accuracy feature hash mismatch')
    q = program.tensors[program.inputs[0]].quantization.encode(feature)
    started = time.monotonic()
    values = evaluate(program, {program.inputs[0]: q})
    first = program.layers[0]
    board_input = values[first.output] if first.op in ('Transpose', 'Reshape') else q
    return board_input.tobytes(), values[program.outputs[0]].tobytes(), sample, time.monotonic()-started


def execute_sample(client, bundle, location, input_data, expected, timeout):
    schedule = bundle['schedule']; base, entry = location['external_base'], location['entry']
    client.write(0x41001c, entry.to_bytes(2, 'little'))
    started = time.monotonic()
    client.write_external(base, input_data)
    uploaded = time.monotonic()
    client.write(0x410000, b'\x01')
    deadline = time.monotonic() + timeout
    while True:
        status = decode_status(client.exchange(STATUS))
        if status['error'] or status['protocol_errors']:
            raise AssertionError(f'FPGA/protocol failure: {status}')
        if not status['busy']:
            break
        if time.monotonic() > deadline:
            raise TimeoutError('ambiguous FPGA launch; not retried')
    result = screening.check_counters(client.read(0x410000, 32), entry + schedule['command_count'])
    output = schedule['final_output']
    actual = client.read_external(base + output['ext'], output['bytes'])
    if actual != expected:
        raise screening.OutputMismatch(f'FPGA logits differ: actual={actual.hex()} expected={expected.hex()}')
    result.update(input_upload_seconds=uploaded-started, wall_seconds=time.monotonic()-started,
                  output_hex=actual.hex(), expected_hex=expected.hex(),
                  prediction=int(np.argmax(np.frombuffer(actual, np.int8))))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'work/phase6/priority-v1')
    parser.add_argument('--port', default='/dev/cu.usbserial-20250303171')
    args = parser.parse_args()
    identity, plan, initial, bundles, combined, locations = prepare()
    if not args.run:
        print(json.dumps({'status': 'preflight-passed', 'confirmation': identity['confirmation_planned'],
                          'accuracy_counts': {m: len(b['source'][5]) for m, b in bundles.items()}})); return
    screening.require_board_free(args.port)
    if args.resume:
        if json.loads((args.output / 'plan.json').read_text()) != identity:
            raise ValueError('resume artifact identity mismatch')
        report = json.loads((args.output / 'report.json').read_text())
        if report.get('output_mismatches'):
            raise ValueError('numerical failure requires investigation and a new campaign')
        confirmation = read_rows(args.output / 'confirmation.jsonl')
        if not confirmation_summary(confirmation, initial)['passed']:
            raise ValueError('resume requires complete passed confirmation')
        if report['status'] == 'passed-priority-campaign':
            raise ValueError('campaign already completed')
    else:
        if args.output.exists(): raise FileExistsError('choose a fresh evidence directory or --resume')
        args.output.mkdir(parents=True)
        screening.save_json(args.output / 'plan.json', identity)
        report = {'schema': 1, 'status': 'running', 'started_at': screening.timestamp(),
                  'plan_sha256': screening.sha(args.output / 'plan.json'), 'physical_board': True,
                  'confirmation': {'status': 'pending', 'completed': 0}, 'accuracy': {},
                  'programming_events': [], 'output_mismatches': 0, 'sessions': [],
                  'phase6_gate_complete': False, 'not_claimed': identity['not_claimed']}
    lock = (ROOT / 'work/phase6/physical-board.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if 'failure' in report:
        report.setdefault('prior_interruptions', []).append({
            'status': report['status'], 'failure': report.pop('failure'),
            'at': report.get('updated_at')})
    report.update(status='running', pid=os.getpid())
    report['sessions'].append({'at': screening.timestamp(), 'resume': args.resume})

    def save():
        report['updated_at'] = screening.timestamp()
        screening.save_json(args.output / 'report.json', report)

    def program(variant, label):
        screening.require_board_free(args.port)
        image = identity['images'][variant]
        screening.verify_files(ROOT, {image['file']: image['sha256']})
        number = len(report['programming_events'])
        log = args.output / f'program-{number}-{label}-{variant}.log'
        if log.exists(): raise FileExistsError('preserve unsuccessful programming log before resuming')
        report['current'] = {'operation': 'programming', 'variant': variant, 'stage': label}; save()
        print(f'{screening.timestamp()} programming {variant}: {label}', flush=True)
        with log.open('x') as stream:
            subprocess.run(['openFPGALoader', '-b', 'tangnano20k', '--ftdi-serial', '2025030317',
                            '--freq', '2500000', '-m', '-v', str(ROOT / image['file'])],
                           stdout=stream, stderr=subprocess.STDOUT, timeout=90, check=True)
        report['programming_events'].append({'variant': variant, 'stage': label,
            'bitstream_sha256': image['sha256'], 'log': log.name, 'log_sha256': screening.sha(log),
            'at': screening.timestamp()}); save(); time.sleep(1)

    def connect():
        uart = serial.Serial(args.port, 750000, timeout=5, write_timeout=5)
        try:
            uart.reset_input_buffer(); client = TiledClient(uart)
            client.capabilities(); client.wait_idle(30)
            if client.read(0x410004, 4) != b'SEQ4': raise ValueError('sequencer missing')
            return uart, client
        except BaseException:
            uart.close(); raise

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}; saved samples remain available')

    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    save()
    try:
        if not args.resume:
            rows = []
            with (args.output / 'confirmation.jsonl').open('x') as stream:
                for block in identity['confirmation_blocks']:
                    variant, session = block['variant'], block['session']
                    program(variant, f'confirmation-session-{session}')
                    uart, client = connect()
                    with uart:
                        for model in block['models']:
                            name = bundles[model]['fixture']; directory = BASE / 'fixtures' / name
                            report['current'] = {'stage': 'confirmation', 'session': session,
                                                 'variant': variant, 'model': model, 'operation': 'load-and-verify'}; save()
                            schedule, load_seconds = screening.load_fixture(client, directory, plan['fixtures'][name])
                            for repeat in range(11):
                                row = screening.execute(client, directory, schedule, 30)
                                row.update(session=session, variant=variant, model=model, fixture=name,
                                    kind='warmup' if repeat == 0 else 'timed', repeat=repeat,
                                    load_seconds=load_seconds if repeat == 0 else 0, at=screening.timestamp())
                                append_row(stream, row); rows.append(row)
                                report['confirmation']['completed'] = len(rows); save()
                            print(f'{screening.timestamp()} confirmation {len(rows)}/88: '
                                  f'session {session} {variant} {model}', flush=True)
            summary = confirmation_summary(rows, initial)
            report['confirmation'].update(status='passed' if summary['passed'] else 'failed-criteria',
                summary=summary, records_sha256=screening.sha(args.output / 'confirmation.jsonl')); save()
            if not summary['passed']:
                raise ValueError('independent confirmation missed criteria; accuracy not started')
            print('Independent confirmation passed; starting complete-set accuracy', flush=True)
        program('spatial', 'full-accuracy')
        uart, client = connect()
        with uart:
            client.exchange(RESET)
            for model, bundle in bundles.items():
                report['current'] = {'stage': 'accuracy-setup', 'model': model, 'operation': 'load-and-verify'}; save()
                base = locations[model]['external_base']; payload = bundle['payload']
                client.write_external(base, payload)
                if client.read_external(base, len(payload)) != payload:
                    raise AssertionError('dual-resident payload readback mismatch')
            client.write(0x500000, combined)
            if client.read(0x500000, len(combined)) != combined:
                raise AssertionError('combined command readback mismatch')
            for model, bundle in bundles.items():
                path = args.output / f'{model}-accuracy.jsonl'
                rows = read_rows(path)
                samples = bundle['source'][5]
                check_prefix(rows, model, samples, identity['images']['spatial']['sha256'], bundle['schedule'], locations[model])
                progress = {'status': 'running', 'planned': len(samples), 'completed': len(rows),
                            'correct': sum(r['prediction'] == r['label'] for r in rows), 'output_mismatches': 0}
                report['accuracy'][model] = progress
                report['current'] = {'stage': 'accuracy', 'model': model, 'operation': 'evaluate'}; save()
                with path.open('a') as stream:
                    for index in range(len(rows), len(samples)):
                        started = time.monotonic()
                        data, expected, sample, oracle_seconds = sample_reference(bundle, index)
                        row = execute_sample(client, bundle, locations[model], data, expected, 30)
                        row.update(sample_index=index, workload=model, sample_id=sample['id'], label=sample['label'],
                            feature_sha256=sample['feature_sha256'], oracle='independent-centered-integer',
                            oracle_seconds=oracle_seconds, sample_total_seconds=time.monotonic()-started,
                            bitstream_sha256=identity['images']['spatial']['sha256'], at=screening.timestamp())
                        append_row(stream, row)
                        progress['completed'] += 1; progress['correct'] += row['prediction'] == row['label']
                        progress['latest_sample_total_seconds'] = row['sample_total_seconds']
                        save()
                        if progress['completed'] % 100 == 0:
                            print(f'{screening.timestamp()} {model}: {progress["completed"]}/{len(samples)}, '
                                  f'correct={progress["correct"]}, zero output mismatches', flush=True)
                final_rows = read_rows(path)
                check_prefix(final_rows, model, samples, identity['images']['spatial']['sha256'], bundle['schedule'], locations[model])
                if len(final_rows) != len(samples): raise ValueError('incomplete accuracy dataset')
                expected_quality = json.loads((ROOT / f'benchmarks/manifests/{model}.static-quality.json').read_text())
                correct = sum(r['prediction'] == r['label'] for r in final_rows)
                if correct != expected_quality['correct']:
                    raise ValueError(f'{model} independent full-set accuracy differs from frozen software quality')
                progress.update(status='passed', correct=correct, accuracy=correct/len(samples),
                    records_sha256=screening.sha(path),
                    median_device_ms=statistics.median(r['device_latency_ms'] for r in final_rows),
                    median_wall_seconds=statistics.median(r['wall_seconds'] for r in final_rows),
                    median_sample_seconds=statistics.median(r['sample_total_seconds'] for r in final_rows))
                save(); print(f'{model} complete accuracy passed: {correct}/{len(samples)}', flush=True)
        report.update(status='passed-priority-campaign', finished_at=screening.timestamp()); save()
        print('Priority campaign complete. No further board tests queued.', flush=True)
    except BaseException as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['failure'] = repr(error)
        if isinstance(error, screening.OutputMismatch): report['output_mismatches'] += 1
        save(); raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()


if __name__ == '__main__':
    main()
