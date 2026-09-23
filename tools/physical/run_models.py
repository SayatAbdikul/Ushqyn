#!/usr/bin/env python3
"""Run pinned fixtures over the real UART, preserving partial failure evidence.

Program the specified bitstream separately first. The protocol capability check
verifies the target ABI; it cannot read back the FPGA configuration hash.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/phase2'))
from host import Client, RESET, TARGET


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fixture(path, jobs):
    metadata = json.loads((path / 'board.json').read_text())
    image = (path / 'board.bin').read_bytes()
    if metadata['target'] != TARGET['name']:
        raise ValueError(f'{path}: target mismatch')
    if metadata['target_manifest_sha256'] != digest(ROOT / 'hardware/targets/tang_nano_20k_v2.json'):
        raise ValueError(f'{path}: target manifest changed')
    if hashlib.sha256(image).hexdigest() != metadata['image_sha256']:
        raise ValueError(f'{path}: image digest mismatch')
    if len(image) != TARGET['memory_bytes']:
        raise ValueError(f'{path}: image capacity mismatch')
    with np.load(path / 'checks.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in ('inputs', 'outputs', 'labels')}
    if not 1 <= jobs <= len(arrays['inputs']):
        raise ValueError(f'{path}: requested job count exceeds fixture')
    if len(metadata['inputs']) != 1 or len(metadata['outputs']) != 1:
        raise ValueError('physical classifier runner requires one input and output')
    if (arrays['inputs'].dtype != np.int8 or arrays['outputs'].dtype != np.int8
            or len(arrays['outputs']) != len(arrays['inputs'])
            or len(arrays['labels']) != len(arrays['inputs'])):
        raise ValueError(f'{path}: invalid fixture array types/counts')
    return metadata, image, arrays


def execute_fixture(client, path, jobs, result, checkpoint=lambda: None):
    metadata, image, arrays = load_fixture(path, jobs)
    result.update(fixture=str(path), requested_jobs=jobs, completed_jobs=0,
                  image_sha256=metadata['image_sha256'],
                  fixture_sha256=digest(path / 'checks.npz'),
                  metadata_sha256=digest(path / 'board.json'),
                  expected_macs=metadata['macs'], records=[], correct=0,
                  integer_mismatches=0, status='running')
    if client.capabilities() != len(image):
        raise ValueError('board SRAM capacity differs from fixture')
    load_start = time.monotonic()
    client.exchange(RESET)
    client.write(0, image)
    if client.read(0, len(image)) != image:
        raise AssertionError('initial full image readback mismatch')
    result['load_and_readback_seconds'] = time.monotonic() - load_start
    input_address = next(iter(metadata['inputs'].values()))
    output_address = next(iter(metadata['outputs'].values()))
    last_progress = time.monotonic()
    for index in range(jobs):
        begin = time.monotonic()
        client.write(input_address, arrays['inputs'][index].tobytes())
        counters = client.run(metadata['entry'])
        expected = arrays['outputs'][index].tobytes()
        actual = client.read(output_address, len(expected))
        result['records'].append(dict(job=index, output_hex=actual.hex(),
                                      host_job_seconds=time.monotonic() - begin,
                                      counters=counters))
        if actual != expected:
            result['integer_mismatches'] += 1
            result['first_mismatch'] = dict(job=index, expected_hex=expected.hex(),
                                           actual_hex=actual.hex())
            raise AssertionError(f'job {index}: integer output mismatch')
        if counters['useful_macs'] != metadata['macs']:
            raise AssertionError(f'job {index}: MAC counter mismatch')
        if counters['elapsed'] != sum(counters[k] for k in ('compute_cycles', 'wait_cycles', 'control_cycles')):
            raise AssertionError(f'job {index}: cycle counters do not reconcile')
        if counters['protocol_errors']:
            raise AssertionError(f'job {index}: unexpected protocol error')
        result['correct'] += int(np.argmax(np.frombuffer(actual, np.int8)) == arrays['labels'][index])
        result['completed_jobs'] += 1
        if time.monotonic() - last_progress >= 10 or index + 1 == jobs:
            print(f'{path.name}: {index + 1}/{jobs} exact physical jobs', flush=True)
            checkpoint()
            last_progress = time.monotonic()
    cycles = [record['counters']['elapsed'] for record in result['records']]
    wall = [record['host_job_seconds'] for record in result['records']]
    result.update(status='passed', accuracy=result['correct'] / jobs,
                  elapsed_cycles_min=min(cycles), elapsed_cycles_max=max(cycles),
                  elapsed_cycles_median=float(np.median(cycles)),
                  host_job_seconds_median=float(np.median(wall)),
                  host_job_seconds_p95=float(np.percentile(wall, 95)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--fixture', type=Path, action='append', required=True,
                        help='may be repeated to test model switches without reprogramming')
    parser.add_argument('--jobs', type=int, default=1000)
    parser.add_argument('--bitstream', type=Path, required=True,
                        help='the separately programmed .fs artifact')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    # Validate every fixture before opening or writing the device.
    for fixture in args.fixture:
        load_fixture(fixture, args.jobs)
    report = dict(scope='physical board UART', status='running', port=args.port,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  target=TARGET['name'], target_id=TARGET['target_id'],
                  declared_clock_hz=TARGET['clock_hz'], baud=TARGET['baud'],
                  supplied_bitstream_sha256=digest(args.bitstream),
                  bitstream_hash_readback_supported=False,
                  programmed_by_this_runner=False, measured_power=False,
                  phases=[])
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint():
        temporary = args.report.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(args.report)

    import serial
    try:
        with serial.Serial(args.port, TARGET['baud'], timeout=2, write_timeout=2) as uart:
            uart.reset_input_buffer()
            client = Client(uart)
            for fixture in args.fixture:
                result = {}
                report['phases'].append(result)
                execute_fixture(client, fixture, args.jobs, result, checkpoint)
                checkpoint()
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        if report['phases'] and report['phases'][-1].get('status') == 'running':
            report['phases'][-1].update(status='failed', error=report['error'])
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        checkpoint()


if __name__ == '__main__':
    main()
