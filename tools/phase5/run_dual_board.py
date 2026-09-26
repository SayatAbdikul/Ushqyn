#!/usr/bin/env python3
"""Run pinned audio/vision accuracy or switch jobs on one resident FPGA image."""

import argparse
import hashlib
import json
import os
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np
import serial

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'compiler'))
sys.path.insert(0, str(ROOT / 'tools/phase4'))
from static_pipeline import execute_layer, run
from tiled_host import TiledClient
from host import STATUS, decode_status
from dual_schedule import combine
from probe_primary import prepared, sha
from schedule import create_plan, build_jobs, load_split


def jobs_for(campaign, limit):
    if campaign == 'switch':
        full = create_plan()
        manifests = {name: load_split(ROOT / f'benchmarks/manifests/{name}.data.json')[0]
                     for name in ('kws', 'vww')}
        jobs = list(build_jobs(manifests))
        identity = full['jsonl_sha256']
    else:
        name = campaign.split('-')[0]
        count = json.loads((ROOT / f'benchmarks/manifests/{name}.data.json').read_text())[
            'splits']['accuracy']['count']
        jobs = [{'job': index, 'workload': name, 'sample_index': index}
                for index in range(count)]
        identity = hashlib.sha256(json.dumps(jobs, sort_keys=True).encode()).hexdigest()
    if limit is not None:
        if limit < 1:
            raise ValueError('positive limit required')
        jobs = jobs[:limit]
    return jobs, identity


def reference_and_input(bundle, index):
    program, _, _, _, features, records, _ = bundle
    input_name = program.inputs[0]
    sample = np.asarray(features[index])
    record = records[index]
    if hashlib.sha256(sample.tobytes()).hexdigest() != record['feature_sha256']:
        raise ValueError(f'feature content differs from manifest: {record["id"]}')
    quantized = program.tensors[input_name].quantization.encode(sample)
    first = program.layers[0]
    board_input = (execute_layer(program, first, {input_name: quantized})
                   if first.op in ('Transpose', 'Reshape') else quantized)
    expected = run(program, {input_name: sample})[program.outputs[0]]
    if board_input.nbytes > 8 * 1024 * 1024:
        raise ValueError('board input exceeds SDRAM')
    return board_input.tobytes(), expected.tobytes(), record


def execute(client, bundle, location, board_input, expected, record, timeout):
    _, commands, _, schedule, _, _, _ = bundle
    base, entry = location['external_base'], location['entry']
    client.write(0x41001c, entry.to_bytes(2, 'little'))
    wall_start = time.monotonic()
    client.write_external(base, board_input)
    client.write(0x410000, b'\x01')
    deadline = time.monotonic() + timeout
    while True:
        status = decode_status(client.exchange(STATUS))
        if not status['busy']:
            break
        if time.monotonic() > deadline:
            raise TimeoutError('ambiguous FPGA launch; the runner will not retry it')
    regs = client.read(0x410000, 32)
    if status['error'] or regs[1] or regs[4:8] != b'SEQ4':
        raise RuntimeError(f'FPGA error: {status}, {regs.hex()}')
    elapsed, engine, dma, overlap, index, _ = struct.unpack('<6I', regs[8:])
    if index != entry + schedule['command_count'] - 1:
        raise AssertionError('FPGA stopped at an unexpected command record')
    output = schedule['final_output']
    actual = client.read_external(base + output['ext'], output['bytes'])
    wall = time.monotonic() - wall_start
    if actual != expected:
        raise AssertionError(f'exact INT8 logit mismatch: {record["id"]}; '
                             f'actual={actual.hex()} expected={expected.hex()}')
    logits = np.frombuffer(actual, dtype=np.int8)
    return {'sample_id': record['id'], 'label': record['label'],
            'prediction': int(np.argmax(logits)), 'output_hex': actual.hex(),
            'elapsed_cycles': elapsed, 'engine_cycles': engine,
            'dma_cycles': dma, 'overlap_cycles': overlap,
            'command_index': index, 'wall_seconds': wall}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True,
                        choices=('kws-accuracy', 'vww-accuracy', 'switch'))
    parser.add_argument('--bitstream', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--baud', type=int, default=750000)
    parser.add_argument('--port', default='/dev/cu.usbserial-20250303171')
    parser.add_argument('--timeout', type=float, default=20)
    parser.add_argument('--resume', action='store_true',
                        help='resume a fully checked accuracy prefix; switch stress always restarts')
    args = parser.parse_args()
    if args.resume and args.campaign == 'switch':
        parser.error('an interrupted switch stress run is not consecutive')
    bundles = {name: prepared(name) for name in ('kws', 'vww')}
    combined, locations = combine(bundles['kws'][1], bundles['kws'][2],
                                  bundles['vww'][1], bundles['vww'][2])
    jobs, plan_sha = jobs_for(args.campaign, args.limit)
    program_sha = hashlib.sha256(combined).hexdigest()
    bit_sha = sha(args.bitstream)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    records_path = args.report.with_suffix('.jsonl')
    identity = {'campaign': args.campaign, 'bitstream_sha256': bit_sha,
                'command_program_sha256': program_sha,
                'job_plan_sha256': plan_sha, 'jobs_planned': len(jobs),
                'full_campaign': args.limit is None,
                'split_sha256': {name: bundles[name][6] for name in bundles},
                'baud': args.baud, 'external_locations': locations}
    completed = 0
    if args.resume:
        old = json.loads(args.report.read_text())
        if any(old.get(key) != value for key, value in identity.items()):
            raise ValueError('resume identity mismatch')
        with records_path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if completed >= len(jobs) or row['job'] != jobs[completed]['job'] or \
                        row['workload'] != jobs[completed]['workload'] or \
                        row['sample_index'] != jobs[completed]['sample_index'] or \
                        row['sample_id'] != bundles[row['workload']][5][row['sample_index']]['id']:
                    raise ValueError('resume prefix differs from frozen job plan')
                completed += 1
    elif args.report.exists() or records_path.exists():
        raise FileExistsError('use a fresh report path or explicitly resume accuracy')
    report = {**identity, 'status': 'running', 'jobs_completed': completed,
              'model_load_events': [], 'reflash_events': 0,
              'physical_board': True, 'output_mismatches': 0}

    def save():
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    save()
    try:
        with serial.Serial(args.port, args.baud, timeout=5, write_timeout=5) as port:
            port.reset_input_buffer()
            client = TiledClient(port)
            client.capabilities()
            if client.read(0x410004, 4) != b'SEQ4':
                raise ValueError('connected image lacks autonomous sequencer')
            client.wait_idle(30)
            for name in ('kws', 'vww'):
                base = locations[name]['external_base']
                payload = bundles[name][2]
                start = time.monotonic()
                client.write_external(base, payload)
                if client.read_external(base, len(payload)) != payload:
                    raise AssertionError(f'{name} immutable image readback mismatch')
                report['model_load_events'].append({
                    'workload': name, 'bytes': len(payload),
                    'seconds': time.monotonic() - start,
                    'image_sha256': hashlib.sha256(payload).hexdigest()})
                save()
            client.write(0x500000, combined)
            if client.read(0x500000, len(combined)) != combined:
                raise AssertionError('combined command-memory readback mismatch')
            for name in ('kws', 'vww'):
                index = locations[name]['entry']
                client.write(0x41001c, index.to_bytes(2, 'little'))
                if client.read(0x41001c, 2) != index.to_bytes(2, 'little'):
                    raise AssertionError('connected image does not support two resident schedules')
            with records_path.open('a') as stream:
                for job in jobs[completed:]:
                    name, sample_index = job['workload'], job['sample_index']
                    board_input, expected, record = reference_and_input(
                        bundles[name], sample_index)
                    result = execute(client, bundles[name], locations[name],
                                     board_input, expected, record, args.timeout)
                    result.update(job=job['job'], workload=name,
                                  sample_index=sample_index)
                    stream.write(json.dumps(result, sort_keys=True) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    report['jobs_completed'] += 1
                    if report['jobs_completed'] % 100 == 0:
                        save()
                        print(f'{args.campaign}: {report["jobs_completed"]}/{len(jobs)}',
                              flush=True)
        counts = {name: {'jobs': 0, 'correct': 0, 'cycles': [], 'wall': []}
                  for name in ('kws', 'vww')}
        with records_path.open() as stream:
            for line in stream:
                item = json.loads(line)
                row = counts[item['workload']]
                row['jobs'] += 1
                row['correct'] += item['prediction'] == item['label']
                row['cycles'].append(item['elapsed_cycles'])
                row['wall'].append(item['wall_seconds'])
        report['metrics'] = {name: {
            'jobs': row['jobs'], 'correct': row['correct'],
            'accuracy': row['correct'] / row['jobs'] if row['jobs'] else None,
            'median_cycles': statistics.median(row['cycles']) if row['cycles'] else None,
            'median_wall_seconds': statistics.median(row['wall']) if row['wall'] else None}
            for name, row in counts.items()}
        report['records_sha256'] = sha(records_path)
        report['status'] = 'passed' if args.limit is None else 'passed-probe'
        save()
    except Exception as error:
        report['status'] = 'failed'
        report['failure'] = repr(error)
        save()
        raise


if __name__ == '__main__':
    main()
