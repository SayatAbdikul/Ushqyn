#!/usr/bin/env python3
"""Measure pinned B1/B2 candidate samples on the common dual-resident FPGA image.

This is a correctness/performance probe, not a complete B03 comparison. Run
only when no other job owns the board UART; each invocation replaces SDRAM and
command contents, but the FPGA bitstream stays unchanged.
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import serial


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/phase4'))
from tiled_host import TiledClient
from probe_primary import prepared, sha
from run_dual_board import execute, reference_and_input
from schedule import draw_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=('kws', 'vww'))
    parser.add_argument('--policy', required=True, choices=('B1', 'B2'))
    parser.add_argument('--bitstream', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--candidate-dir', type=Path,
                        default=ROOT / 'work/phase5/baseline-candidates')
    parser.add_argument('--jobs', type=int, default=64)
    parser.add_argument('--port', default='/dev/cu.usbserial-20250303171')
    parser.add_argument('--baud', type=int, default=750000)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    candidate = json.loads((args.candidate_dir / 'manifest.json').read_text())[
        'models'][args.model][args.policy]
    stem = f'{args.model}-{args.policy.lower()}'
    commands = (args.candidate_dir / f'{stem}.commands.bin').read_bytes()
    payload = (args.candidate_dir / f'{stem}.payload.bin').read_bytes()
    schedule = json.loads((args.candidate_dir / f'{stem}.schedule.json').read_text())
    if (hashlib.sha256(commands).hexdigest() != candidate['commands_sha256']
            or hashlib.sha256(payload).hexdigest() != candidate['payload_sha256']
            or schedule['program_sha256'] != candidate['commands_sha256']
            or schedule['image_sha256'] != candidate['payload_sha256']):
        raise ValueError('baseline candidate files differ from manifest')
    source = prepared(args.model)
    program, _, _, _, features, records, split_sha = source
    if (candidate['source_onnx_sha256'] != sha(ROOT / f'work/phase4/{args.model}-logits.onnx')
            or candidate['calibration_sha256'] != sha(
                ROOT / f'work/phase4/{args.model}-calibration-rebased.json')):
        raise ValueError('baseline source or calibration differs from frozen model')
    sample_indices = draw_indices(len(records), args.jobs, 0xB105)
    sample_sha = hashlib.sha256(json.dumps(sample_indices,
                                          separators=(',', ':')).encode()).hexdigest()
    bundle = (program, commands, payload, schedule, features, records, split_sha)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.report.exists() or args.report.with_suffix('.jsonl').exists():
        raise FileExistsError('choose a fresh baseline report path')
    rows_path = args.report.with_suffix('.jsonl')
    report = {'schema': 1, 'status': 'running', 'scope': 'fixed-seed sample probe only',
              'model': args.model, 'policy': args.policy, 'jobs_planned': args.jobs,
              'jobs_completed': 0, 'sample_indices_sha256': sample_sha,
              'bitstream_sha256': sha(args.bitstream), 'split_sha256': split_sha,
              'commands_sha256': candidate['commands_sha256'],
              'payload_sha256': candidate['payload_sha256'], 'output_mismatches': 0,
              'reflash_events': 0, 'baud': args.baud}

    def save():
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    save()
    try:
        with serial.Serial(args.port, args.baud, timeout=5, write_timeout=5) as port:
            port.reset_input_buffer()
            client = TiledClient(port)
            client.capabilities()
            if client.read(0x410004, 4) != b'SEQ4':
                raise ValueError('board lacks the autonomous sequencer')
            client.wait_idle(30)
            started = time.monotonic()
            client.write_external(0, payload)
            if client.read_external(0, len(payload)) != payload:
                raise AssertionError('baseline payload readback differs')
            report['model_load_seconds'] = time.monotonic() - started
            client.write(0x500000, commands)
            if client.read(0x500000, len(commands)) != commands:
                raise AssertionError('baseline command readback differs')
            with rows_path.open('w') as stream:
                for job, sample_index in enumerate(sample_indices):
                    board_input, expected, record = reference_and_input(bundle, sample_index)
                    result = execute(client, bundle,
                                     {'external_base': 0, 'entry': 0},
                                     board_input, expected, record, 20)
                    result.update(job=job, workload=args.model,
                                  sample_index=sample_index)
                    stream.write(json.dumps(result, sort_keys=True) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    report['jobs_completed'] += 1
                    if report['jobs_completed'] % 16 == 0:
                        save()
                        print(f'{args.model} {args.policy}: {report["jobs_completed"]}/{args.jobs}',
                              flush=True)
        with rows_path.open() as stream:
            rows = [json.loads(line) for line in stream]
        report['records_sha256'] = sha(rows_path)
        report['correct'] = sum(row['prediction'] == row['label'] for row in rows)
        report['median_cycles'] = statistics.median(row['elapsed_cycles'] for row in rows)
        report['median_wall_seconds'] = statistics.median(row['wall_seconds'] for row in rows)
        report['status'] = 'passed-probe'
        save()
    except Exception as error:
        report['status'] = 'failed'
        report['failure'] = repr(error)
        save()
        raise


if __name__ == '__main__':
    main()
