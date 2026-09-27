#!/usr/bin/env python3
"""Paired 256/512-byte UART upload screen on one verified fused FPGA image.

Preflight checks the selected image and pinned VWW fixture without board I/O.
--run programs that image once, then alternates three uploads at each chunk
size. Only the acknowledged upload is timed. Every upload is preceded by an
untimed complement fill and followed by exact full SDRAM readback.
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

import serial

import run_priority as priority
import run_screening as screening
import screen_fused_uart512_candidate as fused
from uart_burst_512 import BURST_BYTES, BurstTiledClient
from variants import ROOT, sha


PAIRS = 3
CHUNKS = (256, 512)
OFFSET = 0


def prepare(route_report):
    plan = fused.prepare(route_report)
    if plan['image'] != 'fused-activation-uart512-v1' or plan['burst_bytes'] != BURST_BYTES:
        raise ValueError('expected the source-verified fused 512-byte UART image')
    name = plan['selected']['vww']['pinned']
    fixture = fused.CORE / 'fixtures' / name
    files = plan['fixtures'][name]['files']
    screening.verify_files(fixture, files)
    data = (fixture / 'input.bin').read_bytes()
    if not data or len(data) > 8 * 1024 * 1024 or len(data) % 512:
        raise ValueError('pinned VWW input is not a complete 512-byte frame set')
    return dict(schema=1, status='preflight-passed', physical_board=False,
        image=plan['image'], bitstream=plan['bitstream'],
        bitstream_sha256=plan['bitstream_sha256'],
        route_report=plan['route_report'],
        route_report_sha256=plan['route_report_sha256'],
        route_inputs_file=plan['route_inputs_file'],
        route_inputs_sha256=plan['route_inputs_sha256'],
        route_sources=plan['route_sources'],
        fixture=str(fixture.relative_to(ROOT)),
        fixture_files=files, input_sha256=sha(fixture / 'input.bin'),
        input_bytes=len(data), offset=OFFSET, baud=750000,
        chunk_bytes=list(CHUNKS), order=list(CHUNKS) * PAIRS,
        repeats_per_chunk=PAIRS, planned=2 * PAIRS,
        runner_sha256=sha(Path(__file__)),
        source_sha256={str(path.relative_to(ROOT)): sha(path) for path in (
            ROOT / 'tools/phase6/screen_fused_uart512_candidate.py',
            ROOT / 'tools/phase6/screen_uart512_candidate.py',
            ROOT / 'tools/phase6/combined_fused_uart512.py',
            ROOT / 'tools/phase6/combined_stream_uart512.py',
            ROOT / 'tools/phase6/uart_burst_512.py',
            ROOT / 'tools/phase6/run_screening.py',
            ROOT / 'tools/phase6/run_priority.py',
            ROOT / 'tools/phase4/tiled_host.py',
            ROOT / 'tools/phase2/host.py')})


def run(plan, output, port):
    screening.require_board_free(port)
    if output.exists():
        raise FileExistsError('preserve previous transfer evidence; choose a new output')
    lock = (ROOT / 'work/phase6/physical-board.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output.mkdir(parents=True)
    screening.save_json(output / 'plan.json', plan)
    report = dict(schema=1, status='running', physical_board=True,
        started_at=screening.timestamp(), image=plan['image'],
        planned=plan['planned'], completed=0,
        plan_sha256=sha(output / 'plan.json'))

    def save():
        report['updated_at'] = screening.timestamp()
        screening.save_json(output / 'report.json', report)

    def interrupted(signum, frame):
        raise KeyboardInterrupt('paired UART transfer interrupted; completed records preserved')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    rows = []
    save()
    try:
        screening.verify_files(ROOT, {plan['bitstream']: plan['bitstream_sha256']})
        report['current'] = 'programming'; save()
        with (output / 'program.log').open('x') as log:
            subprocess.run(['openFPGALoader', '-b', 'tangnano20k',
                '--ftdi-serial', '2025030317', '--freq', '2500000', '-m', '-v',
                str(ROOT / plan['bitstream'])], stdout=log,
                stderr=subprocess.STDOUT, timeout=90, check=True)
        report['program_log_sha256'] = sha(output / 'program.log')
        time.sleep(1)
        folder = ROOT / plan['fixture']
        screening.verify_files(folder, plan['fixture_files'])
        data = (folder / 'input.bin').read_bytes()
        complement = bytes(value ^ 255 for value in data)
        with serial.Serial(port, plan['baud'], timeout=5, write_timeout=5) as uart:
            uart.reset_input_buffer()
            client = BurstTiledClient(uart)
            client.capabilities()  # Read-only FEATURES response must advertise 512.
            report['features_burst_bytes'] = BURST_BYTES
            save()
            with (output / 'records.jsonl').open('x') as stream:
                for index, chunk in enumerate(plan['order']):
                    report['current'] = f'precondition-{index + 1}'; save()
                    # If an upload silently made no writes, a stale matching
                    # input cannot masquerade as a successful readback.
                    client.write_external(plan['offset'], complement)
                    if client.read_external(plan['offset'], len(data)) != complement:
                        raise AssertionError('complement precondition readback mismatch')
                    report['current'] = f'upload-{chunk}-{index + 1}'; save()
                    started = time.monotonic_ns()
                    for begin in range(0, len(data), chunk):
                        client.write_external(plan['offset'] + begin,
                                              data[begin:begin + chunk])
                    upload_seconds = (time.monotonic_ns() - started) / 1e9
                    read_started = time.monotonic_ns()
                    received = client.read_external(plan['offset'], len(data))
                    readback_seconds = (time.monotonic_ns() - read_started) / 1e9
                    if received != data:
                        raise AssertionError('pinned VWW input SDRAM readback mismatch')
                    row = dict(image=plan['image'], pair=index // 2 + 1,
                        sequence=index, chunk_bytes=chunk,
                        frame_count=(len(data) + chunk - 1) // chunk,
                        input_bytes=len(data), input_sha256=plan['input_sha256'],
                        bitstream_sha256=plan['bitstream_sha256'],
                        upload_seconds=upload_seconds,
                        readback_seconds=readback_seconds,
                        readback_verified=True, precondition_verified=True,
                        at=screening.timestamp())
                    priority.append_row(stream, row)
                    rows.append(row)
                    report['completed'] = len(rows); save()
                    print(f'{screening.timestamp()} {len(rows)}/{plan["planned"]} '
                          f'{chunk}-byte upload {upload_seconds:.6f}s', flush=True)
        stored = priority.read_rows(output / 'records.jsonl')
        if stored != rows or len(rows) != plan['planned']:
            raise ValueError('signed transfer record count or identity mismatch')
        medians = {str(chunk): statistics.median(
            row['upload_seconds'] for row in rows if row['chunk_bytes'] == chunk)
            for chunk in CHUNKS}
        summary = dict(median_upload_seconds=medians,
            speedup_512_over_256=medians['256'] / medians['512'],
            input_bytes=plan['input_bytes'], pairs=PAIRS,
            all_readbacks_exact=True)
        screening.verify_files(ROOT, plan['source_sha256'])
        screening.verify_files(ROOT, plan['route_sources'])
        screening.verify_files(ROOT, {plan['bitstream']: plan['bitstream_sha256']})
        screening.verify_files(folder, plan['fixture_files'])
        if (sha(Path(__file__)) != plan['runner_sha256'] or
                sha(ROOT / plan['route_report']) != plan['route_report_sha256'] or
                sha(ROOT / plan['route_inputs_file']) != plan['route_inputs_sha256']):
            raise ValueError('paired transfer source identity changed during run')
        report.update(status='passed-paired-transfer', summary=summary,
            records_sha256=sha(output / 'records.jsonl'),
            finished_at=screening.timestamp())
        save()
        seal = dict(report_sha256=sha(output / 'report.json'),
            plan_sha256=sha(output / 'plan.json'),
            records_sha256=sha(output / 'records.jsonl'),
            runner_sha256=plan['runner_sha256'])
        screening.save_json(output / 'seal.json', seal)
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
        print(json.dumps(dict(status=plan['status'], image=plan['image'],
            bitstream_sha256=plan['bitstream_sha256'],
            input_bytes=plan['input_bytes'], order=plan['order']),
            sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
