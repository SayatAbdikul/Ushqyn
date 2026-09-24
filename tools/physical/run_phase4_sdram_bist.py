#!/usr/bin/env python3
"""Program and capture the Phase 4 full-range, 1-GiB SDRAM board BIST."""

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import time
from pathlib import Path

import serial

EXPECTED_PASSES = 64
EXPECTED_BYTES = 1 << 30
ROOT = Path(__file__).resolve().parents[2]
COMMON_SOURCES = (
    'hardware/phase4_sdram/pll.v',
    'hardware/phase4_sdram/bist.cst',
    'hardware/phase4_sdram/bist.sdc',
    'rtl/v2/sdram_refresh.sv',
    'rtl/v2/uart_tx.sv',
)
OPEN_SOURCES = (
    'hardware/phase4_sdram/bist.sv',
    'hardware/phase4_sdram/build.tcl',
    'hardware/phase4_sdram/third_party/nestang_sdram.v',
)
HS_SOURCES = (
    'hardware/phase4_sdram/bist_hs.sv',
    'hardware/phase4_sdram/hs_byte_adapter.sv',
    'hardware/phase4_sdram/build_hs.tcl',
    'hardware/phase4_sdram/sdram_controller_hs.ipc',
    'work/phase4/ip-generated/sdram_controller_hs.v',
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='/dev/cu.usbserial-20250303171')
    parser.add_argument('--serial-id', default='2025030317')
    parser.add_argument('--bitstream', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--skip-program', action='store_true')
    parser.add_argument('--controller', choices=('open', 'hs'), default='open')
    parser.add_argument('--stop-after-passes', type=int, default=EXPECTED_PASSES)
    args = parser.parse_args()
    if not 1 <= args.stop_after_passes <= EXPECTED_PASSES:
        parser.error('--stop-after-passes must be between 1 and 64')
    bitstream = args.bitstream.resolve(strict=True)
    record = {
        'device': 'GW2AR-LV18QN88C8/I7',
        'serial_id': args.serial_id,
        'uart_port': args.port,
        'bitstream': str(bitstream),
        'bitstream_sha256': sha256(bitstream),
        'source_sha256': {name: sha256(ROOT / name) for name in
                          COMMON_SOURCES + (OPEN_SOURCES if args.controller == 'open'
                                            else HS_SOURCES)},
        'controller': args.controller,
        'expected': {'passes': EXPECTED_PASSES, 'aggregate_bytes': EXPECTED_BYTES,
                     'address_bytes_per_sweep': 1 << 23, 'hold_ms_per_pass': 80},
        'progress': [],
        'status': 'running',
        'start_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if args.controller == 'open':
        record['source_controller'] = {
            'origin': 'https://github.com/nand2mario/sdram-tang-nano-20k',
            'commit': '918ae4143eed676d29b706df6ec7ebcb61e257c1',
            'license': 'Apache-2.0',
        }
    else:
        record['source_controller'] = {
            'origin': 'Gowin Education V1.9.11.03 SDRAM Controller HS IP',
            'encrypted_generated_ip': True,
        }
    save(args.report, record)
    if not args.skip_program:
        command = ['openFPGALoader', '-b', 'tangnano20k', '--ftdi-serial',
                   args.serial_id, '--freq', '2500000', '-m', '-v', str(bitstream)]
        result = subprocess.run(command, text=True, capture_output=True)
        record['programmer_exit_code'] = result.returncode
        record['programmer_log'] = result.stdout + result.stderr
        save(args.report, record)
        if result.returncode:
            raise SystemExit(f'programming failed: {result.stderr}')

    raw = bytearray()
    next_pass = 1
    begin = time.monotonic()
    deadline = begin + args.timeout
    with serial.Serial(args.port, 115200, timeout=0.5) as port:
        while time.monotonic() < deadline:
            chunk = port.read(64)
            if chunk:
                raw.extend(chunk)
            while raw:
                if raw[0] == 0xd4:
                    if len(raw) < 2:
                        break
                    observed = raw[1]
                    del raw[:2]
                    record['progress'].append({'pass': observed,
                                               'elapsed_s': round(time.monotonic()-begin, 3)})
                    print(f'SDRAM sweep {observed}/{EXPECTED_PASSES}', flush=True)
                    if observed != next_pass:
                        record['status'] = 'progress_gap'
                        record['error'] = f'expected pass {next_pass}, got {observed}'
                        save(args.report, record)
                        raise SystemExit(record['error'])
                    next_pass += 1
                    if observed == args.stop_after_passes and observed < EXPECTED_PASSES:
                        record['status'] = 'partial_passed'
                        record['verified_aggregate_bytes'] = observed * (1 << 24)
                        record['elapsed_s'] = round(time.monotonic()-begin, 3)
                        save(args.report, record)
                        print(f'partial_passed: {observed} sweeps', flush=True)
                        return 0
                elif raw[0] == 0xa4:
                    if len(raw) < 6:
                        break
                    count = raw[1]
                    moved = int.from_bytes(raw[2:6], 'little')
                    del raw[:6]
                    record['final_passes'] = count
                    record['aggregate_bytes'] = moved
                    record['elapsed_s'] = round(time.monotonic()-begin, 3)
                    record['status'] = ('passed' if count == EXPECTED_PASSES and
                                        moved == EXPECTED_BYTES and
                                        next_pass == EXPECTED_PASSES else 'invalid_result')
                    save(args.report, record)
                    print(f"{record['status']}: {count} sweeps, {moved} bytes, "
                          f"{record['elapsed_s']} s", flush=True)
                    return 0 if record['status'] == 'passed' else 1
                elif raw[0] == 0xe4:
                    if len(raw) < 7:
                        break
                    count = raw[1]
                    addr = int.from_bytes(raw[2:5], 'little')
                    expected, actual = raw[5], raw[6]
                    record['status'] = 'failed'
                    record['failure'] = {'pass': count, 'address': addr,
                                         'expected': expected, 'actual': actual}
                    record['elapsed_s'] = round(time.monotonic()-begin, 3)
                    save(args.report, record)
                    print(f'SDRAM mismatch: {record["failure"]}', flush=True)
                    return 1
                else:
                    record.setdefault('unexpected_uart_bytes', []).append(raw[0])
                    del raw[0]
            if chunk:
                save(args.report, record)
    record['status'] = 'timeout'
    record['elapsed_s'] = round(time.monotonic()-begin, 3)
    save(args.report, record)
    print(f'SDRAM test timed out after {record["elapsed_s"]} s', flush=True)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
