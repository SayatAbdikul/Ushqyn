#!/usr/bin/env python3
"""Program and capture the directed physical two-word SDRAM burst test."""

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import time
from pathlib import Path

import serial

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    'hardware/phase4_sdram/burst_smoke.sv',
    'hardware/phase4_sdram/pll.v',
    'hardware/phase4_sdram/build_burst.tcl',
    'hardware/phase4_sdram/bist.cst',
    'hardware/phase4_sdram/bist.sdc',
    'hardware/phase4_sdram/sdram_controller_hs.ipc',
    'rtl/v2/hs_sdram_port.sv',
    'rtl/v2/sdram_refresh.sv',
    'rtl/v2/uart_tx.sv',
    'work/phase4/ip-generated/sdram_controller_hs.v',
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', default='/dev/cu.usbserial-20250303171')
    p.add_argument('--serial-id', default='2025030317')
    p.add_argument('--bitstream', required=True, type=Path)
    p.add_argument('--report', required=True, type=Path)
    p.add_argument('--timeout', type=float, default=20)
    p.add_argument('--skip-program', action='store_true')
    p.add_argument('--settle-seconds', type=float, default=2.0,
                   help='drain queued FTDI UART bytes before accepting a frame')
    a = p.parse_args()
    bitstream = a.bitstream.resolve(strict=True)
    report = {
        'device': 'GW2AR-LV18QN88C8/I7',
        'bitstream': str(bitstream), 'bitstream_sha256': sha256(bitstream),
        'source_sha256': {name: sha256(ROOT / name) for name in SOURCES},
        'start_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'tested_byte_addresses': [0, 0x3f8, 0x400, 0x1ffff8,
                                  0x200000, 0x3ffff8, 0x400000, 0x7ffff8],
        'burst_words': 2,
        'distinct_half_patterns': 1,
        'walking_one_patterns': 64,
        'total_write_read_pairs': 73,
        'status': 'running',
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    if not a.skip_program:
        cmd = ['openFPGALoader', '-b', 'tangnano20k', '--ftdi-serial', a.serial_id,
               '--freq', '2500000', '-m', '-v', str(bitstream)]
        programmed = subprocess.run(cmd, capture_output=True, text=True)
        report['programmer_exit_code'] = programmed.returncode
        report['programmer_log'] = programmed.stdout + programmed.stderr
        if programmed.returncode:
            report['status'] = 'programming_failed'
            a.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
            raise SystemExit(1)
    start = time.monotonic()
    frame = bytearray()
    with serial.Serial(a.port, 115200, timeout=0.5) as port:
        port.reset_input_buffer()
        settle_until = time.monotonic() + a.settle_seconds
        while time.monotonic() < settle_until:
            port.read(4096)
        port.reset_input_buffer()
        while time.monotonic() - start < a.timeout:
            raw = port.read(32)
            frame.extend(raw)
            while frame:
                if frame[0] == 0xb9:
                    if len(frame) < 3:
                        break
                    report['status'] = ('passed' if frame[1:3] == bytes((73, 64))
                                        else 'invalid_success_frame')
                    break
                if frame[0] == 0xea:
                    frame_len = 19
                    if len(frame) < frame_len:
                        break
                    report['status'] = 'failed'
                    report['failure'] = {
                        'index': frame[1],
                        'reason': frame[2],
                        'address': (0 if frame[1] <= 64 else
                                    report['tested_byte_addresses'][frame[1]-65]),
                        'actual': int.from_bytes(frame[3:11], 'little'),
                        'expected': int.from_bytes(frame[11:19], 'little'),
                    }
                    break
                del frame[0]
            if report['status'] != 'running':
                break
    if report['status'] == 'running':
        report['status'] = 'timeout'
    report['elapsed_s'] = round(time.monotonic() - start, 3)
    a.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(f"burst smoke {report['status']} in {report['elapsed_s']} s; "
          f"{report.get('failure', '')}", flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
