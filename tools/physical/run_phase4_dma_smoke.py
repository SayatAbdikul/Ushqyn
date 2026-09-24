#!/usr/bin/env python3
"""Program and capture the Phase 4 scratchpad/DMA/SDRAM round-trip test."""

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
    'hardware/phase4_sdram/dma_smoke.sv',
    'hardware/phase4_sdram/pll_dma_20.v',
    'hardware/phase4_sdram/build_dma.tcl',
    'hardware/phase4_sdram/bist.cst',
    'hardware/phase4_sdram/bist.sdc',
    'hardware/phase4_sdram/sdram_controller_hs.ipc',
    'rtl/v2/target_pkg.sv',
    'rtl/v2/requantizer.sv',
    'rtl/v2/scratchpad.sv',
    'rtl/v2/engine.sv',
    'rtl/v2/tile_dma.sv',
    'rtl/v2/tiled_core.sv',
    'rtl/v2/hs_sdram_port.sv',
    'rtl/v2/sdram_refresh.sv',
    'rtl/v2/uart_tx.sv',
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
    parser.add_argument('--bitstream', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--timeout', type=float, default=20)
    parser.add_argument('--skip-program', action='store_true')
    args = parser.parse_args()
    bitstream = args.bitstream.resolve(strict=True)
    record = {
        'device': 'GW2AR-LV18QN88C8/I7',
        'bitstream': str(bitstream),
        'bitstream_sha256': sha256(bitstream),
        'source_sha256': {name: sha256(ROOT / name) for name in SOURCES},
        'start_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'clock_hz': 20250000,
        'cases': [
            {'sram_base': 0, 'sdram_base': 0x7f8000, 'bytes': 32768,
             'property': 'last SDRAM byte'},
            {'sram_base': 0, 'sdram_base': 0x1ffff8, 'bytes': 13,
             'property': 'bank crossing with five-byte final strobe'},
        ],
        'status': 'running',
    }
    save(args.report, record)
    if not args.skip_program:
        command = ['openFPGALoader', '-b', 'tangnano20k', '--ftdi-serial',
                   args.serial_id, '--freq', '2500000', '-m', '-v', str(bitstream)]
        result = subprocess.run(command, capture_output=True, text=True)
        record['programmer_exit_code'] = result.returncode
        record['programmer_log'] = result.stdout + result.stderr
        save(args.report, record)
        if result.returncode:
            record['status'] = 'programming_failed'
            save(args.report, record)
            print('DMA smoke programming failed', flush=True)
            return 1

    start = time.monotonic()
    frame = bytearray()
    with serial.Serial(args.port, 115200, timeout=0.5) as port:
        port.reset_input_buffer()
        settle_until = time.monotonic() + 2.0
        while time.monotonic() < settle_until:
            port.read(4096)
        port.reset_input_buffer()
        while time.monotonic() - start < args.timeout:
            frame.extend(port.read(64))
            while len(frame) >= 2:
                if frame[0] == 0xbd:
                    if len(frame) < 18:
                        break
                    record['reported_cases'] = frame[1]
                    cycles = [int.from_bytes(frame[2+i*4:6+i*4], 'little')
                              for i in range(4)]
                    if frame[1] != 2 or any(value == 0 for value in cycles):
                        record['status'] = 'invalid_success_frame'
                    else:
                        for i, item in enumerate(record['cases']):
                            item['write_cycles'] = cycles[i*2]
                            item['read_cycles'] = cycles[i*2+1]
                            item['write_mib_s'] = round(
                                item['bytes'] * record['clock_hz'] /
                                item['write_cycles'] / (1 << 20), 3)
                            item['read_mib_s'] = round(
                                item['bytes'] * record['clock_hz'] /
                                item['read_cycles'] / (1 << 20), 3)
                        record['status'] = 'passed'
                    break
                if frame[0] == 0xed:
                    record['status'] = 'failed'
                    record['failure_reason'] = frame[1]
                    break
                del frame[0]
            if record['status'] != 'running':
                break
    if record['status'] == 'running':
        record['status'] = 'timeout'
    record['elapsed_s'] = round(time.monotonic() - start, 3)
    save(args.report, record)
    print(f"DMA smoke {record['status']} in {record['elapsed_s']} s; "
          f"reason={record.get('failure_reason', '')}", flush=True)
    return 0 if record['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
