#!/usr/bin/env python3
"""Measure exact tile-DMA cycles over the board's SDRAM controller."""

import argparse
import hashlib
import json
from pathlib import Path

import serial

from tiled_host import TiledClient
from host import RESET

CORE_HZ = 20_250_000
CASES = [(n, 0x4000) for n in (1, 7, 8, 9, 63, 64, 65, 256,
                                1024, 4096, 32768)] + [
    (16, 0x1ffff8),  # 2-MiB bank boundary
    (256, 0x7fff00),  # final SDRAM byte
]


def payload(length, repeat):
    return bytes((i * 37 + length + repeat * 19) & 255
                 for i in range(length))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--bitstream', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--recover-usb', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError('repeats must be positive')
    records = []
    with serial.Serial(args.port, 115200, timeout=5, write_timeout=5) as uart:
        uart.reset_input_buffer()
        client_type=TiledClient
        if args.recover_usb:
            from run_physical_sequence import RecoverableUploadClient
            client_type=RecoverableUploadClient
        client = client_type(uart)
        client.capabilities()
        client.wait_idle(30)
        client.exchange(RESET)
        for repeat in range(args.repeats):
            for length, ext in CASES:
                data = payload(length, repeat)
                client.write_external(ext, data)
                to_status = client.transfer({
                    'ext': ext, 'sram': 0, 'bytes': length,
                    'direction': 'to_sram'})
                if client.read(0, length) != data:
                    raise AssertionError(f'{length}-byte SDRAM-to-SRAM mismatch')
                changed = bytes(byte ^ 0xa5 for byte in data)
                client.write(0, changed)
                from_status = client.transfer({
                    'ext': ext, 'sram': 0, 'bytes': length,
                    'direction': 'from_sram'})
                if client.read_external(ext, length) != changed:
                    raise AssertionError(f'{length}-byte SRAM-to-SDRAM mismatch')
                for direction, status in (('to_sram', to_status),
                                          ('from_sram', from_status)):
                    cycles = status['dma_elapsed_cycles']
                    records.append({
                        'repeat': repeat, 'direction': direction,
                        'length_bytes': length, 'external_offset': ext,
                        'dma_cycles': cycles,
                        'payload_mib_per_second_at_nominal_clock':
                            (length * CORE_HZ / cycles) / (1024 * 1024),
                    })
                print(f'repeat {repeat+1}/{args.repeats}: {length} bytes '
                      f'at 0x{ext:06x} exact, DMA cycles '
                      f'{to_status["dma_elapsed_cycles"]}/'
                      f'{from_status["dma_elapsed_cycles"]}', flush=True)
    result = {
        'status': 'passed', 'physical_board': True,
        'scope': 'byte-exact bidirectional DMA with host readback',
        'bitstream_sha256': hashlib.sha256(args.bitstream.read_bytes()).hexdigest(),
        'nominal_core_clock_hz': CORE_HZ,
        'records': records,
        'limits': ['DMA duration excludes UART upload/readback and compute',
                   'clock period is nominal PLL setting, not oscilloscope-measured'],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
