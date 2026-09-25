#!/usr/bin/env python3
"""Measure a guarded DMA/engine overlap on the connected Tang Nano 20K."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import serial

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'compiler'))
sys.path.insert(0, str(ROOT/'tools/phase2'))
sys.path.insert(0, str(ROOT/'tools/phase4'))
from host import RESET, RUN, STATUS, decode_status
from hardware_v2 import Descriptor
from tiled_host import CONTROL_BASE, TiledClient


def sha(data):
    return hashlib.sha256(data).hexdigest()


def configure(client, ext, sram, length):
    value = (ext.to_bytes(3, 'little') + b'\0' +
             sram.to_bytes(3, 'little') + b'\0' +
             length.to_bytes(4, 'little') + b'\x01')
    client.write(CONTROL_BASE, value)


def counters(client):
    raw = client.read(CONTROL_BASE + 26, 6)
    return {'dma_cycles': int.from_bytes(raw[:4], 'little'),
            'overlap_cycles': int.from_bytes(raw[4:], 'little')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--bitstream', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()

    folder = ROOT/'work/phase4'
    plan = json.loads((folder/'kws-tiled-plan.json').read_text())
    tile = plan['layers'][1]['tiles'][0]
    image = (folder/'kws-parameter-image.bin').read_bytes()
    input_data = (folder/'rtl-kws/input.bin').read_bytes()
    with np.load(folder/'rtl-kws/expected.npz') as expected:
        output = expected['layer_1'].tobytes()
    descriptor = bytes.fromhex(tile['descriptor_hex']) + Descriptor(0).encode()
    # This tile uses [0, 12208). Keep the incoming buffer wholly above it.
    scratch_limit = tile['scratch_bytes']
    spare_sram, spare_ext, spare_length = 0x4000, 0x700000, 8192
    if scratch_limit > spare_sram or spare_sram + spare_length > 32768:
        raise ValueError('no disjoint incoming SRAM region')
    dummy = bytes((i * 73 + 19) & 255 for i in range(spare_length))
    output_sram = tile['transfers'][-1]['sram']

    with serial.Serial(args.port, 115200, timeout=5, write_timeout=5) as uart:
        uart.reset_input_buffer()
        client = TiledClient(uart)
        client.capabilities()
        client.exchange(RESET)
        client.wait_idle(30)
        client.write_external(0, image)
        client.write_external(0, input_data)
        client.write_external(spare_ext, dummy)
        for transfer in tile['transfers'][:-1]:
            client.transfer(transfer)
        client.write(0, descriptor)
        client.write(CONTROL_BASE + 14, scratch_limit.to_bytes(2, 'little'))

        engine_only = client.run(0, timeout=30)
        if client.read(output_sram, len(output)) != output:
            raise AssertionError('engine-only Conv output mismatch')
        configure(client, spare_ext, spare_sram, spare_length)
        client.write(CONTROL_BASE + 13, b'\x01')
        client.wait_idle(30)
        dma_only = counters(client)
        if client.read(spare_sram, spare_length) != dummy:
            raise AssertionError('DMA-only payload mismatch')

        # A completed RUN response precedes the DMA START frame. All DMA
        # registers are staged before RUN so its UART latency is minimized.
        client.exchange(RESET)
        client.exchange(RUN, 0)
        client.write(CONTROL_BASE + 13, b'\x01')
        concurrent = client.wait_idle(30)
        parallel = counters(client)
        if client.read(output_sram, len(output)) != output:
            raise AssertionError('concurrent Conv output mismatch')
        if client.read(spare_sram, spare_length) != dummy:
            raise AssertionError('concurrent DMA payload mismatch')
        if parallel['overlap_cycles'] == 0:
            raise AssertionError('no physical engine/DMA overlap observed')
        sequential_cycles = engine_only['elapsed'] + dma_only['dma_cycles']
        union_cycles = (concurrent['elapsed'] + parallel['dma_cycles'] -
                        parallel['overlap_cycles'])

        # Attempt an overwrite inside the advertised live tile footprint.
        # The guard must reject it before the DMA starts, while RUN continues.
        client.exchange(RESET)
        configure(client, spare_ext, 0x100, spare_length)
        client.exchange(RUN, 0)
        client.write(CONTROL_BASE + 13, b'\x01')
        status = decode_status(client.exchange(STATUS))
        while status['busy']:
            status = decode_status(client.exchange(STATUS))
        if status['error'] != 9 or counters(client)['dma_cycles'] != \
                parallel['dma_cycles']:
            raise AssertionError('live-region DMA guard did not reject overwrite')
        client.exchange(RESET)

    report = {
        'status': 'passed', 'physical_board': True,
        'bitstream_sha256': sha(args.bitstream.read_bytes()),
        'plan_sha256': sha((folder/'kws-tiled-plan.json').read_bytes()),
        'input_sha256': sha(input_data), 'expected_output_sha256': sha(output),
        'case': 'KWS first Conv and disjoint 8192-byte SDRAM-to-SRAM transfer',
        'live_region_limit_bytes': scratch_limit,
        'incoming_region': [spare_sram, spare_sram + spare_length],
        'engine_only_cycles': engine_only['elapsed'],
        'dma_only_cycles': dma_only['dma_cycles'],
        'concurrent_engine_cycles': concurrent['elapsed'],
        'concurrent_dma_cycles': parallel['dma_cycles'],
        'overlap_cycles': parallel['overlap_cycles'],
        'sequential_cycle_sum': sequential_cycles,
        'concurrent_union_cycles': union_cycles,
        'speedup_from_overlap': sequential_cycles / union_cycles,
        'guard_error_for_live_region': status['error'],
        'limits': ['one directed tile, not a full-model ping-pong schedule',
                   'nominal core cycles exclude host UART staging and upload',
                   'no board power measurement'],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: report[key] for key in
                      ('overlap_cycles', 'speedup_from_overlap',
                       'guard_error_for_live_region')}))


if __name__ == '__main__':
    main()
