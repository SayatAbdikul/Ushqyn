#!/usr/bin/env python3
"""Run a compiled Phase 4 tile plan over the CRC-framed UART board link.

The same command protocol exposes 32-KiB SRAM at 0x000000, DMA registers at
0x400000, and 8-MiB SDRAM at 0x800000. This runner is prepared for a future
physical test; simulation alone cannot establish board inference throughput.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/phase2'))
sys.path.insert(0, str(ROOT/'compiler'))
from host import CAPS, RESET, STATUS, Client, decode_status
from hardware_v2 import Descriptor

EXT_BASE = 0x800000
CONTROL_BASE = 0x400000
EXT_BYTES = 8 * 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


class TiledClient(Client):
    def capabilities(self):
        data = self.exchange(CAPS)
        if len(data) != 14 or data[10] != 1 or \
           int.from_bytes(data[11:14], 'little') != EXT_BYTES:
            raise ValueError('board does not advertise the tiled SDRAM protocol')
        super_fields = data[:10]
        from host import TARGET
        if super_fields[:4] != bytes([TARGET['numerics'],
                                     TARGET['descriptor_version'],
                                     TARGET['lanes'], TARGET['max_transfer']]) or \
           int.from_bytes(super_fields[4:7], 'little') != 32768 or \
           super_fields[7] != TARGET['address_bits'] or \
           int.from_bytes(super_fields[8:10], 'little') != TARGET['target_id']:
            raise ValueError('incompatible numerical or descriptor target')
        return EXT_BYTES

    def write_external(self, offset, data):
        if offset < 0 or offset + len(data) > EXT_BYTES:
            raise ValueError('external write exceeds SDRAM')
        self.write(EXT_BASE + offset, data)

    def read_external(self, offset, length):
        if offset < 0 or length < 0 or offset + length > EXT_BYTES:
            raise ValueError('external read exceeds SDRAM')
        return self.read(EXT_BASE + offset, length)

    def wait_idle(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = decode_status(self.exchange(STATUS))
            if not status['busy']:
                if status['error']:
                    raise RuntimeError(f"tile engine/DMA error {status['error']}")
                return status
        raise TimeoutError('tile engine/DMA timeout; inspect status before retry')

    def transfer(self, transfer, timeout=10):
        ext = int(transfer['ext'])
        sram = int(transfer['sram'])
        length = int(transfer['bytes'])
        if transfer['direction'] not in ('to_sram', 'from_sram') or \
           ext < 0 or ext + length > EXT_BYTES or \
           sram < 0 or sram + length > 32768 or \
           (ext | sram) & 7 or length <= 0:
            raise ValueError('invalid tile transfer')
        config = ext.to_bytes(3, 'little') + b'\0' + \
                 sram.to_bytes(3, 'little') + b'\0' + \
                 length.to_bytes(4, 'little') + \
                 bytes([transfer['direction'] == 'to_sram'])
        self.write(CONTROL_BASE, config)
        self.write(CONTROL_BASE + 13, b'\x01')
        status = self.wait_idle(timeout)
        registers = self.read(CONTROL_BASE + 16, 6)
        if registers[1] or int.from_bytes(registers[2:6], 'little') != length:
            raise RuntimeError('DMA status or copied-byte count mismatch')
        status['dma_elapsed_cycles'] = int.from_bytes(
            self.read(CONTROL_BASE + 26, 4), 'little')
        if status['dma_elapsed_cycles'] == 0:
            raise RuntimeError('DMA elapsed-cycle counter did not advance')
        return status


def execute_plan(client, plan, image, input_bytes, expected_layers=None,
                 timeout=30, progress=None):
    """Execute sequential tiles, checking every supplied intermediate tensor."""
    if sha(image) != plan['parameter_image_sha256'] or \
       len(image) != plan['parameter_image_bytes']:
        raise ValueError('parameter image and plan mismatch')
    if len(input_bytes) > plan['activation_slot_bytes']:
        raise ValueError('input exceeds activation slot')
    if expected_layers is not None and len(expected_layers) != len(plan['layers']):
        raise ValueError('expected node count mismatch')
    if client.capabilities() != EXT_BYTES:
        raise ValueError('SDRAM capacity mismatch')
    client.wait_idle(timeout)
    client.exchange(RESET)
    client.write_external(0, image)
    if client.read_external(0, len(image)) != image:
        raise AssertionError('parameter image SDRAM readback mismatch')
    client.write_external(0, input_bytes)
    if client.read_external(0, len(input_bytes)) != input_bytes:
        raise AssertionError('input SDRAM readback mismatch')
    records = []
    slot = 0
    for index, layer in enumerate(plan['layers']):
        start = time.monotonic()
        counters = {name: 0 for name in (
            'elapsed', 'compute_cycles', 'wait_cycles', 'control_cycles',
            'useful_macs', 'read_bytes', 'write_bytes')}
        dma_payload_bytes = 0
        dma_elapsed_cycles = 0
        for tile in layer['tiles']:
            for transfer in tile['transfers'][:-1]:
                dma_status = client.transfer(transfer, timeout) or {}
                dma_payload_bytes += int(transfer['bytes'])
                dma_elapsed_cycles += int(dma_status.get('dma_elapsed_cycles', 0))
            client.write(0, bytes.fromhex(tile['descriptor_hex']) +
                         Descriptor(0).encode())
            tile_status = client.run(0, timeout=timeout)
            for name in counters:
                counters[name] += int(tile_status.get(name, 0))
            dma_status = client.transfer(tile['transfers'][-1], timeout) or {}
            dma_payload_bytes += int(tile['transfers'][-1]['bytes'])
            dma_elapsed_cycles += int(dma_status.get('dma_elapsed_cycles', 0))
        if layer['tiles']:
            slot = layer['output_slot']
        output = None
        if expected_layers is not None:
            expected = expected_layers[index]
            output = client.read_external(slot*plan['activation_slot_bytes'],
                                          len(expected))
            if output != expected:
                mismatch = next((i for i, (a, b) in enumerate(zip(output, expected))
                                 if a != b), None)
                raise AssertionError(f'node {index} output mismatch at byte {mismatch}')
        records.append({'node': index, 'kind': layer['kind'],
                        'tiles': len(layer['tiles']),
                        'host_wall_seconds': time.monotonic() - start,
                        'engine_counters': counters,
                        'dma_payload_bytes': dma_payload_bytes,
                        'dma_elapsed_cycles': dma_elapsed_cycles,
                        'output_sha256': sha(output) if output is not None else None})
        if progress is not None:
            progress(index + 1, len(plan['layers']))
    return {'status': 'passed', 'nodes': records,
            'parameter_image_sha256': sha(image),
            'final_output_sha256': records[-1]['output_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--fixture', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args()
    folder = args.fixture
    name = folder.name.removeprefix('rtl-')
    plan_path = folder.parent/f'{name}-tiled-plan.json'
    image_path = folder.parent/f'{name}-parameter-image.bin'
    plan = json.loads(plan_path.read_text())
    image = image_path.read_bytes()
    manifest = json.loads((folder/'manifest.json').read_text())
    if sha(plan_path.read_bytes()) != manifest['plan_sha256'] or \
       sha(image) != manifest['image_sha256'] or \
       sha((folder/'input.bin').read_bytes()) != manifest['input_sha256'] or \
       sha((folder/'expected.npz').read_bytes()) != manifest['expected_npz_sha256']:
        raise ValueError('fixture hash mismatch')
    with np.load(folder/'expected.npz') as expected:
        outputs = [expected[f'layer_{i}'].tobytes()
                   for i in range(manifest['nodes'])]
    import serial
    with serial.Serial(args.port, 115200, timeout=5, write_timeout=5) as uart:
        uart.reset_input_buffer()
        client = TiledClient(uart)
        result = execute_plan(client, plan, image,
                              (folder/'input.bin').read_bytes(), outputs,
                              timeout=args.timeout,
                              progress=lambda current, total: print(
                                  f'{name}: exact node {current}/{total}', flush=True))
    if result['final_output_sha256'] != manifest['output_sha256']:
        raise AssertionError('final output manifest mismatch')
    result.update({'fixture_sha256': sha((folder/'manifest.json').read_bytes()),
                   'model': name, 'physical_board': True})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'model': name, 'nodes': len(result['nodes']),
                      'final_output_sha256': result['final_output_sha256']}))


if __name__ == '__main__':
    main()
