#!/usr/bin/env python3
"""Physical SRAM, protocol, kernel and layer checks on the current bitstream.

Uses only the accelerator's addressable scratchpad; this does not test SDRAM.
Program the bitstream separately. UART access is exclusive while this runs.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np

from run_models import ROOT, TARGET, digest, load_fixture
sys.path.insert(0, str(ROOT / 'compiler'))
sys.path.insert(0, str(ROOT / 'test/phase4'))
from hardware_v2 import Descriptor
from kernel_vectors import vectors
from host import Client, frame, parse_response, decode_status
from host import CAPS, READ, WRITE, RUN, STATUS, ABORT, RESET


def reconciles(counters, macs=None, protocol_errors=0):
    assert not counters['busy'] and not counters['error'], counters
    assert counters['elapsed'] == sum(counters[k] for k in
                                      ('compute_cycles', 'wait_cycles', 'control_cycles')), counters
    assert counters['protocol_errors'] == protocol_errors, counters
    if macs is not None:
        assert counters['useful_macs'] == macs, counters


def memory_checks(client, record):
    size = client.capabilities()
    assert size == 32768
    client.exchange(RESET)
    address = np.arange(size, dtype=np.uint32)
    rng = np.random.default_rng(20260923)
    patterns = {
        'zeros': bytes(size), 'ones': bytes([255]) * size,
        'walking_one': (1 << (address % 8)).astype(np.uint8).tobytes(),
        'walking_zero': (255 ^ (1 << (address % 8))).astype(np.uint8).tobytes(),
        'address_dependent': ((address * 73) ^ (address >> 8)).astype(np.uint8).tobytes(),
        'seeded_random': rng.integers(0, 256, size, dtype=np.uint8).tobytes(),
    }
    for name, pattern in patterns.items():
        begin = time.monotonic()
        client.write(0, pattern)
        assert client.read(0, size) == pattern, f'SRAM pattern: {name}'
        record(name=name, bytes_written=size, bytes_read=size,
               sha256=hashlib.sha256(pattern).hexdigest(), seconds=time.monotonic()-begin)


def protocol_checks(client, record):
    uart = client.serial
    client.exchange(RESET)
    address = 30001
    sentinel = bytes(range(64))
    client.write(address, sentinel)

    def packet(data, expected):
        uart.write(data)
        header = uart.read(10)
        assert len(header) == 10, 'diagnostic response header timeout'
        size = int.from_bytes(header[8:10], 'little')
        assert 1 <= size <= 65, 'diagnostic response size'
        result = parse_response(header + uart.read(size + 2))
        assert (result['command'], result['sequence'], result['address']) == (
            data[3], data[4], int.from_bytes(data[5:8], 'little')), result
        assert result['status'] == expected, result
        return result

    bad = bytearray(frame(WRITE, 201, address, b'bad'))
    bad[-1] ^= 1
    packet(bad, 1)
    assert client.read(address, 64) == sentinel
    record(name='CRC rejection preserves SRAM')
    nested = b'x' * 64 + frame(WRITE, 202, address, b'evil')
    packet(frame(WRITE, 203, address, nested), 4)
    assert client.read(address, 64) == sentinel
    record(name='oversized frame drains embedded command')
    packet(frame(CAPS, 204, version=3), 2)
    packet(frame(READ, 205, 32760, length=16), 4)
    packet(frame(RUN, 206, address=1), 4)
    packet(frame(99, 207), 5)
    record(name='version, bounds, alignment and opcode rejection')
    uart.write(frame(WRITE, 208, address, b'incomplete')[:-5])
    uart.flush()
    time.sleep(TARGET['timeout_cycles'] / TARGET['clock_hz'] + 0.1)
    assert client.capabilities() == 32768
    assert client.read(address, 64) == sentinel
    status = decode_status(client.exchange(STATUS))
    assert status['protocol_errors'] == 3, status
    record(name='partial frame timeout and recovery', counters=status)

    # Repeat a legal Conv until ABORT, leaving time for actual UART round trips.
    # The hardware watchdog remains enabled as a backstop.
    long = Descriptor(4, input=512, output=16640, weight=32512, params=32640,
                      count=9, outputs=124*126, row_stride=16, next_pc=0,
                      kernel_h=3, kernel_w=3, input_h=126, input_w=128,
                      input_c=1, output_c=1)
    client.write(0, long.encode() + Descriptor(0).encode())
    client.write(32512, bytes(16))
    client.write(32640, struct.pack('<iiBbbbbb', 0, 1 << 30, 30, 0, 0, -128, 127, 0) + b'\0\0')
    client.exchange(RUN)
    assert decode_status(client.exchange(STATUS))['busy'], 'busy test ended too soon'
    packet(frame(WRITE, 209, 100, b'x'), 3)
    packet(frame(READ, 210, 100, length=1), 3)
    packet(frame(RUN, 211), 3)
    client.exchange(ABORT)
    status = decode_status(client.exchange(STATUS))
    assert not status['busy'], status
    record(name='busy ownership and ABORT', counters=status)
    preserved = client.read(32640, 16)
    client.exchange(RESET)
    assert client.read(32640, 16) == preserved
    cleared = decode_status(client.exchange(STATUS))
    assert not any(cleared.values()), cleared
    record(name='RESET clears counters and preserves SRAM')

    wrong = bytearray(Descriptor(0).encode())
    wrong[4] = 1
    client.write(0, wrong)
    client.exchange(RUN)
    status = decode_status(client.exchange(STATUS))
    assert not status['busy'] and status['error'] == 2, status
    client.write(0, Descriptor(0).encode())
    reconciles(client.run(), 0)
    record(name='descriptor version rejection and next-RUN recovery')


def kernel_checks(client, record):
    for desc, inputs, weights, params, expected, macs, label in vectors():
        image = bytearray(32768)
        image[:128] = desc.encode() + Descriptor(0).encode()
        raw = np.asarray(inputs, np.int8).tobytes()
        image[desc.input:desc.input+len(raw)] = raw
        if weights is not None:
            for channel, row in enumerate(weights):
                raw = np.asarray(row, np.int8).tobytes()
                start = desc.weight + channel * desc.row_stride
                image[start:start+len(raw)] = raw
        for channel, raw in enumerate(params):
            start = desc.params + channel * 16
            image[start:start+16] = raw
        client.exchange(RESET)
        client.write(0, image)
        assert client.read(0, len(image)) == image, label
        expected = np.asarray(expected, np.int8).tobytes()
        runs = []
        for _ in range(3):
            counters = client.run()
            actual = client.read(desc.output, len(expected))
            assert actual == expected, label
            reconciles(counters, macs)
            runs.append(counters)
        record(name=label, descriptor=vars(desc), runs=runs,
               image_sha256=hashlib.sha256(image).hexdigest(),
               expected_hex=expected.hex(), actual_hex=actual.hex(), integer_mismatches=0)


def layer_checks(client, fixture, record):
    meta, image, arrays = load_fixture(fixture, 3)
    with np.load(fixture / 'checks.npz', allow_pickle=False) as archive:
        layers = {name: archive['layer_'+name] for name in meta['layer_outputs']}
    client.exchange(RESET)
    client.write(0, image)
    assert client.read(0, len(image)) == image
    halt = len(meta['descriptors']) * 64
    for job in range(3):
        client.write(next(iter(meta['inputs'].values())), arrays['inputs'][job].tobytes())
        for index, (description, name) in enumerate(zip(meta['descriptors'], meta['layer_outputs'])):
            desc = Descriptor(**description)
            client.write(index * 64, replace(desc, next_pc=halt).encode())
            counters = client.run(index * 64)
            reconciles(counters)
            expected = layers[name][job].tobytes()
            actual = client.read(desc.output, len(expected))
            assert actual == expected, f'{fixture.name} job {job} layer {name}'
            client.write(index * 64, desc.encode())
            record(name=name, fixture=str(fixture), job=job, layer=index,
                   image_sha256=meta['image_sha256'], fixture_sha256=digest(fixture/'checks.npz'),
                   counters=counters, bytes_compared=len(expected),
                   expected_sha256=hashlib.sha256(expected).hexdigest(),
                   actual_sha256=hashlib.sha256(actual).hexdigest(), integer_mismatches=0)
    # Leave an ordinary complete program and verify it still executes.
    client.write(next(iter(meta['inputs'].values())), arrays['inputs'][0].tobytes())
    counters = client.run(meta['entry'])
    reconciles(counters, meta['macs'])
    expected = arrays['outputs'][0].tobytes()
    assert client.read(next(iter(meta['outputs'].values())), len(expected)) == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--bitstream', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, action='append', required=True)
    args = parser.parse_args()
    for fixture in args.fixture:
        load_fixture(fixture, 3)
    report = dict(scope='physical board UART; 32-KiB SRAM only', status='running',
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  port=args.port, target_id=TARGET['target_id'],
                  declared_clock_hz=TARGET['clock_hz'],
                  supplied_bitstream_sha256=digest(args.bitstream),
                  bitstream_hash_readback_supported=False, measured_power=False,
                  layer_counter_scope='one isolated layer plus RUN and HALT overhead',
                  checks=[])
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint():
        temporary = args.report.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(args.report)

    def record(**values):
        report['checks'].append(dict(status='passed', **values))
        print(f'PASS: {values["name"]}', flush=True)
        checkpoint()

    import serial
    try:
        with serial.Serial(args.port, TARGET['baud'], timeout=2, write_timeout=2) as uart:
            uart.reset_input_buffer()
            client = Client(uart)
            assert client.capabilities() == TARGET['memory_bytes']
            memory_checks(client, record)
            protocol_checks(client, record)
            kernel_checks(client, record)
            for fixture in args.fixture:
                layer_checks(client, fixture, record)
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        checkpoint()


if __name__ == '__main__':
    main()
