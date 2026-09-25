#!/usr/bin/env python3
"""Profile exact Phase 4 directed kernels through the host-addressable image."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import serial

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'compiler'))
sys.path.insert(0, str(ROOT/'test/phase4'))
from kernel_vectors import vectors
from tiled_host import TiledClient
from host import RESET
from hardware_v2 import Descriptor
from phase4_cost import engine_cycles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--bitstream', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--holdout', action='store_true')
    parser.add_argument('--recover-usb', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError('repeats must be positive')
    cases = []
    source_vectors = vectors
    if args.holdout:
        from holdout_vectors import vectors as source_vectors
    prepared = list(source_vectors())
    predictions = [{'label': v[-1], 'descriptor': vars(v[0]),
                    'predicted_cycles': engine_cycles(v[0])} for v in prepared]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    prediction_path=args.report.with_suffix('.predictions.json')
    prediction_path.write_text(json.dumps(predictions,indent=2,sort_keys=True)+'\n')
    with serial.Serial(args.port, 115200, timeout=5, write_timeout=5) as uart:
        uart.reset_input_buffer()
        client_type=TiledClient
        if args.recover_usb:
            from run_physical_sequence import RecoverableUploadClient
            client_type=RecoverableUploadClient
        client = client_type(uart)
        client.capabilities()
        client.wait_idle(30)
        for desc, inputs, weights, params, expected, macs, label in prepared:
            client.exchange(RESET)
            segments = [(0, desc.encode() + Descriptor(0).encode()),
                        (desc.input, np.asarray(inputs, np.int8).tobytes())]
            if weights is not None:
                for channel, row in enumerate(weights):
                    segments.append((desc.weight + channel*desc.row_stride,
                                     np.asarray(row, np.int8).tobytes()))
            for channel, raw in enumerate(params):
                segments.append((desc.params + channel*16, bytes(raw)))
            for address, data in segments:
                client.write(address, data)
                if client.read(address, len(data)) != data:
                    raise AssertionError(f'{label}: SRAM upload mismatch')
            expected_bytes = np.asarray(expected, np.int8).tobytes()
            runs = []
            for repeat in range(args.repeats):
                counters = client.run(0, timeout=30)
                actual = client.read(desc.output, len(expected_bytes))
                if actual != expected_bytes:
                    raise AssertionError(f'{label}: output mismatch at repeat {repeat}')
                if (counters['elapsed'] != sum(counters[k] for k in
                    ('compute_cycles', 'wait_cycles', 'control_cycles')) or
                    counters['useful_macs'] != macs or counters['protocol_errors']):
                    raise AssertionError(f'{label}: counter mismatch')
                runs.append(counters)
            cases.append({'label': label, 'descriptor': vars(desc),
                          'expected_sha256': hashlib.sha256(expected_bytes).hexdigest(),
                          'runs': runs, 'integer_mismatches': 0})
            cases[-1]['predicted_cycles']=engine_cycles(desc)
            print(f'{label}: {args.repeats} exact physical runs, '
                  f'{runs[-1]["elapsed"]} engine cycles', flush=True)
    result = {
        'status': 'passed', 'physical_board': True,
        'bitstream_sha256': hashlib.sha256(args.bitstream.read_bytes()).hexdigest(),
        'scope': 'directed audio/vision kernels on host-addressable SDRAM image',
        'cases': cases,
        'prospective_holdout': args.holdout,
        'predictions_sha256': hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        'limits': ['SRAM-resident kernel cycles; no DMA or UART time included',
                   'nominal 20.25-MHz core clock'],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
