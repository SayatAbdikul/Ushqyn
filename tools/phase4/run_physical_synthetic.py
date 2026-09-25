#!/usr/bin/env python3
"""Run multi-kernel and over-SRAM tile cases on the SDRAM-connected board."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import serial

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'compiler'))
sys.path.insert(0, str(ROOT/'test/phase4'))
sys.path.insert(0, str(ROOT/'tools/phase4'))
from integer_reference import evaluate
from phase4_compile import compile_tiled
from quantization import Quantization, quantize_parameters
from test_tiled_program import small_program
from tiled_host import TiledClient, execute_plan


def cases():
    small = small_program()
    qx = np.random.default_rng(4242).integers(
        -128, 128, size=small.tensors['x'].shape, dtype=np.int8)
    yield 'conv-depthwise-pointwise-relu-maxpool-avgpool-clip-fc', small, qx

    q = Quantization(.05, -7)
    shape = (1, 16320)
    large = SimpleNamespace(inputs=['x'], outputs=['y'], constants={},
                            tensors={name: SimpleNamespace(shape=shape,
                                                           quantization=q)
                                     for name in ('x', 'y')},
                            layers=[SimpleNamespace(op='Relu', inputs=['x'],
                                                    output='y', attributes={},
                                                    parameters={})])
    qx = np.random.default_rng(4243).integers(-128, 128, size=shape,
                                               dtype=np.int8)
    yield 'two-tile-relu', large, qx

    fq_in, fq_out = Quantization(.03, -11), Quantization(.07, 5)
    frng = np.random.default_rng(4244)
    weight = frng.normal(size=(64, 512))
    fc = SimpleNamespace(inputs=['x'], outputs=['y'], constants={},
                         tensors={'x': SimpleNamespace(shape=(1, 512),
                                                       quantization=fq_in),
                                  'y': SimpleNamespace(shape=(1, 64),
                                                       quantization=fq_out)},
                         layers=[SimpleNamespace(
                             op='Gemm', inputs=['x'], output='y', attributes={},
                             parameters=quantize_parameters(
                                 weight, frng.normal(size=64), fq_in, fq_out))])
    qx = frng.integers(-128, 128, size=(1, 512), dtype=np.int8)
    yield 'multi-tile-fc-32kib-weights', fc, qx

    # Reproduce the frozen ToyCar operator dimensions, not its unavailable
    # source weights or anomaly-scoring dataset. The final Gemm stays linear.
    inventory = json.loads((ROOT/'benchmarks/manifests/ad.canonical-inventory.json')
                           .read_text())
    shapes = {tensor['name']: tuple(tensor['shape'])
              for tensor in inventory['tensors'] if not tensor['constant']}
    aq = Quantization(.05, -7)
    arng = np.random.default_rng(4245)
    layers = []
    for operator in inventory['operators']:
        source, destination = operator['inputs'][0], operator['outputs'][0]
        if operator['op'] == 'Gemm':
            features, neurons = shapes[source][-1], shapes[destination][-1]
            if operator['macs'] != features*neurons:
                raise ValueError('ToyCar inventory MAC/dimension mismatch')
            parameters = quantize_parameters(
                arng.normal(0, .02, size=(neurons, features)),
                arng.normal(0, .01, size=neurons), aq, aq)
        elif operator['op'] == 'Relu':
            parameters = {}
        else:
            raise ValueError(f'unsupported ToyCar inventory operator {operator["op"]}')
        layers.append(SimpleNamespace(op=operator['op'], inputs=[source],
                                      output=destination, attributes={},
                                      parameters=parameters))
    ad_program = SimpleNamespace(
        inputs=['input_1'], outputs=[layers[-1].output], constants={},
        tensors={name: SimpleNamespace(shape=shape, quantization=aq)
                 for name, shape in shapes.items()}, layers=layers)
    qx = arng.integers(-128, 128, size=shapes['input_1'], dtype=np.int8)
    yield 'toycar-frozen-layer-shapes-synthetic-weights', ad_program, qx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--bitstream', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    records = []
    with serial.Serial(args.port, 115200, timeout=5, write_timeout=5) as uart:
        uart.reset_input_buffer()
        client = TiledClient(uart)
        for name, program, qx in cases():
            plan, image = compile_tiled(program)
            reference = evaluate(program, {program.inputs[0]: qx})
            expected = [reference[layer.output].tobytes()
                        for layer in program.layers]
            started = time.monotonic()
            result = execute_plan(client, plan, image, qx.tobytes(), expected,
                                  progress=lambda current, total: print(
                                      f'{name}: exact node {current}/{total}',
                                      flush=True))
            host_wall = time.monotonic() - started
            record = {
                'case': name, 'nodes': len(result['nodes']),
                'tiles': sum(n['tiles'] for n in result['nodes']),
                'parameter_image_bytes': len(image),
                'parameter_image_sha256': result['parameter_image_sha256'],
                'final_output_sha256': result['final_output_sha256'],
                'engine_cycles': sum(n['engine_counters']['elapsed']
                                     for n in result['nodes']),
                'dma_cycles': sum(n['dma_elapsed_cycles'] for n in result['nodes']),
                'dma_payload_bytes': sum(n['dma_payload_bytes']
                                         for n in result['nodes']),
                'host_wall_seconds_including_upload': host_wall,
                'host_node_schedule_seconds': sum(
                    n['host_wall_seconds'] for n in result['nodes'])}
            if name.startswith('toycar-'):
                record['source_inventory_sha256'] = hashlib.sha256(
                    (ROOT/'benchmarks/manifests/ad.canonical-inventory.json')
                    .read_bytes()).hexdigest()
                record['weight_scope'] = ('deterministic synthetic parameters; '
                                          'frozen ToyCar operator shapes')
            records.append(record)
    report = {
        'status': 'passed', 'physical_board': True,
        'bitstream_sha256': hashlib.sha256(args.bitstream.read_bytes()).hexdigest(),
        'scope': 'independent integer oracle at every synthetic node',
        'cases': records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
