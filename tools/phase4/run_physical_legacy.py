#!/usr/bin/env python3
"""Replay the pinned on-chip MLP and SmallCNN fixtures through SDRAM tiling."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import serial

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'compiler'))
from phase4_compile import compile_tiled
from static_pipeline import compile_static
from tiled_host import TiledClient, execute_plan


def sha(data):
    return hashlib.sha256(data).hexdigest()


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
        for name, fixture in (('mlp', ROOT/'work/phase2/mlp'),
                              ('smallcnn', ROOT/'work/phase3/smallcnn')):
            source_path = fixture/'source.onnx'
            calibration_path = fixture/'calibration.json'
            checks_path = fixture/'checks.npz'
            program = compile_static(
                onnx.load(source_path), json.loads(calibration_path.read_text()))
            plan, image = compile_tiled(program)
            with np.load(checks_path) as checks:
                input_bytes = checks['inputs'][0].tobytes()
                expected = [checks['layer_'+layer.output][0].tobytes()
                            for layer in program.layers]
                final_bytes = checks['outputs'][0].tobytes()
            started = time.monotonic()
            result = execute_plan(client, plan, image, input_bytes, expected,
                                  progress=lambda current, total: print(
                                      f'{name}: exact node {current}/{total}',
                                      flush=True))
            host_wall = time.monotonic() - started
            if result['final_output_sha256'] != sha(final_bytes):
                raise AssertionError(f'{name}: final BSRAM/SDRAM output mismatch')
            records.append({
                'model': name, 'status': 'all-node-exact',
                'source_onnx_sha256': sha(source_path.read_bytes()),
                'calibration_sha256': sha(calibration_path.read_bytes()),
                'original_checks_sha256': sha(checks_path.read_bytes()),
                'input_sha256': sha(input_bytes),
                'parameter_image_sha256': result['parameter_image_sha256'],
                'parameter_image_bytes': len(image),
                'final_output_sha256': result['final_output_sha256'],
                'nodes': len(result['nodes']),
                'tiles': sum(node['tiles'] for node in result['nodes']),
                'engine_cycles': sum(node['engine_counters']['elapsed']
                                     for node in result['nodes']),
                'dma_cycles': sum(node['dma_elapsed_cycles']
                                  for node in result['nodes']),
                'dma_payload_bytes': sum(node['dma_payload_bytes']
                                          for node in result['nodes']),
                'host_wall_seconds_including_upload': host_wall,
                'host_node_schedule_seconds': sum(
                    node['host_wall_seconds'] for node in result['nodes']),
            })
    report = {
        'status': 'passed', 'physical_board': True,
        'bitstream_sha256': sha(args.bitstream.read_bytes()),
        'scope': 'one preexisting pinned fixture input per model, '
                 'every node compared with the saved independent oracle',
        'models': records,
        'limits': ['one input/model, not a complete-set accuracy comparison',
                   'sequential host schedule; no autonomous FPS or board energy'],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({r['model']: r['nodes'] for r in records}))


if __name__ == '__main__':
    main()
