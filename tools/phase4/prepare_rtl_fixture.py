#!/usr/bin/env python3
"""Prepare real-model tile RTL stimuli and independent integer layer outputs."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'compiler'))
from integer_reference import evaluate
from phase4_compile import compile_tiled
from static_pipeline import compile_static


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', choices=('kws', 'vww'))
    parser.add_argument('model', type=Path)
    parser.add_argument('calibration', type=Path)
    parser.add_argument('plan', type=Path)
    parser.add_argument('image', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    model_data = args.model.read_bytes()
    program = compile_static(onnx.load_from_string(model_data),
                             json.loads(args.calibration.read_text()))
    plan, image = compile_tiled(program)
    written_plan = json.loads(args.plan.read_text())
    if image != args.image.read_bytes() or \
       plan['layers'] != written_plan['layers'] or \
       sha(image) != written_plan['parameter_image_sha256']:
        raise ValueError('compiled model differs from supplied plan or image')
    input_name = program.inputs[0]
    shape = program.tensors[input_name].shape
    qinput = np.random.default_rng(20260924).integers(
        -128, 128, size=shape, dtype=np.int8)
    oracle = evaluate(program, {input_name: qinput})
    if plan['layers'][0]['kind'] == 'host_layout':
        first = program.layers[0].output
        initial = oracle[first]
    else:
        initial = qinput
    if len(initial.tobytes()) > plan['activation_slot_bytes']:
        raise ValueError('host input exceeds activation slot')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.output_dir/'input.bin'
    expected_path = args.output_dir/'expected.npz'
    input_path.write_bytes(initial.tobytes())
    arrays = {f'layer_{i}': oracle[layer.output]
              for i, layer in enumerate(program.layers)}
    np.savez_compressed(expected_path, **arrays)
    record = {'name': args.name, 'status': 'prepared',
              'scope': 'one deterministic quantized input; independent integer oracle',
              'source_onnx_sha256': sha(model_data),
              'calibration_sha256': sha(args.calibration.read_bytes()),
              'plan_sha256': sha(args.plan.read_bytes()),
              'image_sha256': sha(image),
              'input_sha256': sha(initial.tobytes()),
              'input_bytes': initial.size,
              'nodes': len(program.layers),
              'tiles': sum(len(layer['tiles']) for layer in plan['layers']),
              'expected_npz_sha256': sha(expected_path.read_bytes()),
              'output_sha256': sha(oracle[program.outputs[0]].tobytes()),
              'layers': [{'index': i, 'op': layer.op,
                          'output_sha256': sha(oracle[layer.output].tobytes()),
                          'output_bytes': oracle[layer.output].size}
                         for i, layer in enumerate(program.layers)]}
    (args.output_dir/'manifest.json').write_text(
        json.dumps(record, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k: record[k] for k in ('name', 'nodes', 'tiles',
                                             'output_sha256')}))


if __name__ == '__main__':
    main()
