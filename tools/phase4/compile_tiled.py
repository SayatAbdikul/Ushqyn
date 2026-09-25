#!/usr/bin/env python3
"""Compile a calibrated ONNX graph into the Phase 4 external parameter image.

The source model must match the supplied schema-2 calibration report. The
binary leaves activation slot zero blank for a quantized runtime input.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'compiler'))
from phase4_compile import compile_tiled
from static_pipeline import compile_static


def _atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path, help='pinned converted ONNX graph')
    parser.add_argument('calibration', type=Path, help='schema-2 calibration JSON')
    parser.add_argument('image', type=Path, help='output SDRAM parameter bytes')
    parser.add_argument('plan', type=Path, help='output tile schedule JSON')
    args = parser.parse_args()
    if args.image.resolve() == args.plan.resolve():
        parser.error('image and plan paths must differ')
    model_data = args.model.read_bytes()
    calibration_data = args.calibration.read_bytes()
    program = compile_static(onnx.load_from_string(model_data),
                             json.loads(calibration_data))
    plan, image = compile_tiled(program)
    plan['source_onnx_sha256'] = hashlib.sha256(model_data).hexdigest()
    plan['calibration_report_sha256'] = hashlib.sha256(calibration_data).hexdigest()
    plan_data = (json.dumps(plan, indent=2, sort_keys=True)+'\n').encode()
    _atomic_write(args.image, image)
    _atomic_write(args.plan, plan_data)
    print(json.dumps({'parameter_image_bytes': len(image),
                      'parameter_image_sha256': plan['parameter_image_sha256'],
                      'nodes': len(plan['layers']),
                      'tiles': sum(len(x['tiles']) for x in plan['layers'])}))


if __name__ == '__main__':
    main()
