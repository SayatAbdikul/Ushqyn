#!/usr/bin/env python3
"""Apply the declared integer-logits classifier boundary to converted ONNX."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'compiler'))
from classifier_boundary import logits_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('logits', type=Path)
    parser.add_argument('boundary', type=Path)
    args = parser.parse_args()
    converted, boundary = logits_model(onnx.load(args.source))
    args.logits.parent.mkdir(parents=True, exist_ok=True)
    args.boundary.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(converted, args.logits)
    args.boundary.write_text(json.dumps(boundary, indent=2)+'\n')
    print(json.dumps({'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
                      'logits_sha256': hashlib.sha256(args.logits.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
