#!/usr/bin/env python3
"""Materialize two legal preliminary policies on the identical pinned INT8 models.

B1 uses the conventional largest-fitting layer-wise tile with serialized DMA.
B2 uses 16-KiB live regions and alternate-bank prefetch on the same engine.
These artifacts are candidate preparation, not physical B03 evidence; B3 still
needs an actual, documented adaptation of the pinned DeFiNES policy.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'compiler'))
from phase4_compile import compile_tiled
from phase4_sequence import compile_sequence
from static_pipeline import compile_static


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for model in ('kws', 'vww'):
        source = ROOT / f'work/phase4/{model}-logits.onnx'
        calibration = ROOT / f'work/phase4/{model}-calibration-rebased.json'
        fixture = json.loads((ROOT / f'work/phase4/rtl-{model}/manifest.json').read_text())
        if (digest(source.read_bytes()) != fixture['source_onnx_sha256']
                or digest(calibration.read_bytes()) != fixture['calibration_sha256']):
            raise ValueError(f'{model} source differs from pinned Phase 4 fixture')
        program = compile_static(onnx.load(source),
                                 json.loads(calibration.read_text()))
        results[model] = {}
        for policy, half, overlap in (('B1', False, False),
                                      ('B2', True, True)):
            plan, image = compile_tiled(program, prefer_half=half)
            commands, payload, schedule = compile_sequence(
                plan, image, overlap=overlap, snapshots=False)
            stem = f'{model}-{policy.lower()}'
            (output / f'{stem}.commands.bin').write_bytes(commands)
            (output / f'{stem}.payload.bin').write_bytes(payload)
            (output / f'{stem}.schedule.json').write_text(
                json.dumps(schedule, indent=2, sort_keys=True) + '\n')
            transfers = [transfer for tile in schedule['tiles']
                         for transfer in tile['transfers']]
            results[model][policy] = {
                'source_onnx_sha256': fixture['source_onnx_sha256'],
                'calibration_sha256': fixture['calibration_sha256'],
                'policy': ('largest-fitting layer-wise tiles, serialized DMA'
                           if policy == 'B1' else
                           'prefer 16-KiB alternate-bank regions with 32-KiB fallback and legal DMA prefetch'),
                'half_scratchpad_preference': half,
                'overlap': overlap,
                'commands_sha256': digest(commands),
                'payload_sha256': digest(payload),
                'commands_bytes': len(commands),
                'payload_bytes': len(payload),
                'command_count': schedule['command_count'],
                'tile_count': len(schedule['tiles']),
                'max_live_scratch_bytes': max(
                    tile['scratch_bytes'] for tile in schedule['tiles']),
                'dma_payload_bytes': sum(t['bytes'] for t in transfers),
                'prefetch_payload_bytes': schedule['prefetch_payload_bytes'],
                'final_output': schedule['final_output'],
            }
    manifest = {
        'schema': 1,
        'scope': 'preliminary B1/B2 compilation; no baseline latency or B3 claim',
        'models': results,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2,
                                                     sort_keys=True) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'work/phase5/baseline-candidates')
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2, sort_keys=True))
