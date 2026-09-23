#!/usr/bin/env python3
"""Emit reproducible boardless transfer plans for pinned primary inventories."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'compiler'))
from phase4_tiling import plan_inventory


def canonical(data):
    return (json.dumps(data, indent=2, sort_keys=True)+'\n').encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='compare with checked-in evidence')
    args = parser.parse_args()
    outdir = ROOT/'docs/research/evidence/phase4'
    summary = {'schema': 1, 'evidence_type': 'geometry-only boardless schedule; not inference or SDRAM measurement',
               'models': {}}
    files = {}
    for model in ('kws', 'vww'):
        plan = plan_inventory(ROOT/f'benchmarks/manifests/{model}.canonical-inventory.json')
        raw = canonical(plan)
        files[outdir/f'{model}-tiling-plan.json'] = raw
        tiles = [tile for layer in plan['layers'] for tile in layer['tiles']]
        transfers = [transfer for tile in tiles for transfer in tile['transfers']]
        summary['models'][model] = {
            'inventory_sha256': plan['inventory_sha256'],
            'plan_sha256': hashlib.sha256(raw).hexdigest(),
            'nodes': len(plan['layers']), 'compute_tiles': len(tiles),
            'compute_nodes': sum(bool(layer['tiles']) for layer in plan['layers']),
            'activation_slot_bytes': plan['activation_slot_bytes'],
            'external_image_end': plan['parameter_end'],
            'peak_tile_sram_bytes': max(tile['scratch_bytes'] for tile in tiles),
            'dma_transactions': len(transfers),
            'dma_payload_bytes': sum(t['bytes'] for t in transfers),
            'dma_physical_read_bytes': sum(((t['bytes']+7)//8)*8 for t in transfers if t['direction']=='to_sram'),
            'dma_ideal_data_beats_lower_bound': sum((t['bytes']+7)//8 for t in transfers),
            'max_external_transfer_end': max(t['ext']+t['bytes'] for t in transfers),
            'per_node_tiles': [len(layer['tiles']) for layer in plan['layers']],
        }
    files[outdir/'boardless-tiling-summary.json'] = canonical(summary)
    for path, raw in files.items():
        if args.check:
            if not path.exists() or path.read_bytes() != raw:
                raise SystemExit(f'outdated Phase 4 evidence: {path}')
        else:
            path.write_bytes(raw)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
