#!/usr/bin/env python3
"""Check whether each pinned operator can fit a simple 16-KiB SRAM bank.

This is a geometry feasibility audit, not a double-buffer schedule or a speed
claim. Each operator is planned in isolation under half the physical scratchpad
budget. A failure means ordinary output-channel tiling cannot fit its input.
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'compiler'))
import phase4_tiling


def audit(name):
    path = ROOT/f'benchmarks/manifests/{name}.canonical-inventory.json'
    source = path.read_bytes()
    inventory = json.loads(source)
    records = []
    original = phase4_tiling.SRAM_BYTES
    try:
        # Isolate the modified budget to this diagnostic process. The normal
        # 32-KiB planner and all frozen plans remain byte-for-byte unchanged.
        phase4_tiling.SRAM_BYTES = 16 * 1024
        for index, node in enumerate(inventory['operators']):
            names = set(node['inputs'] + node['outputs'])
            subgraph = {'tensors': [t for t in inventory['tensors']
                                    if t['name'] in names], 'operators': [node]}
            try:
                plan = phase4_tiling.plan_graph(subgraph, 'single-node')
                records.append({'index': index, 'op': node['op'],
                                'fits_16k_bank': True,
                                'tiles': len(plan['layers'][0]['tiles'])})
            except ValueError as error:
                records.append({'index': index, 'op': node['op'],
                                'fits_16k_bank': False,
                                'reason': str(error).replace(
                                    '32 KiB SRAM', '16 KiB SRAM audit limit')})
    finally:
        phase4_tiling.SRAM_BYTES = original
    return {'workload': name, 'status': 'geometry-only-feasibility-audit',
            'inventory_sha256': hashlib.sha256(source).hexdigest(),
            'bank_bytes': 16 * 1024, 'sram_bytes': 32 * 1024,
            'eligible_nodes': sum(r['fits_16k_bank'] for r in records),
            'total_nodes': len(records), 'nodes': records,
            'interpretation': ('A bank fit does not imply safe overlap; '
                               'activation dependencies, liveness, descriptor '
                               'staging and SRAM-port contention remain.')}


def main():
    output = ROOT/'docs/research/evidence/phase4/boardless-pingpong-feasibility.json'
    results = {name: audit(name) for name in ('kws', 'vww')}
    output.write_text(json.dumps(results, indent=2, sort_keys=True)+'\n')
    print(json.dumps({name: {'eligible': value['eligible_nodes'],
                             'total': value['total_nodes'],
                             'ineligible': [r['index'] for r in value['nodes']
                                            if not r['fits_16k_bank']]}
                      for name, value in results.items()}))


if __name__ == '__main__':
    main()
