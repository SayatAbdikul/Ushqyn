#!/usr/bin/env python3
"""Verify that Phase 4 physical reports describe one routed bitstream."""

import hashlib
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/'docs/research/evidence/phase4'
ROUTE = EVIDENCE/'physical-tiled-host-overlap-route.json'
BITSTREAM = ROOT/'hardware/releases/phase4-sdram/hs_tiled_host_overlap.fs'
PROGRAM_LOG = EVIDENCE/'physical-overlap-program.txt.gz'
SUITE_LOG = EVIDENCE/'physical-overlap-suite.txt.gz'
RECORDS = (
    'physical-host-overlap.json',
    'physical-host-overlap-kws.json',
    'physical-host-overlap-vww.json',
    'physical-overlap.json',
    'physical-dma-profile-overlap.json',
    'physical-kernel-profiles-overlap.json',
    'physical-legacy-overlap.json',
    'physical-synthetic-overlap.json',
    'physical-cost-model-overlap.json',
    'physical-static-cost-model.json',
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    route = json.loads(ROUTE.read_text())
    if route['bitstream_sha256'] != sha(BITSTREAM) or \
       route['setup_tns_ns'] != 0 or \
       route['routed_core_fmax_mhz'] <= route['core_clock_constraint_mhz']:
        raise ValueError('bitstream hash or timing mismatch')
    for name, expected in route['source_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'current routed source changed: {name}')
    if sha(ROOT/route['generated_ip_path']) != route['generated_ip_sha256']:
        raise ValueError('Gowin SDRAM IP hash mismatch')
    if b'after program sram' not in gzip.decompress(PROGRAM_LOG.read_bytes()):
        raise ValueError('missing successful programming transcript')
    if not gzip.decompress(SUITE_LOG.read_bytes()):
        raise ValueError('empty simulation suite transcript')
    summary = json.loads((EVIDENCE/'physical-host-overlap.json').read_text())
    raw_model_hashes = {m['workload']: m['raw_report_sha256']
                        for m in summary['models']}
    reports = {}
    for name in RECORDS:
        path = EVIDENCE/name
        report = json.loads(path.read_text())
        if name in ('physical-host-overlap-kws.json',
                    'physical-host-overlap-vww.json'):
            model = name.removeprefix('physical-host-overlap-').removesuffix('.json')
            if sha(path) != raw_model_hashes[model]:
                raise ValueError(f'primary model report hash mismatch: {name}')
        elif report['bitstream_sha256'] != route['bitstream_sha256']:
            raise ValueError(f'mixed-bitstream physical evidence: {name}')
        reports[name] = sha(path)
    if summary['status'] != 'passed' or \
       [m['node_count'] for m in summary['models']] != [22, 58]:
        raise ValueError('primary model node count mismatch')
    overlap = json.loads((EVIDENCE/'physical-overlap.json').read_text())
    if overlap['status'] != 'passed' or overlap['overlap_cycles'] <= 0 or \
       overlap['guard_error_for_live_region'] != 9 or \
       overlap['speedup_from_overlap'] <= 1:
        raise ValueError('directed overlap/guard evidence invalid')
    dma = json.loads((EVIDENCE/'physical-dma-profile-overlap.json').read_text())
    if dma['status'] != 'passed' or len(dma['records']) != 78:
        raise ValueError('physical DMA profile incomplete')
    synthetic = json.loads((EVIDENCE/'physical-synthetic-overlap.json').read_text())
    ad = next((case for case in synthetic['cases']
               if case['case'].startswith('toycar-')), None)
    if (synthetic['status'] != 'passed' or ad is None or ad['nodes'] != 19 or
        ad['source_inventory_sha256'] !=
            sha(ROOT/'benchmarks/manifests/ad.canonical-inventory.json')):
        raise ValueError('ToyCar-shape record incomplete')
    legacy = json.loads((EVIDENCE/'physical-legacy-overlap.json').read_text())
    if [(m['model'], m['nodes']) for m in legacy['models']] != \
       [('mlp', 5), ('smallcnn', 8)]:
        raise ValueError('MLP/SmallCNN SDRAM comparison incomplete')
    output = {
        'status': 'passed', 'scope': 'single routed physical Phase 4 image',
        'bitstream_sha256': route['bitstream_sha256'],
        'route_record_sha256': sha(ROUTE),
        'program_log_sha256': sha(PROGRAM_LOG),
        'suite_log_sha256': sha(SUITE_LOG),
        'reports_sha256': reports,
        'checks': ['routed timing and source/IP hashes',
                   'KWS/VWW node counts', 'guarded physical overlap',
                   '78 exact DMA transfers', 'MLP/SmallCNN SDRAM replay',
                   '19 frozen AD layer shapes with synthetic weights'],
        'limits': ['no full-model ping-pong schedule',
                   'no validated compiler-static cost predictor',
                   'no measured board energy'],
    }
    path = EVIDENCE/'physical-release-audit.json'
    path.write_text(json.dumps(output, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'status': 'passed', 'reports': len(reports)}))


if __name__ == '__main__':
    main()
