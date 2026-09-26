#!/usr/bin/env python3
"""Publish bounded boardless evidence and a pending physical experiment matrix."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(folder, destination, frozen):
    report = json.loads((folder/'report.json').read_text())
    rtl = json.loads((ROOT/'work/phase6/rtl-candidates.json').read_text())
    tests_path = ROOT/'work/phase6/tests.xml'
    cases = ET.parse(tests_path).findall('.//testcase')
    if report['status'] != 'passed' or rtl['status'] != 'passed' or not cases or any(c.find('failure') is not None or c.find('error') is not None for c in cases):
        raise ValueError('incomplete/failing evidence')
    for record in (report, rtl):
        for name, expected in record['source_sha256'].items():
            if sha(ROOT/name) != expected:
                raise ValueError(f'stale evidence source: {name}')
    unchanged = json.loads(frozen.read_text())
    for name, expected in unchanged.items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'active Phase 5 source changed: {name}')
    summary = {'status': 'passed', 'phase6_gate_complete': False, 'physical_improvement_demonstrated': False,
               'host': report['host'], 'tests_passed': len(cases),
               'tests_junit_sha256': sha(tests_path), 'boardless_report_sha256': sha(folder/'report.json'),
               'rtl_report_sha256': sha(ROOT/'work/phase6/rtl-candidates.json'),
               'frozen_phase5_files_verified': len(unchanged), 'frozen_manifest_sha256': sha(frozen),
               'scope': report['scope'], 'models': {}}
    matrix = {'schema': 1, 'status': 'prepared_not_run', 'requires_board': True,
              'run_after': 'Phase 5 campaigns release exclusive UART/JTAG ownership',
              'release_path': 'hardware/releases/phase5/dual_resident_750k.fs',
              'release_sha256': sha(ROOT/'hardware/releases/phase5/dual_resident_750k.fs'),
              'nominal_core_hz': 20250000, 'command_entry': 0,
              'scope': 'single-model current-ABI baseline tuning and overlap ablation, not complete B4/B3 comparison',
              'controls': ['same pinned quantized model and inputs', 'same bitstream/core clock and SDRAM controller',
                           'report accelerator cycles separately from UART transfer time', 'randomize policy order with seed 6107',
                           'minimum 5 independent sessions, 30 repeated jobs per policy/model/session',
                           'check every output against integer oracle; report engine/DMA/overlap counters',
                           'publish paired differences and uncertainty, including negative results'],
              'remaining_research_controls': ['fair B3 adaptation', 'fused/recompute hardware lowering and calibrated costs',
                                               'matched placement/fusion/bank-awareness ablations'], 'models': {}}
    for model, row in report['models'].items():
        variants = {}
        matrix['models'][model] = {}
        for policy, candidate in row['candidates'].items():
            for artifact in candidate['artifacts'].values():
                if sha(folder/artifact['file']) != artifact['sha256']:
                    raise ValueError('changed candidate artifact')
            components = candidate['components']
            variants[policy] = {'component_cycles': (components['engine_cycles']+components['dma_fit_cycles']) if components else None,
                                'components': components, 'prefetch_bytes': candidate['prefetch_bytes'],
                                'verification_cases': len(candidate['verification']), 'physical_status': 'not_run'}
            matrix['models'][model][policy] = {'artifacts': candidate['artifacts'], 'status': 'not_run',
                                              'source_sha256': row['source_sha256']}
        spatial = []
        for tile in ([4, 4], [8, 8]):
            for policy in ('recompute', 'retain-last-region'):
                entries = [r for r in row['spatial'] if r['tile'] == tile and r['cache_policy'] == policy and r['sample'] == 'pinned-fixture']
                macs = sum(v['macs'] for e in entries for v in e['layers'].values())
                spatial.append({'tile': tile, 'policy': policy, 'computed_macs': macs,
                                'mac_recomputation_factor': macs/row['useful_conv_macs_in_spatial_scope'],
                                'source_requested_bytes': sum(e['source_requested_bytes'] for e in entries),
                                'peak_retained_cache_bytes': max(e['peak_retained_cache_bytes'] for e in entries),
                                'physical_sram_fit_proven': False})
        summary['models'][model] = {'variants': variants,
            'candidate_construction_and_search_seconds': row['candidate_construction_and_search_seconds'],
            'beam_gap_in_restricted_catalogue': row['optimization']['beam']['bound_gap_fraction'],
            'candidate_count': len(row['optimization']['choices']),
            'numerical_segment_cases': len(row['spatial']), 'spatial': spatial,
            'spatial_covered_layers': row['spatial_covered_layers'], 'spatial_barrier_layers': row['spatial_barrier_layers'],
            'useful_conv_macs_in_spatial_scope': row['useful_conv_macs_in_spatial_scope'],
            'negative_result': 'restricted mixed search equals full32-serial; no new predicted gain in this catalogue'}
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in (('summary.json', summary), ('physical-matrix.json', matrix)):
        (destination/name).write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')
    for path, name in ((folder/'report.json', 'boardless-report.json.gz'),
                       (ROOT/'work/phase6/rtl-candidates.json', 'rtl-candidates.json.gz'),
                       (tests_path, 'tests.xml.gz'), (frozen, 'phase5-frozen-files.json.gz')):
        (destination/name).write_bytes(gzip.compress(path.read_bytes(), mtime=0))
    print(json.dumps({'tests': len(cases), 'models': list(summary['models']), 'gate_complete': False}, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=ROOT/'work/phase6/boardless')
    parser.add_argument('--destination', type=Path, default=ROOT/'docs/research/evidence/phase6')
    parser.add_argument('--frozen', type=Path, default=ROOT/'work/phase6/phase5-frozen-files.json')
    args = parser.parse_args()
    summarize(args.folder, args.destination, args.frozen)
