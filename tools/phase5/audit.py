#!/usr/bin/env python3
"""Fail-closed Phase 5 evidence inventory; no synthesis estimate is a board run."""

import argparse
import hashlib
import json
from pathlib import Path

from schedule import ROOT, create_plan


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root=ROOT):
    phase4 = json.loads((root / 'docs/research/evidence/phase4/summary.json').read_text())
    inventory = json.loads((root / 'docs/research/evidence/phase4/inventory-audit.json').read_text())
    release = root / 'hardware/releases/phase4/tinyml_v4_kernels.fs'
    if sha256(release) != phase4['bitstream_sha256']:
        raise ValueError('Phase 4 routed bitstream hash mismatch')
    for name in ('kws', 'vww'):
        path = root / f'benchmarks/manifests/{name}.canonical-inventory.json'
        expected = phase4['pinned_inventory_sha256'][name]
        if sha256(path) != expected or inventory[name]['inventory_sha256'] != expected:
            raise ValueError(f'{name} inventory hash mismatch')

    plan = create_plan(root / 'benchmarks/manifests')
    quality = {}
    for name in ('kws', 'vww'):
        result = json.loads((root / f'benchmarks/manifests/{name}.static-quality.json').read_text())
        if (result['count'] != plan['accuracy_split_counts'][name]
                or result['split_sha256'] != plan['accuracy_split_npz_sha256'][name]
                or result['target'] != 'software-v2'
                or not result['calibration_disjoint']
                or abs(result['accuracy'] - result['correct'] / result['count']) > 1e-12
                or result['correct'] > result['count']):
            raise ValueError(f'{name} software quality does not match pinned split')
        quality[name] = {key: result[key] for key in ('count', 'correct', 'accuracy', 'target')}

    for name in ('kws', 'vww'):
        payload = root / f'work/quality/{name}.accuracy.npz'
        if payload.is_file() and sha256(payload) != plan['accuracy_split_npz_sha256'][name]:
            raise ValueError(f'{name} accuracy payload hash mismatch')

    # The original Phase 4 snapshot remains historical. Later board bring-up
    # uses a separately frozen reset-corrected release, without upgrading the
    # complete-primary-model or SDRAM gates.
    physical_path = root / 'docs/research/evidence/physical/summary.json'
    physical = None
    if physical_path.is_file():
        physical = json.loads(physical_path.read_text())
        bitstream = (root / physical['bitstream_path']).resolve()
        if (not bitstream.is_relative_to(root.resolve()) or
                sha256(bitstream) != physical['bitstream_sha256'] or
                physical['status'] != 'passed' or
                not physical['physical_board_programmed']):
            raise ValueError('physical release evidence mismatch')
        for name, expected in physical['artifacts'].items():
            path = (physical_path.parent / name).resolve()
            if (not path.is_relative_to(physical_path.parent.resolve()) or
                    sha256(path) != expected['sha256']):
                raise ValueError(f'physical artifact hash mismatch: {name}')

    checks = {
        'integrated_sdram_controller': bool(phase4['sdram_controller_integrated']),
        'integrated_dma': bool(phase4['dma_integrated']),
        'complete_primary_model_rtl': bool(phase4['complete_kws_vww_rtl']),
        'physical_board_programmed': physical is not None,
        'kws_image_available': (root / 'work/quality/kws.uq2').is_file(),
        'vww_image_available': (root / 'work/quality/vww.uq2').is_file(),
        'kws_accuracy_payload_available': (root / 'work/quality/kws.accuracy.npz').is_file(),
        'vww_accuracy_payload_available': (root / 'work/quality/vww.accuracy.npz').is_file(),
        'primary_board_run_report_available': (root / 'work/phase5/board-run.json').is_file(),
        'ad_full_set_report_available': (root / 'work/phase5/ad-full-set.json').is_file(),
        'matched_baseline_report_available': (root / 'work/phase5/baselines.json').is_file(),
    }
    # File presence alone never certifies any result. Even a fully green
    # inventory needs independent output, provenance and measurement review.
    return {
        'schema': 1,
        'scope': 'readiness inventory; not a G5 certification',
        'phase4_release_sha256': phase4['bitstream_sha256'],
        'physical_onchip_release_sha256': physical['bitstream_sha256'] if physical else None,
        'physical_scope': physical['scope'] if physical else None,
        'schedule_sha256': plan['jsonl_sha256'],
        'schedule_jobs': plan['jobs'],
        'software_quality_only': quality,
        'storage_lower_bounds_bytes': {
            name: inventory[name]['all_persistent_weight_parameter_bytes']
            for name in ('kws', 'vww')
        },
        'scratchpad_bytes': 32768,
        'checks': checks,
        'missing': [name for name, present in checks.items() if not present],
        'g5_certified': False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, help='write a compact pinned summary')
    args = parser.parse_args()
    result = audit()
    rendered = json.dumps(result, indent=2) + '\n'
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(rendered)
    print(rendered, end='')
