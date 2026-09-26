#!/usr/bin/env python3
"""Fail-closed Phase 5 evidence inventory; no synthesis estimate is a board run."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from schedule import ROOT, build_jobs, create_plan, load_split


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_dual_release(root):
    path = root / 'docs/research/evidence/phase5/dual-resident-route.json'
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    bitstream = root / report['bitstream_path']
    if (report['status'] != 'passed-smoke'
            or sha256(bitstream) != report['bitstream_sha256']
            or report['setup_tns_ns'] != 0
            or report['routed_core_fmax_mhz'] <= report['core_clock_mhz']):
        raise ValueError('dual-resident release hash or timing mismatch')
    parent = path.parent
    for entry in report['reports'].values():
        archive = parent / entry['archive']
        if (sha256(archive) != entry['archive_sha256']
                or hashlib.sha256(gzip.decompress(archive.read_bytes())).hexdigest()
                != entry['raw_sha256']):
            raise ValueError('dual-resident routed report hash mismatch')
    program = parent / report['program_log_archive']
    if (sha256(program) != report['program_log_archive_sha256']
            or hashlib.sha256(gzip.decompress(program.read_bytes())).hexdigest()
            != report['program_log_sha256']):
        raise ValueError('dual-resident programming log hash mismatch')
    for name, expected in report['physical_smoke_artifacts'].items():
        if sha256(parent / name) != expected:
            raise ValueError('dual-resident physical smoke hash mismatch')
    smoke = json.loads((parent / 'dual-resident-smoke.json').read_text())
    if (smoke['status'] != 'passed-probe' or smoke['jobs_completed'] != 4
            or smoke['output_mismatches'] or smoke['reflash_events']
            or smoke['bitstream_sha256'] != report['bitstream_sha256']
            or sha256(parent / 'dual-resident-smoke.jsonl') != smoke['records_sha256']):
        raise ValueError('dual-resident physical smoke content mismatch')
    return report


def inspect_full_campaign(root, name, release, plan):
    filename = ('dual-switch-full-v2.json' if name == 'switch'
                else f'dual-{name}-full.json')
    path = root / 'work/phase5' / filename
    if not path.is_file() or release is None:
        return False
    report = json.loads(path.read_text())
    target = {'switch': 10000, 'kws-accuracy': 4890,
              'vww-accuracy': 10961}[name]
    if (report.get('status') != 'passed'
            or not report.get('full_campaign')
            or report.get('jobs_completed') != target
            or report.get('jobs_planned') != target
            or report.get('output_mismatches') != 0
            or report.get('reflash_events') != 0
            or report.get('bitstream_sha256') != release['bitstream_sha256']
            or report.get('split_sha256') != plan['accuracy_split_npz_sha256']):
        return False
    if name == 'switch' and report.get('job_plan_sha256') != plan['jsonl_sha256']:
        return False
    records = path.with_suffix('.jsonl')
    if not records.is_file() or sha256(records) != report.get('records_sha256'):
        return False
    manifests = {model: load_split(root / f'benchmarks/manifests/{model}.data.json')[0]
                 for model in ('kws', 'vww')}
    expected_jobs = (list(build_jobs(manifests)) if name == 'switch'
                     else [{'job': index, 'workload': name.split('-')[0],
                            'sample_index': index}
                           for index in range(target)])
    observed = {model: {'jobs': 0, 'correct': 0} for model in manifests}
    with records.open() as stream:
        rows = (json.loads(line) for line in stream)
        for expected in expected_jobs:
            try:
                row = next(rows)
            except StopIteration:
                return False
            model = expected['workload']
            sample = manifests[model]['splits']['accuracy']['records'][
                expected['sample_index']]
            logits = [byte if byte < 128 else byte - 256
                      for byte in bytes.fromhex(row['output_hex'])]
            if (row['job'] != expected['job'] or row['elapsed_cycles'] <= 0
                    or row['wall_seconds'] <= 0
                    or row['workload'] != model
                    or row['sample_index'] != expected['sample_index']
                    or row['sample_id'] != sample['id']
                    or row['label'] != sample['label']
                    or not logits
                    or row['prediction'] != logits.index(max(logits))):
                return False
            observed[model]['jobs'] += 1
            observed[model]['correct'] += row['prediction'] == row['label']
        if next(rows, None) is not None:
            return False
    for model, counts in observed.items():
        metrics = report['metrics'][model]
        if metrics['jobs'] != counts['jobs'] or metrics['correct'] != counts['correct']:
            return False
    return True


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

    dual_release = inspect_dual_release(root)

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

    dual_smoke = (json.loads((root / 'docs/research/evidence/phase5/dual-resident-smoke.json').read_text())
                  if dual_release else None)
    loaded_models = {event['workload'] for event in dual_smoke['model_load_events']} \
        if dual_smoke else set()
    checks = {
        'integrated_sdram_controller': bool(phase4['sdram_controller_integrated'] or dual_release),
        'integrated_dma': bool(phase4['dma_integrated'] or dual_release),
        'complete_primary_model_rtl': bool(phase4['complete_kws_vww_rtl'] or dual_release),
        'physical_board_programmed': physical is not None or dual_release is not None,
        'kws_image_available': 'kws' in loaded_models or (root / 'work/quality/kws.uq2').is_file(),
        'vww_image_available': 'vww' in loaded_models or (root / 'work/quality/vww.uq2').is_file(),
        'kws_accuracy_payload_available': (root / 'work/quality/kws.accuracy.npz').is_file(),
        'vww_accuracy_payload_available': (root / 'work/quality/vww.accuracy.npz').is_file(),
        'primary_board_run_report_available': bool(dual_release) or (root / 'work/phase5/board-run.json').is_file(),
        'ad_full_set_report_available': (root / 'work/phase5/ad-full-set.json').is_file(),
        'matched_baseline_report_available': (root / 'work/phase5/baselines.json').is_file(),
        'dual_resident_release_smoke': dual_release is not None,
        'kws_complete_board_accuracy': inspect_full_campaign(root, 'kws-accuracy', dual_release, plan),
        'vww_complete_board_accuracy': inspect_full_campaign(root, 'vww-accuracy', dual_release, plan),
        'balanced_10000_board_jobs': inspect_full_campaign(root, 'switch', dual_release, plan),
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
        'dual_resident_release_sha256': dual_release['bitstream_sha256'] if dual_release else None,
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
