#!/usr/bin/env python3
"""Audit and optionally archive the completed direct-writeback experiment."""
import argparse
import gzip
import json
from pathlib import Path

import run_priority as priority
import run_screening as screening
import screen_writeback
from run_writeback import BASE, ENGINE, ROOT
from variants import check_frozen


def audit():
    plan = screen_writeback.prepare()
    directory = BASE / 'physical'
    recorded_plan = json.loads((directory / 'plan.json').read_text())
    if plan != recorded_plan: raise ValueError('physical experiment identities changed')
    physical = json.loads((directory / 'report.json').read_text())
    if (physical['status'] != 'passed-screening' or physical['completed'] != 48
            or physical['output_mismatches']
            or physical['plan_sha256'] != screening.sha(directory / 'plan.json')
            or physical['records_sha256'] != screening.sha(directory / 'records.jsonl')):
        raise ValueError('physical campaign incomplete, rejected or changed')
    screening.verify_files(ROOT, plan['sources'])
    rows = priority.read_rows(directory / 'records.jsonl')
    for row in rows:
        if row['bitstream_sha256'] != plan['images'][row['variant']]['sha256']:
            raise ValueError('record image identity mismatch')
        fixture = screening.BASE / 'fixtures' / row['fixture']
        expected = []
        for line in (fixture / 'checks.txt').read_text().splitlines():
            address, name = line.split()
            path = fixture / name
            expected.append({'file': name, 'bytes': path.stat().st_size, 'sha256': screening.sha(path)})
        if row['tensor_checks'] != expected: raise ValueError('checked tensor identity mismatch')
        schedule = json.loads((fixture / 'schedule.json').read_text())
        if row['command_index'] != schedule['command_count']-1:
            raise ValueError('incomplete command sequence')
        e, c, d, o = [row[k] for k in ('elapsed_cycles', 'engine_cycles', 'dma_cycles', 'overlap_cycles')]
        if min(e, c, d) <= 0 or not 0 <= o <= min(c, d) or c+d-o > e:
            raise ValueError('invalid physical counters')
    summary = screen_writeback.summarize(rows)
    if summary != physical['summary']: raise ValueError('physical summary does not reproduce')
    screening.verify_files(ROOT / 'work/phase6/priority-v1', plan['preserved_accuracy'])
    native = json.loads((BASE / 'native/report.json').read_text())
    route = json.loads((BASE / 'route/report.json').read_text())
    return {'schema': 1, 'status': 'passed-writeback-screen', 'physical_board': True,
            'phase6_gate_complete': False, 'core_clock_hz': plan['core_clock_hz'],
            'frozen_phase5_files_verified': check_frozen(),
            'preserved_accuracy_samples': {m: len(priority.read_rows(
                ROOT / f'work/phase6/priority-v1/{m}-accuracy.jsonl')) for m in ('kws', 'vww')},
            'engine_regression_cases': 3, 'native_full_model_runs': len(native['results']),
            'native_tensor_checks': sum(r['tensor_checks'] for r in native['results']),
            'physical_runs': len(rows), 'physical_correctness_tensor_checks': summary['correctness_tensor_checks'],
            'route': route, 'physical_summary': summary,
            'image': plan['images']['writeback'], 'engine_sha256': screening.sha(ENGINE),
            'serial_cycle_savings': plan['serial_cycle_savings'],
            'scope': 'one paired physical timing session; independent selected-input correctness',
            'not_claimed': ['full accuracy on this new image', 'multi-session confirmation of this change',
                'long-duration stability on this image', 'higher operating clock', 'measured energy', 'SOTA', 'G6 closure']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', action='store_true')
    args = parser.parse_args(); report = audit()
    screening.save_json(BASE / 'summary.json', report)
    if args.archive:
        destination = ROOT / 'docs/research/evidence/phase6/writeback'
        destination.mkdir(parents=True, exist_ok=True)
        files = {'engine-report.json': BASE / 'engine/report.json',
                 'engine-results.xml': BASE / 'engine/results.xml',
                 'native-report.json': BASE / 'native/report.json',
                 'route-report.json': BASE / 'route/report.json',
                 'physical-plan.json': BASE / 'physical/plan.json',
                 'physical-report.json': BASE / 'physical/report.json',
                 'physical-records.jsonl': BASE / 'physical/records.jsonl',
                 'route.txt': BASE / 'route/phase6_tiled_host/impl/pnr/phase6_tiled_host.rpt.txt',
                 'timing.html': BASE / 'route/phase6_tiled_host/impl/pnr/phase6_tiled_host_tr_content.html'}
        archive = {}
        for name, source in files.items():
            path = destination / (name + '.gz')
            path.write_bytes(gzip.compress(source.read_bytes(), mtime=0))
            archive[path.name] = {'sha256': screening.sha(path), 'raw_sha256': screening.sha(source)}
        release = ROOT / 'hardware/releases/phase6/writeback_candidate_750k.fs.gz'
        release.write_bytes(gzip.compress((ROOT / report['image']['file']).read_bytes(), mtime=0))
        report['archives'] = archive
        report['compressed_candidate'] = {'file': str(release.relative_to(ROOT)), 'sha256': screening.sha(release)}
        screening.save_json(destination / 'summary.json', report)
    print(json.dumps(report, indent=2))
