#!/usr/bin/env python3
"""Independent checks for the bounded factor/QoS implementation decision round."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import re
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'work/phase6/decision-round-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def initialize():
    path = BASE / 'baseline.json'
    require(not path.exists(), 'refusing to overwrite start snapshot')
    previous = read(ROOT / 'work/phase6/three-hypothesis-campaign-v1/baseline.json')
    files = {name: sha(ROOT / name) for name in previous['files']}
    frozen = {}
    for model in ('kws', 'vww', 'ad'):
        folder = ROOT / f'work/phase6/representation-screen-v1/{model}'
        for name in ('selected-int8.uq2', 'selection-frozen-before-heldout.json', 'report.json'):
            item = folder / name
            frozen[str(item.relative_to(ROOT))] = sha(item)
    save(path, {'production_files': files, 'frozen_model_files': frozen,
                'git_status_at_start': subprocess.check_output(
                    ['git', 'status', '--short'], cwd=ROOT, text=True)})
    print(json.dumps({'status': 'initialized', 'production_files': len(files),
                      'frozen_model_files': len(frozen)}))


def factor_checks():
    folder = ROOT / 'work/phase6/representation-optimized-v1'
    report = read(folder / 'report.json')
    require(report['status'] == 'passed-matched-optimized-native-comparison',
            'optimized factor round incomplete')
    for field in ('source_sha256', 'model_source_sha256'):
        for path, digest in report[field].items():
            require(sha(path) == digest, 'factor source/input changed: ' + path)
    for item in report['source_bundle']:
        require(sha(ROOT / item['bundled_path']) == item['sha256'],
                'bundled compiler source differs')
    run_count = 0
    diagnostic_checks = {}
    for variant, row in report['variants'].items():
        require(len(row['candidates']) == report['search_budget']['configurations_per_graph'],
                'unequal search budgets')
        feasible = [c for c in row['candidates'] if c['status'] == 'passed-native']
        best = min(feasible, key=lambda c: (c['native']['elapsed_cycles'], c['index']))
        require(row['selected_index'] == best['index'], 'selection not best legal candidate')
        for candidate in feasible:
            fixture = ROOT / candidate['fixture']
            for name, digest in candidate['fixture_sha256'].items():
                require(sha(fixture / name) == digest, 'candidate fixture changed')
            require((fixture / 'commands.bin').stat().st_size <= 32768 and
                    candidate['peak_sram_address'] <= 32768,
                    'candidate memory capacity exceeded')
            raw_path = folder / 'candidates' / variant / f'c{candidate["index"]:02d}' / 'native-s0.json'
            require(sha(raw_path) == candidate['native']['report_sha256'], 'candidate raw report differs')
        require(len(row['validation']) == 8, 'selected input/RAM/diagnostic matrix incomplete')
        keys = set()
        for run in row['validation']:
            keys.add((run['sample'], run['mode'], run['seed']))
            fixture = ROOT / run['fixture']
            raw_path = folder / 'selected' / variant / run['sample'] / f'{run["mode"]}-s{run["seed"]}.json'
            require(sha(raw_path) == run['report_sha256'], 'selected raw report differs')
            raw = read(raw_path)
            require(raw['status'] == 'passed' and raw['elapsed_cycles'] == run['elapsed_cycles'],
                    'selected native counters differ')
            require(raw['tensor_checks'] == len((fixture / 'checks.txt').read_text().splitlines()),
                    'selected tensor checks incomplete')
            if run['mode'] == 'check':
                diagnostic = read(fixture / 'diagnostic.json')
                require(raw['tensor_checks'] == diagnostic['tensor_checks'], 'descriptor checks differ')
                for name, digest in diagnostic['files'].items():
                    require(sha(fixture / name) == digest, 'diagnostic fixture changed')
                diagnostic_checks[variant] = diagnostic['descriptor_output_checks']
            run_count += 1
        require(len(keys) == 8, 'duplicated validation labels')
        require(sha(folder / 'selected' / variant / 'pinned' / 'timed' / 'input.bin') !=
                sha(folder / 'selected' / variant / 'stress' / 'timed' / 'input.bin'),
                'stress input duplicates real input')
    for item in report['comparisons']:
        require(abs(item['elapsed_cycle_reduction_fraction'] -
                    (1 - item['factor_elapsed_cycles'] / item['dense_elapsed_cycles'])) < 1e-12,
                'factor cycle reduction arithmetic error')
        require(item['dense_elapsed_cycles'] == item['strongest_dense_reference_elapsed_cycles'],
                'comparison omits strongest dense control')
    require(report['peak_python_rss_bytes'] < 1 << 30, 'factor memory guard exceeded')
    return {'status': 'passed', 'selected_exact_native_runs': run_count,
            'descriptor_outputs_checked_per_diagnostic': diagnostic_checks,
            'comparisons': report['comparisons'], 'peak_python_rss_bytes': report['peak_python_rss_bytes']}


def schedule_checks(records_path, protocol, report=None, calibrated=None):
    cases = {row['name']: row for row in protocol['cases']}
    rows = read(records_path)
    groups = {}
    identities = set()
    for row in rows:
        key = (row['case'], row['seed'], row['word_cycles'], row['fixed_cycles'])
        require((*key, row['variant']) not in identities, 'duplicate schedule label')
        identities.add((*key, row['variant']))
        result = row['result']
        jobs = cases[row['case']]['releases']
        done = dict(result['completions'])
        require(len(done) == len(jobs) == result['jobs'], 'completion coverage')
        misses = sum(done[identifier] > deadline for model, release, deadline, identifier in jobs)
        lateness = max(done[identifier] - deadline for model, release, deadline, identifier in jobs)
        responses = {model: max(done[identifier] - release for m, release, deadline, identifier in jobs
                                if m == model) for model in (row['background'], row['urgent'])}
        require(misses == result['misses'] and lateness == result['max_lateness_cycles'] and
                responses == result['max_response_cycles'], 'raw completion tally differs')
        require(max(done.values()) == result['finish_cycles'], 'finish does not match completions')
        require(result['finish_cycles'] >= result['service_cycles'] + result['overhead_cycles'],
                'schedule work conservation failure')
        require(result['save_bytes'] == result['restore_bytes'], 'suspended context not restored')
        groups.setdefault(key, {})[row['variant']] = result
    comparisons = {}
    for variant in ('split25000', 'split50000', 'selective-long25000'):
        pairs = [(g['baseline'], g[variant]) for g in groups.values() if variant in g and 'baseline' in g]
        tally = {'cases': len(pairs),
                 'zero_miss_rescues': sum(a['misses'] > 0 and b['misses'] == 0 for a, b in pairs),
                 'cases_fewer_misses': sum(b['misses'] < a['misses'] for a, b in pairs),
                 'cases_more_misses': sum(b['misses'] > a['misses'] for a, b in pairs),
                 'total_misses_baseline': sum(a['misses'] for a, b in pairs),
                 'total_misses_variant': sum(b['misses'] for a, b in pairs)}
        if report:
            for name, value in tally.items():
                require(report['comparisons'][variant][name] == value, 'comparison tally differs')
        comparisons[variant] = tally
    if calibrated:
        for variant, expected in calibrated['summary'].items():
            rr = [g[variant] for g in groups.values()]
            actual = dict(cases=len(rr), total_misses=sum(r['misses'] for r in rr),
                zero_miss_cases=sum(r['misses'] == 0 for r in rr),
                max_metadata_scan_cycles=max(r['metadata_scan_cycles'] for r in rr),
                total_metadata_scan_cycles=sum(r['metadata_scan_cycles'] for r in rr),
                mean_overhead_cycles=sum(r['overhead_cycles'] for r in rr) / len(rr))
            require(actual == expected, 'calibrated summary differs')
        for control, expected in calibrated['pairwise'].items():
            pairs = [(g[control], g['selective-long25000']) for g in groups.values()]
            actual = dict(cases=len(pairs),
                selective_fewer_misses=sum(b['misses'] < a['misses'] for a, b in pairs),
                selective_more_misses=sum(b['misses'] > a['misses'] for a, b in pairs),
                same_misses=sum(b['misses'] == a['misses'] for a, b in pairs),
                selective_zero_miss_rescues=sum(a['misses'] > 0 and b['misses'] == 0 for a, b in pairs),
                control_zero_miss_rescues=sum(b['misses'] > 0 and a['misses'] == 0 for a, b in pairs))
            require(actual == expected, 'calibrated pairwise tally differs')
    return {'status': 'passed', 'raw_schedule_records_checked': len(rows),
            'comparisons': comparisons, 'raw_records_sha256': sha(records_path)}


def policy_checks():
    folder = ROOT / 'work/phase6/deadline-policy-v1'
    protocol = read(folder / 'protocol.json')
    report = read(folder / 'report.json')
    selection = read(folder / 'selection-identity.json')
    screen = ROOT / 'work/phase6/deadline-screen-v1'
    for name, digest in protocol['source_sha256'].items():
        require(sha(screen / name) == digest, 'original trace identity changed')
    for key, filename in (('protocol_sha256', 'protocol.json'), ('profiles_sha256', 'profiles.json'),
                          ('plans_sha256', 'plans.json'), ('fixtures_sha256', 'fixtures.json')):
        require(sha(folder / filename) == selection[key], 'frozen policy selection changed')
    profiles = read(folder / 'profiles.json')
    fixtures = read(folder / 'fixtures.json')
    native = read(folder / 'native-evidence.json')
    for path, digest in native['identity']['sources_sha256'].items():
        require(sha(path) == digest, 'policy native source changed')
    for row in native['runs']:
        fixture = Path(fixtures[row['model']]['selective-long25000'])
        require(sha(fixture / 'commands.bin') == row['commands_sha256'], 'selective command identity changed')
        if row.get('status') == 'reused-identical-baseline':
            require((fixture / 'commands.bin').read_bytes() ==
                    (Path(fixtures[row['model']]['baseline']) / 'commands.bin').read_bytes(),
                    'AD reuse not identical')
        else:
            raw = read(folder / 'native' / f'{row["model"]}-selective-long25000-s{row["seed"]}.json')
            require(raw == row['result'] and raw['status'] == 'passed' and raw['tensor_checks'] == 1,
                    'selective standalone native validation differs')
            require(sha(fixture / 'payload.bin') == row['payload_sha256'], 'selective payload changed')
            require(profiles[row['model']]['selective-long25000'][str(row['seed'])]['native_cycles'] ==
                    raw['elapsed_cycles'], 'selective profile timing differs from native')
    for variants in profiles.values():
        for seeds in variants.values():
            for profile in seeds.values():
                require(sum(row['cycles'] for row in profile['segments']) == profile['native_cycles'],
                        'native profile conservation failure')
    resident_records = 0
    for resident in report['resident_fixtures']:
        fixture = Path(resident['path'])
        manifest = read(fixture / 'policy-manifest.json')
        code = (fixture / 'commands.bin').read_bytes()
        offset = manifest['metadata_offset_bytes']
        require(len(code) == manifest['resident_program_bytes'] <= 32768 and offset % 16 == 0,
                'resident capacity/metadata alignment failure')
        header = struct.unpack_from('<IIII', code, offset)
        require(header == (manifest['urgent_entry'], len(manifest['checkpoints']), 0, 0),
                'serialized resident header differs')
        require(code[offset:] == (fixture / 'metadata.bin').read_bytes(), 'metadata bytes differ')
        for i, item in enumerate(manifest['checkpoints']):
            require(struct.unpack_from('<IIII', code, offset + (i + 1) * 16) ==
                    (item['global_next_index'], item['save_entry'], item['restore_entry'], 0),
                    'serialized resident site differs')
            require(sum(length for base, length in item['spans']) == item['live_dma_bytes'],
                    'checkpoint byte count differs')
            for entry, direction in ((item['save_entry'], 0), (item['restore_entry'], 1)):
                pc = entry
                for base, length in item['spans']:
                    require(base % 8 == length % 8 == 0 and 0 <= base < base + length <= 32768,
                            'invalid live granule range')
                    require(struct.unpack_from('<BBHIII', code, pc * 16) ==
                            (1, direction, 0, manifest['context_external_base'] + base, base, length),
                            'checkpoint DMA bytes differ')
                    require(struct.unpack_from('<BBHIII', code, (pc + 1) * 16) == (3, 2, 0, 0, 0, 0),
                            'checkpoint missing DMA drain')
                    pc += 2
                require(struct.unpack_from('<BBHIII', code, pc * 16) == (0, 0, 0, 0, 0, 0),
                        'checkpoint missing HALT')
            resident_records += 1
        for name, digest in manifest['fixture_sha256'].items():
            require(sha(fixture / name) == digest, 'resident fixture changed')
    result = schedule_checks(folder / 'results.json', protocol, report)
    result['resident_metadata_records_checked'] = resident_records
    result['frozen_arrival_cases'] = len(protocol['cases'])
    require(report['python_max_rss_bytes'] < 1 << 30, 'policy memory guard exceeded')
    calibrated = read(folder / 'scan-calibrated-report.json')
    for path, digest in calibrated['source_sha256'].items():
        require(sha(path) == digest, 'scan timing source changed')
    require(calibrated['raw_records_sha256'] == sha(folder / 'scan-calibrated-results.json'),
            'calibrated raw records changed')
    result['scan_calibrated'] = schedule_checks(folder / 'scan-calibrated-results.json', protocol,
                                                calibrated=calibrated)
    witnesses = calibrated['native_scan_corroboration']
    require(len(witnesses) == 8, 'native scan witness coverage incomplete')
    for row in witnesses:
        require(sha(row['preserved_native_report']) == row['sha256'] == sha(row['native_report_origin']),
                'native scan witness changed')
        raw = read(row['preserved_native_report'])
        require(raw['metadata_scan_cycles'] == row['native_scan_cycles'] == row['modeled_scan_cycles'],
                'native metadata scan mismatch')
    result['native_scan_witnesses_checked'] = len(witnesses)
    require(calibrated['python_max_rss_bytes'] < 1 << 30, 'calibrated memory guard exceeded')
    return result


def resident_checks():
    folder = ROOT / 'work/phase6/deadline-resident-v1'
    report = read(folder / 'report.json')
    require(report['status'] == 'complete-native-pass-physical-fail', 'resident round unfinished')
    identity = read(folder / 'native-build/identity.json')
    require(sha(folder / 'native-build/identity.json') == report['native_identity_sha256'], 'native identity changed')
    for path, digest in identity['source_sha256'].items():
        require(sha(path) == digest, 'resident native source changed')
    require(sha(folder / 'native.cpp') == identity['harness_sha256'], 'resident harness changed')
    require(sha(folder / 'native-build/Vv2_tiled_host_bridge') == identity['executable_sha256'],
            'resident executable changed')
    for item in report['source_bundle']:
        require(sha(item['original_path']) == item['sha256'] == sha(ROOT / item['bundled_path']),
                'resident bundled source changed')
    for variant, files in report['fixture_sha256'].items():
        for name, digest in files.items():
            require(sha(folder / 'fixtures' / variant / name) == digest, 'resident fixture changed')
    keys = set()
    negative = 0
    for row in report['native_runs']:
        key = (row['variant'], row['seed'], row['mode'])
        require(key not in keys, 'duplicate resident native case')
        keys.add(key)
        require(sha(ROOT / row['report_path']) == row['report_sha256'], 'resident raw case changed')
        raw = read(ROOT / row['report_path'])
        require(all(row[k] == v for k, v in raw.items()), 'resident raw counters differ')
        require(raw['status'] == 'passed' and raw['busy_guard_checks'] == 4 and
                raw['post_request_uart_rx_cycles'] == 0, 'busy/host independence failure')
        require(raw['program_bytes'] <= 32768, 'resident code overflow')
        if raw['mode'] == 2:
            require(raw['sequence_error'] or raw['output_byte_mismatches'] or raw['restored_byte_mismatches'],
                    'omitted restore control ineffective')
            negative += 1
        else:
            require(raw['sequence_error'] == raw['output_byte_mismatches'] == raw['restored_byte_mismatches'] == 0,
                    'positive resident exactness failed')
            require(raw['tensor_checks'] == (1 if raw['mode'] == 3 else 2), 'resident output coverage')
        if raw['mode'] != 3:
            require(raw['response_cycles_from_release'] == raw['urgent_done_cycle'] - raw['release_cycle'] and
                    raw['urgent_deadline_missed'] == (raw['response_cycles_from_release'] > raw['urgent_relative_deadline']),
                    'resident response arithmetic differs')
        if raw['mode'] in (1, 2):
            require(raw['clobber_checked'], 'full context clobber missing')
    require(len(keys) == 32 and negative == 8, 'resident native coverage incomplete')
    keys = set()
    late = 0
    for row in report['arrival_runs']:
        key = (row['variant'], row['seed'], row['label'])
        require(key not in keys, 'duplicate arrival case')
        keys.add(key)
        require(sha(ROOT / row['report_path']) == row['report_sha256'], 'arrival raw case changed')
        raw = read(ROOT / row['report_path'])
        require(all(row[k] == v for k, v in raw.items()), 'arrival raw counters differ')
        require(raw['status'] == 'passed' and raw['tensor_checks'] == 2 and
                raw['sequence_error'] == raw['output_byte_mismatches'] == raw['restored_byte_mismatches'] == 0 and
                raw['post_request_uart_rx_cycles'] == 0 and raw['busy_guard_checks'] == 4, 'arrival exactness failure')
        baseline = folder / 'native' / f'{row["variant"]}-s{row["seed"]}-m3.json'
        require(sha(baseline) == row['uninterrupted_baseline_sha256'], 'arrival completion reference changed')
        if row['label'] in ('after-last-site', 'completion-race'):
            require(raw['completion_fallback'], 'late request not completed through fallback')
            late += 1
        if row['label'] == 'completion-race':
            require(row['release_elapsed_target'] == read(baseline)['sequence_elapsed_cycles'] - 20,
                    'completion race uses unmatched trace')
    require(len(keys) == 32 and late == 16, 'arrival coverage incomplete')
    for mode, row in report['physical'].items():
        require(sha(ROOT / row['report_path']) == row['report_sha256'], 'physical raw report changed')
        for path, digest in row['source_sha256'].items():
            require(sha(path) == digest, 'physical source changed')
        base = folder / 'physical' / mode
        require(sha(base / 'build27.tcl') == row['recipe_sha256'] and
                sha(base / 'route.log') == row['log_sha256'], 'physical recipe/log changed')
        raw_report = base / 'phase6_uart_burst/impl/pnr/phase6_uart_burst.rpt.txt'
        require(sha(raw_report) == row['routed_report_sha256'], 'physical resource report changed')
        for name, resources in row['resources'].items():
            text = raw_report.read_text()
            match = re.search(rf'^\s*{name}\s*\|\s*([\d.]+)/([\d.]+)', text, re.M)
            require(match and float(match[1]) == resources['used'] and float(match[2]) == resources['available'],
                    'physical resource summary differs')
        require(row['peak_child_rss_bytes'] < 1 << 30, 'physical memory guard exceeded')
    baseline, candidate = report['physical']['baseline'], report['physical']['candidate']
    timing = (folder / 'physical/baseline/phase6_uart_burst/impl/pnr/phase6_uart_burst_tr_content.html').read_text()
    fmax = re.search(r'<td>27\.000\(MHz\)</td>\s*<td[^>]*>([\d.]+)\(MHz\)</td>', timing)
    require(fmax and float(fmax[1]) == baseline['routed_core_fmax_mhz'], 'baseline timing summary differs')
    for kind in ('Setup', 'Hold'):
        match = re.search(rf'Numbers of {kind} Violated Endpoints</td>\s*<td[^>]*>(\d+)</td>', timing)
        require(match and int(match[1]) == baseline[kind.lower() + '_violations'] == 0,
                'baseline timing violations differ')
    require(baseline['status'] == 'passed-route' and baseline['routed_core_fmax_mhz'] >= 27 and
            baseline['setup_violations'] == baseline['hold_violations'] == 0, 'matched physical baseline failed')
    require(candidate['status'] == 'failed-physical-gate' and candidate['returncode'] != 0 and
            candidate['routed_core_fmax_mhz'] is None and
            "48 LUT(s)(equivalent, include LUT/MUX/ALU) unPlaced" in
            (folder / 'physical/candidate/route.log').read_text(), 'candidate placement failure not preserved')
    return dict(status='passed-evidence-audit', original_native_checks=32, additional_arrival_checks=32,
                effective_negative_controls=negative, late_fallback_checks=late,
                physical_gate='failed-placement', matched_baseline_fmax_mhz=baseline['routed_core_fmax_mhz'])


def audit():
    before = read(BASE / 'baseline.json')
    for group in ('production_files', 'frozen_model_files'):
        changed = [name for name, digest in before[group].items()
                   if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
        require(not changed, group + ' changed: ' + str(changed))
    result = {'status': 'passed-evidence-audit', 'driver_sha256': sha(Path(__file__)),
              'production_files_unchanged': len(before['production_files']),
              'frozen_model_files_unchanged': len(before['frozen_model_files'])}
    result['factor'] = factor_checks()
    result['policy'] = policy_checks()
    result['resident'] = resident_checks()
    regeneration = read(BASE / 'bundled-compiler-regeneration.json')
    require(regeneration['status'] == 'passed-bundled-compiler-regeneration', 'bundled compiler probe failed')
    factor = read(ROOT / 'work/phase6/representation-optimized-v1/report.json')
    for variant, hashes in regeneration['variants'].items():
        row = factor['variants'][variant]
        winner = next(c for c in row['candidates'] if c['index'] == row['selected_index'])
        for name in ('commands', 'payload'):
            require(sha(ROOT / winner['fixture'] / (name + '.bin')) == hashes[name + '_sha256'],
                    'bundled compiler regeneration no longer matches winner')
    result['bundled_compiler_regeneration'] = regeneration
    result['peak_audit_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == 'darwin' else 1024)
    require(result['peak_audit_rss_bytes'] < 1 << 30, 'audit memory guard exceeded')
    save(BASE / 'audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('initialize', 'audit'))
    args = parser.parse_args()
    initialize() if args.stage == 'initialize' else audit()
