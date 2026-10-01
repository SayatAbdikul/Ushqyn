#!/usr/bin/env python3
"""Pin the completed resident feasibility campaign without rerunning experiments."""
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'work/phase6/deadline-resident-v1'
REFERENCE = Path('/Users/sayat/.codex/worktrees/21f0/tinyML_accelerator')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def physical_sources(recipe):
    text = recipe.read_text()
    paths = [REFERENCE / p for p in re.findall(r'^    ((?:rtl|work)/\S+)\s*$', text, re.M)]
    paths.extend(REFERENCE / p for p in (
        'work/phase6/engine-candidate-rtl-v2/engine.sv',
        'work/phase4/ip-generated/sdram_controller_hs.v',
        'hardware/phase4_sdram/tiled_host.cst', 'hardware/phase4_sdram/bist.sdc'))
    paths.extend(BASE / p for p in re.findall(r'add_file \[file join \$resident (\S+)\]', text))
    if len(paths) != 18:
        raise ValueError('physical source inventory does not match frozen recipe')
    return paths


def main():
    historic = BASE / 'native-campaign-report.json'
    if not historic.exists():
        shutil.copyfile(BASE / 'report.json', historic)
    native = read(historic)
    arrivals = read(BASE / 'arrival-report.json')
    identity = read(BASE / 'native-build/identity.json')
    if len(native['native_runs']) != 32 or len(arrivals['runs']) != 32:
        raise ValueError('resident coverage incomplete')
    sources = {Path(p) for p in identity['source_sha256']}
    physical = {}
    for mode in ('baseline', 'candidate'):
        folder = BASE / 'physical' / mode
        row = read(folder / 'report.json')
        paths = physical_sources(folder / 'build27.tcl')
        row['source_sha256'] = {str(p): sha(p) for p in paths}
        sources.update(paths)
        physical[mode] = dict(row, report_path=str((folder / 'report.json').relative_to(ROOT)),
                              report_sha256=sha(folder / 'report.json'))
    source_bundle = []
    for p in sorted(sources):
        origin = 'reference' if p.is_relative_to(REFERENCE) else 'project'
        relative = p.relative_to(REFERENCE if origin == 'reference' else ROOT)
        dest = BASE / 'sources' / origin / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        source_bundle.append(dict(original_path=str(p), bundled_path=str(dest.relative_to(ROOT)), sha256=sha(p)))
    for path, digest in identity['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('native source changed')
    raw_runs = []
    for row in native['native_runs']:
        p = BASE / 'native' / f'{row["variant"]}-s{row["seed"]}-m{row["mode"]}.json'
        raw_runs.append(dict(row, report_path=str(p.relative_to(ROOT)), report_sha256=sha(p)))
    fixtures = {}
    for folder in sorted((BASE / 'fixtures').iterdir()):
        fixtures[folder.name] = {p.name: sha(p) for p in folder.iterdir() if p.is_file()}
    final = dict(status='complete-native-pass-physical-fail', engine_sha256=native['engine_sha256'],
        native_runs=raw_runs, arrival_runs=arrivals['runs'], physical=physical,
        native_identity_sha256=sha(BASE / 'native-build/identity.json'),
        native_campaign_report_sha256=sha(historic), arrival_report_sha256=sha(BASE / 'arrival-report.json'),
        fixture_sha256=fixtures, source_bundle=source_bundle,
        finalizer_sha256=sha(Path(__file__)), physical_board=False,
        native_driver_sha256_at_campaign=native['driver_sha256'],
        current_native_driver_sha256=sha(ROOT / 'tools/phase6/deadline_resident.py'),
        python_peak_rss_bytes=native['python_peak_rss_bytes'],
        placement_failure='PR0003: 48 equivalent LUTs unplaced; all 10368 CLS occupied',
        conclusions=[
            '24 original positive native runs, 8 effective omitted-restore negative controls and 32 additional positive arrival runs pass.',
            'Accepted late requests fall back to urgent execution after background HALT; no host UART after admission.',
            'Matched baseline routes at 27 MHz; this resident implementation fails placement and has no measured routed Fmax.',
            'Program capacity and native correctness do not establish physical feasibility or architectural novelty.'],
        limitations=[
            'Only one outstanding urgent request and one suspended background context; no queued arrival adapter implemented.',
            'Native tests use the v2 UART control path; burst UART path has synthesis/placement evidence only.',
            'One fixed RAM and one sampled stalled RAM mode, VWW background with AD urgent only, immutable repeated inputs.',
            'No board run, measured SDRAM timing, energy, or worst-case deadline bound.'])
    save(BASE / 'report.json', final)
    print(json.dumps(dict(status=final['status'], native_checks=len(raw_runs)+len(arrivals['runs']),
        bundled_sources=len(source_bundle), baseline_fmax_mhz=physical['baseline']['routed_core_fmax_mhz'],
        candidate_status=physical['candidate']['status']), indent=2))


if __name__ == '__main__':
    main()
