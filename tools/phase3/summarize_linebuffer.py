#!/usr/bin/env python3
"""Freeze source-matched RTL, route and artifact hashes for the P3 candidate."""
import hashlib
import gzip
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'docs/research/evidence/phase3-linebuffer'
RELEASE = ROOT / 'hardware/releases/phase3-linebuffer/tinyml_v5_candidate.fs'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resource(report, name):
    match = re.search(rf'^\s*{re.escape(name)}\s*\|\s*([\d.]+)/([\d.]+)', report, re.M)
    if not match:
        raise ValueError(f'missing {name} in route report')
    return [float(match[1]), float(match[2])]


def tests_pass(filename):
    cases = ET.parse(EVIDENCE / filename).findall('.//testcase')
    if not cases or any(c.find('failure') is not None or c.find('error') is not None for c in cases):
        raise ValueError(f'{filename} has no passing test case')
    return [c.attrib['name'] for c in cases]


def main():
    native = json.loads((EVIDENCE / 'smallcnn-1000-rtl.json').read_text())
    full = json.loads(gzip.decompress((EVIDENCE / 'smallcnn-10000-rtl.json.gz').read_bytes()))
    manifest = json.loads((EVIDENCE / 'source-manifest.json').read_text())
    baseline = json.loads((ROOT / 'docs/research/evidence/phase3/smallcnn-1000-rtl.json').read_text())
    board = json.loads((EVIDENCE / 'smallcnn-board.json').read_text())
    full_board = json.loads((EVIDENCE / 'smallcnn-full-board.json').read_text())
    labels = board['layer_outputs']
    if board['target_manifest_sha256'] != manifest['manifest_sha256']:
        raise ValueError('fixture and target manifests differ')
    if native['sources_sha256'] != manifest['sources_sha256']:
        raise ValueError('simulator and route source manifests differ')
    if native['jobs'] != 1000 or native['integer_mismatches'] or native['stage_checks'] != 8:
        raise ValueError('SmallCNN correctness gate is incomplete')
    if len(native['counters']) != 1000 or len({tuple(row) for row in native['counters']}) != 1:
        raise ValueError('inconsistent repeated-job counters')
    if len(native['stage_counters']) != len(labels):
        raise ValueError('missing per-layer stage counters')
    current = native['counters'][0]
    if (full['jobs'] != 10000 or full['integer_mismatches'] or
        len(full['counters']) != 10000 or
        len({tuple(row) for row in full['counters']}) != 1 or
        full['counters'][0] != current or
        full['sources_sha256'] != manifest['sources_sha256'] or
        full['image_sha256'] != native['image_sha256'] or
        full_board['target_manifest_sha256'] != manifest['manifest_sha256'] or
        full_board['integer_correct'] != 9640):
        raise ValueError('10,000-image RTL/oracle evidence is inconsistent')
    previous = baseline['counters'][0]
    if current[5] % 8:
        raise ValueError('nonintegral physical SRAM read transactions')
    port_transactions = current[5] // 8 + current[6]
    if not (current[0] < previous[0] and current[5] < previous[5]):
        raise ValueError('line buffer did not improve cycles and SRAM traffic')

    route = (EVIDENCE / 'gowin-route.txt').read_text()
    timing = gzip.decompress((EVIDENCE / 'gowin-timing.html.gz').read_bytes()).decode()
    ci = (EVIDENCE / 'ci.txt').read_text()
    compiler_pass = re.search(r'(\d+) passed,', ci)
    if not compiler_pass or 'CI tier passed' not in ci:
        raise ValueError('software CI tier did not pass')
    fmax = re.search(r'<td>clk27m</td>\s*<td>27\.000\(MHz\)</td>\s*<td>([\d.]+)\(MHz\)</td>', timing)
    setup = re.search(r'Setup Paths Table.*?<td>1</td>\s*<td>([-\d.]+)</td>', timing, re.S)
    if not fmax or not setup or float(fmax[1]) < 27 or float(setup[1]) < 0:
        raise ValueError('27 MHz post-route timing gate failed')
    profiles = []
    prior = [0] * 9
    for label, counters in zip(labels, native['stage_counters']):
        profiles.append({'layer': label, 'cumulative': counters,
                         'increment': [v - p for v, p in zip(counters, prior)]})
        prior = counters
    if prior != current:
        raise ValueError('layer profile does not reconcile with the full job')

    files = ['source-manifest.json', 'smallcnn-board.json', 'smallcnn-full-board.json',
             'smallcnn-1000-rtl.json', 'smallcnn-10000-rtl.json.gz', 'kernel-profiles.json',
             'kernel-results.xml', 'requantizer-results.xml', 'engine-results.xml',
             'board-results.xml', 'system-results.xml', 'switch-results.xml',
             'gowin-route.txt', 'gowin-build.log', 'gowin-timing.html.gz',
             'ci.txt', 'v2-test.txt']
    summary = {
        'scope': 'unprogrammed Phase 3 Tang Nano 20K candidate; RTL simulation and Gowin post-route only',
        'target_id': manifest['target']['target_id'],
        'tool': 'Gowin Education V1.9.11.03',
        'device': manifest['target']['device'],
        'device_revision': manifest['target']['device_revision'],
        'constraint_mhz': 27.0,
        'routed_fmax_mhz': float(fmax[1]),
        'worst_setup_slack_ns': float(setup[1]),
        'resources': {name: resource(route, name) for name in ('Logic', 'Register', 'BSRAM', 'DSP')},
        'ssram_ram16_used': int(re.search(r'SSRAM\(RAM16\)\s*\|\s*(\d+)', route)[1]),
        'scratchpad_bsram_blocks': int(re.search(r'--SP\s*\|\s*(\d+)', route)[1]),
        'line_buffer_bsram_blocks': int(re.search(r'--SDPB\s*\|\s*(\d+)', route)[1]),
        'jobs': native['jobs'],
        'integer_mismatches': native['integer_mismatches'],
        'stage_checks': native['stage_checks'],
        'full_mnist_rtl_jobs': full['jobs'],
        'full_mnist_integer_mismatches': full['integer_mismatches'],
        'full_mnist_integer_correct': full_board['integer_correct'],
        'full_mnist_integer_accuracy': full_board['integer_correct'] / full['jobs'],
        'full_mnist_fixture_sha256': full['fixture_sha256'],
        'board_image_sha256': native['image_sha256'],
        'fixture_sha256': native['fixture_sha256'],
        'counter_columns': native['counter_columns'],
        'counter_per_job': current,
        'minimum_single_port_service_cycles': port_transactions,
        'single_port_service_fraction_percent': round(100 * port_transactions / current[0], 3),
        'previous_board_tested_rtl_counter': previous,
        'cycle_reduction_percent': round(100 * (previous[0] - current[0]) / previous[0], 3),
        'sram_read_reduction_percent': round(100 * (previous[5] - current[5]) / previous[5], 3),
        'simulated_core_ms_at_27mhz': round(current[0] / 27000, 6),
        'per_layer_profile': profiles,
        'directed_tests': {key: tests_pass(filename) for key, filename in (
            ('requantizer', 'requantizer-results.xml'), ('engine', 'engine-results.xml'),
            ('board_top', 'board-results.xml'), ('system', 'system-results.xml'),
            ('kernels', 'kernel-results.xml'), ('switch', 'switch-results.xml'))},
        'compiler_tests_passed': int(compiler_pass[1]),
        'generic_clock_routing_warning': 'PR1014' in (EVIDENCE / 'gowin-build.log').read_text(),
        'physical_board_programmed': False,
        'measured_power': False,
        'bitstream_path': str(RELEASE.relative_to(ROOT)),
        'bitstream_sha256': digest(RELEASE),
        'bitstream_bytes': RELEASE.stat().st_size,
        'artifact_sha256': {name: digest(EVIDENCE / name) for name in files},
    }
    if summary['resources']['BSRAM'][0] != (summary['scratchpad_bsram_blocks'] +
                                            summary['line_buffer_bsram_blocks']):
        raise ValueError('BSRAM breakdown does not reconcile')
    (EVIDENCE / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('routed_fmax_mhz', 'worst_setup_slack_ns',
          'counter_per_job', 'cycle_reduction_percent', 'sram_read_reduction_percent')}, indent=2))


if __name__ == '__main__':
    main()
