#!/usr/bin/env python3
"""Archive completed physical runs and their exact routed release, fail closed."""
import gzip
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from run_models import load_fixture

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'work/physical'
OUT = ROOT / 'docs/research/evidence/physical'
RELEASE = ROOT / 'hardware/releases/physical'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def validate_phase(phase):
    """Recheck every saved output against the pinned fixture, not a pass flag."""
    path = Path(phase['fixture'])
    if not path.is_absolute():
        path = ROOT / path
    jobs = phase['completed_jobs']
    meta, _, arrays = load_fixture(path, jobs)
    if (phase['status'] != 'passed' or phase['requested_jobs'] != jobs or
            phase['integer_mismatches'] or len(phase['records']) != jobs or
            phase['fixture_sha256'] != sha(path/'checks.npz') or
            phase['metadata_sha256'] != sha(path/'board.json') or
            phase['image_sha256'] != meta['image_sha256'] or
            phase['expected_macs'] != meta['macs']):
        raise ValueError('physical fixture identity or result count differs')
    correct = 0
    cycles = []
    for index, record in enumerate(phase['records']):
        expected = arrays['outputs'][index].tobytes()
        if record['job'] != index or bytes.fromhex(record['output_hex']) != expected:
            raise ValueError(f'archived integer output mismatch: job {index}')
        correct += int(np.frombuffer(expected, np.int8).argmax() == arrays['labels'][index])
        c = record['counters']
        if (c['busy'] or c['error'] or c['protocol_errors'] or
                c['useful_macs'] != meta['macs'] or
                c['elapsed'] != sum(c[k] for k in ('compute_cycles', 'wait_cycles', 'control_cycles'))):
            raise ValueError('invalid per-job counters')
        if not np.isfinite(record['host_job_seconds']) or record['host_job_seconds'] < 0:
            raise ValueError('invalid host timing')
        cycles.append(c['elapsed'])
    if correct != phase['correct'] or phase['accuracy'] != correct / jobs:
        raise ValueError('archived accuracy differs from labels')
    if (phase['elapsed_cycles_min'] != min(cycles) or
            phase['elapsed_cycles_max'] != max(cycles) or
            phase['elapsed_cycles_median'] != float(np.median(cycles))):
        raise ValueError('archived cycle summary differs')


def validate_checks(report):
    """Recheck saved kernel logits and intermediate hashes with their oracles."""
    from run_checks import reconciles, vectors
    checks = report['checks']
    if len(checks) != 59 or any(c['status'] != 'passed' for c in checks):
        raise ValueError('incomplete physical diagnostic suite')
    kernels = [c for c in checks if 'runs' in c]
    expected_kernels = list(vectors())
    if len(kernels) != len(expected_kernels):
        raise ValueError('kernel coverage differs')
    for record, (desc, _, _, _, expected, macs, name) in zip(kernels, expected_kernels):
        expected_hex = np.asarray(expected, np.int8).tobytes().hex()
        if (record['name'] != name or record['descriptor'] != vars(desc) or
                record['expected_hex'] != expected_hex or record['actual_hex'] != expected_hex or
                record['integer_mismatches'] or len(record['runs']) != 3):
            raise ValueError(f'archived kernel mismatch: {name}')
        for counters in record['runs']:
            reconciles(counters, macs)
    layers = [c for c in checks if 'layer' in c]
    seen = set()
    for record in layers:
        path = Path(record['fixture'])
        if not path.is_absolute():
            path = ROOT / path
        meta, _, _ = load_fixture(path, 3)
        index, job = record['layer'], record['job']
        if not 0 <= job < 3 or not 0 <= index < len(meta['layer_outputs']):
            raise ValueError('layer/sample index out of bounds')
        key = (str(path), index, job)
        if key in seen:
            raise ValueError('duplicate layer check')
        seen.add(key)
        name = meta['layer_outputs'][index]
        with np.load(path/'checks.npz', allow_pickle=False) as arrays:
            expected = arrays['layer_'+name][job].tobytes()
        expected_sha = hashlib.sha256(expected).hexdigest()
        if (record['name'] != name or record['integer_mismatches'] or
                record['image_sha256'] != meta['image_sha256'] or
                record['fixture_sha256'] != sha(path/'checks.npz') or
                record['bytes_compared'] != len(expected) or
                record['expected_sha256'] != expected_sha or record['actual_sha256'] != expected_sha):
            raise ValueError(f'archived intermediate mismatch: {name}')
        desc = meta['descriptors'][index]
        macs = desc['outputs'] * desc['count'] if desc['opcode'] in (1, 4, 6) else 0
        reconciles(record['counters'], macs)
    if len(seen) != 39:
        raise ValueError('incomplete layer coverage')


def main():
    build = WORK / 'accelerator-reset-build'
    pnr = build / 'tinyml_v2/impl/pnr'
    bitstream = pnr / 'tinyml_v2.fs'
    bitsha = sha(bitstream)
    source = read(build / 'source-manifest.json')
    if sha(ROOT / 'hardware/targets/tang_nano_20k_v2.json') != source['manifest_sha256']:
        raise ValueError('target changed since build')
    for name, digest in source['sources_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'RTL changed since build: {name}')
    uart_build = WORK / 'uart-release-build'
    for name, digest in read(uart_build / 'source-manifest.json').items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'UART source changed since build: {name}')

    switch = read(WORK / 'model-switch.json')
    quality = read(WORK / 'smallcnn-full.json')
    checks = read(WORK / 'checks.json')
    for result in (switch, quality, checks):
        if result['status'] != 'passed' or result['supplied_bitstream_sha256'] != bitsha:
            raise ValueError('physical result failed or bitstream differs')
    if len(switch['phases']) != 3:
        raise ValueError('three-stage switch result required')
    for phase, expected_macs in zip(switch['phases'], (10112, 61184, 10112)):
        if (phase['completed_jobs'] != 1000 or phase['integer_mismatches'] or
                phase['expected_macs'] != expected_macs):
            raise ValueError('incomplete MLP/SmallCNN/MLP switch result')
    if len(quality['phases']) != 1:
        raise ValueError('one full-set SmallCNN run required')
    q = quality['phases'][0]
    if q['completed_jobs'] != 10000 or q['integer_mismatches'] or q['correct'] != 9640:
        raise ValueError('incomplete or changed full-set quality')
    for phase in switch['phases'] + quality['phases']:
        validate_phase(phase)
    validate_checks(checks)
    loopback = read(WORK / 'uart-release-readback.json')
    if loopback['status'] != 'measured-pass' or loopback['bytes'] != 1280:
        raise ValueError('UART loopback did not pass')
    for name in ('uart-release-program.log', 'accelerator-release-program.log'):
        if 'after program sram: displayReadReg 00006020' not in (WORK / name).read_text():
            raise ValueError(f'missing successful programming status: {name}')

    route = (pnr / 'tinyml_v2.rpt.txt').read_text()
    timing = (pnr / 'tinyml_v2_tr_content.html').read_text()
    def number(pattern, text):
        match = re.search(pattern, text, re.M | re.S)
        if not match:
            raise ValueError(f'report metric missing: {pattern}')
        return float(match.group(1))
    resources = {name: [number(r'^\s+'+name+r'\s+\|\s+([\d.]+)/', route),
                        number(r'^\s+'+name+r'\s+\|\s+[\d.]+/([\d.]+)', route)]
                 for name in ('Logic', 'Register', 'BSRAM', 'DSP')}
    fmax = number(r'<td>27\.000\(MHz\)</td>\s*<td>([\d.]+)\(MHz\)</td>', timing)
    slack = number(r'<td class="label">Slack</td>\s*<td>([-\d.]+)</td>', timing)
    if fmax < 27 or slack < 0 or any(used > limit for used, limit in resources.values()):
        raise ValueError('routed target failed')
    for path in (ROOT / 'work/phase2/sim_board/results.xml', ROOT / 'work/phase4/sim_kernels/results.xml'):
        cases = ET.parse(path).findall('.//testcase')
        if not cases or any(c.find('failure') is not None or c.find('error') is not None for c in cases):
            raise ValueError(f'RTL regression failed: {path}')

    OUT.mkdir(parents=True, exist_ok=True)
    RELEASE.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bitstream, RELEASE / 'tinyml_v4_tang20k.fs')
    shutil.copyfile(uart_build / 'uart_loopback/impl/pnr/uart_loopback.fs', RELEASE / 'uart_loopback.fs')
    archived = {}
    def archive(path, name, compress=False):
        data = path.read_bytes()
        target = OUT / (name + '.gz' if compress else name)
        target.write_bytes(gzip.compress(data, mtime=0) if compress else data)
        archived[target.name] = dict(sha256=sha(target), uncompressed_sha256=hashlib.sha256(data).hexdigest())
    for name in ('model-switch.json', 'smallcnn-full.json', 'checks.json'):
        archive(WORK / name, name, compress=name != 'checks.json')
    for source_path, name in (
        (WORK/'uart-release-readback.json', 'uart-readback.json'),
        (WORK/'wire-readback.json', 'wire-readback.json'),
        (WORK/'uart-release-program.log', 'uart-program.txt'),
        (WORK/'accelerator-release-program.log', 'accelerator-program.txt'),
        (WORK/'accelerator-reset-program.log', 'switch-program.txt'),
        (WORK/'capabilities.json', 'capabilities.json'),
        (WORK/'board-simulation.txt', 'board-simulation.txt'),
        (WORK/'kernel-simulation.txt', 'kernel-simulation.txt'),
        (WORK/'checks-console.txt', 'checks-console.txt'),
        (WORK/'runner-tests.txt', 'runner-tests.txt'),
        (WORK/'phase5-tests.txt', 'phase5-tests.txt'),
        (WORK/'quality-preparation.txt', 'quality-preparation.txt'),
        (ROOT/'work/phase4/kernel-profiles.json', 'kernel-simulation-profiles.json'),
        (build/'build.log', 'gowin-console.txt'),
        (build/'source-manifest.json', 'source-manifest.json'),
        (uart_build/'source-manifest.json', 'uart-source-manifest.json'),
        (pnr/'tinyml_v2.rpt.txt', 'gowin-route.txt'),
        (pnr/'tinyml_v2_tr_content.html', 'gowin-timing.html'),
        (ROOT/'work/phase2/sim_board/results.xml', 'board-results.xml'),
        (ROOT/'work/phase4/sim_kernels/results.xml', 'kernel-results.xml'),
        (WORK/'smallcnn-full/board.json', 'smallcnn-full-metadata.json'),
        (ROOT/'work/phase2/mlp/board.json', 'mlp-metadata.json'),
    ):
        archive(source_path, name, compress=name == 'gowin-timing.html')
    sources = [*source['sources_sha256'], 'hardware/tang_nano_v2.cst',
               'hardware/tang_nano_v2.sdc', 'hardware/targets/tang_nano_20k_v2.json',
               'test/phase2/test_board.py', 'test/phase4/test_kernels.py',
               'test/phase4/kernel_vectors.py', 'tools/phase2/host.py',
               'tools/phase2/build_gowin.py', 'tools/physical/run_models.py',
               'tools/physical/run_checks.py', 'tools/physical/prepare_smallcnn_quality.py',
               'tools/physical/collect_evidence.py', 'compiler/hardware_v2.py',
               'compiler/integer_reference.py', 'compiler/quantization.py',
               'tools/physical/test_run_models.py', 'tools/phase5/audit.py',
               'tools/phase5/test_phase5.py']
    software = {name: sha(ROOT / name) for name in sources}
    (OUT / 'validation-source-manifest.json').write_text(json.dumps(software, indent=2)+'\n')
    def compact(phase):
        return {k: v for k, v in phase.items() if k != 'records'}
    summary = dict(
        scope='physical on-chip accelerator validation; no SDRAM or complete KWS/VWW',
        status='passed', source_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_hashes_authoritative=True, target_id=8196,
        device='GW2AR-LV18QN88C8/I7', device_revision='C', jtag_idcode='0x0000081B',
        user_reported_chip_markings=['GW2AR-LV18', 'QN88C8/I7', '2537C', 'NCWS02.00'],
        usb_serial='2025030317', uart_port='/dev/cu.usbserial-20250303171',
        pcb_revision=None, pcb_revision_note='user could not identify marking',
        power_instrument_available=False, measured_power=False,
        host_platform=platform.platform(), python_version=sys.version,
        tool='Gowin Education V1.9.11.03', programmer='openFPGALoader 1.1.1',
        programming_mode='temporary SRAM configuration; flash untouched',
        physical_board_programmed=True, baud=115200, declared_clock_hz=27000000,
        clock_instrument_measurement=False, generic_clock_routing_warning=True,
        routed_fmax_mhz=fmax, worst_setup_slack_ns=slack, resources=resources,
        bitstream_path='hardware/releases/physical/tinyml_v4_tang20k.fs', bitstream_sha256=bitsha,
        uart_bitstream_sha256=sha(RELEASE/'uart_loopback.fs'),
        model_switch=[compact(p) for p in switch['phases']], smallcnn_full=compact(q),
        diagnostic_checks=len(checks['checks']), physical_layer_checks=39,
        kernel_cases=7, kernel_repetitions=3, sram_pattern_passes=6,
        uart_loopback_bytes=1280, sdram_controller_integrated=False, dma_integrated=False,
        complete_kws_vww=False, baseline_comparison=False,
        validation_source_manifest_sha256=sha(OUT/'validation-source-manifest.json'),
        artifacts=archived)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'Archived 13,000 model jobs and {len(checks["checks"])} physical checks; Fmax {fmax} MHz')


if __name__ == '__main__':
    main()
