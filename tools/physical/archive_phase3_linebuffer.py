#!/usr/bin/env python3
"""Validate and archive the Phase 3 line-buffer run on the physical board."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from collect_evidence import ROOT, validate_phase
from run_checks import reconciles, vectors
from run_models import load_fixture


WORK = ROOT / 'work/physical/phase3-linebuffer'
OUT = ROOT / 'docs/research/evidence/phase3-linebuffer-physical'
BITSTREAM = ROOT / 'hardware/releases/phase3-linebuffer/tinyml_v5_candidate.fs'
SOURCE = ROOT / 'docs/research/evidence/phase3-linebuffer/source-manifest.json'
BASELINE = ROOT / 'docs/research/evidence/physical/summary.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def validate_checks(report):
    checks = report['checks']
    expected_vectors = list(vectors())
    if report['status'] != 'passed' or len(checks) != 13 + len(expected_vectors) + 39:
        raise ValueError('incomplete board diagnostic suite')
    if any(check['status'] != 'passed' for check in checks):
        raise ValueError('failed board diagnostic')
    size = 32768
    address = np.arange(size, dtype=np.uint32)
    patterns = {
        'zeros': bytes(size),
        'ones': bytes([255]) * size,
        'walking_one': (1 << (address % 8)).astype(np.uint8).tobytes(),
        'walking_zero': (255 ^ (1 << (address % 8))).astype(np.uint8).tobytes(),
        'address_dependent': ((address * 73) ^ (address >> 8)).astype(np.uint8).tobytes(),
        'seeded_random': np.random.default_rng(20260923).integers(0, 256, size, dtype=np.uint8).tobytes(),
    }
    for check, (name, pattern) in zip(checks[:6], patterns.items()):
        if (check['name'] != name or check['bytes_written'] != size or
                check['bytes_read'] != size or
                check['sha256'] != hashlib.sha256(pattern).hexdigest()):
            raise ValueError(f'SRAM pattern evidence differs: {name}')
    if [check['name'] for check in checks[6:13]] != [
        'CRC rejection preserves SRAM',
        'oversized frame drains embedded command',
        'version, bounds, alignment and opcode rejection',
        'partial frame timeout and recovery',
        'busy ownership and ABORT',
        'RESET clears counters and preserves SRAM',
        'descriptor version rejection and next-RUN recovery',
    ]:
        raise ValueError('incomplete protocol checks')
    kernels = checks[13:13 + len(expected_vectors)]
    for check, (descriptor, _, _, _, expected, macs, name) in zip(kernels, expected_vectors):
        expected_hex = np.asarray(expected, np.int8).tobytes().hex()
        if (check['name'] != name or check['descriptor'] != vars(descriptor) or
                check['expected_hex'] != expected_hex or
                check['actual_hex'] != expected_hex or
                check['integer_mismatches'] != 0 or len(check['runs']) != 3):
            raise ValueError(f'kernel evidence differs: {name}')
        for counters in check['runs']:
            reconciles(counters, macs)
    seen = set()
    for check in checks[13 + len(expected_vectors):]:
        fixture = ROOT / check['fixture']
        meta, _, _ = load_fixture(fixture, 3)
        job, index = check['job'], check['layer']
        key = (str(fixture), job, index)
        if key in seen or not (0 <= job < 3 and 0 <= index < len(meta['layer_outputs'])):
            raise ValueError('duplicate or out-of-range layer check')
        seen.add(key)
        name = meta['layer_outputs'][index]
        with np.load(fixture / 'checks.npz', allow_pickle=False) as archive:
            expected = archive['layer_' + name][job].tobytes()
        expected_sha = hashlib.sha256(expected).hexdigest()
        if (check['name'] != name or check['integer_mismatches'] != 0 or
                check['image_sha256'] != meta['image_sha256'] or
                check['fixture_sha256'] != digest(fixture / 'checks.npz') or
                check['bytes_compared'] != len(expected) or
                check['expected_sha256'] != expected_sha or
                check['actual_sha256'] != expected_sha):
            raise ValueError(f'layer evidence differs: {name}')
        descriptor = meta['descriptors'][index]
        macs = descriptor['outputs'] * descriptor['count'] if descriptor['opcode'] in (1, 4, 6) else 0
        reconciles(check['counters'], macs)
    if len(seen) != 39:
        raise ValueError('incomplete MLP/SmallCNN layer coverage')


def main():
    bit_sha = digest(BITSTREAM)
    source = read(SOURCE)
    if digest(ROOT / 'hardware/targets/tang_nano_20k_v2.json') != source['manifest_sha256']:
        raise ValueError('target differs from routed image')
    for name, expected_sha in source['sources_sha256'].items():
        if digest(ROOT / name) != expected_sha:
            raise ValueError(f'RTL differs from routed image: {name}')
    switch = read(WORK / 'model-switch.json')
    full = read(WORK / 'smallcnn-full.json')
    checks = read(WORK / 'checks.json')
    for report in (switch, full, checks):
        if report['status'] != 'passed' or report['supplied_bitstream_sha256'] != bit_sha:
            raise ValueError('board report failed or bitstream identity differs')
    if len(switch['phases']) != 3 or len(full['phases']) != 1:
        raise ValueError('expected three switch stages and one full-set stage')
    for stage, expected_macs in zip(switch['phases'], (10112, 61184, 10112)):
        if stage['completed_jobs'] != 1000 or stage['expected_macs'] != expected_macs:
            raise ValueError('incomplete model-switch stage')
        validate_phase(stage)
    quality = full['phases'][0]
    if quality['completed_jobs'] != 10000 or quality['correct'] != 9640:
        raise ValueError('incomplete or changed full-set result')
    validate_phase(quality)
    validate_checks(checks)
    program_log = (WORK / 'program.log').read_text()
    if ('after program sram: displayReadReg 00006020' not in program_log or
            'idcode: 0000081b' not in program_log or
            str(BITSTREAM.relative_to(ROOT)) not in program_log):
        raise ValueError('missing successful SRAM programming record')
    baseline = read(BASELINE)
    old = baseline['smallcnn_full']
    if quality['image_sha256'] != old['image_sha256'] or quality['correct'] != old['correct']:
        raise ValueError('candidate/baseline model or quality differs')
    baseline_raw_path = BASELINE.parent / 'smallcnn-full.json.gz'
    if digest(baseline_raw_path) != baseline['artifacts'][baseline_raw_path.name]['sha256']:
        raise ValueError('baseline raw report hash differs')
    baseline_raw = json.loads(gzip.decompress(baseline_raw_path.read_bytes()))
    old_counters = baseline_raw['phases'][0]['records'][0]['counters']
    baseline_checks_path = BASELINE.parent / 'checks.json'
    if digest(baseline_checks_path) != baseline['artifacts'][baseline_checks_path.name]['sha256']:
        raise ValueError('baseline diagnostic report hash differs')
    baseline_checks = read(baseline_checks_path)
    def smallcnn_first_layers(report):
        return [check for check in report['checks']
                if check.get('fixture') == 'work/phase3/smallcnn' and check.get('job') == 0]
    old_layers = smallcnn_first_layers(baseline_checks)
    new_layers = smallcnn_first_layers(checks)
    if len(old_layers) != 8 or len(new_layers) != 8:
        raise ValueError('incomplete per-layer baseline comparison')
    per_layer = []
    for old_layer, new_layer in zip(old_layers, new_layers):
        if (old_layer['name'] != new_layer['name'] or
                old_layer['expected_sha256'] != new_layer['expected_sha256']):
            raise ValueError('candidate/baseline layer differs')
        per_layer.append(dict(layer=new_layer['name'],
                              baseline_cycles=old_layer['counters']['elapsed'],
                              candidate_cycles=new_layer['counters']['elapsed'],
                              baseline_read_bytes=old_layer['counters']['read_bytes'],
                              candidate_read_bytes=new_layer['counters']['read_bytes']))
    candidate_counters = quality['records'][0]['counters']
    old_cycles = old_counters['elapsed']
    old_reads = old_counters['read_bytes']
    if old_cycles != old['elapsed_cycles_median']:
        raise ValueError('baseline counters differ from summary')
    boardless = read(SOURCE.parent / 'summary.json')
    if bit_sha != boardless['bitstream_sha256']:
        raise ValueError('programmed artifact differs from routed candidate')
    columns = boardless['counter_columns']
    expected_counters = boardless['counter_per_job']
    if any([record['counters'][column] for column in columns] != expected_counters
           for record in quality['records'] + switch['phases'][1]['records']):
        raise ValueError('physical candidate counters differ from board-system RTL')
    if candidate_counters['elapsed'] >= old_cycles or candidate_counters['read_bytes'] >= old_reads:
        raise ValueError('expected line-buffer performance improvement is absent')
    OUT.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name in ('model-switch.json', 'smallcnn-full.json', 'checks.json', 'program.log'):
        src = WORK / name
        data = src.read_bytes()
        compressed = name.endswith('.json') and name != 'checks.json'
        dst = OUT / (name + '.gz' if compressed else name)
        dst.write_bytes(gzip.compress(data, mtime=0) if compressed else data)
        artifacts[dst.name] = dict(sha256=digest(dst), uncompressed_sha256=hashlib.sha256(data).hexdigest())
    summary = dict(
        scope='physical on-chip Phase 3 line-buffer validation; no SDRAM or complete KWS/VWW',
        status='passed', source_base_commit='aa518a488270f6103feffab0b4030705048b2394',
        source_manifest=str(SOURCE.relative_to(ROOT)), source_manifest_sha256=digest(SOURCE),
        target_id=8196, device='GW2AR-LV18QN88C8/I7', device_revision='C',
        jtag_idcode='0x0000081B', usb_serial='2025030317',
        uart_port='/dev/cu.usbserial-20250303171',
        programming_mode='temporary SRAM configuration; flash untouched',
        bitstream_path=str(BITSTREAM.relative_to(ROOT)), bitstream_sha256=bit_sha,
        bitstream_hash_readback_supported=False, declared_clock_hz=27000000,
        clock_instrument_measurement=False, measured_power=False,
        tool=boardless['tool'], programmer='openFPGALoader 1.1.1',
        routed_fmax_mhz=boardless['routed_fmax_mhz'],
        worst_setup_slack_ns=boardless['worst_setup_slack_ns'],
        resources=boardless['resources'],
        model_switch=[{k: v for k, v in stage.items() if k != 'records'} for stage in switch['phases']],
        smallcnn_full={k: v for k, v in quality.items() if k != 'records'},
        smallcnn_full_first_job_counters=candidate_counters,
        diagnostic_checks=len(checks['checks']), physical_layer_checks=39,
        kernel_cases=len(list(vectors())), kernel_repetitions=3, sram_pattern_passes=6,
        baseline_bitstream_sha256=baseline['bitstream_sha256'],
        baseline_smallcnn_cycles=old_cycles, baseline_smallcnn_read_bytes=old_reads,
        cycle_reduction_percent=100 * (old_cycles - candidate_counters['elapsed']) / old_cycles,
        sram_read_reduction_percent=100 * (old_reads - candidate_counters['read_bytes']) / old_reads,
        per_layer_profile=per_layer,
        artifacts=artifacts)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Archived {sum(s["completed_jobs"] for s in switch["phases"]) + quality["completed_jobs"]} jobs and {len(checks["checks"])} checks')


if __name__ == '__main__':
    main()
