#!/usr/bin/env python3
"""Freeze the routed dual-model board image and its build/program evidence."""

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'docs/research/evidence/phase5'
RELEASE = ROOT / 'hardware/releases/phase5/dual_resident_750k.fs'
SOURCES = (
    'rtl/v2/target_pkg.sv',
    'hardware/phase4_sdram/pll_dma_20.v',
    'hardware/phase5/tiled_host.sv',
    'rtl/v2/requantizer.sv',
    'rtl/v2/scratchpad.sv',
    'rtl/v2/engine.sv',
    'rtl/v2/tile_dma.sv',
    'rtl/v2/tiled_core.sv',
    'rtl/v2/command.sv',
    'rtl/v2/tiled_host_bridge.sv',
    'rtl/v2/tile_sequencer.sv',
    'rtl/v2/hs_sdram_burst_port.sv',
    'rtl/v2/sdram_refresh.sv',
    'rtl/v2/uart_rx.sv',
    'rtl/v2/uart_tx.sv',
    'hardware/phase5/build_tiled_host.tcl',
    'hardware/phase4_sdram/tiled_host.cst',
    'hardware/phase4_sdram/bist.sdc',
    'hardware/phase4_sdram/sdram_controller_hs.ipc',
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def required_match(pattern, data, label):
    match = re.search(pattern, data, re.M)
    if match is None:
        raise ValueError(f'missing {label} in route report')
    return match


def archive(build, program_log, smoke):
    pnr = build / 'phase5_tiled_host/impl/pnr'
    prefix = 'phase5_tiled_host'
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    RELEASE.parent.mkdir(parents=True, exist_ok=True)
    reports = {}
    for label, suffix in (('route', '.rpt.txt'),
                          ('timing', '_tr_content.html'),
                          ('power_estimate', '.power.html')):
        raw = (pnr / (prefix + suffix)).read_bytes()
        name = f'dual-resident-{label}.txt.gz'
        (EVIDENCE / name).write_bytes(gzip.compress(raw, mtime=0))
        reports[label] = {'archive': name, 'raw_sha256': digest(raw),
                          'archive_sha256': digest((EVIDENCE / name).read_bytes())}
    route = (pnr / (prefix + '.rpt.txt')).read_text()
    timing = (pnr / (prefix + '_tr_content.html')).read_text()
    resources = {}
    for key, label in (('logic', 'Logic'), ('register', 'Register'),
                       ('bsram', 'BSRAM'), ('dsp', 'DSP')):
        match = required_match(
            rf'^\s*{label}\s*\|\s*([\d.]+)/([\d.]+)', route, label)
        resources[key] = {'used': float(match[1]), 'available': int(match[2])}
    clock = required_match(
        r'<td>20\.250\(MHz\)</td>\s*<td>([\d.]+)\(MHz\)</td>',
        timing, 'core Fmax')
    tns = required_match(
        r'pll/pll_s2/CLKOUT\.default_gen_clk</td>\s*<td>Setup</td>\s*<td>([\d.]+)</td>',
        timing, 'setup TNS')
    fmax_mhz = float(clock[1])
    setup_tns_ns = float(tns[1])
    if fmax_mhz <= 20.25 or setup_tns_ns != 0:
        raise ValueError('dual-resident route fails timing')
    bitstream = (pnr / (prefix + '.fs')).read_bytes()
    RELEASE.write_bytes(bitstream)
    program = program_log.read_bytes()
    if b'after program sram: displayReadReg 00006020' not in program:
        raise ValueError('physical programming success marker absent')
    program_name = 'dual-resident-program.txt.gz'
    (EVIDENCE / program_name).write_bytes(gzip.compress(program, mtime=0))
    smoke_report = json.loads(smoke.read_text())
    if (smoke_report['status'] != 'passed-probe'
            or smoke_report['jobs_completed'] != 4
            or smoke_report['campaign'] != 'switch'
            or smoke_report['bitstream_sha256'] != digest(bitstream)
            or smoke_report['output_mismatches'] != 0
            or smoke_report['reflash_events'] != 0):
        raise ValueError('physical alternating-job smoke report does not match route')
    smoke_records = smoke.with_suffix('.jsonl')
    if digest(smoke_records.read_bytes()) != smoke_report['records_sha256']:
        raise ValueError('smoke record digest mismatch')
    artifacts = {}
    for source, destination in ((smoke, 'dual-resident-smoke.json'),
                                (smoke_records, 'dual-resident-smoke.jsonl')):
        raw = source.read_bytes()
        (EVIDENCE / destination).write_bytes(raw)
        artifacts[destination] = digest(raw)
    ip = ROOT / 'work/phase4/ip-generated/sdram_controller_hs.v'
    result = {
        'schema': 1,
        'scope': 'physical 4-job alternating smoke; full Phase 5 accuracy and stability remain separate',
        'status': 'passed-smoke',
        'device': 'GW2AR-LV18QN88C8/I7',
        'core_clock_mhz': 20.25,
        'routed_core_fmax_mhz': fmax_mhz,
        'setup_tns_ns': setup_tns_ns,
        'uart_baud': 750000,
        'resources': resources,
        'bitstream_path': str(RELEASE.relative_to(ROOT)),
        'bitstream_sha256': digest(bitstream),
        'source_sha256': {name: digest((ROOT / name).read_bytes()) for name in SOURCES},
        'generated_ip_path': str(ip.relative_to(ROOT)),
        'generated_ip_sha256': digest(ip.read_bytes()),
        'reports': reports,
        'program_log_archive': program_name,
        'program_log_sha256': digest(program),
        'program_log_archive_sha256': digest((EVIDENCE / program_name).read_bytes()),
        'physical_smoke_artifacts': artifacts,
        'power_scope': 'Gowin report is an estimate, not physical power or energy',
    }
    (EVIDENCE / 'dual-resident-route.json').write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--program-log', type=Path, required=True)
    parser.add_argument('--smoke', type=Path, required=True)
    args = parser.parse_args()
    result = archive(args.build, args.program_log, args.smoke)
    print(json.dumps({'status': result['status'],
                      'bitstream_sha256': result['bitstream_sha256'],
                      'routed_core_fmax_mhz': result['routed_core_fmax_mhz'],
                      'resources': result['resources']}, sort_keys=True))


if __name__ == '__main__':
    main()
