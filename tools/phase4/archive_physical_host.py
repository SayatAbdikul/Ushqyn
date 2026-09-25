#!/usr/bin/env python3
"""Check and preserve exact physical tiled-host model runs."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/'docs/research/evidence/phase4'
CORE_HZ = 20_250_000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_model(name, report_path, label):
    fixture = ROOT/f'work/phase4/rtl-{name}/manifest.json'
    manifest = json.loads(fixture.read_text())
    report = json.loads(report_path.read_text())
    nodes = report['nodes']
    if (report.get('status') != 'passed' or
            report.get('physical_board') is not True or
            report.get('model') != name or
            report.get('fixture_sha256') != sha(fixture) or
            report.get('parameter_image_sha256') != manifest['image_sha256'] or
            report.get('final_output_sha256') != manifest['output_sha256'] or
            len(nodes) != manifest['nodes']):
        raise ValueError(f'{name}: physical report/fixture mismatch')
    for index, (node, expected) in enumerate(zip(nodes, manifest['layers'])):
        if node['node'] != index or node['output_sha256'] != expected['output_sha256']:
            raise ValueError(f'{name}: node {index} output hash mismatch')
    saved = EVIDENCE/f'physical-host-{label}-{name}.json'
    shutil.copyfile(report_path, saved)
    engine_cycles = sum(n['engine_counters']['elapsed'] for n in nodes)
    dma_cycles = sum(n.get('dma_elapsed_cycles', 0) for n in nodes)
    return {
        'workload': name, 'status': 'all-node-exact-on-physical-board',
        'node_count': len(nodes), 'tile_count': sum(n['tiles'] for n in nodes),
        'fixture_manifest_sha256': sha(fixture),
        'raw_report_sha256': sha(saved),
        'final_output_sha256': report['final_output_sha256'],
        'engine_cycles': engine_cycles,
        'engine_only_seconds_at_nominal_clock': engine_cycles/CORE_HZ,
        'dma_payload_bytes': sum(n['dma_payload_bytes'] for n in nodes),
        'dma_elapsed_cycles': dma_cycles if dma_cycles else None,
        'host_schedule_wall_seconds': sum(n['host_wall_seconds'] for n in nodes),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--kws', required=True, type=Path)
    parser.add_argument('--vww', required=True, type=Path)
    parser.add_argument('--bitstream', required=True, type=Path)
    parser.add_argument('--route', required=True, type=Path)
    args = parser.parse_args()
    route = json.loads(args.route.read_text())
    if sha(args.bitstream) != route['bitstream_sha256']:
        raise ValueError('bitstream does not match route record')
    models = [check_model(name, path, args.label)
              for name, path in (('kws', args.kws), ('vww', args.vww))]
    summary = {
        'status': 'passed', 'board_part': 'GW2AR-LV18QN88C8/I7 revision C',
        'programming': 'openFPGALoader -b tangnano20k -m -v; temporary SRAM image',
        'bitstream_sha256': sha(args.bitstream),
        'route_record_sha256': sha(args.route),
        'nominal_core_clock_hz': CORE_HZ,
        'scope': 'one deterministic input per model; every node checked byte-exactly',
        'models': models,
        'limits': ['host schedule includes stop-and-wait UART and per-node readback',
                   'engine-only clocks exclude DMA and host scheduling',
                   'no whole-board power/energy measurement'],
    }
    output = EVIDENCE/f'physical-host-{args.label}.json'
    output.write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    print(json.dumps({m['workload']: m['node_count'] for m in models}))


if __name__ == '__main__':
    main()
