#!/usr/bin/env python3
"""Validate and archive one full-model framed-host RTL transcript."""

import argparse
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/'docs/research/evidence/phase4'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', choices=('kws', 'vww'))
    parser.add_argument('--log', required=True, type=Path)
    parser.add_argument('--xml', required=True, type=Path)
    args = parser.parse_args()
    fixture = ROOT/f'work/phase4/rtl-{args.name}/manifest.json'
    manifest = json.loads(fixture.read_text())
    folder = fixture.parent
    source = folder.parent
    for path, field in ((source/f'{args.name}-tiled-plan.json', 'plan_sha256'),
                        (source/f'{args.name}-parameter-image.bin', 'image_sha256'),
                        (folder/'input.bin', 'input_sha256'),
                        (folder/'expected.npz', 'expected_npz_sha256')):
        if sha(path) != manifest[field]:
            raise ValueError(f'fixture payload hash mismatch: {path}')
    route = json.loads((EVIDENCE/'boardless-tiled-host-route.json').read_text())
    for name, expected in route['source_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'routed RTL/source has changed: {name}')
    transcript = args.log.read_text()
    events = [int(n) for n in re.findall(
        rf'{args.name} framed host node (\d+)/{manifest["nodes"]} passed',
        transcript)]
    if events != list(range(1, manifest['nodes'] + 1)):
        raise ValueError('missing, duplicated or out-of-order exact node checks')
    if 'TESTS=1 PASS=1 FAIL=0' not in transcript:
        raise ValueError('framed-host simulation did not pass')
    cases = ET.parse(args.xml).findall('.//testcase')
    if len(cases) != 1 or cases[0].find('failure') is not None or \
       cases[0].find('error') is not None:
        raise ValueError('framed-host JUnit result did not pass')
    sim_ns = Decimal(cases[0].attrib['sim_time_ns'])
    if sim_ns <= 0:
        raise ValueError('invalid simulation duration')
    log_copy = EVIDENCE/f'boardless-host-{args.name}-simulation.txt'
    xml_copy = EVIDENCE/f'boardless-host-{args.name}-results.xml'
    log_copy.write_text('\n'.join(line.rstrip() for line in
                                  transcript.splitlines())+'\n')
    if args.xml.resolve() != xml_copy.resolve():
        shutil.copyfile(args.xml, xml_copy)
    report = {
        'status': 'passed', 'workload': args.name,
        'scope': ('one deterministic INT8 input, exact output for every node '
                  'through framed command parser, tiled engine, DMA and '
                  'behavioral 8-MiB external-memory port'),
        'physical_board': False,
        'nodes_checked': len(events), 'tiles': manifest['tiles'],
        'sim_time_ns_including_host': str(sim_ns),
        'fixture_manifest_sha256': sha(fixture),
        'routed_candidate_bitstream_sha256': route['bitstream_sha256'],
        'source_onnx_sha256': manifest['source_onnx_sha256'],
        'parameter_image_sha256': manifest['image_sha256'],
        'plan_sha256': manifest['plan_sha256'],
        'input_sha256': manifest['input_sha256'],
        'expected_npz_sha256': manifest['expected_npz_sha256'],
        'final_output_sha256': manifest['output_sha256'],
        'simulation_log_sha256': sha(log_copy),
        'results_xml_sha256': sha(xml_copy),
        'random_seed': ET.parse(args.xml).find('.//property[@name="random_seed"]').attrib['value'],
        'limits': ['abstract external memory, not physical SDRAM',
                   'simulation includes host packet activity and random stalls',
                   'no complete-set accuracy or physical FPS/energy claim'],
    }
    path = EVIDENCE/f'boardless-host-{args.name}.json'
    path.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'workload': args.name, 'nodes': len(events),
                      'sim_time_ns': str(sim_ns)}))


if __name__ == '__main__':
    main()
