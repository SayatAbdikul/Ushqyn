#!/usr/bin/env python3
"""Isolated direct-writeback candidate. These stages never access the board."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from variants import ROOT, check_frozen, sha
from run_screening import save_json, verify_files

BASE = ROOT / 'work/phase6/writeback-v1'
ENGINE = ROOT / 'rtl/phase6/writeback_engine.sv'


def sources(engine_only=False):
    result = [ROOT / 'rtl/v2/target_pkg.sv', ROOT / 'rtl/v2/requantizer.sv', ENGINE]
    if not engine_only:
        result += [ROOT / 'rtl/v2' / name for name in ('scratchpad.sv', 'tile_dma.sv',
            'tiled_core.sv', 'command.sv', 'tile_sequencer.sv', 'tiled_host_bridge.sv')]
    return result


def engine():
    from cocotb.runner import get_runner
    build = BASE / 'engine'
    runner = get_runner('verilator')
    runner.build(verilog_sources=sources(True), hdl_toplevel='v2_engine',
        build_dir=build, build_args=['--timing', '-Wno-fatal'], timescale=('1ns', '1ps'))
    paths = [str(ROOT / p) for p in ('compiler', 'test/phase2', 'test/phase6')] + sys.path
    sys.path[:] = paths
    runner.test(hdl_toplevel='v2_engine', test_module=['test_engine', 'test_cache', 'test_writeback'],
        test_dir=ROOT / 'test/phase6', build_dir=build, results_xml=str(build / 'results.xml'),
        extra_env={'PYTHONPATH': os.pathsep.join(paths)})
    cases = ET.parse(build / 'results.xml').findall('.//testcase')
    if len(cases) != 3 or any(c.find('failure') is not None or c.find('error') is not None for c in cases):
        raise ValueError('writeback engine regression failed')
    save_json(build / 'report.json', {'status': 'passed', 'tests': len(cases),
        'sources': {str(p.relative_to(ROOT)): sha(p) for p in sources(True) +
            [ROOT / 'test/phase2/test_engine.py', ROOT / 'test/phase6/test_cache.py',
             ROOT / 'test/phase6/test_writeback.py']}, 'results_sha256': sha(build / 'results.xml')})


def native():
    build = BASE / 'native'; build.mkdir(parents=True, exist_ok=True)
    harness = ROOT / 'test/phase6/native.cpp'
    inputs = sources()
    with (build / 'build.log').open('w') as log:
        subprocess.run(['verilator', '--cc', '--exe', '--build', '-j', '2', '--public-flat-rw',
            '-Wno-fatal', '--top-module', 'v2_tiled_host_bridge', '--Mdir', str(build),
            *map(str, inputs), str(harness)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = {'status': 'running', 'physical_board': False,
              'sources': {str(p.relative_to(ROOT)): sha(p) for p in inputs + [harness]}, 'results': []}
    fixtures = json.loads((ROOT / 'work/phase6/optimization/fixtures.json').read_text())['fixtures']
    # Both models, pinned and stress inputs; overlap and serialized timings.
    selected = [f for f in fixtures if '-resident-half-' in f['name']]
    assert len(selected) == 12
    save_json(build / 'report.json', report)
    for fixture in selected:
        directory = ROOT / 'work/phase6/optimization/fixtures' / fixture['name']
        verify_files(directory, fixture['files'])
        for seed in (0, 6063):
            output = build / f'{fixture["name"]}-s{seed}.json'
            start = time.monotonic()
            subprocess.run([str(build / 'Vv2_tiled_host_bridge'), str(directory), str(seed), str(output)], check=True)
            row = json.loads(output.read_text())
            row.update(fixture=fixture['name'], fixture_files=fixture['files'], simulation_seconds=time.monotonic()-start)
            report['results'].append(row); save_json(build / 'report.json', report)
    report['status'] = 'passed'; save_json(build / 'report.json', report)


def route():
    build = BASE / 'route'; build.mkdir(parents=True, exist_ok=True)
    gowin = '/Applications/GowinIDE.app/Contents/Resources/Gowin_EDA/IDE'
    script = ROOT / 'hardware/phase6/build.tcl'
    env = dict(os.environ, DYLD_FRAMEWORK_PATH=f'{gowin}/lib', DYLD_LIBRARY_PATH=f'{gowin}/lib', PHASE6_ENGINE=str(ENGINE))
    start = time.monotonic()
    with (build / 'build.log').open('w') as log:
        subprocess.run([f'{gowin}/bin/gw_sh', str(script)], cwd=build, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    pnr = build / 'phase6_tiled_host/impl/pnr'; prefix = 'phase6_tiled_host'
    route_path = pnr / (prefix + '.rpt.txt'); timing = pnr / (prefix + '_tr_content.html')
    bitstream = pnr / (prefix + '.fs'); text = route_path.read_text(); tr = timing.read_text()
    record = {'status': 'failed-timing', 'physical_board': False, 'resources': {},
              'engine_sha256': sha(ENGINE), 'build_tcl_sha256': sha(script),
              'build_seconds': time.monotonic()-start, 'bitstream': str(bitstream.relative_to(ROOT)),
              'bitstream_sha256': sha(bitstream), 'core_clock_mhz': 20.25,
              'route_sha256': sha(route_path), 'timing_sha256': sha(timing)}
    for key in ('Logic', 'Register', 'BSRAM', 'DSP', 'CLS'):
        m = re.search(rf'^\s*{key}\s*\|\s*([\d.]+)/([\d.]+)', text, re.M)
        if not m: raise ValueError('missing routed resource: ' + key)
        record['resources'][key] = {'used': float(m[1]), 'available': float(m[2])}
    fmax = re.search(r'<td>20\.250\(MHz\)</td>\s*<td[^>]*>([\d.]+)\(MHz\)</td>', tr)
    if not fmax: raise ValueError('missing core Fmax')
    record['routed_core_fmax_mhz'] = float(fmax[1])
    for kind in ('Setup', 'Hold'):
        count = re.search(rf'Numbers of {kind} Violated Endpoints</td>\s*<td[^>]*>(\d+)</td>', tr)
        if not count: raise ValueError('missing timing endpoint count')
        record[kind.lower() + '_violated_endpoints'] = int(count[1])
    if record['routed_core_fmax_mhz'] >= 20.25 and not (record['setup_violated_endpoints'] or record['hold_violated_endpoints']):
        record['status'] = 'passed-route'
    save_json(build / 'report.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('engine', 'native', 'route'))
    args = parser.parse_args()
    check_frozen()
    {'engine': engine, 'native': native, 'route': route}[args.stage]()
    check_frozen()
