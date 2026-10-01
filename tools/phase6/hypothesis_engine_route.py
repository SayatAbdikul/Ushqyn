#!/usr/bin/env python3
"""Isolated 27 MHz route of the native-tested cached-weight co-issue engine.

The selected design is audited before use. Only its engine source changes;
host, PLL, UART, controller IP, pin constraints and timing constraints retain
their sealed hashes. No programmer or serial device is accessed.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
import xml.etree.ElementTree as ET

from matched_campaign import ROOT, POOL, REFERENCE, hardware_contract, read, save, sha, verify

BASE = ROOT / 'work/phase6/engine-candidate-rtl-v2'
ENGINE = BASE / 'engine.sv'
GOWIN = Path('/Applications/GowinIDE.app/Contents/Resources/Gowin_EDA/IDE')


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def prepare():
    hardware = hardware_contract()
    identity = read(BASE / 'identity.json')
    native = read(BASE / 'native/report.json')
    edges = read(BASE / 'edges/report.json')
    if native['status'] != 'passed' or edges['status'] != 'passed':
        raise ValueError('native and focused edge proof required')
    if len(native['results']) != 8 or edges['tests'] != 3:
        raise ValueError('incomplete paired native or edge coverage')
    for proof in (native, edges):
        verify(ROOT, proof['source_sha256'])
    if sha(ENGINE) != native['engine_sha256'] or sha(ENGINE) != identity['engine_sha256']:
        raise ValueError('engine differs from native-tested source')
    if sha(POOL / 'engine.sv') != native['baseline_engine_sha256']:
        raise ValueError('baseline engine differs from matched control')
    expected = {(m, s, n) for m in ('kws', 'vww') for s in ('pinned', 'stress') for n in (0, 6063)}
    if {(r['model'], r['sample'], r['stall_seed']) for r in native['results']} != expected:
        raise ValueError('native matrix changed')
    for row in native['results']:
        for key in ('baseline', 'candidate'):
            if row[key]['status'] != 'passed':
                raise ValueError('native output mismatch')
    coverage = read(REFERENCE / 'plan.json')['pool27_source_coverage']
    old_sources = dict(coverage['source_sha256'])
    old_recipe = POOL / 'build27.tcl'
    old_sources.pop(rel(old_recipe))
    expected_sources = dict(old_sources)
    expected_sources.pop(rel(POOL / 'engine.sv'))
    expected_sources[rel(ENGINE)] = sha(ENGINE)
    if len(expected_sources) != 18:
        raise ValueError('selected source coverage changed')
    verify(ROOT, expected_sources)

    # A root relocation is the only Tcl change. PHASE6_ENGINE selects the
    # separately tested engine without modifying any historical source.
    text = old_recipe.read_text()
    old_root = 'set root {' + hardware['original_checkout'] + '}'
    if text.count(old_root) != 1:
        raise ValueError('route recipe root changed')
    text = text.replace(old_root, 'set root {' + str(ROOT) + '}')
    script = BASE / 'build27.tcl'
    if script.exists() and script.read_text() != text:
        raise ValueError('existing candidate route recipe differs')
    if not script.exists():
        script.write_text(text)
    build = BASE / 'route27'
    build.mkdir(exist_ok=True)
    sources = dict(expected_sources)
    sources[rel(script)] = sha(script)
    sources[rel(Path(__file__))] = sha(Path(__file__))
    evidence = {rel(BASE / name): sha(BASE / name) for name in
                ('identity.json', 'native/report.json', 'edges/report.json')}
    manifest = dict(status='prepared', physical_board=False, core_clock_mhz=27,
        engine=rel(ENGINE), engine_sha256=sha(ENGINE), source_sha256=sources,
        sources=sources, evidence_sha256=evidence, expected_gprj_sources=expected_sources,
        selected_parent=dict(engine=rel(POOL / 'engine.sv'),
            engine_sha256=sha(POOL / 'engine.sv'), bitstream=hardware['image'],
            route_report=rel(POOL / 'route27/report.json'),
            route_report_sha256=sha(POOL / 'route27/report.json'),
            source_sha256=old_sources),
        route_recipe=rel(script), route_recipe_sha256=sha(script),
        recipe_changes=['checkout relocation', 'PHASE6_ENGINE selects tested candidate engine'],
        uart_divider=36, uart_baud=750000, uart_burst_bytes=256, pll_vco_mhz=864,
        tool_executable=str(GOWIN / 'bin/gw_sh'),
        tool_executable_sha256=sha(GOWIN / 'bin/gw_sh'),
        timing_boundary=hardware['timing_boundary'])
    manifest_path = build / 'inputs.json'
    if manifest_path.exists() and read(manifest_path) != manifest:
        raise ValueError('immutable route inputs changed')
    if not manifest_path.exists():
        save(manifest_path, manifest)
    return build, manifest


def finish(build, manifest, process_returncode, seconds):
    verify(ROOT, manifest['source_sha256'])
    verify(ROOT, manifest['evidence_sha256'])
    result = dict(manifest, status='failed-build', build_seconds=seconds,
        process_returncode=process_returncode,
        inputs=rel(build / 'inputs.json'), inputs_sha256=sha(build / 'inputs.json'),
        build_log=rel(build / 'route.log'), build_log_sha256=sha(build / 'route.log'))
    project = build / 'phase6_uart_burst/phase6_uart_burst.gprj'
    if project.exists():
        active = []
        for node in ET.parse(project).findall('.//File'):
            if node.attrib.get('enable') == '1':
                path = Path(node.attrib['path'])
                if not path.is_absolute():
                    path = project.parent / path
                active.append(rel(path))
        expected = manifest['expected_gprj_sources']
        if len(active) != 18 or len(set(active)) != 18 or set(active) != set(expected):
            raise ValueError('active candidate Gowin project differs from expected 18-source set')
        verify(ROOT, expected)
        result['active_gprj_source_coverage'] = dict(
            gowin_project=rel(project), gowin_project_sha256=sha(project),
            active_count=18, source_sha256=expected,
            unchanged_parent_sources=17, replaced_source=rel(POOL / 'engine.sv'),
            replacement_source=rel(ENGINE))
    else:
        result['active_gprj_source_coverage'] = None
    if process_returncode:
        save(build / 'report.json', result)
        return result
    if result['active_gprj_source_coverage'] is None:
        raise ValueError('successful route omitted Gowin project')
    pnr = project.parent / 'impl/pnr'
    routed_file = pnr / 'phase6_uart_burst.rpt.txt'
    timing_file = pnr / 'phase6_uart_burst_tr_content.html'
    bitstream = pnr / 'phase6_uart_burst.fs'
    routed = routed_file.read_text()
    timing = timing_file.read_text()
    resources = {}
    for key in ('Logic', 'Register', 'BSRAM', 'DSP', 'CLS'):
        match = re.search(rf'^\s*{key}\s*\|\s*([\d.]+)/([\d.]+)', routed, re.M)
        if not match:
            raise ValueError(f'missing routed resource {key}')
        resources[key] = dict(used=float(match[1]), available=float(match[2]))
    fmax = re.search(r'<td>27\.000\(MHz\)</td>\s*<td[^>]*>([\d.]+)\(MHz\)</td>', timing)
    if not fmax:
        raise ValueError('27 MHz core timing result absent')
    result.update(resources=resources, routed_core_fmax_mhz=float(fmax[1]),
        bitstream=rel(bitstream), bitstream_sha256=sha(bitstream),
        route_report=rel(routed_file), route_sha256=sha(routed_file),
        timing_report=rel(timing_file), timing_sha256=sha(timing_file))
    for kind in ('Setup', 'Hold'):
        match = re.search(rf'Numbers of {kind} Violated Endpoints</td>\s*<td[^>]*>(\d+)</td>', timing)
        if not match:
            raise ValueError(f'missing {kind} endpoint count')
        result[kind.lower() + '_violated_endpoints'] = int(match[1])
    result['status'] = 'passed-route' if (
        result['routed_core_fmax_mhz'] >= 27 and
        result['setup_violated_endpoints'] == result['hold_violated_endpoints'] == 0 and
        all(r['used'] <= r['available'] for r in resources.values())) else 'failed-timing'
    save(build / 'report.json', result)
    return result


def run():
    build, manifest = prepare()
    if (build / 'report.json').exists():
        raise ValueError('route report already exists; preserve immutable attempt')
    environment = dict(os.environ, DYLD_FRAMEWORK_PATH=str(GOWIN / 'lib'),
        DYLD_LIBRARY_PATH=str(GOWIN / 'lib'), PHASE6_ENGINE=str(ENGINE))
    start = time.monotonic()
    with (build / 'route.log').open('w') as log:
        process = subprocess.run([str(GOWIN / 'bin/gw_sh'), str(BASE / 'build27.tcl')],
            cwd=build, env=environment, stdout=log, stderr=subprocess.STDOUT)
    result = finish(build, manifest, process.returncode, time.monotonic() - start)
    print(json.dumps({k: result.get(k) for k in ('status', 'build_seconds',
        'routed_core_fmax_mhz', 'resources', 'bitstream', 'bitstream_sha256')}, indent=2))
    if result['status'] != 'passed-route':
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'run'))
    args = parser.parse_args()
    if args.stage == 'prepare':
        build, _ = prepare()
        print(rel(build / 'inputs.json'))
    else:
        run()
