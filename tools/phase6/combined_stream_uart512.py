#!/usr/bin/env python3
"""Freeze the stream-mask core with the isolated 512-byte UART host bridge.

Boardless preparation, Cocotb integration, and native grouped-model checks
are separate stages. route-inputs emits immutable Gowin sources but does not
launch EDA or touch the board. route runs Gowin but never programs the board.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
import xml.etree.ElementTree as ET

import run_uart_burst_512_integration as integrated
from run_screening import save_json
from variants import ROOT, check_frozen, sha


BASE = ROOT/'work/phase6/experiments-v1/stream-mask-uart512-v1'
PARENT = ROOT/'work/phase6/experiments-v1/stream-mask-v1'
ENGINE = BASE/'engine.sv'
PARSER = ROOT/'rtl/phase6/uart_burst_512_command.sv'
BRIDGE = ROOT/'rtl/phase6/uart_burst_512_bridge.sv'
TOP = ROOT/'hardware/phase6/uart_burst_512_tiled_host.sv'
BUILD = ROOT/'hardware/phase6/build_uart_burst_512.tcl'
HOST_CLIENT = ROOT/'tools/phase6/uart_burst_512.py'


def clock_label(mhz):
    return str(float(mhz)).replace('.', 'p')


def immutable_write(path, content):
    if path.exists() and path.read_text()!=content:
        raise ValueError(f'immutable candidate artifact differs: {path}')
    if not path.exists():
        path.write_text(content)


def prepare():
    check_frozen()
    parent_identity=json.loads((PARENT/'identity.json').read_text())
    source=PARENT/'engine.sv'
    if parent_identity['engine_sha256']!=sha(source):
        raise ValueError('stream-mask engine source changed')
    BASE.mkdir(parents=True,exist_ok=True)
    immutable_write(ENGINE,source.read_text())
    record=dict(label=BASE.name,parent=str(source.relative_to(ROOT)),
        parent_sha256=sha(source),engine_sha256=sha(ENGINE),
        parser_sha256=sha(PARSER),bridge_sha256=sha(BRIDGE),
        top_sha256=sha(TOP),build_template_sha256=sha(BUILD),
        host_client_sha256=sha(HOST_CLIENT),
        change='byte-identical stream-mask core with 512-byte CRC-committed UART burst bridge',
        physical_board=False)
    identity=BASE/'identity.json'
    if identity.exists():
        if json.loads(identity.read_text())!=record:
            raise ValueError('combined stream UART512 identity changed')
    else: save_json(identity,record)
    return record


def integration():
    identity=prepare()
    integrated.BUILD=BASE/'integration'
    integrated.ENGINE=ENGINE
    integrated.SOURCES[2]=ENGINE
    integrated.main()
    xml=BASE/'integration/results.xml'
    cases=ET.parse(xml).findall('.//testcase')
    if len(cases)!=2 or any(c.find('failure') is not None or
                            c.find('error') is not None for c in cases):
        raise ValueError('combined 512-byte UART Cocotb integration failed')
    tests=[ROOT/'test/phase4/test_tiled_host.py',
           ROOT/'test/phase6/test_uart_burst_512_bridge.py']
    report=dict(status='passed',physical_board=False,
        engine_sha256=identity['engine_sha256'],tests=len(cases),
        cases=[case.attrib['name'] for case in cases],
        sources={str(p.relative_to(ROOT)):sha(p) for p in integrated.SOURCES+tests},
        results_sha256=sha(xml))
    save_json(BASE/'integration/report.json',report)
    check_frozen()
    return report


def native(tail_prefetch=False):
    identity=prepare()
    reference=json.loads((PARENT/'native/report.json').read_text())
    binary=PARENT/'native/Vv2_tiled_host_bridge'
    if reference['status']!='passed' or reference['executable_sha256']!=sha(binary):
        raise ValueError('stream-mask native binary changed')
    if reference['sources'][str((PARENT/'engine.sv').relative_to(ROOT))]!=identity['engine_sha256']:
        raise ValueError('native executable was built from a different engine')
    for file,digest in reference['sources'].items():
        if sha(ROOT/file)!=digest:
            raise ValueError(f'native source changed: {file}')
    manifest_path=ROOT/'work/phase6'/(
        'followup_graph_tail_prefetch' if tail_prefetch else 'followup_graph')/'fixtures.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest['status']!='passed-replay':
        raise ValueError('grouped fixture manifest not passed')
    target=BASE/('native-grouped-tail-prefetch' if tail_prefetch else 'native-grouped')
    target.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',physical_board=False,
        engine_sha256=identity['engine_sha256'],
        executable_sha256=sha(binary),fixture_manifest_sha256=sha(manifest_path),
        results=[])
    for row in manifest['fixtures']:
        fixture_name=row['name'] if tail_prefetch else row['label']
        folder=manifest_path.parent/'fixtures'/fixture_name
        for file,digest in row['files'].items():
            if sha(folder/file)!=digest:
                raise ValueError(f'grouped fixture changed: {fixture_name}/{file}')
        for seed in (0,6063):
            path=target/f'{fixture_name}-s{seed}.json'
            subprocess.run([str(binary),str(folder),str(seed),str(path)],check=True)
            result=json.loads(path.read_text())
            if result['status']!='passed':
                raise ValueError(f'grouped native failure: {fixture_name}, {seed}')
            report['results'].append(dict(fixture=fixture_name,seed=seed,
                result_sha256=sha(path),elapsed_cycles=result['elapsed_cycles'],
                engine_cycles=result['engine_cycles'],dma_cycles=result['dma_cycles'],
                tensor_checks=result['tensor_checks']))
            save_json(target/'report.json',report)
    report['status']='passed'
    save_json(target/'report.json',report)
    check_frozen()
    return report


def route_inputs(mhz):
    identity=prepare()
    if mhz not in (22.5,24,27):
        raise ValueError('unsupported core clock')
    integration_report=json.loads((BASE/'integration/report.json').read_text())
    native_report=json.loads((BASE/'native-grouped/report.json').read_text())
    tail_report=json.loads((BASE/'native-grouped-tail-prefetch/report.json').read_text())
    if (integration_report['status']!='passed' or native_report['status']!='passed' or
            tail_report['status']!='passed' or
            integration_report['engine_sha256']!=identity['engine_sha256'] or
            native_report['engine_sha256']!=identity['engine_sha256'] or
            tail_report['engine_sha256']!=identity['engine_sha256']):
        raise ValueError('combined boardless gates incomplete')
    mhz_name=clock_label(mhz)
    parent_pll=(ROOT/'work/phase6/experiments-v1/combined-spec-scalar-uart-v1/pll.v').read_text()
    if parent_pll.count('.FBDIV_SEL(4), .IDIV_SEL(5)')!=1:
        raise ValueError('PLL source changed')
    div=(4,5) if mhz==22.5 else (7,8) if mhz==24 else (7,7)
    pll=parent_pll.replace('.FBDIV_SEL(4), .IDIV_SEL(5)',
        f'.FBDIV_SEL({div[0]}), .IDIV_SEL({div[1]})').replace(
        '// Experimental 22.5 MHz from 27 MHz.',
        f'// Experimental {mhz:g} MHz from 27 MHz.')
    host=TOP.read_text()
    if host.count('20250000')!=3:
        raise ValueError('UART/SDRAM clock patch sites changed')
    host=host.replace('20250000',str(int(mhz*1000000)))
    script=BUILD.read_text()
    original_root='set root [file normalize [file join [file dirname [info script]] ../..]]'
    if script.count(original_root)!=1:
        raise ValueError('Gowin build root patch site changed')
    script=script.replace(original_root,f'set root {{{ROOT}}}')
    pll_path=BASE/f'pll{mhz_name}.v'
    host_path=BASE/f'host{mhz_name}.sv'
    script_path=BASE/f'build{mhz_name}.tcl'
    script=script.replace('hardware/phase4_sdram/pll_dma_20.v',
                          str(pll_path.relative_to(ROOT)))
    script=script.replace('hardware/phase6/uart_burst_512_tiled_host.sv',
                          str(host_path.relative_to(ROOT)))
    immutable_write(pll_path,pll)
    immutable_write(host_path,host)
    immutable_write(script_path,script)
    record=dict(status='prepared-route-inputs',physical_board=False,
        core_clock_mhz=mhz,engine_sha256=identity['engine_sha256'],
        sources={str(p.relative_to(ROOT)):sha(p) for p in
                 (ENGINE,pll_path,host_path,script_path,PARSER,BRIDGE)},
        build_script=str(script_path.relative_to(ROOT)),
        gowin_project='phase6_uart_burst_512')
    save_json(BASE/f'route-inputs-{mhz_name}.json',record)
    check_frozen()
    return record


def route(mhz):
    """Build, parse, and source-audit one integrated UART512 Gowin image."""
    inputs = route_inputs(mhz)
    label = clock_label(mhz)
    build = BASE/f'route{label}'
    build.mkdir(exist_ok=True)
    report_path = build/'report.json'
    if report_path.exists():
        old = json.loads(report_path.read_text())
        if old['status'] != 'failed-build':
            if (old['engine_sha256'] != inputs['engine_sha256'] or
                    old.get('route_inputs_sha256') != sha(BASE/f'route-inputs-{label}.json')):
                raise ValueError('immutable UART512 route identity changed')
            for file, digest in old.get('sources', {}).items():
                if sha(ROOT/file) != digest:
                    raise ValueError(f'immutable routed source changed: {file}')
            return old
        # Preserve a failed tool-startup attempt before an escalated retry.
        attempt = time.time_ns()
        report_path.rename(build/f'failed-build-{attempt}.json')
        if (build/'build.log').exists():
            (build/'build.log').rename(build/f'failed-build-{attempt}.log')

    gowin = '/Applications/GowinIDE.app/Contents/Resources/Gowin_EDA/IDE'
    env = dict(os.environ, DYLD_FRAMEWORK_PATH=f'{gowin}/lib',
               DYLD_LIBRARY_PATH=f'{gowin}/lib', PHASE6_ENGINE=str(ENGINE))
    started = time.monotonic()
    with (build/'build.log').open('w') as log:
        process = subprocess.run([f'{gowin}/bin/gw_sh',
            str(ROOT/inputs['build_script'])], cwd=build, env=env,
            stdout=log, stderr=subprocess.STDOUT)
    result = dict(status='failed-build', physical_board=False,
        source_label=BASE.name, core_clock_mhz=mhz,
        engine_sha256=inputs['engine_sha256'],
        route_inputs_sha256=sha(BASE/f'route-inputs-{label}.json'),
        build_seconds=time.monotonic()-started,
        build_log_sha256=sha(build/'build.log'))
    if process.returncode:
        save_json(report_path,result)
        raise RuntimeError(f'UART512 Gowin build failed; inspect {build}/build.log')

    project = build/'phase6_uart_burst_512/phase6_uart_burst_512.gprj'
    tree = ET.parse(project)
    selected = {Path(node.attrib['path']).resolve()
                for node in tree.findall('.//File')
                if node.attrib.get('enable') == '1'}
    generated = [BASE/f'pll{label}.v', BASE/f'host{label}.sv']
    static = [ROOT/path for path in (
        'rtl/v2/target_pkg.sv', 'rtl/v2/requantizer.sv',
        'rtl/v2/scratchpad.sv', 'rtl/v2/tile_dma.sv',
        'rtl/v2/tiled_core.sv', 'rtl/v2/tile_sequencer.sv',
        'rtl/v2/hs_sdram_burst_port.sv', 'rtl/v2/sdram_refresh.sv',
        'rtl/v2/uart_rx.sv', 'rtl/v2/uart_tx.sv',
        'work/phase4/ip-generated/sdram_controller_hs.v',
        'hardware/phase4_sdram/tiled_host.cst',
        'hardware/phase4_sdram/bist.sdc')]
    expected = {path.resolve() for path in
                (generated + static + [ENGINE,PARSER,BRIDGE])}
    if selected != expected:
        missing = sorted(str(p) for p in expected-selected)
        extra = sorted(str(p) for p in selected-expected)
        raise ValueError(f'Gowin project source coverage mismatch: missing={missing}, extra={extra}')
    script = ROOT/inputs['build_script']
    source_paths = sorted(expected | {script.resolve()})
    sources = {str(path.relative_to(ROOT)):sha(path) for path in source_paths}
    for file,digest in inputs['sources'].items():
        if sources.get(file) != digest:
            raise ValueError(f'generated UART512 route source changed: {file}')
    result.update(sources=sources,
        source_coverage=dict(status='passed', project_files=len(selected),
            hashed_sources=len(sources), gowin_project=str(project.relative_to(ROOT)),
            gowin_project_sha256=sha(project),
            generated_inputs=sorted(inputs['sources'])))

    pnr = build/'phase6_uart_burst_512/impl/pnr'
    prefix = 'phase6_uart_burst_512'
    route_file = pnr/f'{prefix}.rpt.txt'
    timing_file = pnr/f'{prefix}_tr_content.html'
    bitstream = pnr/f'{prefix}.fs'
    routed = route_file.read_text()
    timing = timing_file.read_text()
    resources = {}
    for key in ('Logic','Register','BSRAM','DSP','CLS'):
        match = re.search(rf'^\s*{key}\s*\|\s*([\d.]+)/([\d.]+)',routed,re.M)
        if not match:
            raise ValueError(f'missing UART512 route resource: {key}')
        resources[key] = dict(used=float(match[1]),available=float(match[2]))
    fmax = re.search(rf'<td>{mhz:.3f}\(MHz\)</td>\s*<td[^>]*>([\d.]+)\(MHz\)</td>',timing)
    if not fmax:
        raise ValueError('UART512 core clock constraint absent')
    result.update(resources=resources,routed_core_fmax_mhz=float(fmax[1]),
        bitstream=str(bitstream.relative_to(ROOT)),
        bitstream_sha256=sha(bitstream),route_sha256=sha(route_file),
        timing_sha256=sha(timing_file))
    for kind in ('Setup','Hold'):
        match = re.search(rf'Numbers of {kind} Violated Endpoints</td>\s*<td[^>]*>(\d+)</td>',timing)
        if not match:
            raise ValueError(f'missing UART512 {kind} endpoint count')
        result[kind.lower()+'_violated_endpoints'] = int(match[1])
    result['status'] = ('passed-route'
        if (result['routed_core_fmax_mhz'] >= mhz and
            result['setup_violated_endpoints'] == 0 and
            result['hold_violated_endpoints'] == 0 and
            all(item['used'] <= item['available'] for item in resources.values()))
        else 'failed-timing')
    save_json(report_path,result)
    check_frozen()
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','integration','native','native-tail','route-inputs','route'))
    parser.add_argument('--mhz',type=float,default=22.5)
    args=parser.parse_args()
    result=(prepare() if args.stage=='prepare' else
            integration() if args.stage=='integration' else
            native() if args.stage=='native' else
            native(True) if args.stage=='native-tail' else
            route_inputs(args.mhz) if args.stage=='route-inputs' else route(args.mhz))
    print(json.dumps(dict(stage=args.stage,status=result.get('status','prepared'),
        engine_sha256=result['engine_sha256']),sort_keys=True),flush=True)
