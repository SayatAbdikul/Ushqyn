#!/usr/bin/env python3
"""Immutable combined exact engine with overflow-safe speculative PW prefetch.

The only RTL delta from combined-next-v1 is the two-hunk prefetch change
independently screened in speculative-v1. No command in this file uses JTAG.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import run_writeback as runner
from chain_resident import native as chain_native
from run_screening import save_json
from variants import ROOT, check_frozen, sha


BASE = ROOT/'work/phase6/experiments-v1/combined-spec-v1'
ENGINE = BASE/'engine.sv'
PARENT = ROOT/'work/phase6/experiments-v1/combined-next-v1/engine.sv'
SPEC_PARENT = ROOT/'work/phase6/experiments-v1/rq-fast-v3/engine.sv'
SPEC = ROOT/'work/phase6/experiments-v1/speculative-v1/engine.sv'

PREFETCH_OLD = """            PW_MAC:if(PW_FETCH_AHEAD && !spatial_dw && col+1<count &&
                !((col[2:0]==7||col+1>=count)&&pw_overflow))begin"""
PREFETCH_NEW = """            // A prefetched input read is side-effect-free. On arithmetic
            // overflow the engine still reports the same failure; any
            // accepted speculative SRAM response is discarded in IDLE.
            PW_MAC:if(PW_FETCH_AHEAD && !spatial_dw && col+1<count)begin"""
FAIL_OLD = '        begin error_code<=code;busy<=0;done<=1;state<=IDLE;end'
FAIL_NEW = '        begin error_code<=code;busy<=0;done<=1;state<=IDLE;pw_prefetch_pending<=0;end'


def _replace_once(source, old, new):
    if source.count(old) != 1 or new in source:
        raise ValueError('speculative prefetch patch site changed')
    return source.replace(old, new, 1)


def add_speculative_prefetch(source):
    return _replace_once(_replace_once(source, PREFETCH_OLD, PREFETCH_NEW), FAIL_OLD, FAIL_NEW)


def prepare():
    check_frozen()
    if add_speculative_prefetch(SPEC_PARENT.read_text()) != SPEC.read_text():
        raise ValueError('speculative-v1 is not the validated two-hunk patch')
    source = add_speculative_prefetch(PARENT.read_text())
    BASE.mkdir(parents=True, exist_ok=True)
    if ENGINE.exists() and ENGINE.read_text() != source:
        raise ValueError('immutable combined-spec-v1 engine differs; choose a new label')
    if not ENGINE.exists():
        ENGINE.write_text(source)
    record = dict(label=BASE.name, parent=str(PARENT.relative_to(ROOT)),
        parent_sha256=sha(PARENT), speculative_parent_sha256=sha(SPEC_PARENT),
        speculative_reference_sha256=sha(SPEC), engine_sha256=sha(ENGINE),
        parameters=json.loads((PARENT.parent/'identity.json').read_text())['parameters'],
        change='only overflow-safe speculative PW prefetch and fail cleanup',
        physical_board=False)
    identity=BASE/'identity.json'
    if identity.exists():
        if json.loads(identity.read_text()) != record:
            raise ValueError('immutable combined-spec-v1 identity changed')
    else:
        save_json(identity, record)
    runner.BASE=BASE
    runner.ENGINE=ENGINE
    return record


def cocotb(stage, module):
    from cocotb.runner import get_runner
    build=BASE/stage
    simulation=get_runner('verilator')
    simulation.build(verilog_sources=runner.sources(True), hdl_toplevel='v2_engine',
        build_dir=build, build_args=['--timing','-Wno-fatal'], timescale=('1ns','1ps'))
    paths=[str(ROOT/p) for p in ('compiler','test/phase6')]+sys.path
    sys.path[:0]=paths
    simulation.test(hdl_toplevel='v2_engine',test_module=module,
        test_dir=ROOT/'test/phase6',build_dir=build,
        results_xml=str(build/'results.xml'),
        extra_env={'PYTHONPATH':os.pathsep.join(paths)})
    cases=ET.parse(build/'results.xml').findall('.//testcase')
    if len(cases)!=1 or any(c.find('failure') is not None or c.find('error') is not None for c in cases):
        raise ValueError(f'{stage} Cocotb regression failed')
    sources={str(p.relative_to(ROOT)):sha(p) for p in runner.sources(True)+[ROOT/f'test/phase6/{module}.py']}
    result=dict(status='passed',physical_board=False,tests=1,sources=sources,
        results_sha256=sha(build/'results.xml'))
    save_json(build/'report.json',result)
    return result


def _write_once(path, content):
    if path.exists() and path.read_text()!=content:
        raise ValueError(f'immutable route source changed: {path}')
    if not path.exists():
        path.write_text(content)


def route_22_5():
    mhz=22.5
    pll=(ROOT/'hardware/phase4_sdram/pll_dma_20.v').read_text()
    pll=pll[pll.index('module '):].replace('.FBDIV_SEL(2), .IDIV_SEL(3)',
                                        '.FBDIV_SEL(4), .IDIV_SEL(5)')
    if '.FBDIV_SEL(4), .IDIV_SEL(5)' not in pll:
        raise ValueError('PLL divisor patch missing')
    _write_once(BASE/'pll.v','// Experimental 22.5 MHz from 27 MHz; unchanged phase selection.\n'+pll)
    host=(ROOT/'hardware/phase5/tiled_host.sv').read_text()
    if host.count('20250000')!=3:
        raise ValueError('UART divider patch site changed')
    _write_once(BASE/'host.sv',host.replace('20250000','22500000'))
    build_script=(ROOT/'hardware/phase6/build.tcl').read_text()
    build_script=build_script.replace(
        'set root [file normalize [file join [file dirname [info script]] ../..]]',
        f'set root {{{ROOT}}}')
    build_script=build_script.replace('hardware/phase4_sdram/pll_dma_20.v',
        str((BASE/'pll.v').relative_to(ROOT)))
    build_script=build_script.replace('hardware/phase5/tiled_host.sv',
        str((BASE/'host.sv').relative_to(ROOT)))
    _write_once(BASE/'build.tcl',build_script)
    build=BASE/'route';build.mkdir(exist_ok=True)
    report=build/'report.json'
    if report.exists():
        old=json.loads(report.read_text())
        if old['engine_sha256']!=sha(ENGINE):
            raise ValueError('route report engine identity mismatch')
        return old
    gowin='/Applications/GowinIDE.app/Contents/Resources/Gowin_EDA/IDE'
    env=dict(os.environ,DYLD_FRAMEWORK_PATH=f'{gowin}/lib',
             DYLD_LIBRARY_PATH=f'{gowin}/lib',PHASE6_ENGINE=str(ENGINE))
    started=time.monotonic()
    with (build/'build.log').open('w') as log:
        completed=subprocess.run([f'{gowin}/bin/gw_sh',str(BASE/'build.tcl')],
            cwd=build,env=env,stdout=log,stderr=subprocess.STDOUT)
    record=dict(status='failed-build',physical_board=False,core_clock_mhz=mhz,
        engine_sha256=sha(ENGINE),build_seconds=time.monotonic()-started,
        sources={str(p.relative_to(ROOT)):sha(p) for p in (
            BASE/'pll.v',BASE/'host.sv',BASE/'build.tcl',
            ROOT/'work/phase4/ip-generated/sdram_controller_hs.ipc',
            ROOT/'rtl/v2/sdram_refresh.sv')},uart_divider=30,pll_vco_mhz=720.0)
    if completed.returncode:
        save_json(report,record)
        raise RuntimeError('Gowin build failed; inspect route/build.log')
    pnr=build/'phase6_tiled_host/impl/pnr';prefix='phase6_tiled_host'
    route_file=pnr/f'{prefix}.rpt.txt';timing=pnr/f'{prefix}_tr_content.html'
    bitstream=pnr/f'{prefix}.fs';tr=timing.read_text();rpt=route_file.read_text()
    resources={}
    for key in ('Logic','Register','BSRAM','DSP','CLS'):
        match=re.search(rf'^\s*{key}\s*\|\s*([\d.]+)/([\d.]+)',rpt,re.M)
        if not match:
            raise ValueError(f'missing routed resource {key}')
        resources[key]=dict(used=float(match[1]),available=float(match[2]))
    fmax=re.search(r'<td>22\.500\(MHz\)</td>\s*<td[^>]*>([\d.]+)\(MHz\)</td>',tr)
    if not fmax:
        raise ValueError('22.5MHz timing constraint missing')
    record.update(resources=resources,routed_core_fmax_mhz=float(fmax[1]),
        bitstream=str(bitstream.relative_to(ROOT)),bitstream_sha256=sha(bitstream),
        route_sha256=sha(route_file),timing_sha256=sha(timing))
    for kind in ('Setup','Hold'):
        match=re.search(rf'Numbers of {kind} Violated Endpoints</td>\s*<td[^>]*>(\d+)</td>',tr)
        if not match:
            raise ValueError(f'missing {kind} endpoint count')
        record[kind.lower()+'_violated_endpoints']=int(match[1])
    if record['routed_core_fmax_mhz']>=mhz and not (record['setup_violated_endpoints'] or record['hold_violated_endpoints']):
        record['status']='passed-route'
    else:
        record['status']='failed-timing'
    save_json(report,record)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','engine','edges','geometry-rejection',
                                         'speculative-safety',
                                         'native','chain-native','route22'))
    args=parser.parse_args()
    identity=prepare()
    if args.stage=='prepare':
        result=identity
    elif args.stage=='engine':
        runner.engine();result=json.loads((BASE/'engine/report.json').read_text())
    elif args.stage=='edges':
        result=cocotb('edges','test_vector_lut_edges')
    elif args.stage=='geometry-rejection':
        result=cocotb('geometry-rejection','test_geometry_rejection')
    elif args.stage=='speculative-safety':
        result=cocotb('speculative-safety','test_speculative_prefetch')
    elif args.stage=='native':
        runner.native();result=json.loads((BASE/'native/report.json').read_text())
    elif args.stage=='chain-native':
        chain_native(BASE.name)
        result=json.loads((BASE.parent/'chain'/f'native-{BASE.name}'/'report.json').read_text())
    else:
        result=route_22_5()
    check_frozen()
    print(json.dumps(dict(stage=args.stage,status=result.get('status','prepared'),
        engine_sha256=identity['engine_sha256']),sort_keys=True),flush=True)


if __name__=='__main__':
    main()
