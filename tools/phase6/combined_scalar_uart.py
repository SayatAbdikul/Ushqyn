#!/usr/bin/env python3
"""Prepare the combined-spec scalar engine with the isolated UART burst bridge.

Run the integrated route only after the separate UART transfer screen passes.
No stage in this driver programs the FPGA.
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

import combined_spec as engine_runner
import run_uart_burst_integration as burst_integration
from run_screening import save_json
from variants import ROOT, check_frozen, sha


BASE=ROOT/'work/phase6/experiments-v1/combined-spec-scalar-uart-v1'
ENGINE=BASE/'engine.sv'
PARENT=ROOT/'work/phase6/experiments-v1/combined-spec-scalar-v1/engine.sv'


def prepare():
    check_frozen()
    source=PARENT.read_text()
    BASE.mkdir(parents=True,exist_ok=True)
    if ENGINE.exists() and ENGINE.read_text()!=source:
        raise ValueError('immutable burst engine differs; choose a new label')
    if not ENGINE.exists():ENGINE.write_text(source)
    identity=dict(label=BASE.name,parent=str(PARENT.relative_to(ROOT)),
        parent_sha256=sha(PARENT),engine_sha256=sha(ENGINE),
        parameters=json.loads((PARENT.parent/'identity.json').read_text())['parameters'],
        change='byte-identical combined-spec scalar engine with UART burst bridge at 22.5 MHz',
        physical_board=False)
    record=BASE/'identity.json'
    if record.exists():
        if json.loads(record.read_text())!=identity:
            raise ValueError('immutable UART integration identity changed')
    else:save_json(record,identity)
    engine_runner.BASE=BASE;engine_runner.ENGINE=ENGINE
    engine_runner.runner.BASE=BASE;engine_runner.runner.ENGINE=ENGINE
    return identity


def integration():
    burst_integration.BUILD=BASE/'integration'
    burst_integration.ENGINE=ENGINE
    burst_integration.SOURCES[2]=ENGINE
    burst_integration.main()
    cases=ET.parse(BASE/'integration/results.xml').findall('.//testcase')
    if len(cases)!=2 or any(c.find('failure') is not None or c.find('error') is not None for c in cases):
        raise ValueError('integrated UART Cocotb regression failed')
    sources={str(p.relative_to(ROOT)):sha(p) for p in burst_integration.SOURCES}
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in (
        ROOT/'test/phase4/test_tiled_host.py',ROOT/'test/phase6/test_uart_burst_bridge.py')})
    result=dict(status='passed',physical_board=False,tests=2,sources=sources,
        results_sha256=sha(BASE/'integration/results.xml'))
    save_json(BASE/'integration/report.json',result)
    return result


def _write_once(path,content):
    if path.exists() and path.read_text()!=content:
        raise ValueError(f'immutable UART route source changed: {path}')
    if not path.exists():path.write_text(content)


def route_22_5():
    parent_route=json.loads((PARENT.parent/'route/report.json').read_text())
    if parent_route['status']!='passed-route' or parent_route['core_clock_mhz']!=22.5:
        raise ValueError('parent scalar 22.5 MHz route has not passed')
    if json.loads((BASE/'integration/report.json').read_text())['status']!='passed':
        raise ValueError('combined UART RTL integration has not passed')
    pll=(ROOT/'hardware/phase4_sdram/pll_dma_20.v').read_text()
    pll=pll[pll.index('module '):].replace('.FBDIV_SEL(2), .IDIV_SEL(3)',
                                        '.FBDIV_SEL(4), .IDIV_SEL(5)')
    if '.FBDIV_SEL(4), .IDIV_SEL(5)' not in pll:
        raise ValueError('PLL divisor patch missing')
    _write_once(BASE/'pll.v','// Experimental 22.5 MHz from 27 MHz.\n'+pll)
    host=(ROOT/'hardware/phase6/uart_burst_tiled_host.sv').read_text()
    if host.count('20250000')!=3:
        raise ValueError('UART burst clock patch site changed')
    _write_once(BASE/'host.sv',host.replace('20250000','22500000'))
    script=(ROOT/'hardware/phase6/build_uart_burst.tcl').read_text()
    script=script.replace('set root [file normalize [file join [file dirname [info script]] ../..]]',
                          f'set root {{{ROOT}}}')
    script=script.replace('hardware/phase4_sdram/pll_dma_20.v',
                          str((BASE/'pll.v').relative_to(ROOT)))
    script=script.replace('hardware/phase6/uart_burst_tiled_host.sv',
                          str((BASE/'host.sv').relative_to(ROOT)))
    _write_once(BASE/'build.tcl',script)
    build=BASE/'route';build.mkdir(exist_ok=True)
    report=build/'report.json'
    if report.exists():
        old=json.loads(report.read_text())
        if old['engine_sha256']!=sha(ENGINE):
            raise ValueError('UART route engine identity mismatch')
        return old
    gowin='/Applications/GowinIDE.app/Contents/Resources/Gowin_EDA/IDE'
    env=dict(os.environ,DYLD_FRAMEWORK_PATH=f'{gowin}/lib',
             DYLD_LIBRARY_PATH=f'{gowin}/lib',PHASE6_ENGINE=str(ENGINE))
    started=time.monotonic()
    with (build/'build.log').open('w') as log:
        completed=subprocess.run([f'{gowin}/bin/gw_sh',str(BASE/'build.tcl')],
            cwd=build,env=env,stdout=log,stderr=subprocess.STDOUT)
    sources=[BASE/'pll.v',BASE/'host.sv',BASE/'build.tcl',
        ROOT/'rtl/phase6/uart_burst_command.sv',ROOT/'rtl/phase6/uart_burst_bridge.sv',
        ROOT/'work/phase4/ip-generated/sdram_controller_hs.ipc',ROOT/'rtl/v2/sdram_refresh.sv']
    record=dict(status='failed-build',physical_board=False,core_clock_mhz=22.5,
        engine_sha256=sha(ENGINE),build_seconds=time.monotonic()-started,
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        uart_divider=30,pll_vco_mhz=720.0,bridge='uart-burst')
    if completed.returncode:
        save_json(report,record)
        raise RuntimeError('Gowin UART burst build failed; inspect route/build.log')
    pnr=build/'phase6_uart_burst/impl/pnr';prefix='phase6_uart_burst'
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
        raise ValueError('22.5MHz UART timing constraint missing')
    record.update(resources=resources,routed_core_fmax_mhz=float(fmax[1]),
        bitstream=str(bitstream.relative_to(ROOT)),bitstream_sha256=sha(bitstream),
        route_sha256=sha(route_file),timing_sha256=sha(timing))
    for kind in ('Setup','Hold'):
        match=re.search(rf'Numbers of {kind} Violated Endpoints</td>\s*<td[^>]*>(\d+)</td>',tr)
        if not match:
            raise ValueError(f'missing {kind} endpoint count')
        record[kind.lower()+'_violated_endpoints']=int(match[1])
    record['status']='passed-route' if (record['routed_core_fmax_mhz']>=22.5 and
        not record['setup_violated_endpoints'] and not record['hold_violated_endpoints']) else 'failed-timing'
    save_json(report,record)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','engine','edges','native','integration','route22'))
    args=parser.parse_args();identity=prepare()
    if args.stage=='prepare':result=identity
    elif args.stage=='engine':
        engine_runner.runner.engine();result=json.loads((BASE/'engine/report.json').read_text())
    elif args.stage=='edges':result=engine_runner.cocotb('edges','test_experiment_edges')
    elif args.stage=='native':
        engine_runner.runner.native();result=json.loads((BASE/'native/report.json').read_text())
    elif args.stage=='integration':result=integration()
    else:result=route_22_5()
    check_frozen()
    print(json.dumps(dict(stage=args.stage,status=result.get('status','prepared'),
        engine_sha256=identity['engine_sha256']),sort_keys=True),flush=True)


if __name__=='__main__':main()
