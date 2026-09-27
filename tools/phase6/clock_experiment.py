#!/usr/bin/env python3
"""Route an isolated clock variant; never programs the board."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from variants import ROOT, check_frozen, sha
from experiments import BASE
from run_screening import save_json, verify_files


def route(source_label, mhz, label=None):
    check_frozen()
    settings={22.5:(4,5),27.0:(0,0)}
    fbdiv,idiv=settings[mhz]
    source=BASE/source_label; target=BASE/(label or f'{source_label}-{mhz:g}mhz')
    if target.exists(): raise FileExistsError('clock evidence already exists')
    target.mkdir(parents=True)
    for stage in ('engine','native'):
        report=json.loads((source/stage/'report.json').read_text())
        if report['status']!='passed':raise ValueError('missing arithmetic regression')
        verify_files(ROOT,report['sources'])
        (target/stage).mkdir();shutil.copy2(source/stage/'report.json',target/stage/'report.json')
    shutil.copy2(source/'engine/results.xml',target/'engine/results.xml')
    shutil.copy2(source/'engine.sv',target/'engine.sv')
    pll=(ROOT/'hardware/phase4_sdram/pll_dma_20.v').read_text()
    pll=pll[pll.index('module '):].replace('.FBDIV_SEL(2), .IDIV_SEL(3)',f'.FBDIV_SEL({fbdiv}), .IDIV_SEL({idiv})')
    (target/'pll.v').write_text(f'// Experimental {mhz:g} MHz from 27 MHz; unchanged phase selection.\n'+pll)
    hz=round(mhz*1e6)
    assert hz%750000==0
    (target/'host.sv').write_text((ROOT/'hardware/phase5/tiled_host.sv').read_text().replace('20250000',str(hz)))
    script=(ROOT/'hardware/phase6/build.tcl').read_text()
    script=script.replace('set root [file normalize [file join [file dirname [info script]] ../..]]',f'set root {{{ROOT}}}')
    script=script.replace('hardware/phase4_sdram/pll_dma_20.v',str((target/'pll.v').relative_to(ROOT)))
    script=script.replace('hardware/phase5/tiled_host.sv',str((target/'host.sv').relative_to(ROOT)))
    (target/'build.tcl').write_text(script)
    build=target/'route';build.mkdir()
    gowin='/Applications/GowinIDE.app/Contents/Resources/Gowin_EDA/IDE'
    env=dict(os.environ,DYLD_FRAMEWORK_PATH=f'{gowin}/lib',DYLD_LIBRARY_PATH=f'{gowin}/lib',PHASE6_ENGINE=str(target/'engine.sv'))
    started=time.monotonic()
    with (build/'build.log').open('w') as log:
        completed=subprocess.run([f'{gowin}/bin/gw_sh',str(target/'build.tcl')],cwd=build,env=env,stdout=log,stderr=subprocess.STDOUT)
    record=dict(status='failed-build',physical_board=False,source_label=source_label,
        core_clock_mhz=mhz,engine_sha256=sha(target/'engine.sv'),build_seconds=time.monotonic()-started,
        sources={str(p.relative_to(ROOT)):sha(p) for p in (target/'pll.v',target/'host.sv',target/'build.tcl',
                ROOT/'work/phase4/ip-generated/sdram_controller_hs.ipc',ROOT/'rtl/v2/sdram_refresh.sv')},
        uart_divider=hz//750000,pll_vco_mhz=mhz*32,
        sdram_cycle_delays_ns={k:float(v)*1000/mhz for k,v in re.findall(r'^(t\w+)=(\d+)$',(ROOT/'work/phase4/ip-generated/sdram_controller_hs.ipc').read_text(),re.M)})
    if completed.returncode:
        save_json(build/'report.json',record);raise RuntimeError('clock build failed')
    pnr=build/'phase6_tiled_host/impl/pnr';prefix='phase6_tiled_host'
    route_file=pnr/f'{prefix}.rpt.txt';timing=pnr/f'{prefix}_tr_content.html';bitstream=pnr/f'{prefix}.fs'
    tr=timing.read_text();text=route_file.read_text();resources={}
    for key in ('Logic','Register','BSRAM','DSP','CLS'):
        match=re.search(rf'^\s*{key}\s*\|\s*([\d.]+)/([\d.]+)',text,re.M)
        if not match:raise ValueError('missing resource')
        resources[key]=dict(used=float(match[1]),available=float(match[2]))
    fmax=re.search(rf'<td>{mhz:.3f}\(MHz\)</td>\s*<td[^>]*>([\d.]+)\(MHz\)</td>',tr)
    if not fmax:raise ValueError('clock constraint not found')
    record.update(resources=resources,routed_core_fmax_mhz=float(fmax[1]),
        bitstream=str(bitstream.relative_to(ROOT)),bitstream_sha256=sha(bitstream),
        route_sha256=sha(route_file),timing_sha256=sha(timing))
    for kind in ('Setup','Hold'):
        count=re.search(rf'Numbers of {kind} Violated Endpoints</td>\s*<td[^>]*>(\d+)</td>',tr)
        if not count:raise ValueError('timing endpoints absent')
        record[kind.lower()+'_violated_endpoints']=int(count[1])
    record['status']='passed-route' if float(fmax[1])>=mhz and not (record['setup_violated_endpoints'] or record['hold_violated_endpoints']) else 'failed-timing'
    save_json(build/'report.json',record);print(json.dumps(record),flush=True);check_frozen()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label');parser.add_argument('mhz',type=float,choices=(22.5,27.0))
    parser.add_argument('--output-label')
    args=parser.parse_args();route(args.label,args.mhz,args.output_label)
