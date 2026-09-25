#!/usr/bin/env python3
"""Archive the autonomous Phase 4 route and exact programmed release bytes."""
import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/'docs/research/evidence/phase4'


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--program-log',type=Path,required=True)
    a=p.parse_args()
    pnr=a.build/'phase4_tiled_host/impl/pnr'
    prefix='phase4_tiled_host'
    report={}
    contents={}
    for name,suffix in (('route','.rpt.txt'),('timing','_tr_content.html'),('power','.power.html')):
        data=(pnr/(prefix+suffix)).read_bytes();contents[name]=data.decode()
        archive=f'physical-sequence-{name}.txt.gz'
        (EVIDENCE/archive).write_bytes(gzip.compress(data,mtime=0))
        report[f'{name}_report_sha256']=sha(data)
    sources=[
        'rtl/v2/target_pkg.sv','hardware/phase4_sdram/pll_dma_20.v',
        'hardware/phase4_sdram/tiled_host.sv','rtl/v2/requantizer.sv',
        'rtl/v2/scratchpad.sv','rtl/v2/engine.sv','rtl/v2/tile_dma.sv',
        'rtl/v2/tiled_core.sv','rtl/v2/command.sv','rtl/v2/tiled_host_bridge.sv',
        'rtl/v2/tile_sequencer.sv','rtl/v2/hs_sdram_burst_port.sv',
        'rtl/v2/sdram_refresh.sv','rtl/v2/uart_rx.sv','rtl/v2/uart_tx.sv',
        'hardware/phase4_sdram/build_tiled_host.tcl','hardware/phase4_sdram/tiled_host.cst',
        'hardware/phase4_sdram/bist.sdc','hardware/phase4_sdram/sdram_controller_hs.ipc']
    report['source_sha256']={name:sha((ROOT/name).read_bytes()) for name in sources}
    ip='work/phase4/ip-generated/sdram_controller_hs.v'
    report['generated_ip_path']=ip;report['generated_ip_sha256']=sha((ROOT/ip).read_bytes())
    report['resources']={}
    for key,label in (('logic','Logic'),('register','Register'),('bsram','BSRAM'),('dsp','DSP')):
        m=re.search(r'^\s*'+label+r'\s*\|\s*([\d.]+)/([\d.]+)',contents['route'],re.M)
        if not m:raise ValueError(f'missing {label}')
        report['resources'][key]={'used':float(m[1]),'available':int(m[2])}
    timing=contents['timing']
    m=re.search(r'<td>20\.250\(MHz\)</td>\s*<td>([\d.]+)\(MHz\)</td>',timing)
    t=re.search(r'pll/pll_s2/CLKOUT\.default_gen_clk</td>\s*<td>Setup</td>\s*<td>([\d.]+)</td>',timing)
    if not m or not t:raise ValueError('missing routed timing')
    report['routed_core_fmax_mhz']=float(m[1]);report['setup_tns_ns']=float(t[1])
    report['core_clock_constraint_mhz']=20.25
    if float(m[1])<=20.25 or float(t[1])!=0:raise ValueError('timing failed')
    raw=(pnr/(prefix+'.fs')).read_bytes()
    release=ROOT/'hardware/releases/phase4-sdram/hs_tiled_autonomous.fs'
    release.write_bytes(raw)
    report['bitstream_sha256']=sha(raw);report['bitstream_path']=str(release.relative_to(ROOT))
    log=a.program_log.read_bytes()
    if b'after program sram: displayReadReg 00006020' not in log:
        raise ValueError('missing successful programming evidence')
    (EVIDENCE/'physical-sequence-program.txt.gz').write_bytes(gzip.compress(log,mtime=0))
    report['program_log_sha256']=sha(log)
    report['status']='passed-post-route';report['scope']='autonomous sequencer and 64-byte SDRAM bursts'
    report['limits']=['physical run reports are separate','Gowin power report is an estimate, not measured energy']
    (EVIDENCE/'physical-sequence-route.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report['resources']))


if __name__=='__main__':main()
