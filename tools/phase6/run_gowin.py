#!/usr/bin/env python3
"""Route isolated cache variants using Gowin; never programs the FPGA."""
import argparse
import gzip
import json
import os
import re
import subprocess
import time
from variants import ROOT, VARIANTS, engine_source, sha, check_frozen

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=['c256p1'])
parser.add_argument('--report-only',action='store_true',help='Parse an existing build without changing its sources or invoking Gowin')
parser.add_argument('--build-label',help='Separate evidence directory for a single historical candidate')
args=parser.parse_args()
if args.build_label and len(args.variants)!=1: parser.error('--build-label requires exactly one variant')
check_frozen()
gowin='/Applications/GowinIDE.app/Contents/Resources/Gowin_EDA/IDE'
for name in args.variants:
    label=args.build_label or name
    build=ROOT/f'work/phase6/optimization/route-{label}';build.mkdir(parents=True,exist_ok=True)
    engine=(build/'engine.sv') if args.report_only else engine_source(name,build)
    env=dict(os.environ,DYLD_FRAMEWORK_PATH=f'{gowin}/lib',DYLD_LIBRARY_PATH=f'{gowin}/lib',PHASE6_ENGINE=str(engine))
    start=time.monotonic()
    if args.report_only:
        returncode=None
    else:
        with (build/'build.log').open('w') as log:
            result=subprocess.run([f'{gowin}/bin/gw_sh',str(ROOT/'hardware/phase6/build.tcl')],cwd=build,env=env,stdout=log,stderr=subprocess.STDOUT)
        returncode=result.returncode
    pnr=build/'phase6_tiled_host/impl/pnr';prefix='phase6_tiled_host'
    record={'variant':name,'label':label,'physical_board':False,'scope':'Gowin synthesis and place-and-route; no measured latency/power',
            'engine_sha256':sha(engine),'build_tcl_sha256':sha(ROOT/'hardware/phase6/build.tcl'),
            'build_seconds':None if args.report_only else time.monotonic()-start,'returncode':returncode,'report_only':args.report_only,'resources':{},'status':'failed-build'}
    route=pnr/(prefix+'.rpt.txt');timing=pnr/(prefix+'_tr_content.html');bitstream=pnr/(prefix+'.fs')
    if (returncode==0 or args.report_only) and route.exists() and timing.exists() and bitstream.exists():
        text=route.read_text();tr=timing.read_text()
        for key in ['Logic','Register','BSRAM','DSP','CLS']:
            m=re.search(rf'^\s*{key}\s*\|\s*([\d.]+)/([\d.]+)',text,re.M)
            if m: record['resources'][key]={'used':float(m[1]),'available':float(m[2])}
        fmax=re.search(r'<td>20\.250\(MHz\)</td>\s*<td[^>]*>([\d.]+)\(MHz\)</td>',tr)
        tns=re.search(r'pll/pll_s2/CLKOUT\.default_gen_clk</td>\s*<td>Setup</td>\s*<td>(-?[\d.]+)</td>',tr)
        if not fmax or not tns: raise ValueError('missing clock timing result')
        record.update(core_clock_mhz=20.25,routed_core_fmax_mhz=float(fmax[1]),setup_tns_ns=float(tns[1]),bitstream_sha256=sha(bitstream))
        for kind in ('Setup','Hold'):
            count=re.search(rf'Numbers of {kind} Violated Endpoints</td>\s*<td[^>]*>(\d+)</td>',tr)
            if not count: raise ValueError('missing timing endpoint count')
            record[kind.lower()+'_violated_endpoints']=int(count[1])
        record['status']='passed-route' if float(fmax[1])>=20.25 and float(tns[1])==0 and not (record['setup_violated_endpoints'] or record['hold_violated_endpoints']) else 'failed-timing'
        record['reports']={}
        for label,p in [('route',route),('timing',timing)]:
            dest=build/f'{label}.gz';dest.write_bytes(gzip.compress(p.read_bytes(),mtime=0));record['reports'][label]={'file':str(dest.relative_to(ROOT)),'sha256':sha(dest),'raw_sha256':sha(p)}
    (build/'report.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    print(json.dumps(record),flush=True)
check_frozen()
