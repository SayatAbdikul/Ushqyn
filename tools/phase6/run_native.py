#!/usr/bin/env python3
"""Run isolated full-model RTL variants; no UART/JTAG/physical board access."""
import argparse
import hashlib
import json
import subprocess
import time
from variants import ROOT, VARIANTS, sources, sha, check_frozen

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=['baseline','c0p0','c0p1','c64p0','c128p0','c256p0','c256p1'])
parser.add_argument('--select',default='')
parser.add_argument('--exclude',action='append',default=[])
parser.add_argument('--seeds',nargs='+',type=int,default=[0,6063])
args=parser.parse_args()
check_frozen()
fixtures=json.loads((ROOT/'work/phase6/optimization/fixtures.json').read_text())['fixtures']
fixtures=[f for f in fixtures if args.select in f['name'] and not any(e in f['name'] for e in args.exclude)]
if not fixtures: raise ValueError('no matching fixtures')
base=ROOT/'work/phase6/optimization'
for variant in args.variants:
    build=base/f'native-{variant}';build.mkdir(parents=True,exist_ok=True)
    inputs=sources(variant,build)
    harness=ROOT/'test/phase6/native.cpp'
    command=['verilator','--cc','--exe','--build','-j','2','--public-flat-rw','-Wno-fatal',
             '--top-module','v2_tiled_host_bridge','--Mdir',str(build),
             *map(str,inputs),str(harness)]
    with (build/'build.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    report={'schema':1,'variant':variant,'scope':'RTL with abstract fixed/stalled external RAM; not physical SDRAM latency',
            'physical_board':False,'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [*inputs,harness]},'results':[]}
    saved=build/'report.json'
    if saved.exists():
        old=json.loads(saved.read_text())
        if old['source_sha256']==report['source_sha256']:
            report['results']=old['results']
    for fixture in fixtures:
        path=base/'fixtures'/fixture['name']
        fixture_hash=hashlib.sha256(json.dumps(fixture['files'],sort_keys=True).encode()).hexdigest()
        for name,digest in fixture['files'].items():
            if sha(path/name)!=digest: raise ValueError(f'fixture changed: {path/name}')
        for seed in args.seeds:
            if any(r['fixture']==fixture['name'] and r['stall_seed']==seed and r.get('fixture_sha256')==fixture_hash for r in report['results']):
                continue
            output=build/f'{fixture["name"]}-s{seed}.json';start=time.monotonic()
            subprocess.run([str(build/'Vv2_tiled_host_bridge'),str(path),str(seed),str(output)],check=True,cwd=ROOT)
            row=json.loads(output.read_text());row.update(fixture=fixture['name'],fixture_sha256=fixture_hash,simulation_seconds=time.monotonic()-start)
            report['results']=[r for r in report['results'] if (r['fixture'],r['stall_seed'])!=(fixture['name'],seed)]
            report['results'].append(row)
            (build/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    report['status']='passed';(build/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
check_frozen()
