#!/usr/bin/env python3
"""Audit cache/retention experiments, predictive errors and frozen provenance."""
import argparse
import gzip
import hashlib
import json
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from variants import ROOT, VARIANTS, sha, check_frozen, configured_engine_text
sys.path.insert(0,str(ROOT/'compiler'))
from scheduler.event_cost import estimate
from scheduler.current_abi import CostModel

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',action='store_true')
args=parser.parse_args();base=ROOT/'work/phase6/optimization'
frozen=check_frozen();fixtures=json.loads((base/'fixtures.json').read_text())
by_name={f['name']:f for f in fixtures['fixtures']}
cost=CostModel(json.loads((ROOT/'docs/research/evidence/phase4/physical-sequence-costs.json').read_text()))
sources=[*sorted((ROOT/'rtl/phase6').glob('*.sv')),*sorted((ROOT/'hardware/phase6').glob('*')),
         *sorted((ROOT/'compiler/scheduler').glob('*.py')),*sorted((ROOT/'compiler').glob('test_scheduler_*.py')),
         *sorted((ROOT/'tools/phase6').glob('*.py')),*sorted((ROOT/'test/phase6').glob('*.py')),*sorted((ROOT/'test/phase6').glob('*.cpp'))]
report={'schema':1,'physical_board':False,'phase6_gate_complete':False,'frozen_phase5_files_verified':frozen,
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},'native':{},'routes':{},'rejected_routes':{},'search':{},'cost_checks':[]}
for variant,setting in VARIANTS.items():
    path=base/f'native-{variant}/report.json'
    if not path.exists(): continue
    data=json.loads(path.read_text())
    if data.get('status')!='passed': raise ValueError(f'variant still running: {variant}')
    generated=ROOT/'rtl/v2/engine.sv' if setting is None else path.parent/'engine.sv'
    if generated.read_text()!=configured_engine_text(variant): raise ValueError('native engine is stale')
    for p,digest in data['source_sha256'].items():
        if sha(ROOT/p)!=digest: raise ValueError(f'changed native source: {p}')
    report['native'][variant]=data
    for row in data['results']:
        require_files=by_name[row['fixture']]['files'];directory=base/'fixtures'/row['fixture']
        for name,digest in require_files.items():
            if sha(directory/name)!=digest: raise ValueError(f'changed fixture: {directory/name}')
        if row['status']!='passed': raise ValueError('failed native result')
        if '-timed' not in row['fixture'] or row['stall_seed']!=0: continue
        entries,params=(0,False) if setting is None else setting
        commands=(directory/'commands.bin').read_bytes();payload=(directory/'payload.bin').read_bytes()
        predicted=estimate(commands,payload,cache_entries=entries,parameter_cache=bool(params),spatial_pw=variant=='spatial')
        error=(predicted['elapsed_cycles']-row['elapsed_cycles'])/row['elapsed_cycles']
        engine_error=predicted['engine_cycles']-row['engine_cycles']
        serial='-serial-' in row['fixture']
        if serial and (error!=0 or engine_error!=0): raise ValueError(f'serial structural model differs: {variant} {row["fixture"]}: {error} {engine_error}')
        if abs(error)>.01: raise ValueError('unexplained >1% fixed-RAM timeline prediction error')
        physical=estimate(commands,payload,cache_entries=entries,parameter_cache=bool(params),spatial_pw=variant=='spatial',dma_cost=cost.dma)
        report['cost_checks'].append({'variant':variant,'fixture':row['fixture'],
            'simulated_cycles':row['elapsed_cycles'],'predicted_fixed_ram_cycles':predicted['elapsed_cycles'],
            'fractional_error':error,'engine_cycle_error':engine_error,
            'physical_dma_fit_projection_cycles':physical['elapsed_cycles'],
            'projection_scope':'old measured DMA fit + new RTL-validated uncontended engine; physical arbitration not recalibrated'})
if args.archive:
    for variant in VARIANTS:
        data=report['native'].get(variant)
        if data is None: raise ValueError(f'missing variant: {variant}')
        completed={(r['fixture'],r['stall_seed']) for r in data['results']}
        for fixture in fixtures['fixtures']:
            name=fixture['name'];extended=bool(fixture.get('resident') or fixture.get('searched'))
            if fixture.get('spatial_search'):
                required=variant in ('baseline','c256p1','spatial')
            else:
                required=(variant=='spatial') or (not extended and '-timed' in name) or (variant in ('baseline','c256p1') and
                           (extended or ('half-overlap-check' in name)))
            if required and any((name,seed) not in completed for seed in (0,6063)):
                raise ValueError(f'missing planned native case: {variant} {name}')
    original={(r['fixture'],r['stall_seed']):r for r in report['native']['baseline']['results']}
    for row in report['native']['c0p0']['results']:
        reference=original[row['fixture'],row['stall_seed']]
        if any(row[k]!=reference[k] for k in ('elapsed_cycles','engine_cycles','dma_cycles','overlap_cycles')):
            raise ValueError('disabled-cache control is not cycle equivalent')
for path in sorted(base.glob('route-*/report.json')):
    d=json.loads(path.read_text());engine=ROOT/'rtl/v2/engine.sv' if d['variant']=='baseline' else path.parent/'engine.sv'
    rejected=d.get('label')=='spatial-initial'
    if d['status']!='passed-route' and not rejected: raise ValueError(f'route did not pass: {path}')
    if sha(engine)!=d['engine_sha256']: raise ValueError('changed routed engine')
    if not rejected and engine.read_text()!=configured_engine_text(d['variant']): raise ValueError('routed engine is stale')
    if sha(ROOT/'hardware/phase6/build.tcl')!=d['build_tcl_sha256']: raise ValueError('changed build TCL')
    for r in d.get('reports',{}).values():
        if sha(ROOT/r['file'])!=r['sha256']: raise ValueError('changed route report')
    timing=gzip.decompress((ROOT/d['reports']['timing']['file']).read_bytes()).decode()
    for kind in ('Setup','Hold'):
        count=re.search(rf'Numbers of {kind} Violated Endpoints</td>\s*<td>(\d+)</td>',timing)
        if not count or (int(count[1])!=0 and not rejected): raise ValueError(f'{kind} timing violation')
        d[kind.lower()+'_violated_endpoints']=int(count[1])
    d['frozen_manifest_sha256']=sha(ROOT/'work/phase6/phase5-frozen-files.json')
    d['generated_ip_sha256']=sha(ROOT/'work/phase4/ip-generated/sdram_controller_hs.v')
    report['rejected_routes' if rejected else 'routes'][d.get('label',d['variant'])]=d
for name in ('kws','vww','kws-spatial','vww-spatial'):
    d=json.loads((base/f'{name}-search.json').read_text())
    for p,digest in d['source_sha256'].items():
        if sha(ROOT/p)!=digest: raise ValueError(f'changed search source: {p}')
    if d['wall_seconds']>60 or d['bound_gap_fraction']>.10: raise ValueError('declared restricted search acceptance missed')
    report['search'][name]=d
tests=ET.parse(base/'tests.xml').findall('.//testcase')
if not tests or any(t.find('failure') is not None or t.find('error') is not None for t in tests): raise ValueError('unit test failure')
report['unit_tests']=len(tests)
report['engine_tests']={}
for path in sorted(base.glob('engine-*/results.xml')):
    cases=ET.parse(path).findall('.//testcase')
    if len(cases)!=2 or any(t.find('failure') is not None or t.find('error') is not None for t in cases): raise ValueError('engine test failure')
    report['engine_tests'][path.parent.name]={'cases':len(cases),'xml_sha256':sha(path)}
if args.archive and not {'engine-c64p0','engine-c128p0','engine-c256p1','engine-spatial'}<=set(report['engine_tests']):
    raise ValueError('missing cache-capacity stress regression')
report['defines_geometry']=json.loads((base/'defines-geometry.json').read_text())
if report['defines_geometry']['status']!='passed': raise ValueError('DeFiNES geometry check failed')
if report['defines_geometry']['adapter_sha256']!=sha(ROOT/'compiler/scheduler/defines_adapter.py'): raise ValueError('changed DeFiNES adapter')
report['native_runs']=sum(len(d['results']) for d in report['native'].values())
report['native_tensor_checks']=sum(r['tensor_checks'] for d in report['native'].values() for r in d['results'])
report['max_fixed_ram_timeline_error_fraction']=max(abs(r['fractional_error']) for r in report['cost_checks'])
report['status']='passed-boardless-milestone'
(base/'summary.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
if args.archive:
    dest=ROOT/'docs/research/evidence/phase6/optimization';dest.mkdir(parents=True,exist_ok=True)
    for name in ('summary.json','fixtures.json','tests.xml','defines-geometry.json','kws-search.json','vww-search.json','kws-spatial-search.json','vww-spatial-search.json'):
        (dest/(name+'.gz')).write_bytes(gzip.compress((base/name).read_bytes(),mtime=0))
    for name,d in {**report['routes'],**report['rejected_routes']}.items():
        for label,r in d['reports'].items():
            (dest/f'{name}-{label}.gz').write_bytes((ROOT/r['file']).read_bytes())
        engine=base/f'route-{name}/engine.sv'
        (dest/f'{name}-engine.sv.gz').write_bytes(gzip.compress(engine.read_bytes(),mtime=0))
    for path in sorted(base.glob('engine-*/results.xml')):
        (dest/f'{path.parent.name}-results.xml.gz').write_bytes(gzip.compress(path.read_bytes(),mtime=0))
    compact={k:v for k,v in report.items() if k not in ('native','defines_geometry','search','source_sha256','cost_checks')}
    compact['cost_checks']=report['cost_checks']
    compact['source_pins_archive']='summary.json.gz'
    (dest/'summary.json').write_text(json.dumps(compact,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:report[k] for k in ('status','unit_tests','native_runs','native_tensor_checks','max_fixed_ram_timeline_error_fraction','frozen_phase5_files_verified')}))
