#!/usr/bin/env python3
"""Freeze an unrun physical comparison plan; never opens or programs a device."""
import argparse
import gzip
import json
import random
from variants import ROOT, sha, check_frozen

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',action='store_true')
args=parser.parse_args();base=ROOT/'work/phase6/optimization'
summary=json.loads((base/'summary.json').read_text())
if summary['status']!='passed-boardless-milestone': raise ValueError('audit required')
for name,digest in summary['source_sha256'].items():
    if sha(ROOT/name)!=digest: raise ValueError(f'changed audited source: {name}')
check_frozen()
fixtures=json.loads((base/'fixtures.json').read_text())['fixtures']
policies={};aliases={}
for f in fixtures:
    if f['sample']!='pinned' or f['snapshots']: continue
    key=(f['model'],*(f['files'][n] for n in ('commands.bin','payload.bin','input.bin','output.bin')))
    if key in aliases:
        policies[aliases[key]]['aliases'].append(f['name']);continue
    aliases[key]=f['name']
    directory=base/'fixtures'/f['name']
    for name,digest in f['files'].items():
        if sha(directory/name)!=digest: raise ValueError('changed fixture')
    policies[f['name']]={'model':f['model'],'directory':str(directory.relative_to(ROOT)),
                        'files':f['files'],'aliases':[],'status':'not_run'}
images={'baseline':{'file':'hardware/releases/phase5/dual_resident_750k.fs',
                    'sha256':sha(ROOT/'hardware/releases/phase5/dual_resident_750k.fs')}}
for name,route in summary['routes'].items():
    p=base/f'route-{name}/phase6_tiled_host/impl/pnr/phase6_tiled_host.fs'
    if sha(p)!=route['bitstream_sha256']: raise ValueError('changed bitstream')
    images[name]={'file':str(p.relative_to(ROOT)),'sha256':sha(p),'routed_core_fmax_mhz':route['routed_core_fmax_mhz']}
sessions=[]
for i in range(5):
    rng=random.Random(6107+i);hardware=list(images);rng.shuffle(hardware);blocks=[]
    for variant in hardware:
        cases=list(policies);rng.shuffle(cases)
        blocks.append({'variant':variant,'policies':cases,'repeats_per_policy':30})
    sessions.append({'session':i+1,'seed':6107+i,'blocks':blocks})
report={'schema':1,'status':'not_run','physical_board':False,'core_clock_mhz':20.25,
        'scope':'engineering hardware and restricted scheduling comparisons; not complete B3/B4',
        'requires':'Finish the original Phase 5 campaigns before any reprogramming.',
        'images':images,'policies':policies,'sessions':sessions,
        'unique_policy_count':len(policies),'planned_timed_jobs':5*30*len(policies)*len(images),
        'controls':['Verify all artifact hashes before programming or uploading.',
                    'Run both pinned and stress snapshot fixtures first; check every tensor against the integer oracle.',
                    'Stop on any output mismatch, protocol failure, counter inconsistency or reset; record it.',
                    'Use a warmup followed by 30 repeats for each policy in each independently initialized session.',
                    'Record engine, DMA, overlap and whole-sequence counters; exclude snapshots from timed runs.',
                    'Report UART/input-loading and preprocessing separately from device inference latency.',
                    'Preserve per-session paired comparisons and uncertainty; publish negative cases.',
                    'Evaluate projected-cycle errors without refitting the model on this measurement set.',
                    'Do not claim measured power or energy without an instrument.'],
        'simulation_summary_sha256':sha(base/'summary.json')}
(base/'board-plan.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
if args.archive:
    dest=ROOT/'docs/research/evidence/phase6/optimization'
    (dest/'board-plan.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    # Preserve the exact two main test candidates, not the rejected revision.
    release=ROOT/'hardware/releases/phase6';release.mkdir(parents=True,exist_ok=True)
    for name in ('c256p1','spatial'):
        p=ROOT/images[name]['file']
        (release/f'{name}_candidate_750k.fs.gz').write_bytes(gzip.compress(p.read_bytes(),mtime=0))
print(json.dumps({'status':report['status'],'unique_policies':len(policies),'images':len(images),'timed_jobs':report['planned_timed_jobs']}))
