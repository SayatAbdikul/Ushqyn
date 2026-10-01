#!/usr/bin/env python3
"""Measure actual avoided lowering work; do not extrapolate from candidate counts."""
import argparse
import json
from pathlib import Path
import random
import time

from hypothesis_search import ROOT,OUT,StateKeys,baseline,lower,outcome,save


def run(report_path=OUT/'report.json',output=OUT/'cost.json'):
    report_path=Path(report_path);report=json.loads(report_path.read_text())
    if report['status']!='passed':raise ValueError('hypothesis proof incomplete')
    for name,expected in report['source_sha256'].items():
        if baseline.sha(ROOT/name)!=expected:raise ValueError('source changed: '+name)
    sources=dict(report['source_sha256'])
    sources['tools/phase6/hypothesis_search_cost.py']=baseline.sha(Path(__file__))
    result=dict(status='running',physical_board=False,source_sha256=sources,
        proof_file=str(report_path.relative_to(ROOT)),proof_sha256=baseline.sha(report_path),models={},
        scope='Measured all avoided lowerings once in shuffled order; subtract independently timed full-catalogue key work. This is not an end-to-end catalogue benchmark.')
    for model,proof in report['models'].items():
        _,_,program,_,_=baseline.matched_model(model)
        catalogue=ROOT/proof['catalogue_file']
        if baseline.sha(catalogue)!=proof['catalogue_sha256']:raise ValueError('catalogue changed')
        document=json.loads(catalogue.read_text());duplicates={};key_seconds={}
        for mode in ('axis_only','exact_halo'):
            keys=StateKeys(program,sources,exact_halo=mode=='exact_halo');seen=set();hits=set()
            began=time.perf_counter()
            for row in document['candidates']:
                key=keys.key(row['configuration'])
                if key in seen:hits.add(row['id'])
                seen.add(key)
            key_seconds[mode]=time.perf_counter()-began;duplicates[mode]=hits
        if not duplicates['axis_only']<=duplicates['exact_halo']:raise AssertionError('non-nested equivalence')
        candidates=[row for row in document['candidates'] if row['id'] in duplicates['exact_halo']]
        random.Random(6167).shuffle(candidates);timings=[]
        for index,row in enumerate(candidates):
            began=time.perf_counter();actual,_=lower(program,row['configuration']);seconds=time.perf_counter()-began
            if actual!=outcome(row):raise AssertionError('changed duplicate outcome '+row['id'])
            timings.append(dict(id=row['id'],seconds=seconds,status=row['status'],axis_only_reuse=row['id'] in duplicates['axis_only']))
            if (index+1)%500==0:print(model,index+1,'/',len(candidates),'avoided lowerings timed',flush=True)
        strategy={}
        for mode in ('axis_only','exact_halo'):
            seconds=sum(row['seconds'] for row in timings if row['id'] in duplicates[mode])
            strategy[mode]=dict(avoided_lowerings=len(duplicates[mode]),measured_avoided_lowering_seconds=seconds,
                measured_full_key_seconds=key_seconds[mode],estimated_net_seconds_saved=seconds-key_seconds[mode])
        result['models'][model]=dict(strategies=strategy,timings=timings,
            preferred_by_measured_removed_work=max(strategy,key=lambda x:strategy[x]['estimated_net_seconds_saved']))
        save(Path(output).with_suffix('.partial.json'),result)
        print(model,strategy,flush=True)
    for name,expected in sources.items():
        if baseline.sha(ROOT/name)!=expected:raise ValueError('source changed: '+name)
    result['status']='passed';save(Path(output),result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=OUT/'report.json')
    parser.add_argument('--output',type=Path,default=OUT/'cost.json')
    args=parser.parse_args();run(args.report,args.output)
