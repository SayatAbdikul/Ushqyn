#!/usr/bin/env python3
"""Reconstruct whole catalogues and compare downstream bounded path winners."""
import argparse
import copy
import json
from pathlib import Path

from hypothesis_search import ROOT,OUT,StateKeys,baseline,digest,outcome,save


def paths(rows):
    best={}
    for row in rows:
        if row['status']!='executable' or row.get('duplicate_within_stack'):continue
        cfg=row['configuration'];edge=best.setdefault((cfg['start'],cfg['stop']),[])
        edge.append(row);edge.sort(key=lambda r:(r['proxy_score'],r['command_bytes'],r['id']));del edge[4:]
    start=min(r['configuration']['start'] for r in rows)
    stop=max(r['configuration']['stop'] for r in rows)
    return [dict(proxy_score=score,command_bytes=size,path=[row['id'] for row in sequence])
            for score,size,sequence in baseline.k_paths(best,start,stop,k=24,max_bytes=30000)]


def run(report_path=OUT/'report.json',output=OUT/'winners.json'):
    report_path=Path(report_path);report=json.loads(report_path.read_text())
    if report['status']!='passed':raise ValueError('hypothesis proof incomplete')
    sources=dict(report['source_sha256'])
    sources['tools/phase6/hypothesis_search_winners.py']=baseline.sha(Path(__file__))
    for name,expected in sources.items():
        if baseline.sha(ROOT/name)!=expected:raise ValueError('source changed: '+name)
    result=dict(status='passed',physical_board=False,source_sha256=sources,
        proof_file=str(report_path.relative_to(ROOT)),proof_sha256=baseline.sha(report_path),
        scope='identical entire generic candidate catalogues and deterministic top24 proxy paths; later resident-weight/manual-incumbent screens are unchanged and outside this catalogue',models={})
    for model,proof in report['models'].items():
        _,_,program,_,_=baseline.matched_model(model)
        catalogue=ROOT/proof['catalogue_file']
        if baseline.sha(catalogue)!=proof['catalogue_sha256']:raise ValueError('catalogue changed')
        original=json.loads(catalogue.read_text())['candidates'];memo={};restored=[]
        keys=StateKeys(program,sources,exact_halo=True)
        for row in original:
            key=keys.key(row['configuration']);memo.setdefault(key,outcome(row))
            reconstructed=copy.deepcopy(row);reconstructed.update(memo[key]);restored.append(reconstructed)
        if restored!=original:raise AssertionError('catalogue changed')
        before=paths(original);after=paths(restored)
        if before!=after:raise AssertionError('global proxy paths changed')
        result['models'][model]=dict(candidate_rows=len(original),original_catalogue_rows_sha256=digest(original),
            reconstructed_catalogue_rows_sha256=digest(restored),global_proxy_paths_identical=True,
            path_budget=dict(k=24,command_bytes=30000),paths=after)
    save(Path(output),result)
    print(json.dumps({m:dict(rows=r['candidate_rows'],paths=len(r['paths']),identical=True)
                      for m,r in result['models'].items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=OUT/'report.json')
    parser.add_argument('--output',type=Path,default=OUT/'winners.json')
    args=parser.parse_args();run(args.report,args.output)
