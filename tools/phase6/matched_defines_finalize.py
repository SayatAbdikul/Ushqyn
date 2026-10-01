#!/usr/bin/env python3
"""Bind final B3 frontier to both exact inputs and both native stall seeds."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'compiler'),str(ROOT/'tools/phase6')]
from matched_defines_baseline import (BASE,REVISION,matched_model,check_oracles,graph_identity,
    compose,fixture,native_measure,save,sha,source_pins,NATIVE)


def merge_sources(destination, incoming):
    for name, digest in incoming.items():
        if name in destination and destination[name] != digest:
            raise ValueError(f'conflicting source freeze: {name}')
        destination[name] = digest


def validate_fixture(row, expected_seeds=(0, 6063)):
    directory = ROOT / row['directory']
    for name, digest in row['files'].items():
        if sha(directory / name) != digest:
            raise ValueError(f'fixture changed: {directory / name}')
    seeds = [native['stall_seed'] for native in row['native']]
    if sorted(seeds) != sorted(expected_seeds):
        raise ValueError('fixture has missing or duplicate native seeds')
    for native in row['native']:
        if native['status'] != 'passed' or native['executable_sha256'] != sha(NATIVE):
            raise ValueError('native executable/result mismatch')
        if native['fixture_files'] != row['files']:
            raise ValueError('native timing is not bound to the declared fixture')
    if row['replay']['status'] != 'passed':
        raise ValueError('fixture replay incomplete')


def run(inputs,output=BASE,frontier_size=3,incumbent=None,baseline_report=None,smoke_report=None,weights_reports=()):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    reports=[(Path(path),json.loads(Path(path).read_text())) for path in inputs]
    weights=[(Path(path),json.loads(Path(path).read_text())) for path in weights_reports]
    for path,r in weights:
        if r['status']!='passed' or r['native_sha256']!=sha(NATIVE):raise ValueError('weight residency screen incomplete or native changed')
        for name,digest in r['compiler_sources'].items():
            if sha(ROOT/name)!=digest:raise ValueError('weight residency source changed')
    incumbent_record=json.loads(Path(incumbent).read_text()) if incumbent is not None else None
    if incumbent_record is not None and incumbent_record['status']!='passed':raise ValueError('manual incumbent not validated')
    for path,r in reports:
        if r['status']!='passed' or r['native_sha256']!=sha(NATIVE):raise ValueError('unvalidated search report')
        for name,digest in r['compiler_sources'].items():
            if sha(ROOT/name)!=digest:raise ValueError(f'search source changed: {name}')
    report=copy.deepcopy(reports[0][1]);report.update(status='running',models={},selected={},frontier={},
        search_reports=[dict(file=str(p.relative_to(ROOT) if p.is_absolute() else p),sha256=sha(p)) for p,r in reports],
        compiler_sources=dict(source_pins(),**{
            'tools/phase6/matched_defines_finalize.py':sha(Path(__file__)),
            'tools/phase6/matched_current.py':sha(ROOT/'tools/phase6/matched_current.py')}))
    upstream=ROOT/f'work/phase6/defines-source/DeFiNES-{REVISION}'
    report['upstream_source_files']={str(p.relative_to(ROOT)):sha(p) for p in sorted(upstream.rglob('*.py'))}
    archive=ROOT/'work/phase6/defines-source.tar.gz'
    if archive.exists():report['upstream_source_archive']=dict(file=str(archive.relative_to(ROOT)),sha256=sha(archive))
    smoke=Path(smoke_report) if smoke_report is not None else ROOT/'work/phase6/matched-defines-native-smoke-v2/report.json'
    smoke_record=json.loads(smoke.read_text())
    if smoke_record['status']!='passed':raise ValueError('generic mode native proof incomplete')
    report['cache_mode_native_proof']=dict(file=str(smoke.relative_to(ROOT)),sha256=sha(smoke))
    inherited=dict(smoke_record['source_sha256'])
    if baseline_report is not None:
        baseline_path=Path(baseline_report)
        baseline=json.loads(baseline_path.read_text())
        if baseline['status']!='passed':raise ValueError('common baseline source freeze incomplete')
        merge_sources(inherited,baseline['compiler_sources'])
        report['shared_baseline_source_proof']=dict(file=str(baseline_path.relative_to(ROOT)),sha256=sha(baseline_path))
    for name,digest in inherited.items():
        if sha(ROOT/name)!=digest:raise ValueError(f'inherited execution source changed: {name}')
        if name in report['compiler_sources'] and report['compiler_sources'][name]!=digest:raise ValueError('conflicting source freeze')
        report['compiler_sources'][name]=digest
    report['weight_residency_screens']=[dict(file=str(p.relative_to(ROOT)),sha256=sha(p)) for p,r in weights]
    for path,r in weights:merge_sources(report['compiler_sources'],r['compiler_sources'])
    report['source_freeze_scope']='union of generic-mode native proof, complete-model run sources, and shared baseline run-start source manifests, rechecked at finalization'
    for path,source in reports:
        for model,info in source['models'].items():
            if model in report['models']:raise ValueError('duplicate model report')
            original,pinned,program,maps,provenance=matched_model(model)
            if graph_identity(program)!=info['graph_sha256']:raise ValueError('graph changed since search')
            stress=np.random.default_rng(6157).integers(-128,128,pinned.shape,dtype=np.int8)
            oracle=check_oracles(original,program,maps,stress)
            pool=[]
            for candidate in info['candidates']:
                timed=ROOT/candidate['directory']
                existing={r['stall_seed']:r for r in candidate['native']}
                if 6063 not in existing:existing[6063]=native_measure(timed,6063,timed.parent/'native-pinned-s6063.json')
                candidate['native']=[existing[k] for k in (0,6063)]
                validate_fixture(candidate)
                candidate['selection_score_cycles']=math.sqrt(existing[0]['elapsed_cycles']*existing[6063]['elapsed_cycles'])
                pool.append(candidate)
            pool.sort(key=lambda r:(r['selection_score_cycles'],r['rank']))
            frontier=[]
            for candidate in pool[:frontier_size]:
                rank=candidate['rank'];timed=ROOT/candidate['directory'];pinned_row=copy.deepcopy(candidate)
                for name,digest in pinned_row['files'].items():
                    if sha(timed/name)!=digest:raise ValueError('candidate changed')
                existing={r['stall_seed']:r for r in pinned_row['native']}
                if 6063 not in existing:existing[6063]=native_measure(timed,6063,timed.parent/'native-pinned-s6063.json')
                pinned_row['native']=[existing[s] for s in (0,6063)]
                chosen=[dict(configuration=cfg) for cfg in candidate['configuration_path']]
                end=max(cfg['stop'] for cfg in candidate['configuration_path'])
                tail=dict(configuration=dict(kind='fallback',start=end,stop=len(program.layers)))
                component_dir=output/'frontier-components'/f'{model}-{rank:03d}'
                code,payload,record=compose(program,chosen,tail,oracle,component_dir)
                directory=output/'frontier-fixtures'/f'{model}-{rank:03d}-stress'
                files=fixture(directory,program,oracle,code,payload,record)
                natives=[native_measure(directory,seed,component_dir/f'native-stress-s{seed}.json') for seed in (0,6063)]
                stress_row=dict(directory=str(directory.relative_to(ROOT)),files=files,replay=record['replay'],native=natives)
                frontier.append(dict(pinned=pinned_row,stress=stress_row))
                print(model,'B3 frontier',rank,'validated both inputs/seeds',flush=True)
            for weight_index,(weight_path,weight_record) in enumerate(weights):
                if model not in weight_record['models']:continue
                wmodel=weight_record['models'][model]
                if wmodel['graph_sha256']!=info['graph_sha256']:raise ValueError('resident graph mismatch')
                from matched_defines_weights import compose as compose_resident
                for pair_index,pair in enumerate(wmodel['pairs']):
                    # Build/check stress only for candidates that can reach the final frontier.
                    candidate=copy.deepcopy(pair['resident'])
                    validate_fixture(candidate)
                    score=math.sqrt(math.prod(n['elapsed_cycles'] for n in candidate['native']))
                    existing_scores=[math.sqrt(math.prod(n['elapsed_cycles'] for n in row['pinned']['native'])) for row in frontier]
                    if len(existing_scores)>=frontier_size and score>sorted(existing_scores)[frontier_size-1]:continue
                    chosen=[dict(configuration=cfg) for cfg in pair['configurations']]
                    cdir=output/'resident-frontier-components'/f'{model}-{weight_index:03d}-{pair_index:03d}'
                    code,payload,record=compose_resident(program,chosen,pair['tail'],oracle,cdir)
                    directory=output/'resident-frontier-fixtures'/f'{model}-{weight_index:03d}-{pair_index:03d}-stress'
                    files=fixture(directory,program,oracle,code,payload,record)
                    natives=[native_measure(directory,seed,cdir/f'native-stress-s{seed}.json') for seed in (0,6063)]
                    frontier.append(dict(pinned=candidate,stress=dict(directory=str(directory.relative_to(ROOT)),files=files,replay=record['replay'],native=natives),
                        incumbent_origin='whole-stack immutable residency allocation variant',paired_speedup=pair['paired_geomean_speedup']))
            if incumbent_record is not None:
                inc=incumbent_record['models'][model]
                if inc['graph_sha256']!=info['graph_sha256']:raise ValueError('manual incumbent graph mismatch')
                row={sample:copy.deepcopy(inc['fixtures'][sample]) for sample in ('pinned','stress')}
                row['incumbent_origin']='manual depth-first full-width two-strip schedules and full-tensor retention; common exact optimizations'
                for sample in ('pinned','stress'):
                    item=row[sample];validate_fixture(item);directory=ROOT/item['directory']
                    for name,digest in item['files'].items():
                        if sha(directory/name)!=digest:raise ValueError('manual incumbent fixture changed')
                    for native in item['native']:
                        if native['status']!='passed' or native['executable_sha256']!=sha(NATIVE):raise ValueError('manual incumbent native mismatch')
                frontier.append(row)
                report['manual_incumbent']=dict(file=str(Path(incumbent).relative_to(ROOT) if Path(incumbent).is_absolute() else Path(incumbent)),sha256=sha(Path(incumbent)),
                    policy_audit='VWW DW/activation/PW/activation full-width two-strip regions instantiate mode1; KWS full-tensor retention is the full-tile degeneracy; source prefetch/alias protection, exact constants and LUT epilogues are shared lowering improvements, not excluded novel mechanisms')
            def score(row):
                seeds={n['stall_seed']:n for n in row['pinned']['native']}
                return math.sqrt(seeds[0]['elapsed_cycles']*seeds[6063]['elapsed_cycles'])
            frontier.sort(key=score)
            distinct=[];identities=set()
            for row in frontier:
                identity=tuple(row['pinned']['files'][k] for k in ('commands.bin','payload.bin'))
                if identity not in identities:distinct.append(row);identities.add(identity)
                if len(distinct)==frontier_size:break
            info['candidates']=pool
            report['models'][model]=info;report['frontier'][model]=distinct;report['selected'][model]=distinct[0]
            save(output/'report.json',report)
    for name,digest in report['compiler_sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'finalized compiler changed: {name}')
    for name,digest in report['upstream_source_files'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'upstream source changed during finalization: {name}')
    for frontier in report['frontier'].values():
        for row in frontier:
            for sample in ('pinned','stress'):validate_fixture(row[sample])
    report['status']='passed';save(output/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,default=BASE)
    parser.add_argument('--frontier-size',type=int,default=3)
    parser.add_argument('--incumbent',type=Path)
    parser.add_argument('--baseline-report',type=Path)
    parser.add_argument('--smoke-report',type=Path)
    parser.add_argument('--weights-reports',nargs='*',type=Path,default=[])
    args=parser.parse_args();run(args.inputs,args.output,args.frontier_size,args.incumbent,args.baseline_report,args.smoke_report,args.weights_reports)
