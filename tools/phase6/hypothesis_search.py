#!/usr/bin/env python3
"""Falsify safe cache-mode equivalence on unchanged executable DeFiNES lowering.

This is a bounded memoization experiment, not a new accelerator mechanism.
Historical catalogues are exhaustive outcome oracles; fresh synthetic/held-out
lowerings independently check the reused state classes. No physical board use.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'compiler'),str(ROOT/'tools/phase6')]
import matched_defines_baseline as baseline
from matched_current import graph_identity
from scheduler.matched_defines import shapes_and_needs,bbox,replay_stack
from scheduler.matched_defines_strip import block_program,macro_stacks
from test_scheduler_spatial import spatial_fixture
from integer_reference import evaluate
from static_pipeline import Program,Layer,Tensor
from quantization import Quantization,quantize_parameters

OUT=ROOT/'work/phase6/hypothesis-search-v1'
CAPACITY_ERRORS={'sram-capacity','command-capacity','external-capacity',
                 'dma-stripe-alignment','constant-tail-alignment','dma-alignment','dma-bounds'}
FIELDS=('status','reason','proxy_score','command_bytes','payload_bytes','code_sha256',
        'payload_sha256','peak_sram_address','counters')
ABI={'sram_bytes':32768,'descriptor_bytes':128,'allocation_alignment':8,
     'preceding_guard_bytes':8,'command_capacity_bytes':32768,
     'external_bytes':8*1024*1024,'initial_live_buffers':[],
     'initial_known_sram':'all unknown','initial_command_bytes':0,
     'entry_contract':'standalone empty arena only; arbitrary partial state is unsupported'}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


class StateKeys:
    """Conservative whole-lowering state keys, with inactive cache actions erased."""
    def __init__(self,program,sources,*,exact_halo=False,abi=ABI):
        self.program=program;self.graph=graph_identity(program)
        self.frozen=digest(dict(graph=self.graph,sources=sources,abi=abi))
        self.exact_halo=exact_halo;self.geometry={};self.fragments={}

    def active_axes(self,cfg):
        token=(cfg['start'],cfg['stop'],cfg['h'],cfg['w'])
        if token in self.geometry:return self.geometry[token]
        shape=self.program.tensors[self.program.layers[cfg['stop']-1].output].shape
        horizontal=cfg['w']<shape[3];vertical=cfg['h']<shape[2]
        if self.exact_halo and (horizontal or vertical):
            stack=token[:2]
            if stack not in self.fragments:self.fragments[stack]=block_program(self.program,*stack)
            fragment=self.fragments[stack];needs=shapes_and_needs(fragment,cfg['h'],cfg['w'])
            stages=range(-1,len(fragment.layers)//2)
            horizontal=any(row[x][j].intersect(row[x+1][j]).area
                           for row in needs for x in range(len(row)-1) for j in stages)
            vertical=any(bbox([n[j] for n in needs[y]]).intersect(
                              bbox([n[j] for n in needs[y+1]])).area
                         for y in range(len(needs)-1) for j in stages)
        self.geometry[token]=(bool(horizontal),bool(vertical))
        return self.geometry[token]

    def key(self,cfg):
        geometry={k:cfg[k] for k in ('start','stop','kind','h','w','retain','prefetch')}
        if cfg['kind']=='rectangle':
            h,v=self.active_axes(cfg)
            mode=cfg['mode']
            geometry.update(horizontal_cache=mode>=2 and h,vertical_cache=mode==3 and v)
        else:geometry['mode']=cfg['mode']
        return digest(dict(frozen=self.frozen,configuration=geometry))


def outcome(row):return {key:row[key] for key in FIELDS if key in row}


def lower(program,cfg):
    try:code,payload,record=baseline.compile_config(program,cfg)
    except ValueError as error:
        if str(error) not in CAPACITY_ERRORS:raise
        return dict(status='infeasible-for-backend',reason=str(error)),None
    return dict(status='executable',proxy_score=baseline.proxy(code,record),command_bytes=len(code),
        payload_bytes=len(payload),code_sha256=baseline.sha(code),payload_sha256=baseline.sha(payload),
        peak_sram_address=record['peak_sram_address'],counters=record.get('counters',{})),(code,payload,record)


def shortlist(rows):
    best={};seen={}
    for row in rows:
        if row['status']!='executable':continue
        cfg=row['configuration'];stack=(cfg['start'],cfg['stop'])
        identity=(row['code_sha256'],row['payload_sha256'])
        if identity in seen.setdefault(stack,set()):continue
        seen[stack].add(identity)
        best.setdefault(stack,[]).append(row)
        best[stack].sort(key=lambda r:(r['proxy_score'],r['command_bytes'],r['id']))
        del best[stack][4:]
    return {f'{a}:{b}':[r['id'] for r in rows] for (a,b),rows in sorted(best.items())}


def audit_catalogue(program,document,sources,exact_halo):
    keys=StateKeys(program,sources,exact_halo=exact_halo)
    groups={};reconstructed=[];began=time.perf_counter()
    for row in document['candidates']:
        key=keys.key(row['configuration']);group=groups.setdefault(key,[])
        if group and outcome(group[0])!=outcome(row):
            raise AssertionError('unsafe equivalence: '+group[0]['id']+' / '+row['id'])
        group.append(row)
        reconstructed.append(dict(row,**outcome(group[0])))
    before=shortlist(document['candidates']);after=shortlist(reconstructed)
    if before!=after:raise AssertionError('changed proxy shortlist')
    return dict(candidates=len(document['candidates']),unique_states=len(groups),
        avoided_lowerings=len(document['candidates'])-len(groups),
        avoided_fraction=1-len(groups)/len(document['candidates']),key_seconds=time.perf_counter()-began,
        equivalence_classes=sum(len(g)>1 for g in groups.values()),
        executable_reuses=sum(len(g)-1 for g in groups.values() if g[0]['status']=='executable'),
        rejection_reuses=sum(len(g)-1 for g in groups.values() if g[0]['status']!='executable'),
        all_candidate_outcomes_identical=True,all_per_stack_proxy_shortlists_identical=True,
        shortlist_sha256=digest(before)),groups


def pointwise_fixture(seed=7):
    rng=np.random.default_rng(seed);qi=Quantization(.07,-37);qo=Quantization(.11,5)
    shape=(1,3,4,5);params=quantize_parameters(rng.normal(0,.2,(3,3,1,1)),
        rng.normal(0,.1,3),qi,qo)
    tensors={'x':Tensor('x',shape,qi,'NCHW'),'y':Tensor('y',shape,qo,'NCHW'),
             'z':Tensor('z',shape,qo,'NCHW')}
    layers=[Layer('Conv',['x'],'y',{'strides':[1,1],'pads':[0]*4,'dilations':[1,1]},params),
            Layer('Relu',['y'],'z',{}, {})]
    return Program(tensors,layers,['x'],['z'],{},{}),rng.integers(-128,128,shape,dtype=np.int8)


def small_configs(program):
    for start,stop in macro_stacks(program):
        shape=program.tensors[program.layers[stop-1].output].shape
        for h in sorted({1,(shape[2]+1)//2,shape[2]}):
            for w in sorted({1,(shape[3]+1)//2,shape[3]}):
                for retain,prefetch in ((False,False),(True,True)):
                    for mode in (1,2,3):
                        yield dict(start=start,stop=stop,kind='rectangle',h=h,w=w,mode=mode,
                                   retain=retain,prefetch=prefetch)


def synthetic(sources):
    results=[]
    cases=[('asymmetric-g1-zp128',spatial_fixture(seed=5,groups=1,zp=-128)),
           ('asymmetric-g2-zp37',spatial_fixture(seed=6,groups=2,zp=-37)),
           ('pointwise',pointwise_fixture())]
    for name,(program,source) in cases:
        oracle=evaluate(program,{program.inputs[0]:source})
        configs=list(small_configs(program));direct=[];began=time.perf_counter();replayed=0
        for cfg in configs:
            result,artifact=lower(program,cfg);direct.append(result)
            if artifact:
                initial=oracle[program.layers[cfg['start']].inputs[0]]
                replay_stack(program,cfg['start'],cfg['stop'],initial,*artifact);replayed+=1
        direct_seconds=time.perf_counter()-began
        keyer=StateKeys(program,sources,exact_halo=True);cache={};reused=[]
        began=time.perf_counter()
        for cfg in configs:
            key=keyer.key(cfg)
            if key not in cache:cache[key]=lower(program,cfg)[0]
            reused.append(cache[key])
        memo_seconds=time.perf_counter()-began
        if reused!=direct:raise AssertionError('small oracle mismatch '+name)
        # Fair timing omits replay from both sides and alternates execution order.
        timings={'plain':[],'memoized':[]}
        for order in (('memoized','plain'),('plain','memoized')):
            for strategy in order:
                began=time.perf_counter();memo={};fresh=StateKeys(program,sources,exact_halo=True)
                for cfg in configs:
                    key=fresh.key(cfg) if strategy=='memoized' else None
                    if strategy=='plain' or key not in memo:memo[key]=lower(program,cfg)[0]
                timings[strategy].append(time.perf_counter()-began)
        results.append(dict(name=name,candidates=len(configs),unique_states=len(cache),
            avoided_lowerings=len(configs)-len(cache),independent_integer_replays=replayed,
            outcomes_identical=True,direct_with_replay_seconds=direct_seconds,
            initial_memoized_seconds=memo_seconds,timings_seconds=timings,
            measured_search_speedup=statistics.median(timings['plain'])/statistics.median(timings['memoized'])))
        print('synthetic',name,results[-1]['measured_search_speedup'],flush=True)
    return results


def heldout(program,groups,count,sources):
    # Prespecified deterministic stratification across executable/rejected
    # state classes. These model candidates were not used in small-case design.
    selected=[]
    for status in ('executable','infeasible-for-backend'):
        pool=[g for g in groups.values() if len(g)>1 and g[0]['status']==status]
        pool.sort(key=lambda g:g[0]['id'])
        if pool:
            selected.extend(pool[i] for i in sorted(set(np.linspace(0,len(pool)-1,min(count,len(pool)),dtype=int))))
    rows=[];direct_seconds=0.;representative_seconds=0.
    for group in selected:
        pair=(group[0],group[-1]);artifacts=[]
        for index,row in enumerate(pair):
            began=time.perf_counter();result,artifact=lower(program,row['configuration']);duration=time.perf_counter()-began
            direct_seconds+=duration
            if index==0:representative_seconds+=duration
            if result!=outcome(row):raise AssertionError('held-out catalogue changed '+row['id'])
            artifacts.append(artifact)
        if (artifacts[0] is None)!=(artifacts[1] is None):raise AssertionError('held-out legality mismatch')
        if artifacts[0]:
            a,b=artifacts
            if a[:2]!=b[:2]:raise AssertionError('held-out byte mismatch')
            ar,br=copy.deepcopy(a[2]),copy.deepcopy(b[2]);ar.pop('mode');br.pop('mode')
            if ar!=br:raise AssertionError('held-out live/transfer/geometry state differs')
        rows.append(dict(representative=pair[0]['id'],heldout=pair[1]['id'],status=pair[0]['status'],
                         exact_bytes_and_record_except_mode=artifacts[0] is not None,outcome=outcome(pair[0])))
    return dict(classes=len(rows),fresh_lowerings=2*len(rows),all_outcomes_identical=True,
                independent_fresh_lowering_seconds=direct_seconds,
                representative_lowering_seconds=representative_seconds,rows=rows)


def sensitivity(sources):
    program,_=pointwise_fixture();cfg=next(small_configs(program))
    original=StateKeys(program,sources,exact_halo=True).key(cfg);changes={}
    for name in ('weights','corrected_bias','multiplier','zero_point','shape'):
        p=copy.deepcopy(program)
        if name=='weights':p.layers[0].parameters['weight'][-1]=0
        elif name in ('corrected_bias','multiplier'):p.layers[0].parameters[name][0]+=1
        elif name=='zero_point':
            p.tensors['x']=replace(p.tensors['x'],quantization=replace(p.tensors['x'].quantization,zero_point=-36))
        else:p.tensors['x']=replace(p.tensors['x'],shape=(1,3,4,6))
        changes[name]=StateKeys(p,sources,exact_halo=True).key(cfg)!=original
    for name,modified in (('command_capacity',dict(ABI,command_capacity_bytes=16384)),
                          ('alignment',dict(ABI,allocation_alignment=16)),
                          ('live_buffers',dict(ABI,initial_live_buffers=[[128,144]]))):
        changes[name]=StateKeys(program,sources,exact_halo=True,abi=modified).key(cfg)!=original
    changes['backend_sources']=StateKeys(program,dict(sources,test='changed'),exact_halo=True).key(cfg)!=original
    for field in ('retain','prefetch'):
        altered=dict(cfg);altered[field]=not altered[field]
        changes[field]=StateKeys(program,sources,exact_halo=True).key(altered)!=original
    if not all(changes.values()):raise AssertionError('unsafe key sensitivity')
    altered=copy.deepcopy(program);altered.layers[0].parameters['weight'][-1]=0
    before,_=lower(program,cfg);after,_=lower(altered,cfg)
    if before==after:raise AssertionError('shape-only weak-key counterexample ineffective')
    return dict(safe_key_changes=changes,shape_only_key_counterexample=dict(
        same_shapes=True,same_configuration=True,changed_last_weight_row_to_zero=True,
        before=before,after=after,conclusion='shape/config-only reuse is unsound'))


def run(output,holdout_per_status=12,exact_halo=True):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('choose fresh hypothesis output')
    output.mkdir(parents=True)
    sources=baseline.source_pins();sources['tools/phase6/hypothesis_search.py']=baseline.sha(Path(__file__))
    sources.update({name:baseline.sha(ROOT/name) for name in (
        'tools/phase6/matched_current.py','compiler/test_scheduler_spatial.py',
        'compiler/phase4_compile.py','compiler/quantization.py','compiler/hardware_v2.py',
        'compiler/integer_reference.py','compiler/static_pipeline.py','compiler/scheduler/spatial.py')})
    report=dict(status='running',physical_board=False,scope='whole-candidate search equivalence; unchanged accelerator/ABI',
        source_sha256=sources,entry_abi=ABI,synthetic=synthetic(sources),sensitivity=sensitivity(sources),models={})
    save(output/'report.partial.json',report)
    for model in ('kws','vww'):
        original,pinned,program,maps,provenance=baseline.matched_model(model)
        catalogue=ROOT/f'work/phase6/matched-baselines-v1/b3-final-{model}/{model}/catalogue.json'
        document=json.loads(catalogue.read_text())
        if document['identity']['graph_sha256']!=graph_identity(program):raise ValueError('catalogue graph changed')
        for path,expected in document['identity']['compiler_sources'].items():
            if baseline.sha(ROOT/path)!=expected:raise ValueError('catalogue compiler changed: '+path)
        axis,groups=audit_catalogue(program,document,sources,False)
        exact,groups=(audit_catalogue(program,document,sources,True) if exact_halo else (axis,groups))
        validation=heldout(program,groups,holdout_per_status,sources)
        report['models'][model]=dict(catalogue_file=str(catalogue.relative_to(ROOT)),catalogue_sha256=baseline.sha(catalogue),
            provenance=provenance,historical_catalogue_seconds=document['seconds'],axis_only=axis,
            exact_halo=exact,heldout=validation,naive_exact_config_memoization_avoided_lowerings=0)
        save(output/'report.partial.json',report)
        print(model,'avoided',exact['avoided_lowerings'],'/',exact['candidates'],'key seconds',exact['key_seconds'],flush=True)
    for path,expected in sources.items():
        if baseline.sha(ROOT/path)!=expected:raise ValueError('experiment source changed: '+path)
    report.update(status='passed',decision='useful standard memoization plus inactive-mode canonicalization; novelty not established',
        limitations=['Historical catalogue proof compares recorded outcomes; only held-out real candidates are freshly lowered.',
                     'Proxy winners preserved by identical outcomes; no newly measured native/board speedup is claimed.',
                     'Whole standalone lowering only: no safe equivalence for arbitrary live partial SRAM states is established.',
                     'Two timing repetitions per strategy are an engineering screen, not a publication performance study.',
                     'Compiler search work reduction does not change inference latency, accuracy, power or hardware resources.'])
    save(output/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--holdout-per-status',type=int,default=12)
    parser.add_argument('--axis-only',action='store_true')
    args=parser.parse_args();run(args.output,args.holdout_per_status,not args.axis_only)
