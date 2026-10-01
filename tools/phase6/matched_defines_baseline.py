#!/usr/bin/env python3
"""Tune an executable DeFiNES policy adaptation on the frozen Tang Nano core.

Enumerates contiguous Conv/activation stacks, rectangle sizes and all three
published halo modes, using the same exact graph transformations as B1/B2.
Native RTL measures final candidate paths; proxy scores only shortlist paths.
No native score is represented as an electrical board measurement.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import time
from collections import Counter

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'compiler'),str(ROOT/'tools/phase6')]
from matched_b1b2_generic import matched_model,compile_candidate,write_fixture,save,sha,native_identity,NATIVE
from channel_compaction import check_oracles
from hardware_v2 import Descriptor
from integer_reference import evaluate
from scheduler.matched_defines import compile_stack,replay_stack
from scheduler.matched_defines_strip import compile_strip_stack,replay_strip_stack,macro_stacks,block_program,align8
from output_pipeline_fusion import decode_fused
from matched_current import graph_identity
from scheduler.fused_verify import replay_fused

BASE=ROOT/'work/phase6/matched-baselines-v1/b3'
CMD=struct.Struct('<BBHIII')
REVISION='7097d6090dc22321e44ce91434e7cc23b065864f'


def source_pins():
    names=['compiler/scheduler/matched_defines.py','compiler/scheduler/matched_defines_strip.py',
           'compiler/scheduler/matched_defines_regions.py','compiler/scheduler/defines_adapter.py',
           'tools/phase6/matched_defines_baseline.py','tools/phase6/matched_b1b2_generic.py',
           'tools/phase6/novelty_constants.py','tools/phase6/channel_compaction.py',
           'tools/phase6/output_pipeline_fusion.py']
    return {n:sha(ROOT/n) for n in names}


def upstream_policy(program):
    """Execute pinned upstream weight-cut and three-mode geometry functions."""
    from scheduler.defines_adapter import segment_workload
    source=ROOT/f'work/phase6/defines-source/DeFiNES-{REVISION}'
    pins=json.loads((ROOT/'docs/research/evidence/phase0/prior-code.json').read_text())
    for entry in pins['files']:
        if sha(source/entry['path'])!=entry['sha256']:raise ValueError('upstream source changed')
    sys.path.insert(0,str(source))
    from classes.workload.dnn_workload import DNNWorkload
    from classes.stages.DepthFirstStage import backpropagate_tilesize
    from classes.stages.DfStackCutIfWeightsOverflowStage import DfStackCutIfWeightsOverflowStage
    from classes.stages.Stage import Stage
    from types import SimpleNamespace
    class Capture(Stage):
        def is_leaf(self):return True
        def run(self):
            yield self.kwargs['df_stack_cuts'],None
    memory=SimpleNamespace(memory_instance=SimpleNamespace(size=32768*8))
    core=SimpleNamespace(memory_hierarchy=SimpleNamespace(get_memory_levels=lambda op:[memory,memory]))
    accelerator=SimpleNamespace(get_core=lambda idx:core)
    stacks=list(macro_stacks(program));maximal=[]
    for a,b in stacks:
        if not any(c<=a and b<=d and (c,d)!=(a,b) for c,d in stacks):maximal.append((a,b))
    records=[]
    for a,b in maximal:
        workload=segment_workload(program,a,b);graph=DNNWorkload(workload)
        cuts=list(DfStackCutIfWeightsOverflowStage([Capture],accelerator=accelerator,workload=graph).run())[0][0]
        shape=program.tensors[program.layers[b-1].output].shape
        geometry=[]
        for mode,hc,vc in ((1,False,False),(2,True,False),(3,True,True)):
            graph=DNNWorkload(workload)
            backpropagate_tilesize(graph,min(4,shape[3]),min(4,shape[2]),hc,hc,vc,vc)
            inp=graph.get_node_with_id(-1)
            geometry.append(dict(mode=mode,horizontal_cache=hc,vertical_cache=vc,
                new_input_tile=[inp.loop_dim_size['OY'],inp.loop_dim_size['OX']]))
        records.append(dict(start=a,stop=b,weight_budget_bytes=32768,source_stack_cuts=cuts,
            geometry=geometry))
    return dict(revision=REVISION,files=pins['files'],executed=['DfStackCutIfWeightsOverflowStage','backpropagate_tilesize'],
        runs=records,adaptation=dict(
            tile_space='bounded explicit height/width grid and all contiguous macro-stack cuts',
            macro='Conv plus exactly requantized Relu/Clip; source quantization barriers preserved',
            cost='fixed descriptor dataflow structural shortlist followed by whole-program native RTL timing',
            memory='physical single 32KiB SRAM plus 8MiB SDRAM, exact COPY/DMA and cache bytes',
            upstream_loma='replaced by fixed executable eight-lane kernel; no freedom to implement unsupported MAC mappings',
            automatic_cuts='source weight-overflow cuts recorded; extra contiguous cuts explored when fixed ABI SRAM/command limits require them',
            objectives='latency only; no energy claim or fabricated SRAM energy coefficients'))


def dimensions(n):
    return sorted({1,n,max(1,(n+1)//2),max(1,(n+3)//4),min(4,n),min(8,n),min(16,n)})


def proxy(code,record):
    """Uncalibrated structural shortlist; never reported as measured cycles."""
    score=0
    for r in record.get('runs',record.get('run_records',[])):
        if r.get('kind')=='copy':
            d=Descriptor.decode(bytes.fromhex(r['descriptor_hex']));score+=36+5*d.outputs;continue
        d,_=decode_fused(bytes.fromhex(r['descriptor_hex']))
        plane=d.outputs//d.output_c
        if d.opcode==4 and d.kernel_h==d.kernel_w==1 and d.count<=256:
            score+=39+d.output_c*((plane+7)//8)*(2*d.input_c+(d.input_c+7)//8+6)
        elif d.opcode==6 and d.kernel_h==d.kernel_w==3:
            score+=39+d.outputs*9
        else:score+=39+d.outputs*(8+4*((d.count+7)//8))
    for i in range(len(code)//16):
        op,flags,_,a,b,c=CMD.unpack_from(code,i*16)
        if op==1:score+=36+2*((c+7)//8)
        elif op in (2,3):score+=2
    return score


def configuration_key(start,stop,kind,h,w,mode,retain=True,prefetch=True):
    return f'{start:02d}-{stop:02d}-{kind}-{h}x{w}-m{mode}-r{int(retain)}-p{int(prefetch)}'


def compile_config(program,cfg):
    if cfg['kind']=='strip':return compile_strip_stack(program,cfg['start'],cfg['stop'],cfg['h'])
    return compile_stack(program,cfg['start'],cfg['stop'],cfg['h'],cfg['w'],cfg['mode'],
                         retain_weights=cfg['retain'],prefetch=cfg['prefetch'])


def enumerate_catalogue(program,output):
    rows=[];best={};began=time.monotonic();counts=Counter()
    identity=dict(graph_sha256=graph_identity(program),compiler_sources=source_pins())
    for start,stop in macro_stacks(program):
        shape=program.tensors[program.layers[stop-1].output].shape
        configs=[]
        for h in dimensions(shape[2]):
            configs.append(dict(start=start,stop=stop,kind='strip',h=h,w=shape[3],mode=1,retain=False,prefetch=False))
            for w in dimensions(shape[3]):
                for mode in (1,2,3):
                    configs.append(dict(start=start,stop=stop,kind='rectangle',h=h,w=w,mode=mode,retain=True,prefetch=True))
        seen=set()
        for cfg in configs:
            key=configuration_key(start,stop,cfg['kind'],cfg['h'],cfg['w'],cfg['mode'],cfg['retain'],cfg['prefetch'])
            row=dict(id=key,configuration=cfg)
            try:
                code,payload,record=compile_config(program,cfg)
            except ValueError as e:
                reason=str(e)
                if reason not in ('sram-capacity','command-capacity','external-capacity','dma-stripe-alignment',
                        'constant-tail-alignment','dma-alignment','dma-bounds'):
                    raise
                row.update(status='infeasible-for-backend',reason=reason);counts[reason]+=1
            else:
                row.update(status='executable',proxy_score=proxy(code,record),command_bytes=len(code),
                    payload_bytes=len(payload),code_sha256=sha(code),payload_sha256=sha(payload),
                    peak_sram_address=record['peak_sram_address'],counters=record.get('counters',{}))
                counts['executable']+=1
                signature=sha(code+payload)
                if signature in seen:row['duplicate_within_stack']=True
                else:
                    seen.add(signature)
                    edge=best.setdefault((start,stop),[]);edge.append(row)
                    edge.sort(key=lambda r:(r['proxy_score'],r['command_bytes'],r['id']))
                    del edge[4:]
            rows.append(row)
        save(output/'catalogue.partial.json',dict(identity=identity,candidates=rows,counts=dict(counts)))
        if len(best)%20==0:
            print(f'B3 enumerate {start}:{stop} candidates={len(rows)} feasible={counts["executable"]}',flush=True)
    report=dict(status='enumerated',identity=identity,seconds=time.monotonic()-began,counts=dict(counts),candidates=rows,
        shortlist_per_stack=4,grid='unique 1/full/ceil-half/ceil-quarter/4/8/16 clamped to shape',
        ranking_scope='uncalibrated structural proxy; every reported final performance is measured by native RTL')
    save(output/'catalogue.json',report)
    return best,report


def k_paths(best,start,end,k=12,max_bytes=32768):
    state={start:[(0,0,[])]}
    for pos in range(start,end):
        for score,size,path in list(state.get(pos,[])):
            for (a,b),options in best.items():
                if a!=pos:continue
                for row in options:
                    n=size+row['command_bytes']-16
                    if n+16>max_bytes:continue
                    dest=state.setdefault(b,[]);dest.append((score+row['proxy_score'],n,path+[row]))
                    dest.sort(key=lambda x:(x[0],x[1],tuple(r['id'] for r in x[2])))
                    unique=[];seen=set()
                    for candidate in dest:
                        sig=tuple(r.get('code_sha256',r['id']) for r in candidate[2])
                        if sig not in seen:unique.append(candidate);seen.add(sig)
                        if len(unique)==k:break
                    state[b]=unique
    return state.get(end,[])


def native_measure(directory,seed,output):
    files={p.name:sha(p) for p in directory.iterdir() if p.is_file()}
    executable=sha(NATIVE)
    with output.with_suffix('.log').open('w') as log:
        subprocess.run([str(NATIVE),str(directory),str(seed),str(output)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    if executable!=sha(NATIVE) or files!={p.name:sha(p) for p in directory.iterdir() if p.is_file()}:
        raise ValueError('native inputs changed during execution')
    result=json.loads(output.read_text())
    if result['status']!='passed':raise ValueError('native mismatch')
    result.update(executable_sha256=executable,fixture_files=files)
    save(output,result)
    return result


def compose(program,path,tail,oracle,output):
    """Relocate standalone verified segments into two shared activation slots."""
    slot=align8(max(math.prod(t.shape) for t in program.tensors.values()))
    payload=bytearray(2*slot);commands=[];components=[];current=0
    for position,row in enumerate(path+[tail]):
        cfg=row['configuration'];start,stop=cfg['start'],cfg['stop']
        fragment=block_program(program,start,stop)
        if cfg['kind']=='fallback':
            directory=output/f'component-{position}'
            code,data,record=compile_candidate(fragment,policy='B2',tile_budget=32768,prefetch=True,snapshots=False,directory=directory)
            initial=oracle[program.layers[start].inputs[0]]
            local_oracle=evaluate(fragment,{fragment.inputs[0]:initial})
            proof=replay_fused(fragment,code,data,{fragment.inputs[0]:initial},oracle=local_oracle,
                run_contracts=record['run_contracts'],constant_contracts=record.get('constant_contracts'),
                final_output=record['final_output'],snapshot_regions=record.get('snapshot_regions'))
            local_slot=align8(max(math.prod(t.shape) for t in fragment.tensors.values()))
            parameter_start=2*local_slot
            final_slot=record['final_output']['ext']//local_slot
            def activation_address(a):
                local=a//local_slot;within=a%local_slot
                return (current if local==0 else 1-current)*slot+within
            output_after=current if final_slot==0 else 1-current
        else:
            code,data,record=compile_config(program,cfg)
            source=oracle[program.layers[start].inputs[0]]
            proof=(replay_strip_stack if cfg['kind']=='strip' else replay_stack)(program,start,stop,source,code,data,record)
            out_base=record['output_external_base'];parameter_start=out_base+align8(record['output_bytes'])
            def activation_address(a):
                return current*slot+a if a<out_base else (1-current)*slot+a-out_base
            output_after=1-current
        append=align8(len(payload));payload.extend(bytes(append-len(payload)));payload.extend(data[parameter_start:])
        first=len(commands)
        for i in range(len(code)//16-1):
            op,flags,res,a,b,c=CMD.unpack_from(code,i*16)
            if op==1:
                if a<parameter_start:
                    if a+c>parameter_start:raise ValueError('DMA straddles activation/immutable boundary')
                    a=activation_address(a)
                else:a=append+a-parameter_start
                if a+c>len(payload):raise ValueError('relocated DMA out of external bounds')
            commands.append((op,flags,res,a,b,c))
        components.append(dict(start=start,stop=stop,kind=cfg['kind'],first_command=first,
            last_command=len(commands),configuration=cfg,standalone_replay=proof,
            standalone_code_sha256=sha(code),standalone_payload_sha256=sha(data),
            input_slot=current,output_slot=output_after,immutable_external_base=append,
            standalone_immutable_base=parameter_start))
        current=output_after
    commands.append((0,0,0,0,0,0));code=b''.join(CMD.pack(*c) for c in commands)
    if len(code)>32768:raise ValueError('composed-command-capacity')
    if len(payload)>8*1024*1024:raise ValueError('composed-external-capacity')
    record=dict(schema=1,components=components,final_output=dict(ext=current*slot,bytes=oracle[program.outputs[0]].nbytes),
        activation_slot_bytes=slot,program_sha256=sha(code),image_sha256=sha(payload),
        replay=dict(status='passed',scope='each segment independently replayed; relocation preserves two disjoint activation slots and immutable data; full composed RTL checked separately'))
    return code,bytes(payload),record


def fixture(directory,program,oracle,code,payload,record):
    directory.mkdir(parents=True,exist_ok=True)
    initial=oracle[program.layers[0].output]
    data={'commands.bin':code,'payload.bin':payload,'input.bin':initial.tobytes(),
          'output.bin':oracle[program.outputs[0]].tobytes(),
          'checks.txt':f'{record["final_output"]["ext"]} output.bin\n'.encode(),
          'schedule.json':(json.dumps(record,indent=2,sort_keys=True)+'\n').encode()}
    for name,raw in data.items():(directory/name).write_bytes(raw)
    return {name:sha(raw) for name,raw in data.items()}


def run(output=BASE,models=('kws','vww'),native_candidates=8):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);sources=source_pins()
    report=dict(schema=1,status='running',physical_board=False,native=native_identity(),native_sha256=sha(NATIVE),
        compiler_sources=sources,coverage=dict(policy_axes=['all contiguous Conv/activation stacks',
            '2D tile grid','mode1 recompute','mode2 horizontal cache','mode3 horizontal and vertical cache',
            'full-width efficient lowering','safe immutable SRAM reuse','safe DMA prefetch'],
            unsupported_dimensions=['MAC dataflow and reduction parallelization fixed by same eight-lane engine',
                'in-place cache rotation not implemented; conservative double cache buffers charged',
                'nonspatial tails use same tuned compiler backend'],
            common_graph='matched_model: exact channel transforms and final constant fold'),
        tuning=dict(native_candidates=native_candidates,validation_input_seed=6157,stall_seeds=[0,6063],
            choice='native seed0 whole-program latency after documented structural shortlist',
            source_native_identity_scope='same pooled27MHz eight-lane engine as B1/B2'),models={},selected={})
    save(output/'report.json',report)
    for model in models:
        directory=output/model;directory.mkdir(parents=True,exist_ok=True)
        original,pinned,program,maps,provenance=matched_model(model)
        oracle=check_oracles(original,program,maps,pinned)
        prior=upstream_policy(program);save(directory/'upstream.json',prior)
        catalogue=directory/'catalogue.json'
        expected_identity=dict(graph_sha256=graph_identity(program),compiler_sources=source_pins())
        if catalogue.exists() and json.loads(catalogue.read_text()).get('identity')==expected_identity:
            cached=json.loads(catalogue.read_text());best={}
            for row in cached['candidates']:
                if row['status']=='executable' and not row.get('duplicate_within_stack'):
                    cfg=row['configuration'];edge=best.setdefault((cfg['start'],cfg['stop']),[]);edge.append(row)
                    edge.sort(key=lambda r:(r['proxy_score'],r['command_bytes'],r['id']));del edge[4:]
            enumeration=cached
        else:best,enumeration=enumerate_catalogue(program,directory)
        ends=[r['stop'] for r in [row['configuration'] for choices in best.values() for row in choices]]
        spatial_end=max(ends)
        tail=dict(id='nonspatial-tail',configuration=dict(kind='fallback',start=spatial_end,stop=len(program.layers)))
        paths=k_paths(best,1,spatial_end,k=max(native_candidates*4,24),max_bytes=30000)
        candidates=[];seen=set()
        for rank,(_,_,path) in enumerate(paths):
            if len(candidates)>=native_candidates:break
            candidate_dir=directory/'candidates'/f'path-{rank:03d}'
            try:code,payload,record=compose(program,path,tail,oracle,candidate_dir)
            except ValueError as e:
                if str(e) not in ('composed-command-capacity','composed-external-capacity'):raise
                continue
            identity=sha(code+payload)
            if identity in seen:continue
            seen.add(identity);timed=candidate_dir/'pinned';files=fixture(timed,program,oracle,code,payload,record)
            native=native_measure(timed,0,candidate_dir/'native-pinned-s0.json')
            row=dict(rank=rank,path=[r['id'] for r in path],configuration_path=[r['configuration'] for r in path],
                directory=str(timed.relative_to(ROOT)),files=files,native=[native],replay=record['replay'])
            candidates.append(row)
            print(model,'B3 path',rank,'cycles',native['elapsed_cycles'],flush=True)
        if not candidates:raise ValueError('no executable complete B3 path')
        candidates.sort(key=lambda r:(r['native'][0]['elapsed_cycles'],r['rank']))
        winner=candidates[0];selected={}
        timed=ROOT/winner['directory'];winner['native'].append(native_measure(timed,6063,timed.parent/'native-pinned-s6063.json'))
        selected['pinned']=winner
        value=np.random.default_rng(6157).integers(-128,128,pinned.shape,dtype=np.int8)
        stress_oracle=check_oracles(original,program,maps,value)
        chosen_path=[dict(configuration=cfg) for cfg in winner['configuration_path']]
        stress_dir=timed.parent/'stress'
        code,payload,record=compose(program,chosen_path,tail,stress_oracle,stress_dir.parent/'stress-components')
        files=fixture(stress_dir,program,stress_oracle,code,payload,record)
        selected['stress']=dict(directory=str(stress_dir.relative_to(ROOT)),files=files,replay=record['replay'],
            native=[native_measure(stress_dir,seed,stress_dir.parent/f'native-stress-s{seed}.json') for seed in (0,6063)])
        report['models'][model]=dict(graph_sha256=graph_identity(program),provenance=provenance,upstream=prior,catalogue_counts=enumeration['counts'],
            catalogue_sha256=sha(catalogue),catalogue_file=str(catalogue.relative_to(ROOT)),candidates=candidates,
            frontier=candidates[:3],winner_rank=winner['rank'])
        report['selected'][model]=selected;save(output/'report.json',report)
    if source_pins()!=sources:raise ValueError('compiler changed during tuning')
    report['status']='passed';save(output/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE)
    parser.add_argument('--models',nargs='+',choices=('kws','vww'),default=['kws','vww'])
    parser.add_argument('--native-candidates',type=int,default=8)
    args=parser.parse_args();run(args.output,args.models,args.native_candidates)
