#!/usr/bin/env python3
"""Bounded paired screen of whole-stack immutable residency after B3 search."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
from types import FunctionType

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'compiler'),str(ROOT/'tools/phase6')]
import matched_defines_baseline as baseline
from matched_b1b2_generic import matched_model
from matched_current import graph_identity
from channel_compaction import check_oracles
from scheduler.matched_defines import compile_stack as generic,replay_stack
from scheduler.matched_defines_weights import compile_stack,implementation
from test_scheduler_spatial import spatial_fixture

BASE=ROOT/'work/phase6/matched-baselines-v1/b3-weight-resident'


def compile_config(program,cfg):
    if cfg.get('resident_weights'):
        return compile_stack(program,cfg['start'],cfg['stop'],cfg['h'],cfg['w'],cfg['mode'],
            retain_weights=cfg['retain'],prefetch=cfg['prefetch'],resident_weights=True)
    return baseline.compile_config(program,cfg)


compose=FunctionType(baseline.compose.__code__,dict(baseline.compose.__globals__,compile_config=compile_config),
                     'compose_with_resident',baseline.compose.__defaults__)


def save_fixture(directory,code,payload,source,expected,record):
    directory.mkdir(parents=True,exist_ok=True)
    data={'commands.bin':code,'payload.bin':payload,'input.bin':source.tobytes(),
          'output.bin':expected.tobytes(),'checks.txt':f'{record["output_external_base"]} output.bin\n'.encode()}
    for name,value in data.items():(directory/name).write_bytes(value)
    baseline.save(directory/'schedule.json',record)


def run(inputs,output=BASE,max_pairs=8):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    sources=dict(baseline.source_pins(),**{
        'compiler/scheduler/matched_defines_weights.py':baseline.sha(ROOT/'compiler/scheduler/matched_defines_weights.py'),
        'tools/phase6/matched_defines_weights.py':baseline.sha(Path(__file__))})
    report=dict(schema=1,status='running',physical_board=False,compiler_sources=sources,
        generated_lowering_sha256=baseline.sha(implementation()[1].encode()),native_sha256=baseline.sha(baseline.NATIVE),
        scope='bounded paired whole-stack weights/parameters/LUT residency screen; all reservation bytes charged',
        budget=dict(max_complete_pairs_per_model=max_pairs,base_paths=3,height_variants='same/ceil-half/ceil-quarter',
                    subsets='all stacks and one stack at a time'),synthetic=[],models={})
    for seed,groups in ((5,1),(6,2)):
        p,x=spatial_fixture(seed=seed,groups=groups)
        expected=baseline.evaluate(p,{p.inputs[0]:x})[p.outputs[0]]
        for mode,ty,tx in ((1,3,8),(2,5,4),(3,3,8)):
            if compile_stack(p,0,4,ty,tx,mode)!=generic(p,0,4,ty,tx,mode):raise ValueError('default variant differs')
            code,payload,record=compile_stack(p,0,4,ty,tx,mode,resident_weights=True)
            proof=replay_stack(p,0,4,x,code,payload,record)
            directory=output/'synthetic'/f's{seed}-g{groups}-m{mode}'
            save_fixture(directory,code,payload,x,expected,record)
            natives=[baseline.native_measure(directory,stall,output/f'synthetic-s{seed}-g{groups}-m{mode}-n{stall}.json') for stall in (0,6063)]
            report['synthetic'].append(dict(seed=seed,groups=groups,mode=mode,default_byte_equivalent=True,
                resident_bytes=record['resident_immutable_bytes'],replay=proof,native=natives,directory=str(directory.relative_to(ROOT))))
    for source in inputs:
        source=Path(source);search=json.loads(source.read_text())
        if search['status']!='passed':raise ValueError('B3 search incomplete')
        for model,info in search['models'].items():
            original,pinned,program,maps,provenance=matched_model(model)
            if graph_identity(program)!=info['graph_sha256']:raise ValueError('model changed')
            oracle=check_oracles(original,program,maps,pinned)
            attempts=[];pairs=[];seen=set()
            for entry in info['candidates'][:3]:
                path=entry['configuration_path']
                choices=[tuple(range(len(path)))]+[(i,) for i in range(len(path))]
                for indices in choices:
                    for divisor in (1,2,4):
                        configs=copy.deepcopy(path)
                        for i in indices:
                            configs[i].update(kind='rectangle',retain=True,prefetch=True,resident_weights=True,
                                h=max(1,(configs[i]['h']+divisor-1)//divisor))
                        signature=json.dumps(configs,sort_keys=True)
                        if signature in seen:continue
                        seen.add(signature);trial=len(attempts)
                        directory=output/model/f'pair-{trial:03d}'
                        tail=dict(configuration=dict(kind='fallback',start=max(c['stop'] for c in configs),stop=len(program.layers)))
                        try:
                            code,payload,record=compose(program,[dict(configuration=c) for c in configs],tail,oracle,directory/'resident-components')
                        except ValueError as e:
                            if str(e) not in ('sram-capacity','command-capacity','composed-command-capacity','external-capacity'):
                                raise
                            attempts.append(dict(configurations=configs,status='infeasible-for-backend',reason=str(e)));continue
                        control=copy.deepcopy(configs)
                        for cfg in control:cfg.pop('resident_weights',None)
                        cc,cp,cr=compose(program,[dict(configuration=c) for c in control],tail,oracle,directory/'control-components')
                        rows={}
                        for label,c,p,r in (('control',cc,cp,cr),('resident',code,payload,record)):
                            fixture=directory/label;files=baseline.fixture(fixture,program,oracle,c,p,r)
                            native=[baseline.native_measure(fixture,stall,directory/f'{label}-s{stall}.json') for stall in (0,6063)]
                            rows[label]=dict(directory=str(fixture.relative_to(ROOT)),files=files,native=native,replay=r['replay'])
                        ratio=math.sqrt(math.prod(a['elapsed_cycles']/b['elapsed_cycles'] for a,b in zip(rows['control']['native'],rows['resident']['native'])))
                        attempts.append(dict(configurations=configs,status='passed-native',paired_geomean_speedup=ratio))
                        pairs.append(dict(configurations=configs,tail=tail,paired_geomean_speedup=ratio,**rows))
                        print(model,'resident weights pair',trial,'speedup',ratio,flush=True)
                        if len(pairs)>=max_pairs:break
                    if len(pairs)>=max_pairs:break
                if len(pairs)>=max_pairs:break
            report['models'][model]=dict(graph_sha256=info['graph_sha256'],attempts=attempts,pairs=pairs,
                source_search=dict(file=str(source.relative_to(ROOT)),sha256=baseline.sha(source)))
            baseline.save(output/'report.json',report)
    for name,digest in sources.items():
        if baseline.sha(ROOT/name)!=digest:raise ValueError('resident variant source changed')
    report['status']='passed';baseline.save(output/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,default=BASE)
    parser.add_argument('--max-pairs',type=int,default=8)
    args=parser.parse_args();run(args.inputs,args.output,args.max_pairs)
