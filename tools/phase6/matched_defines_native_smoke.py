#!/usr/bin/env python3
"""Bounded synthetic arithmetic/cache/DMA screen for the rectangular backend.

The native binary is the frozen 27 MHz image's cycle simulator. These timings
are synthetic fixtures, not full-model speedups or physical-board measurements.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'compiler'),str(ROOT/'tools/phase6')]
from integer_reference import evaluate
from scheduler.matched_defines import compile_stack,replay_stack
from test_scheduler_spatial import spatial_fixture

NATIVE=ROOT/'work/phase6/pool-timing-v1/native/Vv2_tiled_host_bridge'
NATIVE_SHA='b97fbf944827a5345670f3603e7f10368a897e6bcb19d16cc56905d8e06d16ec'
SOURCES=('tools/phase6/matched_defines_native_smoke.py',
    'compiler/scheduler/matched_defines.py','compiler/scheduler/matched_defines_regions.py',
    'compiler/scheduler/matched_defines_strip.py','compiler/scheduler/spatial.py',
    'compiler/test_scheduler_spatial.py','compiler/integer_reference.py',
    'compiler/phase4_compile.py','compiler/quantization.py',
    'tools/phase6/output_pipeline_fusion.py','tools/phase6/matched_b1b2_generic.py',
    'tools/phase6/prefetch_tail.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')


def verify_native():
    report=json.loads((NATIVE.parent/'report.json').read_text())
    if (report['status']!='passed' or report['executable_sha256']!=NATIVE_SHA
            or sha(NATIVE)!=NATIVE_SHA):raise ValueError('unmatched native binary')
    for name,digest in report['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('native source changed: '+name)
    return report['sources']


def run(output,native_seeds=(0,6063),replay_only=False):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    native_sources=verify_native()
    sources={name:sha(ROOT/name) for name in SOURCES}
    report=dict(schema=1,status='running',physical_board=False,source_sha256=sources,
        native_sha256=NATIVE_SHA,native_sources=native_sources,results=[],rejections=[],
        scope='synthetic exactness and cache coverage; no full-model or board performance claim')
    save(output/'report.json',report)
    try:
        for seed,groups,zp in itertools.product((5,6),(1,2),(-128,-37)):
            program,source=spatial_fixture(seed=seed,groups=groups,zp=zp)
            oracle=evaluate(program,{program.inputs[0]:source})
            shape=program.tensors[program.outputs[0]].shape
            tiles=(('full',shape[2],shape[3]),('horizontal',shape[2],4),
                   ('vertical',3,shape[3]),('both',3,4))
            for mode,(tile_name,th,tw) in itertools.product((1,2,3),tiles):
                name=f's{seed}-g{groups}-zp{zp}-m{mode}-{tile_name}'
                params=dict(name=name,seed=seed,groups=groups,zero_point=zp,mode=mode,
                            tile=[th,tw],prefetch=True,retain_weights=True)
                try:
                    code,payload,record=compile_stack(program,0,4,th,tw,mode,
                                                      retain_weights=True,prefetch=True)
                except ValueError as error:
                    if str(error) not in ('command-capacity','sram-capacity','external-capacity'):
                        raise
                    report['rejections'].append(dict(params,reason=str(error)))
                    save(output/'report.json',report)
                    print(name,str(error),flush=True)
                    continue
                replay=replay_stack(program,0,4,source,code,payload,record)
                directory=output/name;directory.mkdir(parents=True,exist_ok=True)
                files={'commands.bin':code,'payload.bin':payload,'input.bin':source.tobytes(),
                    'output.bin':oracle[program.outputs[0]].tobytes(),
                    'checks.txt':f'{record["output_external_base"]} output.bin\n'.encode(),
                    'schedule.json':(json.dumps(record,sort_keys=True,indent=2)+'\n').encode()}
                for filename,data in files.items():(directory/filename).write_bytes(data)
                pins={filename:sha(directory/filename) for filename in files}
                row=dict(params,fixture_files=pins,replay=replay,native=[],
                    directory=str(directory.relative_to(ROOT)),traffic=record['traffic'],
                    counters=record['counters'],command_bytes=record['command_bytes'])
                for native_seed in (() if replay_only else native_seeds):
                    path=directory/f'native-s{native_seed}.json'
                    with (directory/f'native-s{native_seed}.log').open('w') as log:
                        subprocess.run([str(NATIVE),str(directory),str(native_seed),str(path)],
                            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60,cwd=ROOT)
                    value=json.loads(path.read_text())
                    if value['status']!='passed' or value['tensor_checks']!=1:
                        raise ValueError('native did not check exact final tensor')
                    if sha(NATIVE)!=NATIVE_SHA or any(sha(directory/n)!=v for n,v in pins.items()):
                        raise ValueError('native input changed while measuring')
                    value.update(executable_sha256=NATIVE_SHA,fixture_files=dict(pins))
                    save(path,value);row['native'].append(value)
                report['results'].append(row)
                save(output/'report.json',report)
                print(name,'passed',record['command_bytes'],record['counters']['cache_read_bytes'],flush=True)
        if {name:sha(ROOT/name) for name in SOURCES}!=sources:
            raise ValueError('compiler changed during matrix; rerun on stable sources')
        verify_native()
        if len(report['results'])+len(report['rejections'])!=96:
            raise ValueError('missing case in declared matrix')
        for mode in (1,2,3):
            if not any(row['mode']==mode for row in report['results']):
                raise ValueError('no passing coverage for cache mode')
        if not any(row['mode']==3 and row['counters']['cache_read_bytes']>0 for row in report['results']):
            raise ValueError('vertical/horizontal caching never exercised')
        report.update(status='passed-replay' if replay_only else 'passed',
            compiled_cases=len(report['results']),capacity_rejections=len(report['rejections']),
            native_runs=sum(len(r['native']) for r in report['results']))
        save(output/'report.json',report)
        return report
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        save(output/'report.json',report)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'work/phase6/matched-defines-native-smoke-v1')
    parser.add_argument('--replay-only',action='store_true')
    args=parser.parse_args()
    report=run(args.output,replay_only=args.replay_only)
    print(json.dumps({k:report[k] for k in ('status','compiled_cases','capacity_rejections','native_runs')},indent=2))
