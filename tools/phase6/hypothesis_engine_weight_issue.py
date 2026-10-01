#!/usr/bin/env python3
"""Co-issue distributed cached weights with completed operands; isolated RTL v2."""
import argparse
import json
from pathlib import Path
import subprocess

import hypothesis_engine_overlap as common

ROOT=common.ROOT
BASE=ROOT/'work/phase6/engine-candidate-rtl-v2'


def prepare(output=BASE):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if common.sha(common.PARENT)!=common.PARENT_SHA:raise ValueError('selected engine changed')
    source=common.PARENT.read_text()
    source=common.once(source,'    task automatic advance_output;',
        '''    // Operand and distributed weight-cache registers are independent.
    // Consume a valid cached word on the operand-completion edge, avoiding
    // the otherwise empty W_REQ/PW_W_REQ hit cycle. Misses use the original
    // stable request states; col already identifies the completed operand.
    task automatic operand_ready;
        begin
            if((spatial_pw||spatial_dw)&&col[2:0]!=0)state<=PW_MAC;
            else if(cached_weight)begin
                whex<=weight_cache[weight_cache_index];
                if(spatial_pw||spatial_dw)pw_weight<=$signed(weight_cache[weight_cache_index][7:0]);
                state<=(spatial_pw||spatial_dw)?PW_MAC:MAC;
            end else state<=(spatial_pw||spatial_dw)?PW_W_REQ:W_REQ;
        end
    endtask
    task automatic advance_output;''')
    source=common.once(source,
        '                    if(dw_take_calc==0)begin xword<=0;state<=col[2:0]==0?PW_W_REQ:PW_MAC;end',
        '                    if(dw_take_calc==0)begin xword<=0;operand_ready();end')
    source=common.once(source,
        '''                        state<=(32'(pw_input_addr[2:0])+
                            (spatial_dw?32'(dw_take):32'(pw_take))>8)?PW_X2_REQ:
                            (col[2:0]==0?PW_W_REQ:PW_MAC);''',
        '''                        if(32'(pw_input_addr[2:0])+
                            (spatial_dw?32'(dw_take):32'(pw_take))>8)state<=PW_X2_REQ;
                        else operand_ready();''')
    source=common.once(source,
        '''                    state<=(32'(pw_input_addr[2:0])+(spatial_dw?32'(dw_take):32'(pw_take))>8)?PW_X2_REQ:
                        (col[2:0]==0?PW_W_REQ:PW_MAC);''',
        '''                    if(32'(pw_input_addr[2:0])+(spatial_dw?32'(dw_take):32'(pw_take))>8)state<=PW_X2_REQ;
                    else operand_ready();''')
    source=common.once(source,'                    state<=col[2:0]==0?PW_W_REQ:PW_MAC;',
                             '                    operand_ready();')
    begin=source.index('                WIN_PREP:begin')
    end=source.index('                POOL_PREP:begin',begin)
    gather=source[begin:end]
    if gather.count('state<=W_REQ;')!=5:raise ValueError('scalar gather patch sites changed')
    source=source[:begin]+gather.replace('state<=W_REQ;','operand_ready();')+source[end:]
    engine=output/'engine.sv'
    if engine.exists() and engine.read_text()!=source:raise ValueError('candidate already differs')
    if not engine.exists():engine.write_text(source)
    identity=dict(status='prepared',physical_board=False,parent=str(common.PARENT.relative_to(ROOT)),
        parent_sha256=common.sha(common.PARENT),engine_sha256=common.sha(engine),
        driver_sha256=common.sha(Path(__file__)),
        mechanism='co-issue operand and valid distributed weight-cache word into independent registers; skip weight-request state only on hit',
        invariants=['No arithmetic or storage-width change; same eight signed MAC lanes and exact epilogue.',
            'Misses retain original request/wait protocol and error behavior.',
            'PW_MAC next-channel prefetch/bypass logic is untouched; helper runs only after current col has settled.',
            'PW non-boundary operands retain the current weight word and existing byte selection.'],
        scope='independent from v1 LUT/cache-clear overlap; selected routed parent unchanged',
        profile_hit_cycle_headroom=dict(kws=82688,vww=159346),
        limitations=['Device area and maximum clock unmeasured; new cache-hit path may affect placement/timing.'])
    common.save(output/'identity.json',identity)
    return output,engine


def native(output=BASE):
    output,engine=prepare(output);build=output/'native';build.mkdir(exist_ok=True)
    harness=ROOT/'test/phase6/native.cpp';sources=common.source_files(engine)
    pins={str(p.relative_to(ROOT)):common.sha(p) for p in sources+[harness,Path(__file__),Path(common.__file__)]}
    frozen=json.loads(common.FINAL.read_text())
    if frozen['status']!='passed':raise ValueError('final baseline incomplete')
    baseline_report=json.loads((common.NATIVE.parent/'report.json').read_text())
    if common.sha(common.NATIVE)!=baseline_report['executable_sha256']:raise ValueError('baseline native changed')
    with (build/'build.log').open('w') as log:
        subprocess.run(['verilator','--cc','--exe','--build','-j','2','--public-flat-rw','-Wno-fatal',
            '--top-module','v2_tiled_host_bridge','--Mdir',str(build),*map(str,sources),str(harness)],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    executable=build/'Vv2_tiled_host_bridge'
    report=dict(status='running',physical_board=False,source_sha256=pins,engine_sha256=common.sha(engine),
        baseline_engine_sha256=common.sha(common.PARENT),executable_sha256=common.sha(executable),
        baseline_executable_sha256=common.sha(common.NATIVE),
        final_baseline=dict(file=str(common.FINAL.relative_to(ROOT)),sha256=common.sha(common.FINAL)),results=[])
    common.save(build/'report.json',report)
    for model,selected in frozen['selected'].items():
        for sample in ('pinned','stress'):
            fixture=selected[sample];directory=ROOT/fixture['directory']
            for seed in (0,6063):
                a=common.measure(common.NATIVE,directory,fixture['files'],seed,build/f'{model}-{sample}-baseline-s{seed}.json')
                original=next(n for n in fixture['native'] if n['stall_seed']==seed)
                for key in ('elapsed_cycles','engine_cycles','dma_cycles','overlap_cycles','tensor_checks'):
                    if a[key]!=original[key]:raise ValueError('control does not reproduce final baseline')
                b=common.measure(executable,directory,fixture['files'],seed,build/f'{model}-{sample}-candidate-s{seed}.json')
                row=dict(model=model,sample=sample,stall_seed=seed,baseline=a,candidate=b,
                    saved_elapsed_cycles=a['elapsed_cycles']-b['elapsed_cycles'],
                    saved_engine_cycles=a['engine_cycles']-b['engine_cycles'],
                    speedup=a['elapsed_cycles']/b['elapsed_cycles'])
                report['results'].append(row);common.save(build/'report.json',report)
                print(model,sample,seed,'saved',row['saved_elapsed_cycles'],'speedup',row['speedup'],flush=True)
    for path,expected in pins.items():
        if common.sha(ROOT/path)!=expected:raise ValueError('source changed during screen')
    report.update(status='passed',conclusion='exact native mechanism screen; physical feasibility and novelty not established')
    common.save(build/'report.json',report);return report


def edges(output=BASE):
    common.prepare=prepare
    report=common.edges(output)
    report['source_sha256'][str(Path(__file__).relative_to(ROOT))]=common.sha(Path(__file__))
    common.save(Path(output)/'edges/report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','native','edges'])
    parser.add_argument('--output',type=Path,default=BASE)
    args=parser.parse_args();globals()[args.stage](args.output)
