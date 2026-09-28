#!/usr/bin/env python3
"""Exact VWW final-constant propagation into the integer classifier.

Focused experiment: group channels, prove final pointwise zero filters have
input-independent INT8 codes, symbolically propagate through Relu, 3x3
AveragePool and Reshape, fold their centered Gemm contributions into INT32
bias, then compact the now-dead channels. No model retraining or FPGA RTL
change. Preparation/replay and native execution are separate commands.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'compiler'),str(ROOT/'tools/phase6')]

from channel_compaction import compact_channels,check_oracles,compile_graph,command_stats,graph_stats
from followup_graph import group_channels
from integer_reference import coefficients,evaluate,rounded
from run_boardless import load_model
from scheduler.resident_verify import replay_resident
from variants import check_frozen,sha

BASE=ROOT/'work/phase6/novelty-constants-v1'
VWW=(53,54,55,56,57)
I32_MIN=-(1<<31)
I32_MAX=(1<<31)-1


def one_rounded(numerator,shift,zero_point):
    return int(rounded(np.asarray([numerator],dtype=np.int64),shift,zero_point)[0])


def fold_final_constants(grouped):
    """Return folded graph and a certificate valid for all INT8 model inputs."""
    program=copy.deepcopy(grouped)
    conv,act,pool,reshape,gemm=(program.layers[i] for i in VWW)
    if ([x.op for x in (conv,act,pool,reshape,gemm)]!=
            ['Conv','Relu','AveragePool','Reshape','Gemm'] or
            any(program.layers[j].inputs!=[program.layers[j-1].output]
                for j in range(54,58))):
        raise ValueError('unsupported final VWW graph')
    if (conv.attributes.get('group',1)!=1 or
            program.tensors[conv.output].shape!=(1,256,3,3) or
            program.tensors[pool.output].shape!=(1,256,1,1) or
            program.tensors[reshape.output].shape!=(1,256) or
            program.tensors[gemm.output].shape!=(1,2)):
        raise ValueError('final VWW dimensions changed')
    attrs=pool.attributes
    if (attrs.get('kernel_shape')!=[3,3] or
            attrs.get('pads',[0,0,0,0])!=[0,0,0,0] or
            attrs.get('strides',[1,1])!=[3,3]):
        raise ValueError('unsupported pooled constant geometry')

    conv_w=np.asarray(conv.parameters['weight'],dtype=np.int8)
    zero=np.all(conv_w==0,axis=(1,2,3))
    candidates=np.flatnonzero(zero).tolist()
    if not candidates:
        raise ValueError('no constant final filters')
    cparam=conv.parameters
    cq=program.tensors[conv.output].quantization
    aq_in=program.tensors[act.inputs[0]].quantization
    aq_out=program.tensors[act.output].quantization
    pq_in=program.tensors[pool.inputs[0]].quantization
    pq_out=program.tensors[pool.output].quantization
    gq_in=program.tensors[gemm.inputs[0]].quantization
    if cq!=aq_in or aq_out!=pq_in or pq_out!=gq_in:
        raise ValueError('quantization edge mismatch')
    am,ash=coefficients(aq_in.scale/aq_out.scale)
    pm,psh=coefficients(pq_in.scale/(pq_out.scale*9))
    codes=[]
    for ch in candidates:
        if int(cparam['corrected_bias'][ch])!=int(cparam['bias'][ch]):
            raise ValueError('zero-filter corrected bias differs from bias')
        raw_bias=int(cparam['bias'][ch])
        mul=int(cparam['multiplier'][ch]);shift=int(cparam['shift'][ch])
        conv_code=one_rounded(raw_bias*mul,shift,cq.zero_point)
        act_code=one_rounded((max(aq_in.zero_point,conv_code)-aq_in.zero_point)*am,
                             ash,aq_out.zero_point)
        pool_sum=9*(act_code-pq_in.zero_point)
        pool_code=one_rounded(pool_sum*pm,psh,pq_out.zero_point)
        codes.append(dict(channel=ch,conv_code=conv_code,activation_code=act_code,
                          pool_code=pool_code,reshape_code=pool_code,
                          centered_classifier_input=pool_code-gq_in.zero_point))

    old_weight=np.asarray(gemm.parameters['weight'],dtype=np.int8)
    old_bias=np.asarray(gemm.parameters['bias'],dtype=np.int64)
    old_corrected=np.asarray(gemm.parameters['corrected_bias'],dtype=np.int64)
    if old_weight.shape!=(2,256):
        raise ValueError('classifier weight shape changed')
    old_formula=old_bias-gq_in.zero_point*old_weight.astype(np.int64).sum(axis=1)
    if not np.array_equal(old_corrected,old_formula):
        raise ValueError('original Gemm corrected bias identity failed')
    contribution=np.zeros(2,dtype=np.int64)
    new_weight=old_weight.copy()
    for row in codes:
        ch=row['channel']
        contribution+=row['centered_classifier_input']*old_weight[:,ch].astype(np.int64)
        new_weight[:,ch]=0
    new_bias=old_bias+contribution
    new_corrected=new_bias-gq_in.zero_point*new_weight.astype(np.int64).sum(axis=1)
    flat=new_weight.astype(np.int64)
    raw_bound=np.abs(new_corrected)+128*np.abs(flat).sum(axis=1)
    centered_factor=max(abs(-128-gq_in.zero_point),abs(127-gq_in.zero_point))
    centered_bound=np.abs(new_bias)+centered_factor*np.abs(flat).sum(axis=1)
    if (np.any(new_bias<I32_MIN) or np.any(new_bias>I32_MAX) or
            np.any(new_corrected<I32_MIN) or np.any(new_corrected>I32_MAX) or
            np.any(raw_bound>I32_MAX) or np.any(centered_bound>I32_MAX)):
        raise ValueError('all-input INT32 partial-sum proof failed')
    gemm.parameters['weight']=new_weight
    gemm.parameters['bias']=new_bias.astype('<i4')
    gemm.parameters['corrected_bias']=new_corrected.astype('<i4')
    certificate=dict(schema=1,scope='VWW final Conv 53 -> Relu 54 -> AveragePool 55 -> Reshape 56 -> Gemm 57',
        proof='zero Conv rows imply bias-only requantized code for every input; exact Relu and full 3x3 pool maps are deterministic; Gemm centered contributions move into bias; triangle bounds cover every raw/centered INT32 partial sum',
        constant_channels=len(codes),constant_codes=codes,
        nonzero_classifier_weights_removed=int(np.count_nonzero(old_weight[:,candidates])),
        classifier_input_zero_point=gq_in.zero_point,
        gemm_bias_before=old_bias.tolist(),gemm_bias_delta=contribution.tolist(),
        gemm_bias_after=new_bias.tolist(),
        corrected_bias_before=old_corrected.tolist(),
        corrected_bias_after=new_corrected.tolist(),
        raw_partial_sum_upper_bound=raw_bound.tolist(),
        centered_partial_sum_upper_bound=centered_bound.tolist(),
        accumulator_limit=I32_MAX,
        conv_output_zero_point=cq.zero_point,activation_input_zero_point=aq_in.zero_point,
        activation_output_zero_point=aq_out.zero_point,
        pool_input_zero_point=pq_in.zero_point,pool_output_zero_point=pq_out.zero_point,
        activation_multiplier=am,activation_shift=ash,
        pool_multiplier=pm,pool_shift=psh,
        input_independent=True)
    return program,certificate


def model():
    original,pinned,_,sources=load_model('vww')
    grouped,group_maps,_=group_channels(original)
    folded,certificate=fold_final_constants(grouped)
    compacted,keep,changes,alignment=compact_channels(folded)
    maps={name:tuple(group_maps[name][j] for j in retained)
          for name,retained in keep.items()}
    return original,pinned,grouped,folded,compacted,maps,group_maps,changes,alignment,certificate,sources


def verify_sample(original,folded,compacted,maps,group_maps,certificate,value):
    initial=evaluate(original,{original.inputs[0]:value})
    folded_values=evaluate(folded,{folded.inputs[0]:value})
    for i,layer in enumerate(original.layers):
        if not np.array_equal(folded_values[layer.output],
                              np.take(initial[layer.output],group_maps[layer.output],axis=1)):
            raise ValueError(f'folded full-width layer {i} differs from original')
    for item in certificate['constant_codes']:
        ch=item['channel']
        for i,key in ((53,'conv_code'),(54,'activation_code'),
                      (55,'pool_code'),(56,'reshape_code')):
            layer=folded.layers[i]
            if not np.all(folded_values[layer.output][:,ch]==item[key]):
                raise ValueError(f'channel {ch} layer {i} violates constant proof')
    compact_values=check_oracles(original,compacted,maps,value)
    return compact_values


def prepare(output=BASE):
    check_frozen();output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    original,pinned,grouped,folded,compacted,maps,group_maps,changes,alignment,certificate,sources=model()
    (output/'certificate.json').write_text(json.dumps(certificate,sort_keys=True,indent=2)+'\n')
    result=dict(schema=1,status='running',physical_board=False,model='vww',
        source_sha256=sha(Path(__file__)),frozen_sources=sources,
        certificate_sha256=sha(output/'certificate.json'),
        changes=changes,alignment_retained=alignment,
        grouped_stats=graph_stats(grouped),
        compaction_only_stats=json.loads((ROOT/'work/phase6/channel-compaction-v1/report.json').read_text())['models']['vww']['new'],
        folded_compacted_stats=graph_stats(compacted),fixtures=[])
    for sample,value in (('pinned',pinned),('stress',
        np.random.default_rng(6257).integers(-128,128,pinned.shape,dtype=np.int8))):
        oracle=verify_sample(original,folded,compacted,maps,group_maps,certificate,value)
        first=compacted.layers[0]
        initial=oracle[first.output] if first.op in ('Transpose','Reshape') else value
        for snapshots in ((True,False) if sample=='pinned' else (True,)):
            name=f'vww-{sample}-known-compacted-'+('check' if snapshots else 'timed')
            code,payload,schedule=compile_graph(compacted,snapshots)
            check=replay_resident(compacted,code,payload,{compacted.inputs[0]:value},
                run_contracts=schedule['run_contracts'],
                constant_contracts=schedule.get('constant_contracts'),
                final_output=schedule['final_output'],
                snapshot_regions=schedule['snapshot_regions'],oracle=oracle)
            dest=output/'fixtures'/name;dest.mkdir(parents=True,exist_ok=True)
            files=dict(**{'commands.bin':code,'payload.bin':payload,'input.bin':initial.tobytes(),
                         'output.bin':oracle[compacted.outputs[0]].tobytes()},
                **{f'layer-{i}.bin':oracle[layer.output].tobytes()
                   for i,layer in enumerate(compacted.layers) if i in schedule['snapshot_regions']})
            checks=[f'{schedule["final_output"]["ext"]} output.bin']
            checks += [f'{region["ext"]} layer-{i}.bin'
                       for i,region in schedule['snapshot_regions'].items()]
            files['checks.txt']=('\n'.join(checks)+'\n').encode()
            files['schedule.json']=(json.dumps(schedule,sort_keys=True,indent=2)+'\n').encode()
            for filename,data in files.items():(dest/filename).write_bytes(data)
            result['fixtures'].append(dict(name=name,sample=sample,snapshots=snapshots,
                verification=check,commands=command_stats(code),
                files={filename:sha(dest/filename) for filename in files}))
            print(name,command_stats(code),check['status'],flush=True)
    result['status']='passed-replay'
    (output/'report.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    check_frozen();return result


def native(output=BASE):
    from fused_activation import lower_fixture,check_table_lifetimes
    from prefetch_tail import analyze_fixture,commands,reorder
    from scheduler.fused_verify import replay_fused
    check_frozen();output=output.resolve()
    prepared=json.loads((output/'report.json').read_text())
    if (prepared['status']!='passed-replay' or
            prepared['source_sha256']!=sha(Path(__file__)) or
            prepared['certificate_sha256']!=sha(output/'certificate.json')):
        raise ValueError('prepared fixture/certificate source identity changed')
    original,pinned,_,folded,compacted,maps,group_maps,_,_,certificate,_=model()
    engine=ROOT/'work/phase6/experiments-v1/fused-activation-v1/native/Vv2_tiled_host_bridge'
    base_native=json.loads((engine.parent/'report.json').read_text())
    if base_native['status']!='passed' or sha(engine)!=base_native['executable_sha256']:
        raise ValueError('selected native engine changed')
    target=output/'fused';target.mkdir(exist_ok=True)
    report=dict(schema=1,status='running',physical_board=False,
        prepared_report_sha256=sha(output/'report.json'),
        certificate_sha256=prepared['certificate_sha256'],
        executable_sha256=sha(engine),results=[])
    for row in prepared['fixtures']:
        src=output/'fixtures'/row['name']
        for name,digest in row['files'].items():
            if sha(src/name)!=digest:raise ValueError('prepared fixture changed')
        name=row['name'].replace('known-compacted','known-compacted-fused')
        dest=target/'fixtures'/name;dest.mkdir(parents=True,exist_ok=True)
        code,payload,schedule=lower_fixture(src)
        (dest/'commands.bin').write_bytes(code)
        (dest/'payload.bin').write_bytes(payload)
        (dest/'schedule.json').write_text(json.dumps(schedule,sort_keys=True,indent=2)+'\n')
        analysis=analyze_fixture(dest)
        changed,mapping=reorder(commands(code),analysis['candidates'])
        code=b''.join(struct.pack('<BBHIII',*cmd) for cmd in changed)
        schedule['run_contracts']={str(mapping[int(k)]):v for k,v in schedule['run_contracts'].items()}
        schedule['constant_contracts']={str(mapping[int(k)]):v for k,v in schedule['constant_contracts'].items()}
        schedule['program_sha256']=hashlib.sha256(code).hexdigest()
        schedule['tail_prefetch']=dict(bytes=analysis['immediately_legal_prefetch_bytes'],
                                       transfers=analysis['immediately_legal_count'])
        lifetimes=check_table_lifetimes(code,payload,schedule)
        value=(pinned if row['sample']=='pinned' else
               np.random.default_rng(6257).integers(-128,128,pinned.shape,dtype=np.int8))
        oracle=verify_sample(original,folded,compacted,maps,group_maps,certificate,value)
        replay=replay_fused(compacted,code,payload,{compacted.inputs[0]:value},
            run_contracts=schedule['run_contracts'],constant_contracts=schedule['constant_contracts'],
            final_output=schedule['final_output'],snapshot_regions=schedule['snapshot_regions'],
            oracle=oracle)
        (dest/'commands.bin').write_bytes(code)
        (dest/'schedule.json').write_text(json.dumps(schedule,sort_keys=True,indent=2)+'\n')
        for filename in ('input.bin','output.bin'):
            (dest/filename).write_bytes((src/filename).read_bytes())
        for i in schedule['snapshot_regions']:
            filename=f'layer-{i}.bin'
            (dest/filename).write_bytes((src/filename).read_bytes())
        checks=[f'{schedule["final_output"]["ext"]} output.bin']
        checks += [f'{region["ext"]} layer-{i}.bin' for i,region in schedule['snapshot_regions'].items()]
        (dest/'checks.txt').write_text('\n'.join(checks)+'\n')
        entry=dict(name=name,sample=row['sample'],snapshots=row['snapshots'],
            lifetimes=lifetimes,verification=replay,commands=command_stats(code),
            fusion=schedule['fusion'],tail_prefetch=schedule['tail_prefetch'],
            files={p.name:sha(p) for p in dest.iterdir() if p.is_file()},native=[])
        for seed in (0,6063):
            path=target/f'{name}-s{seed}.json'
            subprocess.run([str(engine),str(dest),str(seed),str(path)],check=True)
            native_result=json.loads(path.read_text())
            if native_result['status']!='passed':raise ValueError('native RTL mismatch')
            entry['native'].append(dict(seed=seed,**native_result))
        report['results'].append(entry)
        (target/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
        print(name,[v['elapsed_cycles'] for v in entry['native']],flush=True)
    report['status']='passed'
    baseline=json.loads((ROOT/'work/phase6/channel-compaction-v1/fused/report.json').read_text())
    base_row=next(r for r in baseline['results'] if r['label']=='vww-pinned-compacted-fused-timed')
    new_row=next(r for r in report['results'] if r['name']=='vww-pinned-known-compacted-fused-timed')
    if baseline['executable_sha256']!=report['executable_sha256']:
        raise ValueError('comparison uses a different RTL executable')
    report['matched_native_comparison']={str(seed):dict(
        baseline_cycles=next(r['elapsed_cycles'] for r in base_row['native'] if r['seed']==seed),
        candidate_cycles=next(r['elapsed_cycles'] for r in new_row['native'] if r['seed']==seed))
        for seed in (0,6063)}
    for value in report['matched_native_comparison'].values():
        value['saved_cycles']=value['baseline_cycles']-value['candidate_cycles']
        value['latency_reduction_fraction']=value['saved_cycles']/value['baseline_cycles']
    (target/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    check_frozen();return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','native'))
    parser.add_argument('--output',type=Path,default=BASE)
    args=parser.parse_args()
    result=prepare(args.output) if args.stage=='prepare' else native(args.output)
    print(result['status'],len(result['fixtures'] if args.stage=='prepare' else result['results']))
