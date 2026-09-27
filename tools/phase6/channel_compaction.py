#!/usr/bin/env python3
"""Exact dead-channel compaction on the frozen, linear INT8 graphs.

An input channel is removed from a dense operator only when all weights from
that channel into every retained output are literally zero. The corresponding
producer channel is then removed, along with intervening channel-wise layers.
Public input/output layouts, quantizers, and all remaining integer operations
are unchanged. This is an isolated research experiment, not production code.
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'compiler'), str(ROOT/'tools/phase6')]

from constant_filter import compile_constant_chain
from followup_graph import group_channels, trim_constant_only_loads
from integer_reference import evaluate
from run_boardless import load_model
from scheduler.resident_verify import replay_resident
from variants import check_frozen, sha


def compact_channels(program, protected_producer_layers=()):
    """Return a reduced Program and new-channel -> old-channel tensor maps.

    The graph must be a linear chain with an initial host-layout alias. A
    channel kept because a consumer needs it is never silently replaced by a
    constant: only exactly zero weight columns are discarded. The original
    corrected biases therefore stay bit-identical after shrinking reductions.
    """
    if len(program.inputs) != 1 or len(program.outputs) != 1 or program.constants:
        raise ValueError('expected one-input, one-output static chain')
    if program.layers[0].op not in ('Transpose','Reshape','Identity'):
        raise ValueError('host boundary requires an initial layout alias')
    protected=frozenset(protected_producer_layers)
    if any(type(i) is not int or not 0<=i<len(program.layers) or
           program.layers[i].op!='Conv' or program.layers[i].attributes.get('group',1)!=1
           for i in protected):
        raise ValueError('protected producer must be an ordinary Conv layer index')
    for i, layer in enumerate(program.layers):
        previous = program.inputs[0] if i == 0 else program.layers[i-1].output
        if layer.inputs != [previous]:
            raise ValueError('dead-channel analysis requires a linear chain')

    keep = {program.outputs[0]: tuple(range(program.tensors[program.outputs[0]].shape[1]))}
    alignment_retained=[]
    for i in range(len(program.layers)-1, -1, -1):
        layer = program.layers[i]
        input_name = layer.inputs[0]
        output_name = layer.output
        out = keep[output_name]
        old_input = program.tensors[input_name]
        old_output = program.tensors[output_name]
        if layer.op in ('Conv','Gemm'):
            weights = np.asarray(layer.parameters['weight'])
            if layer.op == 'Conv' and layer.attributes.get('group',1) != 1:
                if (layer.attributes['group'] != old_input.shape[1] or
                        weights.shape[0] != old_output.shape[1] or weights.shape[1] != 1):
                    raise ValueError('unsupported grouped Conv')
                needed = out
            else:
                if weights.shape[1] != old_input.shape[1]:
                    raise ValueError('weight/input geometry mismatch')
                active = np.any(weights[list(out)] != 0, axis=(0,2,3) if layer.op=='Conv' else 0)
                needed = tuple(np.flatnonzero(active).tolist())
                # Existing ABI cannot encode an empty reduction. In this
                # corner case retain one zero column without changing output.
                if not needed:
                    needed = (0,)
                # An upstream pointwise Conv whose retained outputs include
                # constant filters is lowered by an aligned fill DMA. Its
                # final output tile must end on an eight-byte boundary. Keep
                # otherwise dead zero-filter channels where needed to satisfy
                # that ABI constraint; their consumer columns remain zero.
                upstream=next((j for j in range(i-1,-1,-1)
                               if program.layers[j].op=='Conv' and
                               program.layers[j].attributes.get('group',1)==1),None)
                if upstream is not None:
                    producer=program.layers[upstream]
                    producer_shape=program.tensors[producer.output].shape
                    if (len(producer_shape)==4 and producer_shape[1]==old_input.shape[1] and
                            all(program.tensors[program.layers[j].output].shape[1]==old_input.shape[1]
                                for j in range(upstream+1,i))):
                        if upstream in protected:
                            needed=tuple(range(old_input.shape[1]))
                        pweights=np.asarray(producer.parameters['weight'])
                        zero=np.all(pweights==0,axis=(1,2,3))
                        plane=math.prod(producer_shape[2:])
                        if any(zero[list(needed)]) and len(needed)*plane%8:
                            before=len(needed)
                            choices=[j for j in range(old_input.shape[1]) if j not in needed and zero[j]]
                            choices += [j for j in range(old_input.shape[1]) if j not in needed and not zero[j]]
                            selected=set(needed)
                            for extra in choices:
                                selected.add(extra)
                                if len(selected)*plane%8==0:break
                            if len(selected)*plane%8:
                                raise ValueError('cannot align constant-fill output')
                            needed=tuple(sorted(selected))
                            alignment_retained.append(dict(producer_layer=upstream,consumer_layer=i,
                                                           extra_channels=len(needed)-before))
        elif layer.op in ('Relu','Clip','MaxPool','AveragePool','GlobalAveragePool','Identity'):
            if old_input.shape[1] != old_output.shape[1]:
                raise ValueError('channel-wise layer changes channel count')
            needed = out
        elif layer.op in ('Flatten','Reshape'):
            if len(old_input.shape)==4 and len(old_output.shape)==2:
                plane=math.prod(old_input.shape[2:])
                if old_output.shape[1] != old_input.shape[1]*plane:
                    raise ValueError('unsupported flatten order')
                channels=tuple(sorted({j//plane for j in out}))
                expected=tuple(c*plane+k for c in channels for k in range(plane))
                if expected != out:
                    # The current channel-only lowering cannot remove part of
                    # a spatial plane. Retain all planes rather than guessing.
                    channels=tuple(range(old_input.shape[1]))
                    keep[output_name]=tuple(range(old_output.shape[1]))
                needed=channels
            elif len(old_input.shape)==len(old_output.shape)==4 and old_input.shape[1]==old_output.shape[1]:
                needed=out
            elif i==0:
                needed=tuple(range(old_input.shape[1]))
                keep[output_name]=tuple(range(old_output.shape[1]))
            else:
                raise ValueError('unsupported reshape geometry')
        elif layer.op=='Transpose' and i==0:
            needed=tuple(range(old_input.shape[1]))
            keep[output_name]=tuple(range(old_output.shape[1]))
        else:
            raise ValueError(f'unsupported layer {i}: {layer.op}')
        keep[input_name]=needed

    # The first operator is performed by the host when preparing input.bin.
    # Retain that complete layout to avoid an unimplemented gather operation.
    first_output = program.layers[0].output
    if keep[first_output] != tuple(range(program.tensors[first_output].shape[1])):
        raise ValueError('host layout would need a gather')
    if keep[program.inputs[0]] != tuple(range(program.tensors[program.inputs[0]].shape[1])):
        raise ValueError('public input order changed')
    for i in protected:
        name=program.layers[i].output
        if keep[name]!=tuple(range(program.tensors[name].shape[1])):
            raise ValueError(f'protected producer {i} was not retained in full')

    compacted=copy.deepcopy(program)
    changed=[]
    for i, layer in enumerate(compacted.layers):
        old=program.layers[i]
        input_map=keep[layer.inputs[0]]
        output_map=keep[layer.output]
        old_output=program.tensors[layer.output]
        if len(old_output.shape)==4:
            shape=(old_output.shape[0],len(output_map),*old_output.shape[2:])
        elif len(old_output.shape)==2:
            shape=(old_output.shape[0],len(output_map))
        else:
            raise ValueError('unexpected output rank')
        compacted.tensors[layer.output]=replace(old_output,shape=shape)
        if layer.op in ('Conv','Gemm'):
            weights=np.asarray(old.parameters['weight'])
            if layer.op=='Conv' and old.attributes.get('group',1)!=1:
                if input_map!=output_map:
                    raise ValueError('depthwise input/output channel maps differ')
                reduced=weights[list(output_map)].copy()
                layer.attributes['group']=len(input_map)
            else:
                removed=sorted(set(range(weights.shape[1]))-set(input_map))
                if removed and np.any(weights[np.ix_(output_map,removed)]):
                    raise ValueError(f'layer {i}: nonzero discarded weight')
                reduced=weights[list(output_map)][:,list(input_map)].copy()
            layer.parameters['weight']=reduced
            for key in ('bias','corrected_bias','multiplier','shift'):
                layer.parameters[key]=np.asarray(old.parameters[key])[list(output_map)].copy()
            if layer.op=='Conv' and old.attributes.get('group',1)==1 or layer.op=='Gemm':
                old_c=np.asarray(old.parameters['corrected_bias'])[list(output_map)].astype(np.int64)
                raw_b=np.asarray(layer.parameters['bias']).astype(np.int64)
                zp=program.tensors[layer.inputs[0]].quantization.zero_point
                new_c=raw_b-zp*reduced.astype(np.int64).reshape(len(output_map),-1).sum(axis=1)
                if not np.array_equal(old_c,new_c):
                    raise ValueError(f'layer {i}: corrected bias changed')
        if len(output_map)!=old_output.shape[1] or len(input_map)!=program.tensors[layer.inputs[0]].shape[1]:
            changed.append(dict(layer=i,op=layer.op,old_input_channels=program.tensors[layer.inputs[0]].shape[1],
                                input_channels=len(input_map),old_output_channels=old_output.shape[1],
                                output_channels=len(output_map)))
    return compacted,keep,changed,alignment_retained


def check_oracles(original, compacted, maps, value):
    expected=evaluate(original,{original.inputs[0]:value})
    actual=evaluate(compacted,{compacted.inputs[0]:value})
    for i,layer in enumerate(original.layers):
        selected=np.take(expected[layer.output],maps[layer.output],axis=1)
        if not np.array_equal(actual[layer.output],selected):
            raise ValueError(f'layer {i}: exact integer oracle mismatch')
    return actual


def graph_stats(program):
    macs=0; executed_macs=0; weight_payload=0; activation_bytes=0
    for layer in program.layers:
        out=program.tensors[layer.output]
        if layer.op in ('Conv','Gemm'):
            w=np.asarray(layer.parameters['weight'])
            reduction=math.prod(w.shape[1:]);plane=math.prod(out.shape[2:]) if len(out.shape)==4 else 1
            macs+=len(w)*plane*reduction
            if layer.op=='Conv' and layer.attributes.get('group',1)==1:
                live=np.count_nonzero(np.any(w!=0,axis=(1,2,3)))
            else:
                live=len(w)
            executed_macs+=int(live)*plane*reduction
            weight_payload+=len(w)*((reduction+7)&~7)
        activation_bytes+=math.prod(out.shape)
    return dict(logical_macs=macs,estimated_macs_after_existing_constant_filter=executed_macs,
                aligned_weight_payload_bytes=weight_payload,materialized_output_bytes=activation_bytes)


def command_stats(code):
    cmds=[struct.unpack_from('<BBHIII',code,i) for i in range(0,len(code),16)]
    loads=[c for c in cmds if c[0]==1 and c[1]==1]
    stores=[c for c in cmds if c[0]==1 and c[1]==0]
    return dict(commands=len(cmds),engine_runs=sum(c[0]==2 for c in cmds),
                dma_load_commands=len(loads),dma_store_commands=len(stores),
                dma_load_bytes=sum(c[5] for c in loads),dma_store_bytes=sum(c[5] for c in stores))


def compile_graph(program,snapshots=False):
    code,payload,schedule=compile_constant_chain(program,snapshots,
                                                 reuse_sibling_inputs=True)
    code,schedule=trim_constant_only_loads(code,schedule)
    return code,payload,schedule


def run(output,protected_producer_layers=()):
    check_frozen();output.mkdir(parents=True,exist_ok=True)
    report=dict(schema=1,status='running',physical_board=False,models={},fixtures=[],
                protected_producer_layers=sorted(set(protected_producer_layers)),
                claim='exact dead-channel graph and executable existing-ABI schedule; physical result pending')
    for name in ('kws','vww'):
        original,pinned,_,sources=load_model(name)
        grouped,group_maps,_=group_channels(original)
        compacted,kept,changed,alignment_retained=compact_channels(
            grouped,protected_producer_layers if name=='vww' else ())
        maps={key:tuple(group_maps[key][j] for j in selected) for key,selected in kept.items()}
        model=dict(changes=changed,alignment_retained=alignment_retained,
                   old=graph_stats(grouped),new=graph_stats(compacted),
                   source_sha256=sources)
        for sample,value in (('pinned',pinned),('stress',np.random.default_rng(6157).integers(-128,128,pinned.shape,dtype=np.int8))):
            oracle=check_oracles(original,compacted,maps,value)
            first=compacted.layers[0]
            initial=oracle[first.output] if first.op in ('Transpose','Reshape') else value
            for snapshots in ((True,False) if sample=='pinned' else (True,)):
                label=f'{name}-{sample}-compacted-'+('check' if snapshots else 'timed')
                code,payload,schedule=compile_graph(compacted,snapshots)
                verification=replay_resident(compacted,code,payload,{compacted.inputs[0]:value},
                    run_contracts=schedule['run_contracts'],
                    constant_contracts=schedule.get('constant_contracts'),
                    final_output=schedule['final_output'],
                    snapshot_regions=schedule['snapshot_regions'],oracle=oracle)
                target=output/'fixtures'/label;target.mkdir(parents=True,exist_ok=True)
                files=dict(**{'commands.bin':code,'payload.bin':payload,'input.bin':initial.tobytes(),
                             'output.bin':oracle[compacted.outputs[0]].tobytes()},
                    **{f'layer-{i}.bin':oracle[layer.output].tobytes()
                       for i,layer in enumerate(compacted.layers) if i in schedule['snapshot_regions']})
                checks=[f'{schedule["final_output"]["ext"]} output.bin']
                checks += [f'{region["ext"]} layer-{i}.bin' for i,region in schedule['snapshot_regions'].items()]
                files['checks.txt']=('\n'.join(checks)+'\n').encode()
                files['schedule.json']=(json.dumps(schedule,sort_keys=True,indent=2)+'\n').encode()
                for filename,data in files.items():(target/filename).write_bytes(data)
                entry=dict(label=label,model=name,sample=sample,snapshots=snapshots,
                           verification=verification,commands=command_stats(code),
                           image_bytes=len(payload),files={key:sha(target/key) for key in files})
                report['fixtures'].append(entry)
                if sample=='pinned' and not snapshots:
                    model['compacted_timed']=entry['commands']
                print(label,entry['commands'],verification['status'],flush=True)
        report['models'][name]=model
        (output/'report.partial.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    report['status']='passed-replay'
    report['source_sha256']=sha(Path(__file__))
    (output/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    check_frozen();return report


def fused_native(output):
    """Screen compaction with the already-selected fused RTL and prefetch."""
    from fused_activation import lower_fixture, check_table_lifetimes
    from prefetch_tail import analyze_fixture, commands, reorder
    from scheduler.fused_verify import replay_fused

    check_frozen();output=output.resolve()
    source=json.loads((output/'report.json').read_text())
    if source['status']!='passed-replay':
        raise ValueError('compacted replay missing')
    native_root=ROOT/'work/phase6/experiments-v1/fused-activation-v1/native'
    native_report=json.loads((native_root/'report.json').read_text())
    executable=native_root/'Vv2_tiled_host_bridge'
    if (native_report['status']!='passed' or
            native_report['executable_sha256']!=sha(executable)):
        raise ValueError('selected fused native engine changed')
    fused_root=output/'fused'
    report=dict(schema=1,status='running',physical_board=False,
                source_report_sha256=sha(output/'report.json'),
                executable_sha256=sha(executable),results=[])
    model_cache={}
    for item in source['fixtures']:
        src=output/'fixtures'/item['label']
        for key,digest in item['files'].items():
            if sha(src/key)!=digest:raise ValueError('fixture mutated')
        name=item['label'].replace('compacted','compacted-fused')
        dest=fused_root/'fixtures'/name;dest.mkdir(parents=True,exist_ok=True)
        code,payload,schedule=lower_fixture(src)
        (dest/'commands.bin').write_bytes(code)
        (dest/'payload.bin').write_bytes(payload)
        (dest/'schedule.json').write_text(json.dumps(schedule,sort_keys=True,indent=2)+'\n')
        analysis=analyze_fixture(dest)
        reordered,mapping=reorder(commands(code),analysis['candidates'])
        code=b''.join(struct.pack('<BBHIII',*cmd) for cmd in reordered)
        schedule['run_contracts']={str(mapping[int(k)]):v for k,v in schedule['run_contracts'].items()}
        schedule['constant_contracts']={str(mapping[int(k)]):v for k,v in schedule['constant_contracts'].items()}
        schedule['program_sha256']=hashlib.sha256(code).hexdigest()
        schedule['tail_prefetch']=dict(bytes=analysis['immediately_legal_prefetch_bytes'],
                                       transfers=analysis['immediately_legal_count'])
        lifetimes=check_table_lifetimes(code,payload,schedule)
        model=item['model'];sample=item['sample']
        if model not in model_cache:
            original,pinned,_,_=load_model(model)
            grouped,gm,_=group_channels(original)
            compacted,kept,_,_=compact_channels(grouped,
                source.get('protected_producer_layers',()) if model=='vww' else ())
            maps={key:tuple(gm[key][j] for j in selected) for key,selected in kept.items()}
            model_cache[model]=(original,pinned,compacted,maps)
        original,pinned,compacted,maps=model_cache[model]
        value=(pinned if sample=='pinned' else
               np.random.default_rng(6157).integers(-128,128,pinned.shape,dtype=np.int8))
        oracle=check_oracles(original,compacted,maps,value)
        verification=replay_fused(compacted,code,payload,{compacted.inputs[0]:value},
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
        entry=dict(label=name,model=model,sample=sample,snapshots=item['snapshots'],
                   verification=verification,lifetimes=lifetimes,commands=command_stats(code),
                   fusion=schedule['fusion'],tail_prefetch=schedule['tail_prefetch'],
                   files={p.name:sha(p) for p in dest.iterdir() if p.is_file()},native=[])
        for seed in (0,6063):
            result_path=fused_root/f'{name}-s{seed}.json'
            subprocess.run([str(executable),str(dest),str(seed),str(result_path)],check=True)
            result=json.loads(result_path.read_text())
            if result['status']!='passed':raise ValueError(f'{name} seed {seed}: RTL mismatch')
            entry['native'].append(dict(seed=seed,**result))
        report['results'].append(entry)
        (fused_root/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
        print(name,entry['fusion']['pairs'],[x['elapsed_cycles'] for x in entry['native']],flush=True)
    report['status']='passed'
    (fused_root/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    check_frozen();return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'work/phase6/channel-compaction-v1')
    parser.add_argument('--fused-native',action='store_true')
    parser.add_argument('--protect-producer',type=int,action='append',default=[],
        help='retain all output channels of this VWW pointwise Conv block')
    args=parser.parse_args()
    if args.fused_native and args.protect_producer:
        parser.error('--protect-producer applies when preparing the graph, not --fused-native')
    report=fused_native(args.output) if args.fused_native else run(args.output,args.protect_producer)
    print(report['status'],len(report['results'] if args.fused_native else report['fixtures']))
