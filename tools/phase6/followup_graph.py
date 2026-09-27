#!/usr/bin/env python3
"""Exact experimental VWW channel grouping on the existing constant-filter ABI.

Each mapping records which original channel occupies each transformed channel
position. Convolution weights and all per-output parameters follow that map.
The public input/output channel order and every quantization boundary remain
unchanged. This experiment leaves the production compiler unchanged.
"""
import argparse
import bisect
import copy
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
from integer_reference import evaluate, rounded, coefficients
from run_boardless import load_model
from scheduler.resident_verify import replay_resident
from variants import check_frozen, sha


def group_channels(program):
    """Return an equivalent program and new-index -> original-index maps."""
    grouped = copy.deepcopy(program)
    mappings = {name: tuple(range(t.shape[1])) for name, t in program.tensors.items()
                if len(t.shape) >= 2}
    changes = []
    for index, layer in enumerate(grouped.layers):
        original = program.layers[index]
        input_map = mappings.get(layer.inputs[0])
        if layer.op == 'Conv':
            weights = np.asarray(original.parameters['weight'])
            depthwise = layer.attributes.get('group', 1) != 1
            if depthwise:
                if layer.attributes['group'] != weights.shape[0] or input_map is None:
                    raise ValueError('unsupported depthwise channel mapping')
                output_map = input_map
                new_weights = weights[list(input_map)].copy()
            else:
                if input_map is None or len(input_map) != weights.shape[1]:
                    raise ValueError('unsupported dense Conv input mapping')
                zero = np.all(weights == 0, axis=tuple(range(1, weights.ndim)))
                output_map = tuple(np.flatnonzero(~zero).tolist() + np.flatnonzero(zero).tolist())
                new_weights = weights[list(output_map)][:, list(input_map)].copy()
                if any(zero):
                    changes.append(dict(layer=index, channels=len(zero),
                                        constant_channels=int(zero.sum()),
                                        original_runs=int(sum(a != b for a,b in zip(zero[:-1],zero[1:]))+1),
                                        grouped_runs=2 if any(~zero) else 1))
            layer.parameters['weight'] = new_weights
            for key in ('bias', 'corrected_bias', 'multiplier', 'shift'):
                layer.parameters[key] = np.asarray(original.parameters[key])[list(output_map)].copy()
            mappings[layer.output] = output_map
        elif layer.op == 'Gemm':
            weights = np.asarray(original.parameters['weight'])
            if input_map is None or len(input_map) != weights.shape[1]:
                raise ValueError('unsupported Gemm input mapping')
            layer.parameters['weight'] = weights[:, list(input_map)].copy()
            mappings[layer.output] = tuple(range(weights.shape[0]))
        elif layer.op in ('Relu', 'Clip', 'MaxPool', 'AveragePool', 'GlobalAveragePool',
                          'Identity', 'Flatten', 'Reshape'):
            if input_map is None:
                raise ValueError('missing mapped input')
            input_shape = program.tensors[layer.inputs[0]].shape
            output_shape = program.tensors[layer.output].shape
            if layer.op in ('Flatten', 'Reshape') and len(input_shape) == 4 and len(output_shape) == 4:
                if input_map != tuple(range(input_shape[1])):
                    raise ValueError('unsupported spatial reshape after channel permutation')
                output_map = tuple(range(output_shape[1]))
            elif layer.op in ('Flatten', 'Reshape') and len(input_shape) == 4:
                plane = math.prod(input_shape[2:])
                if len(output_shape) != 2 or output_shape[1] != len(input_map)*plane:
                    raise ValueError('unsupported channel-flatten mapping')
                output_map = tuple(ch*plane+pixel for ch in input_map for pixel in range(plane))
            else:
                output_map = input_map
            mappings[layer.output] = output_map
        elif layer.op == 'Transpose':
            if index != 0:
                raise ValueError('internal transpose channel mapping unsupported')
            mappings[layer.output] = tuple(range(program.tensors[layer.output].shape[1]))
        else:
            raise ValueError(f'unsupported channel propagation through {layer.op}')
    if mappings[program.outputs[0]] != tuple(range(program.tensors[program.outputs[0]].shape[1])):
        raise ValueError('public output order changed')
    return grouped, mappings, changes


def _known_spatial_conv(image, weight, bias, multiplier, shift, iq_zp, oq_zp,
                        out_h, out_w, attrs):
    """Evaluate a known depthwise plane, including zero-centered padding."""
    kh,kw=weight.shape
    pt,pl,pb,pr=attrs.get('pads',[0,0,0,0])
    sh,sw=attrs.get('strides',[1,1])
    dh,dw=attrs.get('dilations',[1,1])
    centered=np.pad(image.astype(np.int64)-iq_zp,((pt,pb),(pl,pr)))
    result=np.empty((out_h,out_w),dtype=np.int8)
    for y in range(out_h):
        for x in range(out_w):
            total=int(bias)
            for ky in range(kh):
                for kx in range(kw):
                    total+=int(centered[y*sh+ky*dh,x*sw+kx*dw])*int(weight[ky,kx])
            result[y,x]=rounded(np.asarray([total*int(multiplier)],dtype=np.int64),shift,oq_zp)[0]
    return result


def fold_known_inputs(program):
    """Fold proven uniform input channels into pointwise biases.

    A known pattern is derived from weights/biases and exact INT8 arithmetic,
    never from a sample. Nonuniform known patterns are propagated through
    depthwise and activation layers but not folded into a scalar Conv bias.
    """
    folded=copy.deepcopy(program)
    known={name:[None]*tensor.shape[1] for name,tensor in program.tensors.items()
           if len(tensor.shape)>=2}
    changes=[]
    for index,layer in enumerate(folded.layers):
        original=program.layers[index]
        input_known=known[layer.inputs[0]]
        iq=program.tensors[layer.inputs[0]].quantization
        oq=program.tensors[layer.output].quantization
        output_shape=program.tensors[layer.output].shape
        if layer.op=='Conv' and layer.attributes.get('group',1)==1:
            weights=np.asarray(original.parameters['weight'])
            candidates=[]
            kh,kw=weights.shape[2:]
            pads=layer.attributes.get('pads',[0,0,0,0])
            for channel,pattern in enumerate(input_known):
                if pattern is None or np.unique(pattern).size!=1: continue
                code=int(pattern.reshape(-1)[0])
                if kh*kw==1 or not any(pads) or code==iq.zero_point:
                    candidates.append(channel)
            if candidates:
                new_weights=weights.copy()
                new_bias=np.asarray(original.parameters['bias'],dtype=np.int64).copy()
                for channel in candidates:
                    code=int(input_known[channel].reshape(-1)[0])
                    centered=code-iq.zero_point
                    new_bias+=centered*weights[:,channel].astype(np.int64).sum(axis=(1,2))
                    new_weights[:,channel]=0
                flat=new_weights.astype(np.int64).reshape(len(new_weights),-1)
                corrected=new_bias-iq.zero_point*flat.sum(axis=1)
                bound=128*np.abs(flat).sum(axis=1)+np.abs(corrected)
                centered_bound=max(abs(-128-iq.zero_point),abs(127-iq.zero_point))*np.abs(flat).sum(axis=1)+np.abs(new_bias)
                if np.any(new_bias<-(1<<31)) or np.any(new_bias>(1<<31)-1) or np.any(corrected<-(1<<31)) or np.any(corrected>(1<<31)-1) or np.any(bound>(1<<31)-1) or np.any(centered_bound>(1<<31)-1):
                    raise ValueError(f'layer {index}: folded INT32 accumulator bound failed')
                layer.parameters['weight']=new_weights
                layer.parameters['bias']=new_bias.astype('<i4')
                layer.parameters['corrected_bias']=corrected.astype('<i4')
                newly_zero=int(np.count_nonzero(np.all(new_weights==0,axis=(1,2,3)) &
                                  ~np.all(weights==0,axis=(1,2,3))))
                changes.append(dict(layer=index, folded_input_channels=len(candidates),
                    newly_constant_outputs=newly_zero,
                    removed_weight_values=int(np.count_nonzero(weights[:,candidates]))))
            else:
                new_weights=weights
            zero=np.all(new_weights==0,axis=(1,2,3))
            output=[]
            for channel,is_zero in enumerate(zero):
                if is_zero:
                    bias=int(layer.parameters['bias'][channel])
                    mult=int(layer.parameters['multiplier'][channel])
                    shift=int(layer.parameters['shift'][channel])
                    code=int(rounded(np.asarray([bias*mult],dtype=np.int64),shift,oq.zero_point)[0])
                    output.append(np.full(output_shape[2:],code,dtype=np.int8))
                else: output.append(None)
            known[layer.output]=output
        elif layer.op=='Conv':
            weights=np.asarray(original.parameters['weight'])
            if layer.attributes.get('group')!=len(input_known) or len(weights)!=len(input_known):
                raise ValueError('unsupported depthwise grouping')
            known[layer.output]=[
                _known_spatial_conv(pattern,weights[ch,0],
                    original.parameters['bias'][ch],original.parameters['multiplier'][ch],
                    original.parameters['shift'][ch],iq.zero_point,oq.zero_point,
                    output_shape[2],output_shape[3],layer.attributes)
                if pattern is not None else None for ch,pattern in enumerate(input_known)]
        elif layer.op in ('Relu','Clip'):
            lower,upper=((iq.zero_point,127) if layer.op=='Relu' else
                         tuple(int(v) for v in layer.parameters['clip_bounds']))
            mult,shift=coefficients(iq.scale/oq.scale)
            known[layer.output]=[
                rounded((np.clip(pattern.astype(np.int64),lower,upper)-iq.zero_point)*mult,
                        shift,oq.zero_point).astype(np.int8)
                if pattern is not None else None for pattern in input_known]
        elif layer.op in ('Identity','Reshape','Flatten','AveragePool','GlobalAveragePool','MaxPool'):
            if layer.op in ('Reshape','Flatten') and len(output_shape)==4:
                if any(pattern is not None for pattern in input_known):
                    raise ValueError('spatial reshape of known values unsupported')
                known[layer.output]=[None]*output_shape[1]
            elif layer.op in ('Reshape','Flatten'):
                if len(output_shape)!=2 or math.prod(program.tensors[layer.inputs[0]].shape[2:])!=1:
                    raise ValueError('known flatten requires unit spatial plane')
                known[layer.output]=[
                    np.asarray([int(pattern.reshape(-1)[0])],dtype=np.int8)
                    if pattern is not None else None for pattern in input_known]
            elif layer.op in ('AveragePool','GlobalAveragePool','MaxPool'):
                # VWW's only pool is a full, unpadded 3x3 average to 1x1.
                attrs=layer.attributes
                if (layer.op not in ('AveragePool','GlobalAveragePool') or
                        any(attrs.get('pads',[0,0,0,0])) or output_shape[2:]!=(1,1)):
                    known[layer.output]=[None]*output_shape[1]
                else:
                    kh,kw=attrs.get('kernel_shape',program.tensors[layer.inputs[0]].shape[2:])
                    mult,shift=coefficients(iq.scale/(oq.scale*kh*kw))
                    known[layer.output]=[
                        np.asarray([[int(rounded(np.asarray([(pattern.astype(np.int64)-iq.zero_point).sum()*mult],dtype=np.int64),shift,oq.zero_point)[0])]],dtype=np.int8)
                        if pattern is not None else None for pattern in input_known]
            else: known[layer.output]=list(input_known)
        elif layer.op=='Transpose':
            if index!=0: raise ValueError('internal transpose unsupported')
            known[layer.output]=[None]*output_shape[1]
        elif layer.op=='Gemm':
            weights=np.asarray(original.parameters['weight'])
            candidates=[ch for ch,pattern in enumerate(input_known) if pattern is not None]
            if candidates:
                bias=np.asarray(original.parameters['bias'],dtype=np.int64).copy()
                new_weights=weights.copy()
                for ch in candidates:
                    code=int(input_known[ch].reshape(-1)[0])
                    bias+=(code-iq.zero_point)*weights[:,ch].astype(np.int64)
                    new_weights[:,ch]=0
                flat=new_weights.astype(np.int64)
                corrected=bias-iq.zero_point*flat.sum(axis=1)
                bound=128*np.abs(flat).sum(axis=1)+np.abs(corrected)
                centered_bound=max(abs(-128-iq.zero_point),abs(127-iq.zero_point))*np.abs(flat).sum(axis=1)+np.abs(bias)
                if np.any(np.abs(bias)>(1<<31)-1) or np.any(np.abs(corrected)>(1<<31)-1) or np.any(bound>(1<<31)-1) or np.any(centered_bound>(1<<31)-1):
                    raise ValueError('folded Gemm accumulator bound failed')
                layer.parameters['weight']=new_weights
                layer.parameters['bias']=bias.astype('<i4')
                layer.parameters['corrected_bias']=corrected.astype('<i4')
                changes.append(dict(layer=index,folded_input_channels=len(candidates),
                    newly_constant_outputs=0,removed_weight_values=int(np.count_nonzero(weights[:,candidates]))))
            known[layer.output]=[None]*output_shape[1]
        else: raise ValueError(f'unsupported fold traversal {layer.op}')
    return folded,known,changes


def check_oracles(original, grouped, mappings, value):
    expected = evaluate(original, {original.inputs[0]: value})
    actual = evaluate(grouped, {grouped.inputs[0]: value})
    for index, layer in enumerate(original.layers):
        got = actual[layer.output]
        want = expected[layer.output]
        mapping = mappings[layer.output]
        if len(want.shape) < 2 or got.shape != want.shape or len(mapping) != want.shape[1]:
            raise ValueError(f'layer {index}: invalid channel mapping')
        if not np.array_equal(got, np.take(want, mapping, axis=1)):
            raise ValueError(f'layer {index}: grouped oracle mismatch')
    return actual


def trim_constant_only_loads(code, schedule):
    """Drop unused operand DMAs when every output in a Conv tile is constant."""
    commands = [struct.unpack_from('<BBHIII',code,i) for i in range(0,len(code),16)]
    stages = copy.deepcopy(schedule['stages'])
    removed = set()
    cursor = 0
    saved_bytes = 0

    def dma(transfer, remove=False):
        nonlocal cursor, saved_bytes
        cmd,wait = commands[cursor:cursor+2]
        expected = (1,int(transfer['direction']=='to_sram'),0,
                    transfer['ext'],transfer['sram'],transfer['bytes'])
        if cmd != expected or wait != (3,2,0,0,0,0):
            raise ValueError(f'command cursor {cursor}: unexpected DMA')
        if remove:
            removed.update((cursor,cursor+1))
            saved_bytes += transfer['bytes']
        cursor += 2

    def engine():
        nonlocal cursor
        if commands[cursor][0] != 2 or commands[cursor+1] != (3,3,0,0,0,0):
            raise ValueError(f'command cursor {cursor}: unexpected engine run')
        cursor += 2

    for stage in stages:
        parts = stage.get('constant_filter_parts')
        if parts is None:
            for transfer in stage['loads']: dma(transfer)
            engine()
        else:
            all_constant = all(part['constant'] for part in parts)
            retained = []
            for transfer in stage['loads']:
                if transfer['role'] == 'descriptor': continue
                drop = all_constant and transfer['direction'] == 'to_sram'
                dma(transfer,remove=drop)
                if not drop: retained.append(transfer)
            if all_constant:
                stage['loads'] = [t for t in stage['loads'] if t['role']=='descriptor']+retained
            for part in parts:
                if part['constant']:
                    # Constant fill has its own appended payload DMA.
                    if commands[cursor][0] != 1 or commands[cursor+1] != (3,2,0,0,0,0):
                        raise ValueError('missing constant fill')
                    cursor += 2
            for part in parts:
                if not part['constant']:
                    if commands[cursor][0] != 1 or commands[cursor+1] != (3,2,0,0,0,0):
                        raise ValueError('missing split descriptor')
                    cursor += 2
                    engine()
        if schedule['snapshots_enabled']:
            region = schedule['snapshot_regions'][stage['layer']]
            dma(dict(direction='from_sram',ext=region['ext']+stage['first_element'],
                     sram=stage['output']['sram'],bytes=stage['output']['bytes']))
        if stage['store'] is not None: dma(stage['store'])
    if cursor != len(commands)-1 or commands[cursor] != (0,0,0,0,0,0):
        raise ValueError('command traversal did not reach HALT')
    sorted_removed = sorted(removed)
    remap = lambda x: str(int(x)-bisect.bisect_left(sorted_removed,int(x)))
    updated = dict(schedule,stages=stages,
        run_contracts={remap(k):v for k,v in schedule['run_contracts'].items()},
        constant_contracts={remap(k):v for k,v in schedule.get('constant_contracts',{}).items()},
        redundant_loads_removed_bytes=saved_bytes,
        redundant_loads_removed_commands=len(removed))
    packed = b''.join(struct.pack('<BBHIII',*cmd) for i,cmd in enumerate(commands) if i not in removed)
    updated['command_count']=len(packed)//16
    updated['program_bytes']=len(packed)
    updated['program_sha256']=hashlib.sha256(packed).hexdigest()
    return packed,updated


def prepare(output,fold=False):
    check_frozen()
    output.mkdir(parents=True, exist_ok=True)
    all_fixtures = []
    model_records = {}
    for name in ('kws', 'vww'):
        original, pinned, _, sources = load_model(name)
        intermediate, known, fold_changes = (fold_known_inputs(original) if fold else
            (original,{},[]))
        grouped, mappings, changes = group_channels(intermediate)
        model_records[name] = dict(changes=changes,fold_changes=fold_changes,
                                   pinned_sources=sources)
        for sample, value in (('pinned', pinned), ('stress',
            np.random.default_rng(6078).integers(-128,128,pinned.shape,dtype=np.int8))):
            oracle = check_oracles(original, grouped, mappings, value)
            initial = oracle[grouped.layers[0].output] if grouped.layers[0].op == 'Transpose' else value
            for snapshots in ((True, False) if sample == 'pinned' else (True,)):
                label = f'{name}-{sample}-grouped'+('-fold' if fold else '')+'-'+('check' if snapshots else 'timed')
                code, payload, schedule = compile_constant_chain(grouped, snapshots,
                    reuse_sibling_inputs=True)
                code, schedule = trim_constant_only_loads(code,schedule)
                replay = replay_resident(grouped, code, payload, {grouped.inputs[0]:value},
                    run_contracts=schedule['run_contracts'],
                    constant_contracts=schedule.get('constant_contracts'),
                    final_output=schedule['final_output'],
                    snapshot_regions=schedule['snapshot_regions'], oracle=oracle)
                directory = output/'fixtures'/label
                directory.mkdir(parents=True, exist_ok=True)
                files = dict(**{'commands.bin':code, 'payload.bin':payload,
                    'input.bin':initial.tobytes(),
                    'output.bin':oracle[grouped.outputs[0]].tobytes()},
                    **{f'layer-{i}.bin':oracle[layer.output].tobytes()
                       for i,layer in enumerate(grouped.layers) if i in schedule['snapshot_regions']})
                checks = [f'{schedule["final_output"]["ext"]} output.bin']
                checks += [f'{region["ext"]} layer-{i}.bin'
                    for i,region in schedule['snapshot_regions'].items()]
                files['checks.txt'] = ('\n'.join(checks)+'\n').encode()
                files['schedule.json'] = (json.dumps(schedule,sort_keys=True,indent=2)+'\n').encode()
                for filename,data in files.items():
                    (directory/filename).write_bytes(data)
                row = dict(label=label, model=name, sample=sample, snapshots=snapshots,
                    verification=replay, command_count=schedule['command_count'],
                    program_bytes=len(code), payload_bytes=len(payload),
                    constants=schedule['constant_filter'],
                    files={filename:sha(directory/filename) for filename in files})
                all_fixtures.append(row)
                print(label, 'commands',len(code)//16, 'replay',replay['status'],flush=True)
    manifest = dict(schema=1,status='passed-replay',fold=fold,
                    models=model_records,fixtures=all_fixtures)
    (output/'fixtures.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    check_frozen()
    return manifest


def native(output, label='combined-spec-scalar-v1'):
    check_frozen()
    manifest_path = output/'fixtures.json'
    manifest = json.loads(manifest_path.read_text())
    candidate = ROOT/'work/phase6/experiments-v1'/label
    baseline = json.loads((candidate/'native/report.json').read_text())
    executable = candidate/'native/Vv2_tiled_host_bridge'
    if (baseline['status'] != 'passed' or not executable.is_file() or
            (baseline.get('executable_sha256') and sha(executable) != baseline['executable_sha256'])):
        raise ValueError('native engine baseline changed')
    for filename,digest in baseline['sources'].items():
        if sha(ROOT/filename) != digest:
            raise ValueError(f'native source changed: {filename}')
    target = output/f'native-{label}'
    target.mkdir(parents=True,exist_ok=True)
    report = dict(status='running',executable_sha256=sha(executable),
                  fixture_manifest_sha256=sha(manifest_path),results=[])
    path = target/'report.json'
    for fixture in manifest['fixtures']:
        directory = output/'fixtures'/fixture['label']
        for filename,digest in fixture['files'].items():
            if sha(directory/filename) != digest:
                raise ValueError('fixture changed')
        for seed in (0,6079):
            output_path = target/f'{fixture["label"]}-s{seed}.json'
            subprocess.run([str(executable),str(directory),str(seed),str(output_path)],check=True)
            result = json.loads(output_path.read_text())
            if result['status'] != 'passed':
                raise ValueError(f'native mismatch: {fixture["label"]}, seed {seed}')
            report['results'].append(dict(fixture=fixture['label'],seed=seed,result=result))
            path.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    report['status']='passed'
    path.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    check_frozen()
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','native'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--fold',action='store_true')
    parser.add_argument('--engine',default='combined-spec-scalar-v1')
    args=parser.parse_args()
    if args.output is None:
        args.output=ROOT/'work/phase6'/('followup_graph_fold' if args.fold else 'followup_graph')
    result=prepare(args.output,args.fold) if args.stage=='prepare' else native(args.output,args.engine)
    print(result['status'], len(result['fixtures'] if args.stage=='prepare' else result['results']))
