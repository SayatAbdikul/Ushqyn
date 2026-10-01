#!/usr/bin/env python3
"""Frozen VWW factors versus the strongest dense spatial compiler control.

Imports the validated spatial compiler read-only from the referenced worktree.
Both graphs receive the same exact channel transformations and path catalogue.
The selected co-issue RTL executable is unchanged. This is boardless timing.
"""
import os
for _n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_n] = '1'

import argparse
import copy
from dataclasses import replace
import hashlib
import inspect
import json
import math
from pathlib import Path
import resource
import struct
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'work/phase6/representation-optimized-v1'
ORIGINAL_REFERENCE = Path('/Users/sayat/.codex/worktrees/21f0/tinyML_accelerator')
REFERENCE = Path(os.environ.get('PHASE6_SPATIAL_COMPILER_ROOT',str(ORIGINAL_REFERENCE))).resolve()
if not (REFERENCE/'compiler/scheduler/matched_defines.py').exists():
    REFERENCE=OUT/'sources/reference'
sys.path[:0] = [str(REFERENCE/'compiler'),str(REFERENCE/'tools/phase6')]
import matched_defines_baseline as b3
import matched_defines_weights as resident
import novelty_constants
from matched_b1b2 import matched_model,compile_candidate
from matched_current import graph_identity
from channel_compaction import compact_channels,graph_stats
from followup_graph import group_channels
from hardware_v2 import Descriptor,TARGET_PATH
from integer_reference import evaluate
from output_pipeline_fusion import decode_fused
from program_image import load_image
from scheduler.matched_defines import replay_stack
# The generated resident lowering imports this module lazily. Import it before
# the source snapshot so the portable bundle includes that dependency too.
from scheduler import matched_defines_regions as _rectangle_dependencies
from scheduler.matched_defines_strip import block_program,replay_strip_stack,align8
from scheduler.fused_verify import replay_fused
from static_pipeline import Layer
import run_boardless
# The source bundle carries compiler code; original benchmark inputs remain in
# the authorized project dataset locations rather than duplicated in evidence.
run_boardless.ROOT=ROOT

CMD = struct.Struct('<BBHIII')
ENGINE_SHA = '9f934dcc27ae891e4c7b8b84bcc4b65e82d903492c1867572784fbe1cd9484ee'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for piece in iter(lambda:f.read(1<<20),b''):h.update(piece)
    return h.hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def native_identity():
    directory = ROOT/'work/phase6/deadline-screen-v1/native-build'
    ident = json.loads((directory/'identity.json').read_text())
    exe = directory/'Vv2_tiled_host_bridge'
    if ident['engine_sha256'] != ENGINE_SHA or sha(exe) != ident['executable_sha256']:
        raise ValueError('selected co-issue native executable changed')
    for name,digest in ident['sources_sha256'].items():
        path=Path(name)
        if not path.exists():path=OUT/'sources/native'/path.name
        if sha(path)!=digest:raise ValueError('selected native source changed: '+name)
    if sha(ROOT/'work/phase6/deadline-screen-v1/trace.cpp')!=ident['harness_sha256']:
        raise ValueError('selected harness changed')
    return exe,ident


def fold_factor_tail(grouped):
    """Reuse the same exact final-constant proof, changing only layer indices."""
    indices = tuple(range(len(grouped.layers)-5,len(grouped.layers)))
    source = inspect.getsource(novelty_constants.fold_final_constants)
    old = 'range(54,58)'
    if source.count(old)!=1:raise ValueError('final constant folder source changed')
    source = source.replace(old,f'range({indices[1]},{indices[-1]+1})')
    namespace = dict(vars(novelty_constants),VWW=indices)
    exec(compile(source,'<factor-final-constants.generated>','exec'),namespace)
    folded,certificate = namespace['fold_final_constants'](grouped)
    certificate['actual_layer_indices'] = list(indices)
    certificate['generated_function_sha256'] = hashlib.sha256(source.encode()).hexdigest()
    return folded,certificate


def normalize_factor_pairs(program):
    """Expose Conv->Conv as Conv->identity Clip->Conv without losing INT8 rounding.

    The existing rectangular backend uses Conv/(Relu|Clip) macros. A full INT8
    Clip with identical quantizers has an identity 256-byte LUT. It costs the
    same existing epilogue lookup as ordinary activations; no new RTL exists.
    """
    result = copy.deepcopy(program)
    result.layers=[]
    inserted=[]
    for i,old in enumerate(program.layers):
        layer=copy.deepcopy(old)
        if layer.op=='Conv' and i+1<len(program.layers) and program.layers[i+1].op=='Conv':
            original_name=layer.output
            pre=original_name+'/identity-lut-input'
            result.tensors[pre]=replace(result.tensors[original_name],name=pre)
            layer.output=pre
            result.layers.extend([layer,Layer('Clip',[pre],original_name,{},
                {'clip_bounds':np.asarray([-128,127],dtype=np.int8)})])
            inserted.append({'original_factor_layer':i,'normalized_conv_layer':len(result.layers)-2,
                             'normalized_identity_clip_layer':len(result.layers)-1,'tensor':original_name})
        else:result.layers.append(layer)
    return result,inserted


def factor_model(original):
    grouped,group_maps,_=group_channels(original)
    folded,certificate=fold_factor_tail(grouped)
    compact,keep,changes,alignment=compact_channels(folded)
    maps={n:tuple(group_maps[n][j] for j in selected) for n,selected in keep.items()}
    normalized,inserted=normalize_factor_pairs(compact)
    return normalized,maps,dict(compaction=changes,alignment_retained=alignment,
        final_constant_certificate=certificate,identity_clip_normalization=inserted,
        compact_graph_sha256=graph_identity(compact))


def verified_oracle(original,program,maps,value,certificate):
    before=evaluate(original,{original.inputs[0]:value})
    after=evaluate(program,{program.inputs[0]:value})
    # The last Gemm's weights/biases change by the same proved constant fold.
    # All other original tensors agree under the exact channel maps.
    for layer in original.layers[:-1]:
        expected=np.take(before[layer.output],maps[layer.output],axis=1)
        if not np.array_equal(expected,after[layer.output]):
            raise ValueError('graph transformation changed tensor: '+layer.output)
    if not np.array_equal(before[original.outputs[0]],after[program.outputs[0]]):
        raise ValueError('constant fold changed output')
    for row in certificate.get('identity_clip_normalization',[]):
        tensor=row['tensor'];pre=tensor+'/identity-lut-input'
        if not np.array_equal(after[pre],after[tensor]):raise ValueError('identity Clip differs')
    return after


def map_path(configs,dense,factor):
    by_output={l.output:i for i,l in enumerate(factor.layers)}
    def endpoint(i):
        if i==0:return 0
        if i==1:return 1
        return by_output[dense.layers[i-1].output]+1
    answer=copy.deepcopy(configs)
    for cfg in answer:cfg.update(start=endpoint(cfg['start']),stop=endpoint(cfg['stop']))
    return answer


def configurations(incumbent):
    """Same bounded last-stack geometry/cache/residency choices on both graphs."""
    configs=[]
    for h,w,mode,residency in ((3,3,1,False),(3,3,1,True),(2,3,1,False),
            (1,3,1,False),(3,2,1,False),(3,1,1,False),(2,2,1,False),
            (3,3,2,False),(3,3,3,False),(1,3,1,True),(3,1,1,True),(1,1,1,False)):
        path=copy.deepcopy(incumbent)
        path[-1].update(h=h,w=w,mode=mode)
        path[-1].pop('resident_weights',None)
        if residency:path[-1]['resident_weights']=True
        configs.append(path)
    return configs


def compose(program,configs,oracle,directory):
    """Strong B3 relocation plus complete per-RUN contracts for diagnostics."""
    slot=align8(max(math.prod(t.shape) for t in program.tensors.values()))
    payload=bytearray(2*slot);commands=[];components=[];contracts={};current=0;peak=0;identity_elisions=[]
    tail=dict(kind='fallback',start=configs[-1]['stop'],stop=len(program.layers))
    for pos,cfg in enumerate(configs+[tail]):
        start,stop=cfg['start'],cfg['stop'];fragment=block_program(program,start,stop)
        if cfg['kind']=='fallback':
            code,data,record=compile_candidate(fragment,policy='B2',tile_budget=32768,prefetch=True,
                snapshots=False,directory=directory/f'component-{pos}')
            source=oracle[program.layers[start].inputs[0]]
            local=evaluate(fragment,{fragment.inputs[0]:source})
            proof=replay_fused(fragment,code,data,{fragment.inputs[0]:source},oracle=local,
                run_contracts=record['run_contracts'],constant_contracts=record.get('constant_contracts'),
                final_output=record['final_output'],snapshot_regions=record.get('snapshot_regions'))
            local_slot=align8(max(math.prod(t.shape) for t in fragment.tensors.values()))
            parameter_start=2*local_slot;final_slot=record['final_output']['ext']//local_slot
            def address(a):return (current if a//local_slot==0 else 1-current)*slot+a%local_slot
            output_after=current if final_slot==0 else 1-current
            records={int(i):dict(kind='fallback',layer=start+v['layer'],first_element=v['first_element'])
                     for i,v in record['run_contracts'].items()}
            peak=max(peak,max(s['live'][1] for s in record['stages']))
        else:
            code,data,record=resident.compile_config(program,cfg)
            source=oracle[program.layers[start].inputs[0]]
            proof=(replay_strip_stack if cfg['kind']=='strip' else replay_stack)(program,start,stop,source,code,data,record)
            out_base=record['output_external_base'];parameter_start=out_base+align8(record['output_bytes'])
            def address(a):return current*slot+a if a<out_base else (1-current)*slot+a-out_base
            output_after=1-current;records={r['command']:dict(r) for r in record['runs']}
            if cfg['kind']=='strip':
                for r in records.values():
                    r['kind']='strip'
                    stripe=record['stripes'][r['strip']]
                    r['output_rows']=stripe['rectangles'][program.layers[r['layer']+1].output]
            peak=max(peak,record['peak_sram_address'])
            # The macro backend requires an activation node, but a signed
            # latent already has its complete quantization in the Conv. Clear
            # the proved identity LUT tag so native execution pays no extra
            # lookup. Remove its now-unused LUT DMA too. SRAM reservation stays
            # conservative; no allocator redesign or unsupported dataflow.
            mutable=bytearray(data)
            for run_index,r in records.items():
                layer=r.get('layer')
                if r.get('kind')=='copy' or layer is None:continue
                act=program.layers[layer+1]
                if act.op!='Clip' or not np.array_equal(act.parameters.get('clip_bounds'),[-128,127]):continue
                if program.tensors[act.inputs[0]].quantization!=program.tensors[act.output].quantization:
                    continue
                raw=bytes.fromhex(r['descriptor_hex']);d,table=decode_fused(raw)
                lut=bytes.fromhex(r['lut_hex'])
                if lut!=bytes(range(256)):raise ValueError('full-range equal-quantizer LUT is not identity')
                loads=[t for t in record['transfers'] if t['command']<run_index and t['role']=='descriptor']
                descriptor_load=max(loads,key=lambda t:t['command'])
                loc=descriptor_load['ext']
                if mutable[loc:loc+64]!=raw:raise ValueError('identity descriptor load does not match')
                untagged=d.encode();mutable[loc:loc+64]=untagged
                r['descriptor_hex']=untagged.hex();r['identity_lut_elided']=True
                identity_elisions.append(dict(component=pos,layer=layer,
                    original_descriptor_hex=raw.hex(),untagged_descriptor_hex=untagged.hex(),
                    table_sram=table,lut_sha256=hashlib.sha256(lut).hexdigest(),
                    proof='all 256 LUT bytes are identity; Conv already preserves frozen INT8 rounding'))
            data=bytes(mutable)
            removed=set()
            if any(r.get('identity_lut_elided') for r in records.values()):
                decoded=list(struct.iter_unpack('<BBHIII',code))
                for t in record['transfers']:
                    if (t['direction']=='to_sram' and t['role'] in ('lut','activation_table')
                            and t['bytes']==256 and data[t['ext']:t['ext']+256]==bytes(range(256))):
                        index=t['command']
                        if decoded[index][0:2]!=(1,1) or decoded[index+1]!=(3,2,0,0,0,0):
                            raise ValueError('identity LUT load no longer has standalone wait')
                        removed.update((index,index+1))
                remap={i:j for j,i in enumerate(i for i in range(len(decoded)) if i not in removed)}
                records={remap[i]:r for i,r in records.items()}
                code=b''.join(CMD.pack(*r) for i,r in enumerate(decoded) if i not in removed)
                for item in identity_elisions:
                    if item['component']==pos:
                        item['removed_identity_lut_dma_commands']=len(removed)//2
        append=align8(len(payload));payload.extend(bytes(append-len(payload)));payload.extend(data[parameter_start:])
        first=len(commands)
        for i in range(len(code)//16-1):
            op,flags,res,a,b,c=CMD.unpack_from(code,i*16)
            if op==1:
                if a<parameter_start:
                    if a+c>parameter_start:raise ValueError('DMA straddles activation/immutable boundary')
                    a=address(a)
                else:a=append+a-parameter_start
                if a+c>len(payload):raise ValueError('relocated DMA out of bounds')
            if op==2:
                if i not in records:raise ValueError('missing RUN diagnostic contract')
                contracts[str(len(commands))]=records[i]
            commands.append((op,flags,res,a,b,c))
        components.append(dict(start=start,stop=stop,kind=cfg['kind'],configuration=cfg,
            first_command=first,last_command=len(commands),standalone_replay=proof,
            standalone_code_sha256=hashlib.sha256(code).hexdigest(),
            standalone_payload_sha256=hashlib.sha256(data).hexdigest(),
            input_slot=current,output_slot=output_after,immutable_external_base=append,
            standalone_immutable_base=parameter_start))
        current=output_after
    commands.append((0,0,0,0,0,0));code=b''.join(CMD.pack(*c) for c in commands)
    if len(code)>32768:raise ValueError('composed-command-capacity')
    if len(payload)>8*1024*1024:raise ValueError('composed-external-capacity')
    return code,bytes(payload),dict(schema=1,components=components,run_contracts=contracts,
        final_output=dict(ext=current*slot,bytes=oracle[program.outputs[0]].nbytes),
        activation_slot_bytes=slot,peak_sram_address=peak,identity_lut_elisions=identity_elisions,
        program_sha256=hashlib.sha256(code).hexdigest(),image_sha256=hashlib.sha256(payload).hexdigest(),
        replay=dict(status='passed',scope='independent standalone oracle replay plus exact relocation; native full model separately'))


def fixture(directory,program,oracle,code,payload,record):
    directory.mkdir(parents=True,exist_ok=True)
    files={'commands.bin':code,'payload.bin':payload,'input.bin':oracle[program.layers[0].output].tobytes(),
        'output.bin':oracle[program.outputs[0]].tobytes(),
        'checks.txt':f'{record["final_output"]["ext"]} output.bin\n'.encode(),
        'schedule.json':(json.dumps(record,sort_keys=True,indent=2)+'\n').encode()}
    for name,data in files.items():(directory/name).write_bytes(data)
    return {name:sha(directory/name) for name in files}


def diagnostics(directory,program,oracle,code,payload,record):
    """Independently replay every produced/COPY tensor and snapshot actual RTL outputs.

    Timed programs are untouched. Diagnostic stores occur after a drained WAIT
    and check every descriptor output, including factor latents and COPY results.
    """
    directory.mkdir(parents=True,exist_ok=True)
    ext=bytearray(payload);host=oracle[program.layers[0].output].tobytes();ext[:len(host)]=host
    sram=bytearray(32768);commands=[];checks=[];pending=None;layers=set()
    for i,row in enumerate(struct.iter_unpack('<BBHIII',code)):
        op,flags,_,a,b,c=row;commands.append(row)
        if op==1:
            if flags:sram[b:b+c]=ext[a:a+c]
            else:ext[a:a+c]=sram[b:b+c]
        elif op==2:
            r=record['run_contracts'][str(i)];raw=bytes(sram[a:a+64])
            if int.from_bytes(raw[6:8],'little')&1:d,table=decode_fused(raw)
            else:d,table=Descriptor.decode(raw),None
            if table is not None and r.get('lut_hex') is not None:
                if bytes(sram[table:table+256])!=bytes.fromhex(r['lut_hex']):
                    raise ValueError('nonidentity LUT differs after removing identity loads')
            if d.opcode==3:value=bytes(sram[d.input:d.input+d.count])
            elif r['kind']=='conv':
                y0,x0,y1,x1=r['rect'];layer=r['layer']+1
                value=oracle[program.layers[layer].output][:,:d.output_c,y0:y1,x0:x1].copy().tobytes();layers.add(layer)
            elif r['kind']=='strip':
                y0,y1=r['output_rows'];layer=r['layer']+1
                value=oracle[program.layers[layer].output][:,:d.output_c,y0:y1,:].copy().tobytes();layers.add(layer)
            elif r['kind']=='fallback':
                layer=r['layer'];arr=oracle[program.layers[layer].output].reshape(-1)
                value=arr[r['first_element']:r['first_element']+d.outputs].tobytes()
                if table is not None:
                    value=bytes(sram[table+v] for v in value)
                    layer+=1
                layers.add(layer)
            else:raise ValueError('unknown diagnostic RUN kind')
            if len(value)!=d.outputs:raise ValueError('diagnostic output shape differs')
            sram[d.output:d.output+d.outputs]=value;pending=(d.output,value,i)
        elif op==3 and flags&1 and pending is not None:
            base,value,run=pending
            # Drain any simultaneous prefetch before starting a snapshot DMA.
            commands.append((3,3,0,0,0,0))
            lo=base//8*8;size=align8(base-lo+len(value));dest=align8(len(ext))
            ext.extend(bytes(dest-len(ext)));ext.extend(bytes(size))
            commands.extend([(1,0,0,dest,lo,size),(3,2,0,0,0,0)])
            name=f'run-{run}.bin';(directory/name).write_bytes(value)
            checks.append(f'{dest+base-lo} {name}');pending=None
    if pending is not None:raise ValueError('RUN did not drain')
    if ext[record['final_output']['ext']:record['final_output']['ext']+oracle[program.outputs[0]].nbytes]!=oracle[program.outputs[0]].tobytes():
        raise ValueError('composed independent replay final output differs')
    checks.append(f'{record["final_output"]["ext"]} output.bin')
    payload=payload+bytes(len(ext)-len(payload));code=b''.join(CMD.pack(*c) for c in commands)
    if len(code)>32768 or len(payload)>8*1024*1024:raise ValueError('diagnostic capacity')
    for name,data in {'commands.bin':code,'payload.bin':payload,'input.bin':host,
        'output.bin':oracle[program.outputs[0]].tobytes(),'checks.txt':('\n'.join(checks)+'\n').encode()}.items():
        (directory/name).write_bytes(data)
    metadata=dict(descriptor_output_checks=len(checks)-1,layer_outputs_checked=sorted(layers),
        tensor_checks=len(checks),program_bytes=len(code),payload_bytes=len(payload),
        scope='every Conv/Gemm/pool/COPY descriptor output, including latent requantization; fused raw Conv outputs are not physically materialized',
        files={p.name:sha(p) for p in directory.iterdir() if p.is_file() and p.name!='diagnostic.json'})
    save(directory/'diagnostic.json',metadata)
    return metadata


def native(exe,directory,dest,seed):
    dest.parent.mkdir(parents=True,exist_ok=True)
    began=time.perf_counter()
    call=subprocess.run([str(exe),str(directory),str(seed),str(dest)],cwd=ROOT,
        capture_output=True,text=True,timeout=180)
    if call.returncode:
        save(dest,dict(status='failed',returncode=call.returncode,stdout=call.stdout[-2000:],stderr=call.stderr[-2000:]))
        raise RuntimeError('native rejected fixture: '+str(directory)+' '+call.stderr)
    result=json.loads(dest.read_text())
    count=len((directory/'checks.txt').read_text().splitlines())
    if result['status']!='passed' or result['tensor_checks']!=count:raise ValueError('native tensor coverage')
    result.update(seed=seed,simulation_seconds=time.perf_counter()-began,report_sha256=sha(dest))
    return result


def imported_pins():
    paths={TARGET_PATH.resolve()}
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f:
            p=Path(f).resolve()
            if p.suffix=='.py' and (p.is_relative_to(REFERENCE) or p.is_relative_to(ROOT)):
                paths.add(p)
    paths.add(Path(__file__).resolve())
    return {str(p):sha(p) for p in sorted(paths)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-configurations',type=int,default=12)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter();exe,identity=native_identity()
    original,pinned,dense,dense_maps,dense_provenance=matched_model('vww')
    factor_path=ROOT/'work/phase6/representation-screen-v1/vww/selected-int8.uq2'
    screen=json.loads((factor_path.parent/'report.json').read_text())
    if sha(factor_path)!=screen['export']['sha256']:raise ValueError('frozen factor export changed')
    frozen_factor=load_image(factor_path.read_bytes());factor,factor_maps,factor_provenance=factor_model(frozen_factor)
    old_fixture=REFERENCE/'work/phase6/matched-baselines-v1/b3-weights-vww/vww/pair-005/resident'
    if not old_fixture.exists():old_fixture=OUT/'sources/strong-dense-control'
    old_schedule=json.loads((old_fixture/'schedule.json').read_text())
    incumbent=[c['configuration'] for c in old_schedule['components'][:-1]]
    paths=configurations(incumbent)[:args.max_configurations]
    pins=imported_pins();model_pins=dict(dense_provenance['sources'])
    model_pins={str(ROOT/p):v for p,v in model_pins.items()}
    model_pins.update({str(factor_path):sha(factor_path),str(factor_path.parent/'report.json'):sha(factor_path.parent/'report.json'),
        str(old_fixture/'commands.bin'):sha(old_fixture/'commands.bin'),
        str(old_fixture/'payload.bin'):sha(old_fixture/'payload.bin'),str(old_fixture/'schedule.json'):sha(old_fixture/'schedule.json')})
    report=dict(status='running',physical_board=False,selected_engine_changed=False,native_identity=identity,
        source_sha256=pins,model_source_sha256=model_pins,threads=1,native_build_jobs=0,
        memory_limit_target_bytes=1<<30,search_budget=dict(configurations_per_graph=len(paths),
            selection='lowest native fixed-RAM cycles on pinned input',
            space='incumbent four spatial region cuts; same 12 final-region height/width/cache/residency settings; no model tuning'),
        strongest_dense_reference_cycles=2014205,
        variants={},comparisons=[],limitations=[
            'Restricted matched compiler search, not exhaustive schedule optimality.',
            'No physical SDRAM, board latency, energy or new accuracy evaluation.',
            'The factor graph uses an exact identity Clip to express signed latent INT8 rounding in the existing Conv/activation backend.',
            'Classification candidate stays frozen; previously observed held-out data is not used for tuning.'])
    for name,p,orig,maps,provenance in (('dense',dense,original,dense_maps,dense_provenance),
                                      ('factor',factor,frozen_factor,factor_maps,factor_provenance)):
        oracle=verified_oracle(orig,p,maps,pinned,provenance)
        row=dict(graph_sha256=graph_identity(p),arithmetic=graph_stats(p),transformations=provenance,candidates=[])
        report['variants'][name]=row
        for index,path in enumerate(paths):
            configs=copy.deepcopy(path) if name=='dense' else map_path(path,dense,factor)
            dest=OUT/'candidates'/name/f'c{index:02d}'
            try:code,payload,schedule=compose(p,configs,oracle,dest/'components')
            except ValueError as error:
                if str(error) not in ('sram-capacity','command-capacity','composed-command-capacity','external-capacity',
                    'composed-external-capacity','constant-tail-alignment','dma-stripe-alignment','dma-bounds/alignment'):
                    raise
                row['candidates'].append(dict(index=index,configurations=configs,status='infeasible',reason=str(error)))
                print(name,index,'infeasible',str(error),flush=True)
            else:
                files=fixture(dest/'timed',p,oracle,code,payload,schedule)
                result=native(exe,dest/'timed',dest/'native-s0.json',0)
                row['candidates'].append(dict(index=index,configurations=configs,status='passed-native',
                    native=result,fixture=str((dest/'timed').relative_to(ROOT)),fixture_sha256=files,
                    program_bytes=len(code),payload_bytes=len(payload),peak_sram_address=schedule['peak_sram_address'],
                    scheduled_dma_bytes=sum(r[5] for r in struct.iter_unpack('<BBHIII',code) if r[0]==1)))
                if name=='dense' and index==0:
                    if (code!=(old_fixture/'commands.bin').read_bytes() or payload!=(old_fixture/'payload.bin').read_bytes()
                            or result['elapsed_cycles']!=2014205):raise ValueError('strongest dense control not reproduced')
                print(name,index,result['elapsed_cycles'],flush=True)
            save(OUT/'report.json',report)
        passed=[r for r in row['candidates'] if r['status']=='passed-native']
        if not passed:raise ValueError('no feasible matched schedule for '+name)
        winner=min(passed,key=lambda r:(r['native']['elapsed_cycles'],r['index']))
        row['selected_index']=winner['index'];row['validation']=[]
        cfg=winner['configurations']
        stress=np.random.default_rng(6062).integers(-128,128,pinned.shape,dtype=np.int8)
        for sample,value in (('pinned',pinned),('stress',stress)):
            oracle=verified_oracle(orig,p,maps,value,provenance)
            dest=OUT/'selected'/name/sample
            code,payload,schedule=compose(p,cfg,oracle,dest/'components')
            fixture(dest/'timed',p,oracle,code,payload,schedule)
            diag=diagnostics(dest/'check',p,oracle,code,payload,schedule)
            for mode in ('timed','check'):
                for seed in (0,6063):
                    result=native(exe,dest/mode,dest/f'{mode}-s{seed}.json',seed)
                    result.update(sample=sample,mode=mode,fixture=str((dest/mode).relative_to(ROOT)))
                    row['validation'].append(result)
                    print('validate',name,sample,mode,seed,result['elapsed_cycles'],flush=True)
            save(dest/'diagnostic-coverage.json',diag)
            save(OUT/'report.json',report)
    for sample in ('pinned','stress'):
        for seed in (0,6063):
            results={n:next(r for r in v['validation'] if (r['sample'],r['mode'],r['seed'])==(sample,'timed',seed))
                     for n,v in report['variants'].items()}
            d,f=(results[n]['elapsed_cycles'] for n in ('dense','factor'))
            report['comparisons'].append(dict(sample=sample,seed=seed,dense_elapsed_cycles=d,factor_elapsed_cycles=f,
                elapsed_cycle_reduction_fraction=(d-f)/d,
                strongest_dense_reference_elapsed_cycles=2014205 if seed==0 else 2059395))
    ablation=OUT/'identity-lut-ablation/tagged-fixture'
    if ablation.exists():
        report['identity_lut_ablation']={'scope':'same frozen factor graph/path; tagged identity lookup versus ordinary signed Conv latent',
            'tagged_fixture_sha256':{p.name:sha(p) for p in ablation.iterdir() if p.is_file()},'native':[]}
        for seed in (0,6063):
            report['identity_lut_ablation']['native'].append(native(exe,ablation,OUT/f'identity-lut-ablation/tagged-timed-s{seed}.json',seed))
    source_bundle=[]
    for path,digest in pins.items():
        p=Path(path);tree='reference' if p.is_relative_to(REFERENCE) else 'workspace'
        relative=p.relative_to(REFERENCE if tree=='reference' else ROOT)
        dest=OUT/'sources'/tree/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(p.read_bytes())
        if sha(dest)!=digest:raise ValueError('source bundle copy differs')
        source_bundle.append(dict(original_path=path,bundled_path=str(dest.relative_to(ROOT)),sha256=digest))
    # Preserve the generated index-only constant folder and strongest control,
    # without original datasets or executable/build files.
    folder=inspect.getsource(novelty_constants.fold_final_constants).replace('range(54,58)',
        f'range({len(frozen_factor.layers)-4},{len(frozen_factor.layers)})')
    generated=OUT/'sources/generated-factor-final-constants.py';generated.write_text(folder)
    report['source_bundle']=source_bundle
    report['generated_constant_folder']=dict(file=str(generated.relative_to(ROOT)),sha256=sha(generated),
        VWW=list(range(len(frozen_factor.layers)-5,len(frozen_factor.layers))))
    control=OUT/'sources/strong-dense-control';control.mkdir(parents=True,exist_ok=True)
    report['strong_dense_control_source_bundle']={}
    for p in old_fixture.iterdir():
        if p.is_file():
            dest=control/p.name;dest.write_bytes(p.read_bytes())
            report['strong_dense_control_source_bundle'][str(dest.relative_to(ROOT))]=sha(dest)
    report['native_source_bundle']={}
    for name,digest in identity['sources_sha256'].items():
        path=Path(name)
        if path.is_relative_to(ROOT):continue
        if not path.exists():path=OUT/'sources/native'/path.name
        dest=OUT/'sources/native'/path.name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
        if sha(dest)!=digest:raise ValueError('native source bundle differs')
        report['native_source_bundle'][str(dest.relative_to(ROOT))]=digest
    dense_best=next(c for c in report['variants']['dense']['candidates'] if c['index']==report['variants']['dense']['selected_index'])
    factor_best=next(c for c in report['variants']['factor']['candidates'] if c['index']==report['variants']['factor']['selected_index'])
    def stats(c):
        code=(ROOT/c['fixture']/'commands.bin').read_bytes();rows=list(struct.iter_unpack('<BBHIII',code))
        return dict(engine_runs=sum(r[0]==2 for r in rows),command_count=len(rows),
            elapsed_cycles=c['native']['elapsed_cycles'],engine_cycles=c['native']['engine_cycles'],
            dma_cycles=c['native']['dma_cycles'],overlap_cycles=c['native']['overlap_cycles'],
            dispatch_and_wait_cycles=c['native']['elapsed_cycles']-c['native']['engine_cycles']-c['native']['dma_cycles']+c['native']['overlap_cycles'],
            useful_macs=report['variants']['dense' if c is dense_best else 'factor']['arithmetic']['logical_macs'],
            scheduled_dma_bytes=c['scheduled_dma_bytes'],program_bytes=c['program_bytes'],payload_bytes=c['payload_bytes'])
    report['cycle_breakdown']={n:stats(c) for n,c in (('dense',dense_best),('factor',factor_best))}
    for path,digest in dict(pins,**model_pins).items():
        if sha(path)!=digest:raise ValueError('source/model changed during experiment: '+path)
    native_identity()
    report.update(status='passed-matched-optimized-native-comparison',elapsed_seconds=time.perf_counter()-start,
        peak_python_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        peak_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        gate='factor model optimization useful if faster than best dense; no new hardware mechanism or architecture novelty established')
    if report['peak_python_rss_bytes']>1<<30:raise ValueError('Python memory limit exceeded')
    save(OUT/'report.json',report)


if __name__=='__main__':main()
