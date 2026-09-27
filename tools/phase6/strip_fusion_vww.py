#!/usr/bin/env python3
"""Executable same-slot VWW DW/Relu/PW/Relu strip-retention probe.

The selected tagged activation engine is reused unchanged. This is a bounded
four-layer block experiment, not a full-model performance claim. A two-strip
anti-dependence prefetch preserves source rows before the first output store.
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'compiler'), str(ROOT/'tools/phase6')]

from static_pipeline import Program
from hardware_v2 import Descriptor
from integer_reference import evaluate
from phase4_compile import _parameter_rows
from scheduler.resident import compile_resident
from scheduler.fused_verify import replay_fused
from followup_graph import group_channels
from fused_activation import lower_fixture
from output_pipeline_fusion import activation_table, decode_fused, encode_fused
from run_boardless import load_model

OUT = ROOT/'work/phase6/strip-fusion-vww-v1'
NATIVE = ROOT/'work/phase6/pool-timing-v1/native/Vv2_tiled_host_bridge'
NATIVE_REPORT = NATIVE.parent/'report.json'
CMD = struct.Struct('<BBHIII')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def align8(value):
    return (value+7)&~7


def block_program():
    original, pinned, _, sources = load_model('vww')
    grouped, _, _ = group_channels(original)
    layers = copy.deepcopy(grouped.layers[3:7])
    names = [layers[0].inputs[0]]+[layer.output for layer in layers]
    tensors = {name: copy.deepcopy(grouped.tensors[name]) for name in names}
    block = Program(tensors,layers,[names[0]],[names[-1]],{},copy.deepcopy(grouped.provenance))
    return original, grouped, block, pinned, sources


def local_oracle(block, source, start):
    assert start in (0,24)
    # 3x3 DW needs a one-row halo at the shared strip boundary. Full width
    # keeps all external DMA starts/lengths eight-byte aligned.
    source_first = 0 if start == 0 else 23
    local_input = source[:,:,source_first:source_first+25,:].copy()
    local = copy.deepcopy(block)
    local.tensors[local.inputs[0]] = replace(local.tensors[local.inputs[0]], shape=(1,8,25,48))
    for layer in local.layers:
        channels = 16 if layer is local.layers[2] or layer is local.layers[3] else 8
        local.tensors[layer.output] = replace(local.tensors[layer.output],shape=(1,channels,24,48))
    local.layers[0].attributes['pads']=[1,1,0,1] if start == 0 else [0,1,1,1]
    oracle=evaluate(local,{local.inputs[0]:local_input})
    return local_input, oracle


def rows_and_params(block):
    return [_parameter_rows(block,layer) for layer in block.layers]


def build_candidate(block):
    assert tuple(block.tensors[block.inputs[0]].shape)==(1,8,48,48)
    assert tuple(block.tensors[block.outputs[0]].shape)==(1,16,48,48)
    slot=36864
    payload=bytearray(slot)
    def put(data):
        pos=align8(len(payload));payload.extend(bytes(pos-len(payload)));payload.extend(data)
        return pos
    values=rows_and_params(block)
    external=[]
    for weights,params in values:
        external.append(dict(weight=put(weights) if weights else None,
                             params=put(params) if params else None,
                             weight_bytes=weights,params_bytes=params))
    tables=[activation_table(2,external[i]['params_bytes']) for i in (1,3)]
    table_ext=[put(t) for t in tables]
    commands=[];run_records=[];transfer_records=[]
    def emit(op,flags=0,a=0,b=0,c=0):
        commands.append((op,flags,0,a,b,c))
    def dma(direction,ext,sram,size,role):
        assert ext%8==sram%8==0 and 0<size<=32768 and sram+size<=32768
        assert ext+size<=len(payload)
        index=len(commands);emit(1,1 if direction=='to_sram' else 0,ext,sram,size);emit(3,2)
        transfer_records.append(dict(command=index,direction=direction,role=role,ext=ext,sram=sram,bytes=size))
    def run(layer,strip,desc,table_sram,table_index,expected_input,output_bytes,weight_bytes,params_bytes):
        assert desc.next_pc==64
        desc.validate()
        encoded=encode_fused(desc,table_sram)
        again,pointer=decode_fused(encoded)
        assert again==desc and pointer==table_sram
        desc_ext=put(encoded+Descriptor(0).encode())
        dma('to_sram',desc_ext,0,128,'descriptor')
        weight=external[layer]['weight']
        params=external[layer]['params']
        dma('to_sram',weight,desc.weight,len(weight_bytes),'weight')
        dma('to_sram',params,desc.params,len(params_bytes),'params')
        dma('to_sram',table_ext[table_index],table_sram,256,'activation_table')
        index=len(commands);emit(2,0,0,32768<<16);emit(3,3)
        run_records.append(dict(command=index,layer=layer,strip=strip,
            descriptor_hex=encoded.hex(),table_sram=table_sram,
            table_sha256=digest(tables[table_index]),
            expected_input=expected_input,output_bytes=output_bytes,
            weight_hex=weight_bytes.hex(),params_hex=params_bytes.hex()))

    # Two orientations permit the next strip's 9,600-byte source to be loaded
    # after PW completes but before output stores overwrite its external halo.
    source_bases=(128,18560)
    resident_bases=(18560,128)
    output_bases=(128,9344)
    for strip,start in enumerate((0,24)):
        source_base=source_bases[strip]
        resident=resident_bases[strip]
        output=output_bases[strip]
        local_pad=(1,0) if strip==0 else (0,1)
        if strip==0:
            for channel in range(8):
                dma('to_sram',channel*2304,source_base+channel*1200,1200,'source_strip0')
        dw_weight=9728 if strip==0 else 9344
        dw_params=dw_weight+128
        dw_table=dw_params+128
        dw=Descriptor(6,input=source_base,output=resident,weight=dw_weight,params=dw_params,
            count=9,outputs=9216,row_stride=16,next_pc=64,kernel_h=3,kernel_w=3,
            stride_h=1,stride_w=1,pad_top=local_pad[0],pad_bottom=local_pad[1],
            pad_left=1,pad_right=1,input_h=25,input_w=48,input_c=8,output_c=8)
        run(0,strip,dw,dw_table,0,'source',9216,*values[0])
        pw_weight=27776
        pw_params=pw_weight+128
        pw_table=pw_params+256
        pw=Descriptor(4,input=resident,output=output,weight=pw_weight,params=pw_params,
            count=8,outputs=18432,row_stride=8,next_pc=64,kernel_h=1,kernel_w=1,
            input_h=24,input_w=48,input_c=8,output_c=16)
        run(2,strip,pw,pw_table,1,'dw_relu',18432,*values[2])
        if strip==0:
            # Before writing rows 0..23 over the source slot, capture rows
            # 23..47 (including the shared halo) in the dead DW-output bank.
            for channel in range(8):
                dma('to_sram',channel*2304+23*48,source_bases[1]+channel*1200,
                    1200,'prefetch_source_strip1')
        for channel in range(16):
            dma('from_sram',channel*2304+start*48,output+channel*1152,
                1152,f'output_strip{strip}')
    emit(0)
    code=b''.join(CMD.pack(*c) for c in commands)
    assert len(code)<=32768 and len(payload)<=8*1024*1024
    assert len(commands)<=2048
    result=dict(schema=1,kind='vww-layers3-6-two-full-width-strips',
        output_external_base=0,source_external_base=0,command_count=len(commands),
        program_bytes=len(code),payload_bytes=len(payload),
        max_sram_end=28416,bridge_intermediate_external_bytes=0,
        source_halo_bytes=768,run_records=run_records,transfers=transfer_records,
        code_sha256=digest(code),payload_sha256=digest(payload))
    return code,bytes(payload),result


def replay_candidate(block,source,expected,code,payload,record):
    assert digest(code)==record['code_sha256'] and digest(payload)==record['payload_sha256']
    assert len(code)%16==0 and len(code)<=32768
    ext=bytearray(payload);ext[:source.size]=source.tobytes()
    sram=bytearray(32768)
    locals_=[local_oracle(block,source,start) for start in (0,24)]
    records={r['command']:r for r in record['run_records']}
    seen=[]
    for index in range(len(code)//16):
        op,flags,reserved,a,b,c=CMD.unpack_from(code,index*16)
        assert reserved==0
        if op==0:
            assert index==len(code)//16-1 and (flags,a,b,c)==(0,0,0,0)
            break
        if op==3:
            assert flags in (2,3) and (a,b,c)==(0,0,0)
            continue
        if op==1:
            assert flags in (0,1) and a%8==b%8==0 and 0<c<=32768
            assert a+c<=len(ext) and b+c<=len(sram)
            if flags:sram[b:b+c]=ext[a:a+c]
            else:ext[a:a+c]=sram[b:b+c]
            continue
        assert op==2 and flags==c==a==0 and b==32768<<16
        stage=records[index];seen.append(index)
        descriptor,pointer=decode_fused(bytes(sram[:64]))
        assert bytes(sram[:64]).hex()==stage['descriptor_hex']
        assert pointer==stage['table_sram'] and digest(sram[pointer:pointer+256])==stage['table_sha256']
        assert Descriptor.decode(bytes(sram[64:128])).opcode==0
        assert bytes(sram[descriptor.weight:descriptor.weight+len(bytes.fromhex(stage['weight_hex']))])==bytes.fromhex(stage['weight_hex'])
        assert bytes(sram[descriptor.params:descriptor.params+len(bytes.fromhex(stage['params_hex']))])==bytes.fromhex(stage['params_hex'])
        local_input,values=locals_[stage['strip']]
        if stage['expected_input']=='source':operand=local_input
        else:operand=values[block.layers[1].output]
        assert bytes(sram[descriptor.input:descriptor.input+operand.size])==operand.tobytes()
        producer=values[block.layers[stage['layer']].output].tobytes()
        activated=values[block.layers[stage['layer']+1].output].tobytes()
        table=sram[pointer:pointer+256]
        result=bytes(table[byte] for byte in producer)
        assert result==activated and len(result)==stage['output_bytes']==descriptor.outputs
        sram[descriptor.output:descriptor.output+len(result)]=result
    assert seen==[r['command'] for r in record['run_records']]
    assert ext[:expected.size]==expected.tobytes()
    return dict(status='passed',command_count=len(code)//16,engine_runs=len(seen),
                final_output_sha256=digest(expected.tobytes()),
                strip_output_sha256=[digest(values[block.layers[-1].output].tobytes()) for _,values in locals_],
                scope='serialized command replay with independent INT8 four-layer oracle and checked LUT/operand bytes')


def fixture(path,code,payload,source,expected,checks_base,extra=None):
    path.mkdir(parents=True,exist_ok=True)
    files={'commands.bin':code,'payload.bin':payload,'input.bin':source.tobytes(),
           'output.bin':expected.tobytes(),
           'checks.txt':f'{checks_base} output.bin\n'.encode()}
    if extra is not None:files['schedule.json']=(json.dumps(extra,sort_keys=True,indent=2)+'\n').encode()
    for name,data in files.items():(path/name).write_bytes(data)
    return {name:digest(data) for name,data in files.items()}


def splice_full_fixture(block, grouped, compacted, source_folder, target_folder,
                        candidate_code, candidate_payload, candidate, host_input,
                        run_native):
    """Replace only layers 3..6; preserve the compacted graph's external slots.

    The old tail-prefetch of layer 7 parameters is serialized after the new
    block. This avoids an SRAM write into the new block's live region.
    """
    old_code=(source_folder/'commands.bin').read_bytes()
    old_payload=(source_folder/'payload.bin').read_bytes()
    old_schedule=json.loads((source_folder/'schedule.json').read_text())
    old=[CMD.unpack_from(old_code,i) for i in range(0,len(old_code),16)]
    stages=old_schedule['stages']
    first=next(s for s in stages if s['layer']==3)
    following=next(s for s in stages if s['layer']==7)
    def descriptor_command(stage):
        load=next(t for t in stage['loads'] if t['role']=='descriptor')
        hits=[i for i,c in enumerate(old) if c==(1,1,0,load['ext'],load['sram'],load['bytes'])]
        assert len(hits)==1
        return hits[0]
    cut_first,cut_end=descriptor_command(first),descriptor_command(following)
    assert old[cut_first-1]==(3,2,0,0,0,0) and old[cut_end][0]==1
    removed=old[cut_first:cut_end]
    first_external=first['loads']
    source_load=next(t for t in first_external if t['role']=='input')
    activation_ext=source_load['ext']
    assert activation_ext==36864
    assert all(s['store'] is None or s['store']['ext'] in (0,36864,50688,64512)
               for s in stages if 3<=s['layer']<=6)
    next_params=[t for t in following['loads'] if t['role']=='parameter']
    preserved=[]
    for load in next_params:
        item=(1,1,0,load['ext'],load['sram'],load['bytes'])
        matches=[i for i,c in enumerate(removed) if c==item]
        assert len(matches)==1 and removed[matches[0]+1]==(3,2,0,0,0,0)
        preserved.extend((item,(3,2,0,0,0,0)))
    assert len(preserved)==6
    # Every outside-layer run begins after the cut. Layer-7 parameter prefetch
    # is the only load from its first stage already issued inside the cut.
    assert all(not (cut_first<=int(i)<cut_end) or 3<=v['layer']<=6
               for i,v in old_schedule['run_contracts'].items())
    payload=bytearray(old_payload)
    appended=align8(len(payload))
    payload.extend(bytes(appended-len(payload)))
    payload.extend(candidate_payload[36864:])
    candidate_commands=[CMD.unpack_from(candidate_code,i) for i in range(0,len(candidate_code),16)]
    transfers={t['command']:t for t in candidate['transfers']}
    runs={r['command']:r for r in candidate['run_records']}
    new_block=[];new_runs=[]
    snapshots={int(k):v for k,v in old_schedule['snapshot_regions'].items()}
    def add_dma(ext,sram,size):
        assert ext%8==sram%8==0 and ext+size<=len(payload) and sram+size<=32768
        new_block.extend(((1,0,0,ext,sram,size),(3,2,0,0,0,0)))
    for old_index,cmd in enumerate(candidate_commands[:-1]):
        op,flags,reserved,a,b,c=cmd
        if op==1:
            transfer=transfers[old_index]
            if transfer['role'].startswith(('source_strip','prefetch_source','output_strip')):
                a+=activation_ext
            else:
                assert flags==1 and a>=36864
                a=appended+(a-36864)
            cmd=(op,flags,reserved,a,b,c)
        if op==2:
            stage=runs[old_index]
            new_runs.append((len(new_block),stage))
        new_block.append(cmd)
        if op==3 and flags==3 and old_index-1 in runs:
            stage=runs[old_index-1]
            layer=4 if stage['layer']==0 else 6
            if layer in snapshots:
                d,_=decode_fused(bytes.fromhex(stage['descriptor_hex']))
                rows=8 if layer==4 else 16
                for channel in range(rows):
                    ext=snapshots[layer]['ext']+channel*2304+stage['strip']*1152
                    add_dma(ext,d.output+channel*1152,1152)
    new_block.extend(preserved)
    combined=old[:cut_first]+new_block+old[cut_end:]
    code=b''.join(CMD.pack(*c) for c in combined)
    assert len(code)<=32768 and len(payload)<=8*1024*1024 and len(combined)<=2048
    delta=len(new_block)-(cut_end-cut_first)
    schedule=copy.deepcopy(old_schedule)
    schedule['stages']=[s for s in stages if not 3<=s['layer']<=6]
    for key in ('run_contracts','constant_contracts','pack_contracts'):
        remapped={}
        for index,value in old_schedule.get(key,{}).items():
            i=int(index)
            if cut_first<=i<cut_end:continue
            remapped[str(i if i<cut_first else i+delta)]=value
        schedule[key]=remapped
    for local_index,stage in new_runs:
        global_index=cut_first+local_index
        schedule['run_contracts'][str(global_index)]=dict(
            layer=3 if stage['layer']==0 else 5,
            fused_activation_layer=4 if stage['layer']==0 else 6,
            fused_table_sha256=stage['table_sha256'],
            strip_origin_y=stage['strip']*24,strip_height=24)
    schedule.update(command_count=len(combined),program_bytes=len(code),
                    program_sha256=digest(code),image_sha256=digest(payload),
                    catalogue='existing-ABI two-strip DW/activation/PW/activation retention',
                    strip_splice=dict(source_fixture=source_folder.name,cut_first=cut_first,
                        cut_end=cut_end,inserted_commands=len(new_block),
                        preserved_next_layer_prefetch=len(preserved)//2,
                        activation_external_base=activation_ext,
                        source_sha256=digest(old_code),source_payload_sha256=digest(old_payload)))
    target_folder.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source_folder,target_folder,dirs_exist_ok=True)
    (target_folder/'commands.bin').write_bytes(code)
    (target_folder/'payload.bin').write_bytes(payload)
    (target_folder/'schedule.json').write_text(json.dumps(schedule,sort_keys=True,indent=2)+'\n')
    # The pinned/stress public input and every expected integer tensor are
    # copied from the unchanged compacted fixture. Native full-model execution
    # validates them against the actual spliced command stream.
    expected=(target_folder/'output.bin').read_bytes()
    assert len(host_input.tobytes())==(target_folder/'input.bin').stat().st_size==27648
    assert (target_folder/'input.bin').read_bytes()==host_input.tobytes()
    assert len(expected)==schedule['final_output']['bytes']
    native=[]
    if run_native:
        for seed in (0,6063):
            path=target_folder.parent.parent/f'{target_folder.name}-seed{seed}.json'
            subprocess.run([str(NATIVE),str(target_folder),str(seed),str(path)],check=True)
            result=json.loads(path.read_text())
            assert result['status']=='passed'
            native.append(result)
    return dict(label=target_folder.name,baseline_fixture=source_folder.name,
                files={p.name:digest(p.read_bytes()) for p in target_folder.iterdir() if p.is_file()},
                verification=dict(status='passed',scope='block INT8 replay, unchanged full prefix/suffix and exact full-model native checks'),
                native=native,splice=schedule['strip_splice'])


def full_model(output,run_native):
    output.mkdir(parents=True,exist_ok=True)
    original,grouped,block,pinned,sources=block_program()
    from channel_compaction import compact_channels,check_oracles
    compacted,kept,_,_=compact_channels(grouped)
    # The early DW/PW pair is identical after exact channel compaction. The
    # host-boundary KWS fixtures are copied without modification.
    for i in range(3,7):
        a,b=block.layers[i-3],compacted.layers[i]
        assert a.op==b.op and block.tensors[a.output].shape==compacted.tensors[b.output].shape
        assert a.attributes==b.attributes
        assert set(a.parameters)==set(b.parameters)
        assert all(np.array_equal(a.parameters[k],b.parameters[k]) for k in a.parameters)
    source_root=ROOT/'work/phase6/channel-compaction-v1/fused/fixtures'
    source_report=json.loads((source_root.parent/'report.json').read_text())
    assert source_report['status']=='passed'
    source_rows={r['label']:r for r in source_report['results']}
    target_root=output/'fixtures'
    target_root.mkdir(parents=True,exist_ok=True)
    code,payload,candidate=build_candidate(block)
    report=dict(schema=1,status='running',physical_board=False,model='vww',layers=[3,6],
        scope='full compacted VWW command splice on selected tagged RTL; native simulation and board short screen separate',
        model_source_sha256=sources,source_sha256={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__).read_bytes())},
        script_sha256=digest(Path(__file__).read_bytes()),
        source_compaction_report_sha256=digest((source_root.parent/'report.json').read_bytes()),
        selected_native_executable_sha256=digest(NATIVE.read_bytes()),fixtures={})
    if run_native:
        prior=json.loads(NATIVE_REPORT.read_text())
        assert prior['status']=='passed' and prior['executable_sha256']==digest(NATIVE.read_bytes())
    for suffix,sample in (('pinned-compacted-fused-timed','pinned'),
                          ('pinned-compacted-fused-check','pinned'),
                          ('stress-compacted-fused-check','stress')):
        label='vww-'+suffix
        target=label.replace('compacted-fused','compacted-strip')
        value=pinned if sample=='pinned' else np.random.default_rng(6157).integers(-128,128,pinned.shape,dtype=np.int8)
        compact_oracle=evaluate(compacted,{compacted.inputs[0]:value})
        local_source=compact_oracle[compacted.layers[3].inputs[0]]
        block_oracle=evaluate(block,{block.inputs[0]:local_source})
        assert all(np.array_equal(block_oracle[block.layers[i-3].output],
                                  compact_oracle[compacted.layers[i].output]) for i in range(3,7))
        replay_candidate(block,local_source,block_oracle[block.outputs[0]],code,payload,candidate)
        source_folder=source_root/label
        assert all(digest((source_folder/name).read_bytes())==value
                   for name,value in source_rows[label]['files'].items())
        host_input=compact_oracle[compacted.layers[0].output]
        row=splice_full_fixture(block,grouped,compacted,source_folder,target_root/target,
                                code,payload,candidate,host_input,run_native)
        report['fixtures'][target]=row
        (output/'report.partial.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    for label in ('kws-pinned-compacted-fused-timed','kws-stress-compacted-fused-check'):
        source_folder=source_root/label;target_folder=target_root/label
        assert all(digest((source_folder/name).read_bytes())==value
                   for name,value in source_rows[label]['files'].items())
        shutil.copytree(source_folder,target_folder,dirs_exist_ok=True)
        assert sorted((p.name,digest(p.read_bytes())) for p in source_folder.iterdir() if p.is_file())==\
               sorted((p.name,digest(p.read_bytes())) for p in target_folder.iterdir() if p.is_file())
        report['fixtures'][label]=dict(label=label,baseline_fixture=label,
            files={p.name:digest(p.read_bytes()) for p in target_folder.iterdir() if p.is_file()},
            verification=dict(status='passed',scope='byte-identical previously verified compacted KWS fixture'),
            native=[])
    report['status']='passed-native' if run_native else 'passed-replay'
    (output/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],fixtures={k:[r['elapsed_cycles'] for r in v['native']]
        for k,v in report['fixtures'].items()}),indent=2))


def main(output,run_native):
    output.mkdir(parents=True,exist_ok=True)
    _,grouped,block,pinned,sources=block_program()
    code,payload,candidate=build_candidate(block)
    baseline_code,baseline_payload,baseline_schedule=compile_resident(block,fused={0,2},prefer_half=True,overlap=False)
    baseline_raw=output/'baseline-raw'
    first_oracle=evaluate(grouped,{grouped.inputs[0]:pinned})
    first_source=first_oracle[block.inputs[0]]
    first_result=evaluate(block,{block.inputs[0]:first_source})
    fixture(baseline_raw,baseline_code,baseline_payload,first_source,first_result[block.outputs[0]],
            baseline_schedule['final_output']['ext'],baseline_schedule)
    baseline_code,baseline_payload,baseline_schedule=lower_fixture(baseline_raw)
    if len(baseline_code)>32768:raise AssertionError('baseline program over capacity')
    baseline_activation_bytes=sum(
        load['bytes'] for stage in baseline_schedule['stages']
        for load in stage['loads'] if load['role']=='input')+sum(
        stage['store']['bytes'] for stage in baseline_schedule['stages'] if stage['store'] is not None)
    candidate_activation_bytes=sum(t['bytes'] for t in candidate['transfers']
                                   if t['role'].startswith(('source_strip','prefetch_source','output_strip')))
    report=dict(schema=1,status='running',physical_board=False,model='vww',layers=[3,6],
        scope='Executable existing-ABI two-strip four-layer block, same external source/output slot. No full-model or board result.',
        model_source_sha256=sources,code_sha256=digest(code),payload_sha256=digest(payload),
        baseline_code_sha256=digest(baseline_code),baseline_payload_sha256=digest(baseline_payload),
        selected_native_executable_sha256=digest(NATIVE.read_bytes()) if NATIVE.exists() else None,
        candidate=candidate,baseline=dict(command_count=len(baseline_code)//16,program_bytes=len(baseline_code),
                                          payload_bytes=len(baseline_payload)),
        activation_dma_bytes=dict(baseline=baseline_activation_bytes,candidate=candidate_activation_bytes),samples=[])
    if run_native:
        prior=json.loads(NATIVE_REPORT.read_text())
        assert prior['status']=='passed' and prior['executable_sha256']==digest(NATIVE.read_bytes())
    for sample in ('pinned','stress'):
        if sample=='pinned':input_value=pinned
        else:input_value=np.random.default_rng(6157).integers(-128,128,pinned.shape,dtype=np.int8)
        whole=evaluate(grouped,{grouped.inputs[0]:input_value})
        source=whole[block.inputs[0]]
        oracle=evaluate(block,{block.inputs[0]:source})
        for local_start in (0,24):
            _,local=local_oracle(block,source,local_start)
            for layer in block.layers:
                a=local[layer.output]
                b=oracle[layer.output][:,:,local_start:local_start+24,:]
                assert np.array_equal(a,b),f'{sample} strip{local_start} {layer.op} differs'
        candidate_replay=replay_candidate(block,source,oracle[block.outputs[0]],code,payload,candidate)
        baseline_replay=replay_fused(block,baseline_code,baseline_payload,{block.inputs[0]:source},
            run_contracts=baseline_schedule['run_contracts'],final_output=baseline_schedule['final_output'],
            snapshot_regions=baseline_schedule['snapshot_regions'],oracle=oracle)
        entry=dict(sample=sample,candidate_replay=candidate_replay,baseline_replay=baseline_replay,
                   fixtures={},native={})
        for label,c,p,base in (('candidate',code,payload,0),
                                ('baseline',baseline_code,baseline_payload,baseline_schedule['final_output']['ext'])):
            path=output/f'{label}-{sample}'
            entry['fixtures'][label]=fixture(path,c,p,source,oracle[block.outputs[0]],base)
            if run_native:
                runs=[]
                for seed in (0,6063):
                    result_path=output/f'{label}-{sample}-seed{seed}.json'
                    subprocess.run([str(NATIVE),str(path),str(seed),str(result_path)],check=True)
                    result=json.loads(result_path.read_text())
                    assert result['status']=='passed' and result['tensor_checks']==1
                    runs.append(result)
                entry['native'][label]=runs
        report['samples'].append(entry)
        (output/'report.partial.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    report['status']='passed-native' if run_native else 'passed-replay'
    report['script_sha256']=digest(Path(__file__).read_bytes())
    (output/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],candidate_commands=candidate['command_count'],
        baseline_commands=report['baseline']['command_count'],native={e['sample']:{name:[r['elapsed_cycles'] for r in runs]
        for name,runs in e['native'].items()} for e in report['samples']}),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--native',action='store_true')
    parser.add_argument('--full',action='store_true',help='splice into complete compacted VWW fixtures')
    args=parser.parse_args()
    full_model(args.output/'full',args.native) if args.full else main(args.output,args.native)
