#!/usr/bin/env python3
"""Existing-ABI VWW 7–10 strip retention added to proven 3–6 and 11–14.

This uses two full-width 12-row strips, exact INT8 activation tables, and a
same-slot source prefetch before the first output store. No RTL is modified.
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

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'compiler'),str(ROOT/'tools/phase6')]

from static_pipeline import Program
from hardware_v2 import Descriptor
from integer_reference import evaluate
from phase4_compile import _parameter_rows
from followup_graph import group_channels
from channel_compaction import compact_channels
from output_pipeline_fusion import activation_table,decode_fused,encode_fused
from run_boardless import load_model

BASE=ROOT/'work/phase6/strip-fusion-pair7-v1'
SOURCE=ROOT/'work/phase6/strip-fusion-pair11-v1/full'
NATIVE=ROOT/'work/phase6/pool-timing-v1/native/Vv2_tiled_host_bridge'
CMD=struct.Struct('<BBHIII')


def sha(data):return hashlib.sha256(data).hexdigest()
def align8(value):return (value+7)&~7


def model_block():
    original,pinned,_,sources=load_model('vww')
    grouped,_,_=group_channels(original)
    compacted,_,_,_=compact_channels(grouped)
    layers=copy.deepcopy(compacted.layers[7:11])
    names=[layers[0].inputs[0]]+[layer.output for layer in layers]
    block=Program({n:copy.deepcopy(compacted.tensors[n]) for n in names},layers,
                  [names[0]],[names[-1]],{},copy.deepcopy(compacted.provenance))
    assert block.tensors[block.inputs[0]].shape==(1,16,48,48)
    assert block.tensors[block.outputs[0]].shape==(1,32,24,24)
    assert [l.op for l in layers]==['Conv','Relu','Conv','Relu']
    assert layers[0].attributes['strides']==[2,2] and layers[0].attributes['pads']==[0,0,1,1]
    assert layers[0].parameters['weight'].shape==(16,1,3,3)
    assert layers[2].attributes['strides']==[1,1] and layers[2].parameters['weight'].shape==(32,16,1,1)
    return compacted,block,pinned,sources


def local_oracle(block,source,strip):
    assert strip in (0,1)
    row=0 if strip==0 else 24
    height=25 if strip==0 else 24
    x=source[:,:,row:row+height,:].copy()
    local=copy.deepcopy(block)
    local.tensors[local.inputs[0]]=replace(local.tensors[local.inputs[0]],shape=(1,16,height,48))
    for index,layer in enumerate(local.layers):
        channels=16 if index<2 else 32
        local.tensors[layer.output]=replace(local.tensors[layer.output],shape=(1,channels,12,24))
    local.layers[0].attributes['pads']=[0,0,0,1] if strip==0 else [0,0,1,1]
    return x,evaluate(local,{local.inputs[0]:x})


def build(block):
    payload=bytearray(36864)
    def put(data):
        pos=align8(len(payload));payload.extend(bytes(pos-len(payload)));payload.extend(data)
        return pos
    rows=[_parameter_rows(block,l) for l in block.layers]
    ext=[]
    for weight,params in rows:
        ext.append(dict(weight=put(weight) if weight else None,
                        params=put(params) if params else None,
                        weight_bytes=weight,params_bytes=params))
    tables=[activation_table(2,ext[i]['params_bytes']) for i in (1,3)]
    table_ext=[put(t) for t in tables]
    commands=[];transfers=[];runs=[]
    def emit(op,flags=0,a=0,b=0,c=0):commands.append((op,flags,0,a,b,c))
    def dma(direction,external,sram,size,role):
        assert external%8==sram%8==0 and 0<size<=32768
        assert external+size<=len(payload) and sram+size<=32768
        i=len(commands);emit(1,int(direction=='to_sram'),external,sram,size);emit(3,2)
        transfers.append(dict(command=i,direction=direction,role=role,ext=external,sram=sram,bytes=size))
    def run(layer,strip,d,table_sram,table_id,input_role):
        d.validate()
        encoded=encode_fused(d,table_sram)
        assert decode_fused(encoded)==(d,table_sram)
        desc_ext=put(encoded+Descriptor(0).encode())
        dma('to_sram',desc_ext,0,128,'descriptor')
        dma('to_sram',ext[layer]['weight'],d.weight,len(ext[layer]['weight_bytes']),'weight')
        dma('to_sram',ext[layer]['params'],d.params,len(ext[layer]['params_bytes']),'params')
        dma('to_sram',table_ext[table_id],table_sram,256,'activation_table')
        index=len(commands);emit(2,0,0,32768<<16);emit(3,3)
        runs.append(dict(command=index,layer=layer,strip=strip,input_role=input_role,
                         descriptor_hex=encoded.hex(),table_sha256=sha(tables[table_id]),
                         table_sram=table_sram,weight_hex=ext[layer]['weight_bytes'].hex(),
                         params_hex=ext[layer]['params_bytes'].hex()))
    source_bases=(128,9344)
    resident_bases=(19328,128)
    output_bases=(128,9344)
    for strip in (0,1):
        source_base=source_bases[strip]
        resident=resident_bases[strip]
        output=output_bases[strip]
        if strip==0:
            for channel in range(16):
                dma('to_sram',channel*2304,source_base+channel*1200,1200,'source_strip0')
        dw_weight=23936 if strip==0 else 4736
        dw_params=dw_weight+256
        dw_table=dw_params+256
        dw=Descriptor(6,input=source_base,output=resident,weight=dw_weight,params=dw_params,
            count=9,outputs=4608,row_stride=16,next_pc=64,kernel_h=3,kernel_w=3,
            stride_h=2,stride_w=2,pad_top=0,pad_bottom=0 if strip==0 else 1,
            pad_left=0,pad_right=1,input_h=25 if strip==0 else 24,
            input_w=48,input_c=16,output_c=16)
        run(0,strip,dw,dw_table,0,'source')
        pw_weight=23936 if strip==0 else 18560
        pw_params=pw_weight+512
        pw_table=pw_params+512
        pw=Descriptor(4,input=resident,output=output,weight=pw_weight,params=pw_params,
            count=16,outputs=9216,row_stride=16,next_pc=64,
            input_h=12,input_w=24,input_c=16,output_c=32)
        run(2,strip,pw,pw_table,1,'dw_relu')
        if strip==0:
            # Source/output share the external slot. Capture all lower source
            # rows in dead DW SRAM before storing the upper PW strip.
            for channel in range(16):
                dma('to_sram',channel*2304+24*48,source_bases[1]+channel*1152,
                    1152,'prefetch_source_strip1')
        for channel in range(32):
            dma('from_sram',channel*576+strip*288,output+channel*288,
                288,f'output_strip{strip}')
    emit(0)
    code=b''.join(CMD.pack(*c) for c in commands)
    assert len(code)<=32768 and len(commands)<=2048 and len(payload)<=8*1024*1024
    record=dict(schema=1,scope='existing-ABI exact two-strip VWW 7-10 stride2 block',
        command_count=len(commands),program_bytes=len(code),payload_bytes=len(payload),
        source_halo_bytes=768,activation_dma_bytes=56064,
        max_sram_address_exclusive=27776,code_sha256=sha(code),payload_sha256=sha(payload),
        transfers=transfers,runs=runs)
    return code,bytes(payload),record


def replay(block,source,expected,code,payload,record):
    assert sha(code)==record['code_sha256'] and sha(payload)==record['payload_sha256']
    ext=bytearray(payload);ext[:source.size]=source.tobytes();sram=bytearray(32768)
    locals_=[local_oracle(block,source,s) for s in (0,1)]
    planned={r['command']:r for r in record['runs']};seen=[]
    for i in range(len(code)//16):
        op,flags,reserved,a,b,c=CMD.unpack_from(code,16*i)
        assert reserved==0
        if op==0:
            assert i==len(code)//16-1 and (flags,a,b,c)==(0,0,0,0);break
        if op==3:
            assert flags in (2,3) and (a,b,c)==(0,0,0);continue
        if op==1:
            assert flags in (0,1) and a%8==b%8==0 and 0<c<=32768
            assert a+c<=len(ext) and b+c<=32768
            if flags:sram[b:b+c]=ext[a:a+c]
            else:ext[a:a+c]=sram[b:b+c]
            continue
        assert op==2 and flags==c==a==0 and b==32768<<16
        stage=planned[i];seen.append(i)
        d,pointer=decode_fused(bytes(sram[:64]));d.validate()
        assert bytes(sram[:64]).hex()==stage['descriptor_hex']
        assert Descriptor.decode(bytes(sram[64:128])).opcode==0
        assert pointer==stage['table_sram'] and sha(sram[pointer:pointer+256])==stage['table_sha256']
        weights=bytes.fromhex(stage['weight_hex']);params=bytes.fromhex(stage['params_hex'])
        assert bytes(sram[d.weight:d.weight+len(weights)])==weights
        assert bytes(sram[d.params:d.params+len(params)])==params
        x,values=locals_[stage['strip']]
        operand=x if stage['input_role']=='source' else values[block.layers[1].output]
        assert bytes(sram[d.input:d.input+operand.size])==operand.tobytes()
        producer=values[block.layers[stage['layer']].output].tobytes()
        activated=values[block.layers[stage['layer']+1].output].tobytes()
        table=sram[pointer:pointer+256]
        result=bytes(table[value] for value in producer)
        assert result==activated and len(result)==d.outputs
        sram[d.output:d.output+len(result)]=result
    assert seen==[r['command'] for r in record['runs']]
    assert ext[:expected.size]==expected.tobytes()
    return dict(status='passed',runs=len(seen),final_output_sha256=sha(expected.tobytes()))


def splice(source_folder,target_folder,candidate_code,candidate_payload,candidate,host_input,native):
    old_code=(source_folder/'commands.bin').read_bytes()
    old_payload=(source_folder/'payload.bin').read_bytes()
    old_schedule=json.loads((source_folder/'schedule.json').read_text())
    old=[CMD.unpack_from(old_code,i) for i in range(0,len(old_code),16)]
    stages=old_schedule['stages']
    first=next(s for s in stages if s['layer']==7)
    def descriptor_index(stage):
        load=next(t for t in stage['loads'] if t['role']=='descriptor')
        matches=[i for i,c in enumerate(old) if c==(1,1,0,load['ext'],load['sram'],load['bytes'])]
        assert len(matches)==1
        return matches[0]
    # The compiler hoists the first DW weights/table before its descriptor.
    # Include those loads in the cut, then stop immediately before the already
    # proved pair11 command block. That block must remain byte-identical.
    descriptor=descriptor_index(first)
    first_parameter=next(t for t in first['loads'] if t['role']=='parameter')
    first_parameter_command=(1,1,0,first_parameter['ext'],first_parameter['sram'],first_parameter['bytes'])
    matches=[i for i,c in enumerate(old[:descriptor]) if c==first_parameter_command]
    assert len(matches)==1
    cut_first=matches[0]
    cut_end=old_schedule['pair11_splice']['cut_first']
    assert cut_first<descriptor<cut_end
    assert old[cut_first-1]==(3,2,0,0,0,0)
    assert all(not cut_first<=int(k)<cut_end or 7<=v['layer']<=10
               for k,v in old_schedule['run_contracts'].items())
    activation_ext=next(t['ext'] for t in first['loads'] if t['role']=='input')
    assert activation_ext==36864
    assert old[cut_end]==(1,1,0,activation_ext,128,312)
    assert old_schedule['pair11_splice']['inserted_commands']>0
    preserved=[]
    payload=bytearray(old_payload)
    appended=align8(len(payload));payload.extend(bytes(appended-len(payload)))
    payload.extend(candidate_payload[36864:])
    candidate_commands=[CMD.unpack_from(candidate_code,i) for i in range(0,len(candidate_code),16)]
    transfers={t['command']:t for t in candidate['transfers']}
    runs={r['command']:r for r in candidate['runs']}
    snapshots={int(k):v for k,v in old_schedule['snapshot_regions'].items()}
    block=[];new_runs=[]
    def snapshot(ext,sram,size):
        assert ext%8==sram%8==0 and ext+size<=len(payload) and sram+size<=32768
        block.extend(((1,0,0,ext,sram,size),(3,2,0,0,0,0)))
    for i,command in enumerate(candidate_commands[:-1]):
        op,flags,reserved,a,b,c=command
        if op==1:
            transfer=transfers[i]
            if transfer['role'].startswith(('source_strip','prefetch_source','output_strip')):
                a+=activation_ext
            else:
                assert flags==1 and a>=36864
                a=appended+(a-36864)
            command=(op,flags,reserved,a,b,c)
        if op==2:new_runs.append((len(block),runs[i]))
        block.append(command)
        if op==3 and flags==3 and i-1 in runs:
            stage=runs[i-1];layer=8 if stage['layer']==0 else 10
            if layer in snapshots:
                d,_=decode_fused(bytes.fromhex(stage['descriptor_hex']))
                for channel in range(16 if layer==8 else 32):
                    external=snapshots[layer]['ext']+channel*576+stage['strip']*288
                    snapshot(external,d.output+channel*288,288)
    block.extend(preserved)
    combined=old[:cut_first]+block+old[cut_end:]
    code=b''.join(CMD.pack(*c) for c in combined)
    assert len(combined)<=2048 and len(code)<=32768 and len(payload)<=8*1024*1024
    delta=len(block)-(cut_end-cut_first)
    schedule=copy.deepcopy(old_schedule)
    schedule['stages']=[s for s in stages if not 7<=s['layer']<=10]
    for key in ('run_contracts','constant_contracts','pack_contracts'):
        remapped={}
        for index,value in old_schedule.get(key,{}).items():
            i=int(index)
            if cut_first<=i<cut_end:continue
            remapped[str(i if i<cut_first else i+delta)]=value
        schedule[key]=remapped
    for index,stage in new_runs:
        schedule['run_contracts'][str(cut_first+index)]=dict(
            layer=7 if stage['layer']==0 else 9,
            fused_activation_layer=8 if stage['layer']==0 else 10,
            fused_table_sha256=stage['table_sha256'],strip_origin_y=stage['strip']*12,strip_height=12)
    schedule.update(command_count=len(combined),program_bytes=len(code),
        program_sha256=sha(code),image_sha256=sha(payload),
        catalogue='exact VWW 3-6 plus 7-10 plus 11-14 full-width SRAM strip retention',
        pair7_splice=dict(source_fixture=source_folder.name,cut_first=cut_first,cut_end=cut_end,
            inserted_commands=len(block),preserved_next_layer_prefetch=len(preserved)//2,
            activation_external_base=activation_ext,source_sha256=sha(old_code),
            source_payload_sha256=sha(old_payload)))
    target_folder.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source_folder,target_folder,dirs_exist_ok=True)
    (target_folder/'commands.bin').write_bytes(code)
    (target_folder/'payload.bin').write_bytes(payload)
    (target_folder/'schedule.json').write_text(json.dumps(schedule,sort_keys=True,indent=2)+'\n')
    assert (target_folder/'input.bin').read_bytes()==host_input.tobytes()
    assert (target_folder/'output.bin').stat().st_size==schedule['final_output']['bytes']
    results=[]
    if native:
        for seed in (0,6063):
            path=target_folder.parent.parent/f'{target_folder.name}-seed{seed}.json'
            subprocess.run([str(NATIVE),str(target_folder),str(seed),str(path)],check=True)
            result=json.loads(path.read_text())
            assert result['status']=='passed' and result['tensor_checks']>=1
            results.append(result)
    return dict(label=target_folder.name,baseline_fixture=source_folder.name,
        files={p.name:sha(p.read_bytes()) for p in target_folder.iterdir() if p.is_file()},
        verification=dict(status='passed',scope='independent INT8 strip oracle, serialized command replay and full-model native tensor checks'),
        native=results,splice=schedule['pair7_splice'])


def main(output,native):
    output.mkdir(parents=True,exist_ok=True)
    compacted,block,pinned,sources=model_block()
    code,payload,candidate=build(block)
    source_report=json.loads((SOURCE/'report.json').read_text())
    assert source_report['status']=='passed-native'
    assert source_report['script_sha256']==sha((ROOT/'tools/phase6/strip_fusion_pair11.py').read_bytes())
    if native:
        prior=json.loads((NATIVE.parent/'report.json').read_text())
        assert prior['status']=='passed' and prior['executable_sha256']==sha(NATIVE.read_bytes())
    report=dict(schema=1,status='running',physical_board=False,model='vww',layers=[3,6,7,10,11,14],
        scope='pair7 current-ABI stride2 strips added to exact pair3+11 full-model schedule; board result separate',
        model_source_sha256=sources,source_sha256={
            'tools/phase6/strip_fusion_pair7.py':sha(Path(__file__).read_bytes()),
            'tools/phase6/strip_fusion_pair11.py':sha((ROOT/'tools/phase6/strip_fusion_pair11.py').read_bytes()),
            'tools/phase6/strip_fusion_vww.py':sha((ROOT/'tools/phase6/strip_fusion_vww.py').read_bytes())},
        script_sha256=sha(Path(__file__).read_bytes()),
        parent_report_sha256=sha((SOURCE/'report.json').read_bytes()),
        selected_native_executable_sha256=sha(NATIVE.read_bytes()),
        pair7=candidate,fixtures={})
    fixtures=output/'fixtures';fixtures.mkdir(parents=True,exist_ok=True)
    for sample,label in (('pinned','vww-pinned-compacted-strip3-11-timed'),
                         ('pinned','vww-pinned-compacted-strip3-11-check'),
                         ('stress','vww-stress-compacted-strip3-11-check')):
        value=pinned if sample=='pinned' else np.random.default_rng(6157).integers(-128,128,pinned.shape,dtype=np.int8)
        oracle=evaluate(compacted,{compacted.inputs[0]:value})
        source=oracle[block.inputs[0]]
        local=evaluate(block,{block.inputs[0]:source})
        assert all(np.array_equal(local[layer.output],oracle[compacted.layers[7+i].output])
                   for i,layer in enumerate(block.layers))
        for strip in (0,1):
            _,values=local_oracle(block,source,strip)
            for layer in block.layers:
                assert np.array_equal(values[layer.output],
                                      local[layer.output][:,:,strip*12:(strip+1)*12,:])
        replay_result=replay(block,source,local[block.outputs[0]],code,payload,candidate)
        old_folder=SOURCE/'fixtures'/label
        assert all(sha((old_folder/name).read_bytes())==digest
                   for name,digest in source_report['fixtures'][label]['files'].items())
        new_label=label.replace('strip3-11','strip3-7-11')
        target=fixtures/new_label
        row=splice(old_folder,target,code,payload,candidate,
                   oracle[compacted.layers[0].output],native)
        row['verification']['block_replay']=replay_result
        report['fixtures'][new_label]=row
        (output/'report.partial.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    for label in ('kws-pinned-compacted-fused-timed','kws-stress-compacted-fused-check'):
        src=SOURCE/'fixtures'/label;dst=fixtures/label
        shutil.copytree(src,dst,dirs_exist_ok=True)
        checks={p.name:sha(p.read_bytes()) for p in dst.iterdir() if p.is_file()}
        assert checks==source_report['fixtures'][label]['files']
        report['fixtures'][label]=dict(label=label,baseline_fixture=label,files=checks,
            verification=dict(status='passed',scope='byte-identical prior compacted KWS fixture'),native=[])
    report['status']='passed-native' if native else 'passed-replay'
    (output/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],pair7_commands=candidate['command_count'],
        fixtures={name:[r['elapsed_cycles'] for r in row['native']]
                  for name,row in report['fixtures'].items()}),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE/'full')
    parser.add_argument('--native',action='store_true')
    args=parser.parse_args()
    main(args.output,args.native)
