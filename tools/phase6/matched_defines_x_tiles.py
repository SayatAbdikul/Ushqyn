#!/usr/bin/env python3
"""Isolated executable VWW 11–14 8-row, 16+8-column spatial cut.

The unchanged 27 MHz image executes 8x16 and 8x8 tiles. Chained COPY runs
pack aligned 24-byte source rows into exact narrow halos and scatter output
tiles into an 8-row SRAM band. The band is stored contiguously to temporary
SDRAM; after all bands, one bulk transfer restores the original layer-14 slot.
No frozen baseline or RTL source is modified.
"""
from dataclasses import replace
import copy
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

from hardware_v2 import Descriptor
from integer_reference import evaluate
from output_pipeline_fusion import activation_table,decode_fused,encode_fused
from phase4_compile import _parameter_rows
from scheduler.defines_verify import replay_defines
import strip_fusion_pair11 as pair11

BASE=ROOT/'work/phase6/matched-baselines-v1/b3-x-tiles-v2'
SOURCE=ROOT/'work/phase6/strip-fusion-pair7-v1/full/fixtures/vww-pinned-compacted-strip3-7-11-timed'
NATIVE=ROOT/'work/phase6/pool-timing-v1/native/Vv2_tiled_host_bridge'
CMD=struct.Struct('<BBHIII')
WAIT_DMA=(3,2,0,0,0,0)
WAIT_ENGINE=(3,3,0,0,0,0)
HALT=(0,0,0,0,0,0)


def sha(data):return hashlib.sha256(data).hexdigest()
def file_sha(path):return sha(path.read_bytes())
def align8(value):return (value+7)&~7


def local_tile(block, source, oracle, y0, x0, width):
    ybeg=max(0,y0-1);yend=min(24,y0+9)
    xbeg=0 if x0==0 else 15
    xend=17 if x0==0 else 24
    input_value=source[:,:,ybeg:yend,xbeg:xend].copy()
    local=copy.deepcopy(block)
    local.tensors[local.inputs[0]]=replace(local.tensors[local.inputs[0]],
        shape=(1,32,yend-ybeg,xend-xbeg))
    for layer in local.layers:
        local.tensors[layer.output]=replace(local.tensors[layer.output],
            shape=(1,32,8,width))
    local.layers[0].attributes['pads']=[int(y0==0),int(x0==0),
                                        int(y0==16),int(x0==16)]
    values=evaluate(local,{local.inputs[0]:input_value})
    assert np.array_equal(values[local.outputs[0]],oracle[:,:,y0:y0+8,x0:x0+width])
    return input_value,values


def build(block, old_payload):
    payload=bytearray(old_payload)
    temp=align8(len(payload));payload.extend(bytes(temp-len(payload)))
    payload.extend(bytes(18432))
    commands,transfers,runs=[] ,[],[]
    dw_weight,dw_params=_parameter_rows(block,block.layers[0])
    pw_weight,pw_params=_parameter_rows(block,block.layers[2])
    dw_table=activation_table(2,_parameter_rows(block,block.layers[1])[1])
    pw_table=activation_table(2,_parameter_rows(block,block.layers[3])[1])
    assert len(dw_weight)==len(dw_params)==512
    assert len(pw_weight)==1024 and len(pw_params)==512

    def put(data):
        pos=align8(len(payload));payload.extend(bytes(pos-len(payload)));payload.extend(data)
        return pos
    def emit(op,flags=0,a=0,b=0,c=0):commands.append((op,flags,0,a,b,c))
    def dma(direction,ext,sram,size,role,tile=None):
        assert ext%8==sram%8==0 and 0<size<=32768
        assert ext+size<=len(payload) and sram+size<=32768
        i=len(commands);emit(1,int(direction=='to_sram'),ext,sram,size)
        commands.append(WAIT_DMA)
        transfers.append(dict(command=i,direction=direction,ext=ext,sram=sram,
                              bytes=size,role=role,tile=tile))
    def run(pc,kind,tile,**details):
        i=len(commands);emit(2,0,pc,32768<<16);commands.append(WAIT_ENGINE)
        runs.append(dict(command=i,pc=pc,kind=kind,tile=tile,**details))

    for y0 in (0,8,16):
        for x0,width in ((0,16),(16,8)):
            tile=[y0,x0,8,width]
            iy0=max(0,y0-1);iy1=min(24,y0+9);ih=iy1-iy0
            iw=17 if x0==0 else 9
            offset=0 if x0==0 else 15
            plane=8*width
            for first,n in ((0,24),(24,8)):
                packed=6288;staging=10384;dw_output=16160+first*plane
                table=17152 if first==0 else 20480
                input_plane=ih*iw;source_plane=ih*24
                for ch in range(n):
                    dma('to_sram',36864+(first+ch)*576+iy0*24,
                        staging+ch*source_plane,source_plane,'aligned_source',tile)
                order=[(ch,row) for ch in range(n) for row in range(ih)]
                if offset:order.reverse()
                descriptors=[]
                for index,(ch,row) in enumerate(order):
                    desired=packed+ch*input_plane+row*iw
                    d=Descriptor(3,input=staging+ch*source_plane+row*24,
                        output=desired-offset,count=24,outputs=24,
                        next_pc=table+64*(index+1))
                    d.validate();descriptors.append(d.encode())
                descriptors.append(Descriptor(0).encode())
                table_bytes=b''.join(descriptors)
                assert table+len(table_bytes)<=32768
                dma('to_sram',put(table_bytes),table,len(table_bytes),'pack_chain',tile)
                run(table,'pack',tile,first_channel=first,channels=n,
                    copy_descriptors=len(order),input_h=ih,input_w=iw,
                    source_x_offset=offset,packed_sram=packed,
                    table_sha256=sha(table_bytes))
                dw=Descriptor(6,input=packed,output=dw_output,weight=21000,
                    params=21400,count=9,outputs=n*plane,row_stride=16,
                    next_pc=64,kernel_h=3,kernel_w=3,
                    pad_top=int(y0==0),pad_bottom=int(y0==16),
                    pad_left=int(x0==0),pad_right=int(x0==16),
                    input_h=ih,input_w=iw,input_c=n,output_c=n)
                dw.validate()
                raw=encode_fused(dw,21800)+Descriptor(0).encode()
                dma('to_sram',put(raw),0,128,'dw_descriptor',tile)
                dma('to_sram',put(dw_weight[first*16:(first+n)*16]),dw.weight,n*16,'dw_weight',tile)
                dma('to_sram',put(dw_params[first*16:(first+n)*16]),dw.params,n*16,'dw_params',tile)
                dma('to_sram',put(dw_table),21800,256,'dw_table',tile)
                run(0,'depthwise',tile,first_channel=first,channels=n,
                    descriptor_hex=raw[:64].hex(),table_sha256=sha(dw_table))

            pw=Descriptor(4,input=16160,output=6288,weight=21000,
                params=22024,count=32,outputs=32*plane,row_stride=32,
                next_pc=64,input_h=8,input_w=width,input_c=32,output_c=32)
            pw.validate()
            raw=encode_fused(pw,22536)+Descriptor(0).encode()
            dma('to_sram',put(raw),0,128,'pw_descriptor',tile)
            dma('to_sram',put(pw_weight),pw.weight,len(pw_weight),'pw_weight',tile)
            dma('to_sram',put(pw_params),pw.params,len(pw_params),'pw_params',tile)
            dma('to_sram',put(pw_table),22536,256,'pw_table',tile)
            run(0,'pointwise',tile,descriptor_hex=raw[:64].hex(),
                table_sha256=sha(pw_table))

            table=10432
            descriptors=[]
            for ch in range(32):
                for row in range(8):
                    d=Descriptor(3,input=6288+ch*plane+row*width,
                        output=128+ch*192+row*24+x0,count=width,outputs=width,
                        next_pc=table+64*(len(descriptors)+1))
                    d.validate();descriptors.append(d.encode())
            descriptors.append(Descriptor(0).encode())
            scatter=b''.join(descriptors)
            assert len(scatter)==16448 and table+len(scatter)<=32768
            dma('to_sram',put(scatter),table,len(scatter),'scatter_chain',tile)
            run(table,'scatter',tile,copy_descriptors=256,table_sha256=sha(scatter))
        for ch in range(32):
            dma('from_sram',temp+ch*576+y0*24,128+ch*192,192,'output_band',[y0,0,8,24])
    dma('to_sram',temp,128,18432,'final_output_reload')
    dma('from_sram',36864,128,18432,'final_output_store')
    code=b''.join(CMD.pack(*command) for command in commands)
    assert len(commands)<2048 and len(payload)<8*1024*1024
    report=dict(schema=1,status='lowered',policy='three 8-row bands; 16+8 x tiles',
        output_temp_ext=temp,commands=len(commands),payload_bytes=len(payload),
        max_copy_chain_descriptors=max(x.get('copy_descriptors',0) for x in runs),
        runs=runs,transfers=transfers,code_sha256=sha(code),payload_sha256=sha(payload))
    return code,bytes(payload),report


def replay_segment(block,source,expected,code,payload,record):
    """Byte replay of every segment DMA/COPY plus independent INT8 tile oracle."""
    assert sha(code)==record['code_sha256'] and sha(payload)==record['payload_sha256']
    ext=bytearray(payload);ext[36864:36864+18432]=source.tobytes()
    original_prefix=bytes(ext[:36864])
    sram=bytearray(32768)
    transfers={x['command']:x for x in record['transfers']}
    runs={x['command']:x for x in record['runs']}
    local={}
    for y in (0,8,16):
        for x,w in ((0,16),(16,8)):
            local[(y,x)]=local_tile(block,source,expected,y,x,w)
    commands=[CMD.unpack_from(code,i) for i in range(0,len(code),16)]
    assert len(commands)==record['commands']
    counts={'pack':0,'depthwise':0,'pointwise':0,'scatter':0}
    for i,(op,flags,reserved,a,b,c) in enumerate(commands):
        assert reserved==0
        if op==3:
            assert (flags,a,b,c) in ((2,0,0,0),(3,0,0,0));continue
        if op==1:
            t=transfers[i]
            assert (flags,a,b,c)==(int(t['direction']=='to_sram'),t['ext'],t['sram'],t['bytes'])
            if flags:sram[b:b+c]=ext[a:a+c]
            else:ext[a:a+c]=sram[b:b+c]
            continue
        assert op==2 and (flags,b,c)==(0,32768<<16,0)
        r=runs[i];tile=r['tile'];y,x,_,w=tile
        if r['kind'] in ('pack','scatter'):
            pc=a;n=0
            while True:
                d=Descriptor.decode(bytes(sram[pc:pc+64]));d.validate()
                if d.opcode==0:break
                assert d.opcode==3 and d.next_pc==pc+64
                operand=bytes(sram[d.input:d.input+d.count])
                sram[d.output:d.output+d.outputs]=operand
                pc=d.next_pc;n+=1
            assert n==r['copy_descriptors']
            counts[r['kind']]+=1
            if r['kind']=='pack':
                operand=local[(y,x)][0][:,r['first_channel']:
                    r['first_channel']+r['channels']].tobytes()
                assert bytes(sram[r['packed_sram']:r['packed_sram']+len(operand)])==operand
            continue
        d,table=decode_fused(bytes(sram[a:a+64]));d.validate()
        assert bytes(sram[a:a+64]).hex()==r['descriptor_hex']
        assert Descriptor.decode(bytes(sram[a+64:a+128])).opcode==0
        assert sha(bytes(sram[table:table+256]))==r['table_sha256']
        _,values=local[(y,x)]
        if r['kind']=='depthwise':
            first,n=r['first_channel'],r['channels']
            wanted=values[block.layers[1].output][:,first:first+n].tobytes()
            counts['depthwise']+=1
        else:
            wanted=values[block.outputs[0]].tobytes()
            assert bytes(sram[d.input:d.input+d.input_c*8*w])==values[block.layers[1].output].tobytes()
            counts['pointwise']+=1
        assert len(wanted)==d.outputs
        sram[d.output:d.output+d.outputs]=wanted
    assert counts=={'pack':12,'depthwise':12,'pointwise':6,'scatter':6}
    assert bytes(ext[36864:36864+18432])==expected.tobytes()
    assert bytes(ext[:36864])==original_prefix
    return dict(status='passed',counts=counts,output_sha256=sha(expected.tobytes()),
                prefix_preserved=True,external_temp_bytes=18432)


def splice(source,code,payload,record,output):
    old=(source/'commands.bin').read_bytes()
    original=json.loads((source/'schedule.json').read_text())
    commands=[CMD.unpack_from(old,i) for i in range(0,len(old),16)]
    first,end=418,714
    assert commands[first]==(1,1,0,36864,128,312)
    assert commands[end]==(1,1,0,161472,24192,256)
    assert all(not first<=int(k)<end or v['layer'] in (11,13)
               for k,v in original['run_contracts'].items())
    changed=old[:first*16]+code+old[end*16:]
    count=len(changed)//16
    assert count<=2048 and len(payload)<=8*1024*1024
    delta=record['commands']-(end-first)
    schedule=copy.deepcopy(original)
    schedule['run_contracts']={str(j if j<first else j+delta):v
        for key,v in original['run_contracts'].items()
        if not first<=(j:=int(key))<end}
    for key in ('constant_contracts','pack_contracts'):
        schedule[key]={str(j if j<first else j+delta):v
            for k,v in original.get(key,{}).items()
            if not first<=(j:=int(k))<end}
    schedule.update(command_count=count,program_bytes=len(changed),
        program_sha256=sha(changed),image_sha256=sha(payload),
        catalogue='isolated existing-ABI 8x16+8x8 VWW spatial cut',
        defines_x_tile=dict(cut_first=first,cut_end=end,inserted=record['commands'],
            old_commands_sha256=sha(old),segment_sha256=sha(code),
            output_temp_ext=record['output_temp_ext']))
    output.mkdir(parents=True,exist_ok=False)
    for name in ('input.bin','output.bin','checks.txt'):
        shutil.copy2(source/name,output/name)
    (output/'commands.bin').write_bytes(changed)
    (output/'payload.bin').write_bytes(payload)
    (output/'schedule.json').write_text(json.dumps(schedule,sort_keys=True,indent=2)+'\n')
    return dict(command_count=count,program_sha256=sha(changed),
        payload_sha256=sha(payload),source_commands_sha256=sha(old),
        run_contracts_shifted=delta)


def main():
    assert not BASE.exists(), 'choose a fresh output directory'
    compacted,block,pinned,model_sources=pair11.model_block()
    oracle=evaluate(compacted,{compacted.inputs[0]:pinned})
    source=oracle[block.inputs[0]]
    expected=oracle[block.outputs[0]]
    old_payload=(SOURCE/'payload.bin').read_bytes()
    segment,payload,record=build(block,old_payload)
    replay=replay_segment(block,source,expected,segment,payload,record)
    BASE.mkdir(parents=True)
    fixture=BASE/'fixture'
    spliced=splice(SOURCE,segment,payload,record,fixture)
    source_schedule=json.loads((SOURCE/'schedule.json').read_text())
    full_replay=replay_defines(compacted,(SOURCE/'commands.bin').read_bytes(),old_payload,
        {compacted.inputs[0]:pinned},run_contracts=source_schedule['run_contracts'],
        constant_contracts=source_schedule.get('constant_contracts'),
        pack_contracts=source_schedule.get('pack_contracts'),
        final_output=source_schedule['final_output'],oracle=oracle)
    assert full_replay['status']=='passed'
    native=[]
    for seed in (0,6063):
        path=BASE/f'native-s{seed}.json'
        subprocess.run([str(NATIVE),str(fixture),str(seed),str(path)],cwd=ROOT,check=True)
        raw=json.loads(path.read_text());native.append(raw)
    report=dict(schema=1,status='passed-native' if all(x['status']=='passed' for x in native) else 'failed',
        physical_board=False,scope='full-model VWW 8x16+8x8 existing-ABI spatial cut; compositional replay',
        model_source_sha256=model_sources,source_sha256=sha(Path(__file__).read_bytes()),
        native_executable_sha256=file_sha(NATIVE),frozen_source_fixture=str(SOURCE.relative_to(ROOT)),
        frozen_source_files_sha256={name:file_sha(SOURCE/name) for name in
            ('commands.bin','payload.bin','input.bin','output.bin','schedule.json')},
        segment=record,segment_replay=replay,source_full_model_replay=full_replay,
        splice=spliced,fixture_files_sha256={p.name:file_sha(p) for p in fixture.iterdir()},
        native=native)
    (BASE/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],commands=spliced['command_count'],
        segment_commands=record['commands'],cycles=[x['elapsed_cycles'] for x in native]),
        sort_keys=True))


if __name__=='__main__':main()
