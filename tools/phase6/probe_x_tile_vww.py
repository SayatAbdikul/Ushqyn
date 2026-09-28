#!/usr/bin/env python3
"""Isolated 8x16 VWW 11–14 x-tile packing probe on the frozen 27 MHz ABI.

This is a block-level feasibility probe, not a matched full-model baseline.
Aligned 24-byte source rows are packed by chained SRAM COPY descriptors into
unaligned 17-byte rows. No RTL or frozen evidence is changed.
"""
from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'compiler'), str(ROOT/'tools/phase6')]

from hardware_v2 import Descriptor
from integer_reference import evaluate
from output_pipeline_fusion import activation_table, encode_fused
from phase4_compile import _parameter_rows
import strip_fusion_pair11 as pair11

BASE = ROOT/'work/phase6/x-tile-probe-v1'
NATIVE = ROOT/'work/phase6/pool-timing-v1/native/Vv2_tiled_host_bridge'
CMD = struct.Struct('<BBHIII')
WAIT_DMA = (3,2,0,0,0,0)
WAIT_ENGINE = (3,3,0,0,0,0)
HALT = (0,0,0,0,0,0)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def align8(value):
    return (value+7)&~7


def build(block, source):
    y0,x0,out_h,out_w = 8,8,8,16
    in_h,in_w = 10,17
    assert source.shape == (1,32,24,24)
    payload = bytearray(40960)
    commands, transfers, runs = [], [], []

    def put(data):
        pos=align8(len(payload))
        payload.extend(bytes(pos-len(payload)))
        payload.extend(data)
        return pos

    def emit(op, flags=0, a=0, b=0, c=0):
        commands.append((op,flags,0,a,b,c))

    def dma(direction, ext, sram, size, role):
        assert ext%8==sram%8==0 and 0<size<=32768 and sram+size<=32768
        assert ext+size<=len(payload)
        index=len(commands)
        emit(1,int(direction=='to_sram'),ext,sram,size)
        commands.append(WAIT_DMA)
        transfers.append(dict(command=index,direction=direction,ext=ext,sram=sram,
                              bytes=size,role=role))

    def run(pc, kind, **details):
        index=len(commands)
        emit(2,0,pc,32768<<16)
        commands.append(WAIT_ENGINE)
        runs.append(dict(command=index,pc=pc,kind=kind,**details))

    rows=[_parameter_rows(block,layer) for layer in block.layers]
    dw_weights,dw_params=rows[0]
    pw_weights,pw_params=rows[2]
    dw_table=activation_table(2,rows[1][1])
    pw_table=activation_table(2,rows[3][1])
    assert len(dw_weights)==len(dw_params)==512
    assert len(pw_weights)==1024 and len(pw_params)==512

    # Chained COPY outputs may be unaligned even though input and DMA bases
    # remain 8-byte aligned. Reverse row/channel order prevents a 7-byte
    # leading overfetch from overwriting an already packed neighboring row.
    groups=(dict(first=0,channels=24,packed=256,staging=7808,dw_output=4480),
            dict(first=24,channels=8,packed=9000,staging=11000,dw_output=7552))
    copy_table=15360
    for group in groups:
        first,n,packed,staging,dw_output=(group[key] for key in
            ('first','channels','packed','staging','dw_output'))
        source_stride=in_h*24
        packed_stride=in_h*in_w
        for channel in range(n):
            dma('to_sram',(first+channel)*576+7*24,
                staging+channel*source_stride,source_stride,'aligned_source_strip')
        order=[(channel,row) for channel in reversed(range(n))
               for row in reversed(range(in_h))]
        descriptors=[]
        for index,(channel,row) in enumerate(order):
            source_addr=staging+channel*source_stride+row*24
            desired=packed+channel*packed_stride+row*in_w
            d=Descriptor(3,input=source_addr,output=desired-7,
                         count=24,outputs=24,next_pc=copy_table+64*(index+1))
            d.validate()
            descriptors.append(d.encode())
        descriptors.append(Descriptor(0).encode())
        table_bytes=b''.join(descriptors)
        assert copy_table+len(table_bytes)<=32768
        dma('to_sram',put(table_bytes),copy_table,len(table_bytes),'copy_chain')
        run(copy_table,'pack',first_channel=first,channels=n,
            copy_descriptors=len(order),packed_sram=packed,
            packed_bytes=n*packed_stride,copy_table_sha256=sha(table_bytes))

        dw=Descriptor(6,input=packed,output=dw_output,weight=13000,
            params=13400,count=9,outputs=n*out_h*out_w,row_stride=16,
            next_pc=64,kernel_h=3,kernel_w=3,pad_right=1,
            input_h=in_h,input_w=in_w,input_c=n,output_c=n)
        dw.validate()
        desc=encode_fused(dw,13800)+Descriptor(0).encode()
        dma('to_sram',put(desc),0,128,'dw_descriptor')
        dma('to_sram',put(dw_weights[first*16:(first+n)*16]),dw.weight,n*16,'dw_weights')
        dma('to_sram',put(dw_params[first*16:(first+n)*16]),dw.params,n*16,'dw_params')
        dma('to_sram',put(dw_table),13800,256,'dw_activation_table')
        run(0,'depthwise',first_channel=first,channels=n,
            descriptor_hex=desc[:64].hex(),table_sha256=sha(dw_table))

    pw=Descriptor(4,input=4480,output=9000,weight=14000,
        params=15024,count=32,outputs=32*out_h*out_w,row_stride=32,
        next_pc=64,input_h=out_h,input_w=out_w,input_c=32,output_c=32)
    pw.validate()
    desc=encode_fused(pw,15536)+Descriptor(0).encode()
    dma('to_sram',put(desc),0,128,'pw_descriptor')
    dma('to_sram',put(pw_weights),pw.weight,len(pw_weights),'pw_weights')
    dma('to_sram',put(pw_params),pw.params,len(pw_params),'pw_params')
    dma('to_sram',put(pw_table),15536,256,'pw_activation_table')
    run(0,'pointwise',descriptor_hex=desc[:64].hex(),table_sha256=sha(pw_table))
    dma('from_sram',36864,pw.output,pw.outputs,'output_tile')
    commands.append(HALT)
    code=b''.join(CMD.pack(*item) for item in commands)
    assert len(code)<=32768 and len(commands)<=2048
    record=dict(schema=1,status='lowered',tile=[y0,x0,out_h,out_w],
        source_halo=[7,7,10,17],source_overfetch_per_row=7,
        command_count=len(commands),payload_bytes=len(payload),
        max_copy_chain_descriptors=max(row.get('copy_descriptors',0) for row in runs),
        max_sram_address_exclusive=copy_table+15424,
        commands_sha256=sha(code),payload_sha256=sha(payload),
        runs=runs,transfers=transfers)
    return code,bytes(payload),record


def oracle_tile(block, source, full):
    local=copy.deepcopy(block)
    local.tensors[local.inputs[0]]=replace(local.tensors[local.inputs[0]],
                                          shape=(1,32,10,17))
    for layer in local.layers:
        local.tensors[layer.output]=replace(local.tensors[layer.output],
                                            shape=(1,32,8,16))
    local.layers[0].attributes['pads']=[0,0,0,1]
    value=source[:,:,7:17,7:24].copy()
    result=evaluate(local,{local.inputs[0]:value})[local.outputs[0]]
    expected=full[:,:,8:16,8:24].copy()
    assert np.array_equal(result,expected)
    return expected


def write_fixture(directory,code,payload,source,expected,record):
    directory.mkdir(parents=True,exist_ok=False)
    files={'commands.bin':code,'payload.bin':payload,
           'input.bin':source.tobytes(),'output.bin':expected.tobytes(),
           'checks.txt':b'36864 output.bin\n',
           'schedule.json':(json.dumps(record,sort_keys=True,indent=2)+'\n').encode()}
    for name,data in files.items():
        (directory/name).write_bytes(data)
    return {name:sha(data) for name,data in files.items()}


def main():
    assert not BASE.exists(), 'choose a fresh output directory'
    compacted,block,pinned,model_sources=pair11.model_block()
    full=evaluate(compacted,{compacted.inputs[0]:pinned})
    source=full[block.inputs[0]]
    expected=oracle_tile(block,source,full[block.outputs[0]])
    code,payload,record=build(block,source)
    files=write_fixture(BASE/'fixture',code,payload,source,expected,record)
    native=[]
    for seed in (0,6063):
        path=BASE/f'native-s{seed}.json'
        subprocess.run([str(NATIVE),str(BASE/'fixture'),str(seed),str(path)],
                       cwd=ROOT,check=True)
        result=json.loads(path.read_text())
        native.append(result)
    report=dict(schema=1,status='passed-native' if all(x['status']=='passed' for x in native) else 'failed',
        physical_board=False,scope='isolated VWW 11–14 8x16 interior tile; not a full-model schedule',
        model_source_sha256=model_sources,source_sha256=sha(Path(__file__).read_bytes()),
        native_executable_sha256=sha(NATIVE.read_bytes()),
        fixture_files_sha256=files,record=record,native=native)
    (BASE/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],commands=record['command_count'],
        copy_chain_descriptors=record['max_copy_chain_descriptors'],
        cycles=[x['elapsed_cycles'] for x in native]),sort_keys=True))


if __name__=='__main__':
    main()
