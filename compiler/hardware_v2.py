"""Board descriptor ABI and lowering. Software VM containers are never flashed."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import operator
import math
import struct
import numpy as np
from program_image import validate_program
from quantization import multiplier_shift
from memory_planner import allocate, regions_for_program, report as memory_report

ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = ROOT/'hardware/targets/tang_nano_20k_v2.json'
TARGET = json.loads(TARGET_PATH.read_text())
MAGIC = 0x32445355


def validate_target(target):
    if (target['numerics'],target['descriptor_version'],target['protocol_version'],target['address_bits'],target['lanes'],target['descriptor_bytes']) != (2,2,2,24,8,64):
        raise ValueError('incompatible target widths/numerical ABI')
    size=target['memory_bytes']
    if not 1024<=size<=32768 or size&(size-1):raise ValueError('unsupported physical SRAM geometry')
    if target['max_transfer']!=64 or not 10<=target['timeout_cycles']<2**32:raise ValueError('protocol geometry')
    if target['opcodes']!={'HALT':0,'GEMM':1,'RELU':2,'COPY':3,'CONV':4,'MAXPOOL':5}:raise ValueError('unsupported opcode ABI')
    if len(target['sources'])!=len(set(target['sources'])):raise ValueError('duplicate active RTL sources')


validate_target(TARGET)


def uint(value, bits):
    value = operator.index(value)
    if not 0 <= value < 1 << bits: raise ValueError(f'unsigned {bits}-bit field out of range')
    return value


@dataclass
class Descriptor:
    opcode: int
    input: int = 0
    output: int = 0
    weight: int = 0
    params: int = 0
    count: int = 0
    outputs: int = 0
    row_stride: int = 0
    next_pc: int = 0
    kernel_h: int = 1
    kernel_w: int = 1
    stride_h: int = 1
    stride_w: int = 1
    pad_top: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_right: int = 0
    input_h: int = 1
    input_w: int = 1
    input_c: int = 1
    output_c: int = 1

    def encode(self):
        if self.opcode not in TARGET['opcodes'].values(): raise ValueError('unsupported opcode')
        for n in ('input','output','weight','params','next_pc'): uint(getattr(self,n),24)
        for n in ('count','outputs','row_stride'): uint(getattr(self,n),32)
        geometry = [getattr(self,n) for n in list(asdict(self))[9:]]
        for v in geometry: uint(v,16)
        return struct.pack('<IBBBB8I12H', MAGIC,2,self.opcode,0,0,
                           self.input,self.output,self.weight,self.params,
                           self.count,self.outputs,self.row_stride,self.next_pc,*geometry)

    @classmethod
    def decode(cls,data):
        if len(data)!=64: raise ValueError('descriptor length')
        fields=struct.unpack('<IBBBB8I12H',data)
        if fields[:2]!=(MAGIC,2) or fields[3:5]!=(0,0): raise ValueError('descriptor magic/version/flags')
        d=cls(fields[2],*fields[5:]);d.encode();return d

    def validate(self,capacity=TARGET['memory_bytes']):
        self.encode()
        def region(base,size,align=1):
            if base%align or size<0 or base+size>capacity: raise ValueError('unaligned/out-of-range memory region')
        if self.opcode==0: return
        region(self.next_pc,64,64)
        if min(self.count,self.outputs)==0: raise ValueError('empty geometry')
        spatial=self.opcode in (4,5)
        if spatial:
            if (min(self.kernel_h,self.kernel_w,self.input_h,self.input_w,self.input_c,self.output_c)<=0
                    or self.stride_h not in (1,2) or self.stride_w not in (1,2)
                    or self.pad_top>=self.kernel_h or self.pad_bottom>=self.kernel_h
                    or self.pad_left>=self.kernel_w or self.pad_right>=self.kernel_w
                    or self.kernel_h>7 or self.kernel_w>7
                    or max(self.input_h,self.input_w,self.input_c,self.output_c)>255):
                raise ValueError('unsupported convolution/pooling geometry')
            numer_h=self.input_h+self.pad_top+self.pad_bottom-self.kernel_h
            numer_w=self.input_w+self.pad_left+self.pad_right-self.kernel_w
            if numer_h<0 or numer_w<0:raise ValueError('empty spatial output')
            oh=numer_h//self.stride_h+1;ow=numer_w//self.stride_w+1
            if self.outputs!=self.output_c*oh*ow:raise ValueError('spatial output count mismatch')
            if self.opcode==4 and self.count!=self.input_c*self.kernel_h*self.kernel_w:
                raise ValueError('convolution reduction mismatch')
            if self.opcode==5 and (self.count!=self.input_c*self.input_h*self.input_w or self.output_c!=self.input_c):
                raise ValueError('pool input/channel mismatch')
            region(self.input,self.input_c*self.input_h*self.input_w,8)
        else:
            if (self.kernel_h,self.kernel_w,self.stride_h,self.stride_w,self.pad_top,self.pad_bottom,self.pad_left,self.pad_right,self.input_h,self.input_w,self.input_c,self.output_c)!=(1,1,1,1,0,0,0,0,1,1,1,1):
                raise ValueError('unexpected spatial geometry')
            region(self.input,(self.count+7)//8*8,8)
        region(self.output,self.outputs)
        if self.opcode in (1,4):
            if self.row_stride%8 or self.row_stride<self.count: raise ValueError('invalid FC row stride')
            weight_rows=self.output_c if spatial else self.outputs
            region(self.weight,weight_rows*self.row_stride,8)
            region(self.params,weight_rows*16,8)
        else:
            if not spatial and self.outputs!=self.count: raise ValueError('elementwise count mismatch')
            if self.opcode in (2,5): region(self.params,16,8)


def lower(program):
    """Pack a verified FC/ordinary-Conv/MaxPool program with live SRAM reuse."""
    validate_program(program)
    supported={'Gemm','Conv','MaxPool','Relu','Flatten','Reshape','Identity'}
    if any(l.op not in supported for l in program.layers): raise ValueError('unsupported hardware graph')
    if program.constants: raise ValueError('runtime constants unsupported in board hardware')
    n=len(program.layers)
    if n>=TARGET['max_descriptors']: raise ValueError('descriptor budget')
    payloads=[]
    for l in program.layers:
        packed={}
        if l.op in ('Gemm','Conv'):
            w=l.parameters['weight'];reduction=math.prod(w.shape[1:]);stride=(reduction+7)&~7
            rows=np.zeros((w.shape[0],stride),np.int8);rows[:,:reduction]=w.reshape(w.shape[0],reduction)
            packed['weights']=rows.tobytes()
            iq=program.tensors[l.inputs[0]].quantization;oq=program.tensors[l.output].quantization
            packed['params']=b''.join(struct.pack('<iiBbbbbb',int(l.parameters['corrected_bias'][c]),int(l.parameters['multiplier'][c]),int(l.parameters['shift'][c]),oq.zero_point,iq.zero_point,-128,127,0)+b'\0\0' for c in range(w.shape[0]))
        elif l.op in ('Relu','MaxPool'):
            iq=program.tensors[l.inputs[0]].quantization;oq=program.tensors[l.output].quantization
            m,s=multiplier_shift(iq.scale/oq.scale)
            packed['params']=struct.pack('<iiBbbbbb',0,m,s,oq.zero_point,iq.zero_point,iq.zero_point,127,0)+b'\0\0'
        else:
            if program.tensors[l.inputs[0]].quantization!=program.tensors[l.output].quantization:
                raise ValueError('copy quantization changed')
        payloads.append(packed)
    regions=regions_for_program(program,payloads)
    segments=allocate(regions,TARGET['memory_bytes'],(n+1)*64)
    addresses={s['name']:s['offset'] for s in segments}
    memory=bytearray(TARGET['memory_bytes'])
    for i,p in enumerate(payloads):
        for name,data in p.items():memory[addresses[f'{i}/{name}']:addresses[f'{i}/{name}']+len(data)]=data
    descriptors=[]
    for i,l in enumerate(program.layers):
        count=math.prod(program.tensors[l.inputs[0]].shape); outputs=math.prod(program.tensors[l.output].shape)
        opname={'Gemm':'GEMM','Conv':'CONV','MaxPool':'MAXPOOL','Relu':'RELU'}.get(l.op,'COPY')
        d=Descriptor(TARGET['opcodes'][opname],addresses[l.inputs[0]],addresses[l.output],count=count,outputs=outputs,next_pc=(i+1)*64)
        iq=program.tensors[l.inputs[0]].quantization;oq=program.tensors[l.output].quantization
        if l.op in ('Conv','MaxPool'):
            xshape=program.tensors[l.inputs[0]].shape;yshape=program.tensors[l.output].shape
            if (len(xshape)!=4 or len(yshape)!=4 or xshape[0]!=1 or yshape[0]!=1
                    or program.tensors[l.inputs[0]].layout!='NCHW'
                    or program.tensors[l.output].layout!='NCHW'):
                raise ValueError('board spatial tensors require batch-one NCHW')
            a=l.attributes;kh,kw=(l.parameters['weight'].shape[2:] if l.op=='Conv' else a['kernel_shape'])
            pt,pl,pb,pr=a.get('pads',[0,0,0,0]);sh,sw=a.get('strides',[1,1])
            if a.get('group',1)!=1 or a.get('dilations',[1,1])!=[1,1] or a.get('ceil_mode',0):
                raise ValueError('unsupported convolution/pooling grouping or dilation')
            d.kernel_h=kh;d.kernel_w=kw;d.stride_h=sh;d.stride_w=sw
            d.pad_top=pt;d.pad_bottom=pb;d.pad_left=pl;d.pad_right=pr
            d.input_h=xshape[2];d.input_w=xshape[3];d.input_c=xshape[1];d.output_c=yshape[1]
            if l.op=='Conv':d.count=math.prod(l.parameters['weight'].shape[1:])
        if l.op in ('Gemm','Conv'):
            w=l.parameters['weight'];reduction=math.prod(w.shape[1:]);stride=(reduction+7)&~7
            if l.op=='Gemm' and w.shape!=(outputs,count): raise ValueError('FC shape')
            if l.op=='Conv' and (w.shape[0]!=d.output_c or w.shape[1]!=d.input_c):
                raise ValueError('Conv channels')
            d.weight=addresses[f'{i}/weights'];d.row_stride=stride;d.params=addresses[f'{i}/params']
        elif l.op in ('Relu','MaxPool'):d.params=addresses[f'{i}/params']
        d.validate();descriptors.append(d)
        memory[i*64:(i+1)*64]=d.encode()
    memory[n*64:(n+1)*64]=Descriptor(0).encode()
    mem=memory_report(segments,TARGET['memory_bytes'],(n+1)*64)
    return bytes(memory),dict(target=TARGET['name'],target_manifest_sha256=hashlib.sha256(TARGET_PATH.read_bytes()).hexdigest(),numerics=2,entry=0,used_bytes=mem['allocated_high_watermark'],memory=mem,segments=segments,
                             inputs={n:addresses[n] for n in program.inputs},outputs={n:addresses[n] for n in program.outputs},tensors={n:addresses[n] for n in program.tensors},layer_outputs=[l.output for l in program.layers],descriptors=[asdict(d) for d in descriptors])
