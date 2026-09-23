"""Phase-2 hardware ABI and lowering. Software VM containers are never flashed.

One manifest drives compiler, RTL package, simulator and Gowin. The hardware
subset is deliberately FC/ReLU/copy; CNN descriptors reject until phase 3/4.
"""
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
    if target['opcodes']!={'HALT':0,'GEMM':1,'RELU':2,'COPY':3}:raise ValueError('unsupported opcode ABI')
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
        # Reserved convolution geometry is represented independently but not
        # silently treated as FC; the implementation subset is explicit.
        if (self.kernel_h,self.kernel_w,self.stride_h,self.stride_w,self.pad_top,self.pad_bottom,self.pad_left,self.pad_right,self.input_h,self.input_w,self.input_c,self.output_c)!=(1,1,1,1,0,0,0,0,1,1,1,1):
            raise ValueError('convolution geometry not implemented by phase-2 target')
        region(self.input,(self.count+7)//8*8,8)
        region(self.output,self.outputs)
        if self.opcode==1:
            if self.row_stride%8 or self.row_stride<self.count: raise ValueError('invalid FC row stride')
            region(self.weight,self.outputs*self.row_stride,8)
            region(self.params,self.outputs*16,8)
        else:
            if self.outputs!=self.count: raise ValueError('elementwise count mismatch')
            if self.opcode==2: region(self.params,16,8)


def lower(program):
    """Pack a fully verified software Program into nonoverlapping board SRAM."""
    validate_program(program)
    supported={'Gemm','Relu','Flatten','Reshape','Identity'}
    if any(l.op not in supported for l in program.layers): raise ValueError('unsupported hardware graph')
    if program.constants: raise ValueError('runtime constants unsupported in phase-2 hardware')
    n=len(program.layers)
    if n>=TARGET['max_descriptors']: raise ValueError('descriptor budget')
    memory=bytearray(TARGET['memory_bytes']); offset=(n+1)*64; segments=[]
    def alloc(name,data,size=None):
        nonlocal offset
        offset=(offset+7)&~7; size=len(data) if size is None else size
        padded=(size+7)&~7
        if offset+padded>len(memory): raise ValueError('model exceeds target SRAM')
        start=offset;memory[start:start+len(data)]=data;offset+=padded
        segments.append(dict(name=name,offset=start,size=padded))
        return start
    addresses={name:alloc(name,b'',math.prod(t.shape)) for name,t in program.tensors.items()}
    descriptors=[]
    for i,l in enumerate(program.layers):
        count=math.prod(program.tensors[l.inputs[0]].shape); outputs=math.prod(program.tensors[l.output].shape)
        d=Descriptor(TARGET['opcodes'][{'Gemm':'GEMM','Relu':'RELU'}.get(l.op,'COPY')],addresses[l.inputs[0]],addresses[l.output],count=count,outputs=outputs,next_pc=(i+1)*64)
        iq=program.tensors[l.inputs[0]].quantization;oq=program.tensors[l.output].quantization
        if l.op=='Gemm':
            w=l.parameters['weight'];stride=(count+7)&~7
            if w.shape!=(outputs,count): raise ValueError('FC shape')
            packed=np.zeros((outputs,stride),np.int8);packed[:,:count]=w
            d.weight=alloc(f'{i}/weights',packed.tobytes());d.row_stride=stride
            params=b''.join(struct.pack('<iiBbbbbb',int(l.parameters['corrected_bias'][c]),int(l.parameters['multiplier'][c]),int(l.parameters['shift'][c]),oq.zero_point,iq.zero_point,-128,127,0)+b'\0\0' for c in range(outputs))
            d.params=alloc(f'{i}/params',params)
        elif l.op=='Relu':
            m,s=multiplier_shift(iq.scale/oq.scale)
            params=struct.pack('<iiBbbbbb',0,m,s,oq.zero_point,iq.zero_point,iq.zero_point,127,0)+b'\0\0'
            d.params=alloc(f'{i}/params',params)
        elif iq!=oq: raise ValueError('copy quantization changed')
        d.validate();descriptors.append(d)
        memory[i*64:(i+1)*64]=d.encode()
    memory[n*64:(n+1)*64]=Descriptor(0).encode()
    return bytes(memory),dict(target=TARGET['name'],target_manifest_sha256=hashlib.sha256(TARGET_PATH.read_bytes()).hexdigest(),numerics=2,entry=0,used_bytes=offset,segments=segments,
                             inputs={n:addresses[n] for n in program.inputs},outputs={n:addresses[n] for n in program.outputs},tensors=addresses,descriptors=[asdict(d) for d in descriptors])
