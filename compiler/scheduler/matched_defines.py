"""DeFiNES-style rectangular depth-first backend for the fixed descriptor core.

All quantization boundaries remain explicit exact LUT maps. Published cache
modes are applied to quantized producer tensors; aligned DMA and COPY implement
packing without pretending an ideal multiported SRAM. SRAM allocation includes
cache buffers, all active recursive requests, descriptor bytes and temporaries.
"""
from dataclasses import dataclass
import copy
import hashlib
import math
import struct

import numpy as np

from hardware_v2 import Descriptor
from integer_reference import evaluate
from phase4_compile import _parameter_rows
from quantization import requantize
from .spatial import Rect,input_halo
from .matched_defines_strip import block_program,macro_stacks,align8,sha

CMD=struct.Struct('<BBHIII')


@dataclass
class Buffer:
    base:int
    size:int
    tag:str


class Arena:
    def __init__(self):
        self.buffers=[];self.peak=128;self.peak_bytes=128
    def alloc(self,size,tag):
        if size<=0:raise ValueError('empty-buffer')
        total=8+align8(size);pos=128
        for b in sorted(self.buffers,key=lambda b:b.base):
            if pos+total<=b.base-8:break
            pos=max(pos,b.base+align8(b.size))
        if pos+total>32768:raise ValueError('sram-capacity')
        b=Buffer(pos+8,size,tag);self.buffers.append(b)
        self.peak=max(self.peak,b.base+align8(size))
        self.peak_bytes=max(self.peak_bytes,128+sum(8+align8(b.size) for b in self.buffers))
        return b
    def free(self,b):self.buffers.remove(b)
    def live(self):
        return [(b.base-8,b.base+align8(b.size),b.tag) for b in self.buffers]


@dataclass
class Cache:
    rect:Rect
    buffer:Buffer
    valid:list


def shapes_and_needs(block,tile_h,tile_w):
    macros=list(range(0,len(block.layers),2));last=block.tensors[block.outputs[0]].shape
    all_rows=[]
    for y in range(0,last[2],tile_h):
        row=[]
        for x in range(0,last[3],tile_w):
            wanted=Rect(y,x,min(y+tile_h,last[2]),min(x+tile_w,last[3]))
            needs={len(macros)-1:wanted}
            for j in range(len(macros)-1,-1,-1):
                conv=block.layers[macros[j]];ish=block.tensors[conv.inputs[0]].shape
                a=conv.attributes
                needed=input_halo(needs[j],conv.parameters['weight'].shape[2:],
                    tuple(a.get('strides',[1,1])),tuple(a.get('pads',[0]*4)))
                needs[j-1]=needed.intersect(Rect(0,0,ish[2],ish[3]))
            row.append(needs)
        all_rows.append(row)
    return all_rows


def bbox(rects):
    rects=[r for r in rects if r.area]
    if not rects:return Rect(0,0,0,0)
    return Rect(min(r.y0 for r in rects),min(r.x0 for r in rects),max(r.y1 for r in rects),max(r.x1 for r in rects))


def compile_stack(program,start,stop,tile_h,tile_w,mode=1,*,retain_weights=True,prefetch=True):
    from output_pipeline_fusion import activation_table,encode_fused
    from .matched_defines_regions import emit_read_rectangle,emit_write_rectangle,emit_copy_rectangle,rectangle_temp_bytes
    if (start,stop) not in set(macro_stacks(program)):raise ValueError('unsupported-stack')
    if mode not in (1,2,3):raise ValueError('invalid-mode')
    block=block_program(program,start,stop);macros=list(range(0,len(block.layers),2))
    ishape=block.tensors[block.inputs[0]].shape;oshape=block.tensors[block.outputs[0]].shape
    if not (1<=tile_h<=oshape[2] and 1<=tile_w<=oshape[3]):raise ValueError('invalid-tile')
    for k in macros:
        conv=block.layers[k];a=conv.attributes
        if a.get('dilations',[1,1])!=[1,1]:raise ValueError('dilation-backend-gap')
        if a.get('group',1) not in (1,block.tensors[conv.inputs[0]].shape[1]):raise ValueError('group-backend-gap')
    source_size=math.prod(ishape);output_size=math.prod(oshape);output_ext=align8(source_size)
    payload=bytearray(output_ext+align8(output_size));commands=[];runs=[];transfers=[]
    arena=Arena();known=[None]*32768;traffic={'input':0,'output':0,'immutable':0,'retained_immutable':0,'copy':0}
    counters={'cache_read_bytes':0,'cache_write_bytes':0,'computed_output_bytes':0,'computed_macs':0}
    cache_manifest=[]
    def put(data):
        pos=align8(len(payload));payload.extend(bytes(pos-len(payload)));payload.extend(data)
        return pos
    parameters=[]
    for k in macros:
        conv,act=block.layers[k:k+2];weights,params=_parameter_rows(block,conv)
        _,ap=_parameter_rows(block,act);lut=activation_table(2 if act.op=='Relu' else 8,ap)
        w=conv.parameters['weight'];live=len(w)
        if conv.attributes.get('group',1)==1:
            nz=np.flatnonzero(np.any(w!=0,axis=(1,2,3)));live=int(nz[-1])+1 if len(nz) else 0
        stride=align8(math.prod(w.shape[1:]));weights=weights[:live*stride];params=params[:live*16]
        constant=[]
        for ch in range(live,len(w)):
            raw=requantize(np.asarray([int(conv.parameters['corrected_bias'][ch])],dtype=np.int64),
                int(conv.parameters['multiplier'][ch]),int(conv.parameters['shift'][ch]),
                block.tensors[conv.output].quantization.zero_point)
            constant.append(lut[int(raw[0])&255])
        parameters.append(dict(weight=weights,params=params,lut=lut,live=live,stride=stride,constant=constant,
            weight_ext=put(weights),params_ext=put(params),lut_ext=put(lut)))
    def emit(op,flags=0,a=0,b=0,c=0):
        commands.append((op,flags,0,a,b,c))
        if len(commands)>2048:raise ValueError('command-capacity')
    def dma(direction,ext,sram,size,role):
        if ext%8 or sram%8 or not 0<size<=32768 or ext+size>len(payload) or sram+size>32768:
            raise ValueError('dma-bounds/alignment')
        static=(direction=='to_sram' and role in ('weight','params','lut','descriptor','constant'))
        data=list(payload[ext:ext+size]) if static else [None]*size
        if static and role!='descriptor' and retain_weights and known[sram:sram+size]==data:
            traffic['retained_immutable']+=size;return
        at=len(commands);emit(1,int(direction=='to_sram'),ext,sram,size);emit(3,2)
        transfers.append(dict(command=at,direction=direction,ext=ext,sram=sram,bytes=size,role=role))
        if direction=='to_sram':
            known[sram:sram+size]=data
            traffic['immutable' if static else 'input']+=size
        else:traffic['output']+=size
    def run(d,details):
        raw=details.get('descriptor',d.encode());loc=put(raw+Descriptor(0).encode())
        dma('to_sram',loc,0,128,'descriptor')
        live=arena.live();high=max([128]+[r[1] for r in live]);at=len(commands)
        emit(2,0,0,high<<16);emit(3,3)
        row=dict(command=at,descriptor_hex=raw.hex(),live_ranges=live,**{k:v for k,v in details.items() if k!='descriptor'})
        runs.append(row)
        if d.opcode==3:
            # COPY behavior is byte-serial, but all backend source/destination
            # pairs are disjoint; no dependence on memmove semantics.
            known[d.output:d.output+d.outputs]=list(known[d.input:d.input+d.count])
        else:known[d.output:d.output+d.outputs]=[None]*d.outputs
    def copy_bytes(src,dst,size,role):
        d=Descriptor(3,input=src,output=dst,count=size,outputs=size,next_pc=64)
        d.validate();run(d,dict(kind='copy',role=role));traffic['copy']+=size
    def channels(j):
        return ishape[1] if j==-1 else block.tensors[block.layers[macros[j]+1].output].shape[1]
    def fullshape(j):
        return ishape[1:] if j==-1 else block.tensors[block.layers[macros[j]+1].output].shape[1:]
    def region_copy(source,source_rect,dest,dest_rect,piece,j,role):
        if not piece.area:return
        temp=arena.alloc(16,'copy-prefix')
        try:
            emit_copy_rectangle(source.base,(channels(j),*source_rect.shape),
                (piece.y0-source_rect.y0,piece.x0-source_rect.x0,piece.y1-source_rect.y0,piece.x1-source_rect.x0),
                dest.base,(channels(j),*dest_rect.shape),
                (piece.y0-dest_rect.y0,piece.x0-dest_rect.x0),temp.base,copy_bytes)
        finally:arena.free(temp)
    needs=shapes_and_needs(block,tile_h,tile_w)
    h_old={};h_new={};v_old={};v_new={}
    active_tile=(0,0)
    def cache_alloc(rect,j,tag):
        if not rect.area:return None
        b=arena.alloc(channels(j)*rect.area,tag)
        return Cache(rect,b,[])
    def free_cache(cache):
        if cache is not None:arena.free(cache.buffer)
    def publish(j,rect,result):
        for cache in (h_new.get(j),v_new.get(j)):
            if cache is None:continue
            piece=rect.intersect(cache.rect)
            if not piece.area:continue
            region_copy(result,rect,cache.buffer,cache.rect,piece,j,'cache-write')
            cache.valid.append(piece);counters['cache_write_bytes']+=channels(j)*piece.area
    def compute(j,wanted):
        if not wanted.area:raise ValueError('empty-request')
        result=None
        missing=[wanted]
        for cache in (h_old.get(j),v_old.get(j)):
            if cache is None:continue
            for valid in cache.valid:
                todo=[]
                for piece in missing:
                    common=piece.intersect(valid)
                    if common.area:
                        if result is None:result=arena.alloc(channels(j)*wanted.area,f'request-{j}')
                        region_copy(cache.buffer,cache.rect,result,wanted,common,j,'cache-read')
                        counters['cache_read_bytes']+=channels(j)*common.area
                    todo.extend(piece.subtract(common))
                missing=todo
        for piece in missing:
            if j==-1:
                if result is None:result=arena.alloc(channels(j)*wanted.area,f'request-{j}')
                temp=arena.alloc(rectangle_temp_bytes(piece.shape[1]),'dma-gather')
                gathered=result if piece==wanted else arena.alloc(channels(-1)*piece.area,'input-piece')
                try:
                    emit_read_rectangle(0,fullshape(-1),(piece.y0,piece.x0,piece.y1,piece.x1),
                        gathered.base,temp.base,dma,copy_bytes)
                    if gathered is not result:region_copy(gathered,piece,result,wanted,piece,-1,'input-assemble')
                finally:
                    if gathered is not result:arena.free(gathered)
                    arena.free(temp)
            else:
                conv=block.layers[macros[j]];a=conv.attributes;kh,kw=conv.parameters['weight'].shape[2:]
                sh,sw=a.get('strides',[1,1]);pt,pl,pb,pr=a.get('pads',[0]*4)
                halo=input_halo(piece,(kh,kw),(sh,sw),(pt,pl,pb,pr))
                fs=fullshape(j-1);clipped=halo.intersect(Rect(0,0,fs[1],fs[2]))
                source=compute(j-1,clipped)
                if result is None:result=arena.alloc(channels(j)*wanted.area,f'request-{j}')
                produced=result if piece==wanted else arena.alloc(channels(j)*piece.area,'conv-piece')
                row=parameters[j];live=[]
                try:
                    if row['live']:
                        for key in ('weight','params','lut'):
                            b=arena.alloc(len(row[key]),key);live.append(b)
                            dma('to_sram',row[key+'_ext'],b.base,len(row[key]),key)
                        wb,pb_,lb=live;depthwise=a.get('group',1)!=1
                        d=Descriptor(6 if depthwise else 4,input=source.base,output=produced.base,
                            weight=wb.base,params=pb_.base,count=kh*kw*(1 if depthwise else fs[0]),
                            outputs=row['live']*piece.area,row_stride=row['stride'],next_pc=64,
                            kernel_h=kh,kernel_w=kw,stride_h=sh,stride_w=sw,
                            pad_top=clipped.y0-halo.y0,pad_bottom=halo.y1-clipped.y1,
                            pad_left=clipped.x0-halo.x0,pad_right=halo.x1-clipped.x1,
                            input_h=clipped.shape[0],input_w=clipped.shape[1],input_c=fs[0],output_c=row['live'])
                        d.validate();raw=encode_fused(d,lb.base)
                        run(d,dict(kind='conv',macro=j,layer=start+macros[j],rect=list((piece.y0,piece.x0,piece.y1,piece.x1)),
                            input_rect=list((clipped.y0,clipped.x0,clipped.y1,clipped.x1)),
                            descriptor=raw,weight_hex=row['weight'].hex(),params_hex=row['params'].hex(),lut_hex=row['lut'].hex()))
                        counters['computed_macs']+=d.outputs*d.count
                    for b in reversed(live):arena.free(b)
                    live=[]
                    if row['constant']:
                        data=b''.join(bytes([v])*piece.area for v in row['constant']);constant=arena.alloc(len(data),'constant')
                        try:
                            dma('to_sram',put(data),constant.base,len(data),'constant')
                            copy_bytes(constant.base,produced.base+row['live']*piece.area,len(data),'constant-tail')
                        finally:arena.free(constant)
                    counters['computed_output_bytes']+=channels(j)*piece.area
                    if produced is not result:region_copy(produced,piece,result,wanted,piece,j,'assemble-result')
                finally:
                    for b in reversed(live):arena.free(b)
                    if produced is not result:arena.free(produced)
                    arena.free(source)
        publish(j,wanted,result)
        return result
    for yi,row_needs in enumerate(needs):
        if mode==3 and yi+1<len(needs):
            for j in range(-1,len(macros)):
                here=bbox([n[j] for n in row_needs]);nxt=bbox([n[j] for n in needs[yi+1]])
                rect=here.intersect(nxt)
                v_new[j]=cache_alloc(rect,j,f'v-next-{j}')
        for xi,tile_needs in enumerate(row_needs):
            active_tile=(yi,xi)
            if mode>=2 and xi+1<len(row_needs):
                for j in range(-1,len(macros)):
                    rect=tile_needs[j].intersect(row_needs[xi+1][j])
                    h_new[j]=cache_alloc(rect,j,f'h-next-{j}')
            target=tile_needs[len(macros)-1]
            result=compute(len(macros)-1,target)
            temp=arena.alloc(rectangle_temp_bytes(target.shape[1]),'dma-scatter')
            try:
                emit_write_rectangle(output_ext,fullshape(len(macros)-1),
                    (target.y0,target.x0,target.y1,target.x1),result.base,temp.base,dma,copy_bytes)
            finally:
                arena.free(temp);arena.free(result)
            cache_manifest.append(dict(tile=[yi,xi],horizontal={str(j):dict(rect=[c.rect.y0,c.rect.x0,c.rect.y1,c.rect.x1],
                valid=[[v.y0,v.x0,v.y1,v.x1] for v in c.valid]) for j,c in h_new.items() if c is not None},
                vertical={str(j):dict(rect=[c.rect.y0,c.rect.x0,c.rect.y1,c.rect.x1],
                valid=[[v.y0,v.x0,v.y1,v.x1] for v in c.valid]) for j,c in v_new.items() if c is not None}))
            for cache in h_old.values():free_cache(cache)
            h_old,h_new=h_new,{}
        for cache in h_old.values():free_cache(cache)
        h_old={}
        for cache in v_old.values():free_cache(cache)
        v_old,v_new=v_new,{}
    for cache in v_old.values():free_cache(cache)
    if arena.buffers:raise ValueError('leaked-SRAM-buffer')
    emit(0);code=b''.join(CMD.pack(*c) for c in commands)
    if len(code)>32768:raise ValueError('command-capacity')
    if len(payload)>8*1024*1024:raise ValueError('external-capacity')
    prefetch_rows=[]
    if prefetch:
        from matched_b1b2_generic import prefetch_candidates
        from prefetch_tail import reorder
        prefetch_rows=prefetch_candidates(code)
        reordered,mapping=reorder(commands,prefetch_rows)
        code=b''.join(CMD.pack(*c) for c in reordered)
        for records in (runs,transfers):
            for record in records:record['command']=mapping[record['command']]
    return code,bytes(payload),dict(schema=1,kind='rectangular-depth-first-stack',start=start,stop=stop,
        tile_h=tile_h,tile_w=tile_w,mode=mode,retain_weights=retain_weights,prefetch=prefetch,
        output_external_base=output_ext,input_bytes=source_size,output_bytes=output_size,
        code_sha256=sha(code),payload_sha256=sha(payload),command_bytes=len(code),payload_bytes=len(payload),
        peak_sram_address=arena.peak,peak_live_bytes=arena.peak_bytes,traffic=traffic,counters=counters,
        prefetch_transfers=len(prefetch_rows),runs=runs,transfers=transfers,cache_manifest=cache_manifest,
        cache_implementation='separate previous/next horizontal and vertical halo buffers with valid rectangles',
        limitations=['fixed input/output-channel and reduction dataflow of Tang Nano engine',
                    'conservative separate old/new cache storage may reject a schedule that in-place cache rotation can fit'])


def replay_stack(program,start,stop,source,code,payload,record):
    from output_pipeline_fusion import decode_fused
    if (sha(code),sha(payload))!=(record['code_sha256'],record['payload_sha256']):raise ValueError('artifact changed')
    block=block_program(program,start,stop);oracle=evaluate(block,{block.inputs[0]:source})
    ext=bytearray(payload);ext[:source.nbytes]=source.tobytes();sram=bytearray(32768)
    runs={r['command']:r for r in record['runs']};seen=[]
    for i in range(len(code)//16):
        op,flags,_,a,b,c=CMD.unpack_from(code,i*16)
        if op==0:
            if i!=len(code)//16-1:raise ValueError('early HALT')
            break
        if op==3:continue
        if op==1:
            if flags:sram[b:b+c]=ext[a:a+c]
            else:ext[a:a+c]=sram[b:b+c]
        elif op==2:
            r=runs[i];seen.append(i)
            if bytes(sram[:64]).hex()!=r['descriptor_hex']:raise ValueError('descriptor overwritten')
            if r['kind']=='copy':
                d=Descriptor.decode(bytes(sram[:64]));sram[d.output:d.output+d.outputs]=sram[d.input:d.input+d.count]
                continue
            d,table=decode_fused(bytes(sram[:64]));k=r['macro']*2;conv,act=block.layers[k:k+2]
            y0,x0,y1,x1=r['input_rect'];inp=oracle[conv.inputs[0]][:,:,y0:y1,x0:x1].copy().tobytes()
            if bytes(sram[d.input:d.input+len(inp)])!=inp:raise ValueError(f'input mismatch macro{k} command{i}')
            for base,key in ((d.weight,'weight_hex'),(d.params,'params_hex'),(table,'lut_hex')):
                value=bytes.fromhex(r[key])
                if bytes(sram[base:base+len(value)])!=value:raise ValueError(f'{key} overwritten')
            y0,x0,y1,x1=r['rect'];raw=oracle[conv.output][:,:d.output_c,y0:y1,x0:x1].copy().tobytes()
            value=bytes(sram[table+b] for b in raw)
            expected=oracle[act.output][:,:d.output_c,y0:y1,x0:x1].copy().tobytes()
            if value!=expected:raise ValueError('epilogue differs from oracle')
            sram[d.output:d.output+d.outputs]=value
        else:raise ValueError('unknown command')
    expected=oracle[block.outputs[0]].tobytes();base=record['output_external_base']
    if bytes(ext[base:base+len(expected)])!=expected:raise ValueError('final rectangle result mismatch')
    return dict(status='passed',runs=len(seen),output_sha256=sha(expected))
