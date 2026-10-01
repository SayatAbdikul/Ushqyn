"""Exact full-width depth-first strip lowering on the frozen eight-lane ABI.

This is a backend for DeFiNES mode 1 and the equivalent mode 2 when there is
one horizontal tile. It is not a complete DeFiNES adaptation: width tiling,
vertical overlap caching and channel/reduction tiling remain explicit gaps.
Every INT8 activation boundary is preserved by the shared epilogue LUT.
"""
from dataclasses import replace
import copy
import hashlib
import math
import struct

import numpy as np

from hardware_v2 import Descriptor
from integer_reference import evaluate
from phase4_compile import _parameter_rows
from quantization import requantize
from static_pipeline import Program

CMD = struct.Struct('<BBHIII')


def align8(n):
    return (n + 7) & ~7


def sha(data):
    return hashlib.sha256(data).hexdigest()


def macro_stacks(program, max_macros=None):
    """Every contiguous Conv/(Relu|Clip) stack, not hand-selected model pairs."""
    runs=[]; current=[]
    i=0
    while i < len(program.layers)-1:
        a,b=program.layers[i:i+2]
        if (a.op=='Conv' and b.op in ('Relu','Clip') and b.inputs==[a.output]
                and program.tensors[a.output].shape==program.tensors[b.output].shape):
            if current and a.inputs!=[program.layers[current[-1]+1].output]:
                runs.append(current);current=[]
            current.append(i);i+=2
        else:
            if current:runs.append(current);current=[]
            i+=1
    if current:runs.append(current)
    for run in runs:
        for i in range(len(run)):
            for j in range(i+1, len(run)+1):
                if max_macros is None or j-i<=max_macros:
                    yield run[i],run[j-1]+2


def block_program(program, start, stop):
    layers=copy.deepcopy(program.layers[start:stop])
    names=[layers[0].inputs[0]]+[l.output for l in layers]
    return Program({n:copy.deepcopy(program.tensors[n]) for n in names},layers,
                   [names[0]],[names[-1]],{},copy.deepcopy(program.provenance))


def local_program(block, y0, y1):
    """Backward halo recurrence with exact border clipping and local padding."""
    local=copy.deepcopy(block)
    rectangles={block.outputs[0]:(y0,y1)}
    for layer in reversed(block.layers):
        lo,hi=rectangles[layer.output]
        ishape=block.tensors[layer.inputs[0]].shape
        if layer.op=='Conv':
            kh,kw=layer.parameters['weight'].shape[2:]
            sh,sw=layer.attributes.get('strides',[1,1])
            pt,pl,pb,pr=layer.attributes.get('pads',[0,0,0,0])
            if layer.attributes.get('dilations',[1,1])!=[1,1]:
                raise ValueError('dilation-backend-gap')
            need_lo=lo*sh-pt;need_hi=(hi-1)*sh-pt+kh
            rectangles[layer.inputs[0]]=(max(0,need_lo),min(ishape[2],need_hi))
        else:
            if layer.op not in ('Relu','Clip'):raise ValueError('operator-backend-gap')
            rectangles[layer.inputs[0]]=(lo,hi)
    for name,(lo,hi) in rectangles.items():
        t=local.tensors[name]
        local.tensors[name]=replace(t,shape=(1,t.shape[1],hi-lo,t.shape[3]))
    for layer in local.layers:
        if layer.op!='Conv':continue
        a=layer.attributes;kh,kw=layer.parameters['weight'].shape[2:]
        sh,sw=a.get('strides',[1,1]);pt,pl,pb,pr=a.get('pads',[0,0,0,0])
        oy0,oy1=rectangles[layer.output];iy0,iy1=rectangles[layer.inputs[0]]
        top=iy0-(oy0*sh-pt);bottom=(oy1-1)*sh-pt+kh-iy1
        a['pads']=[top,pl,bottom,pr]
    return local,rectangles


def allocate(size, live, capacity=32768, reverse=False):
    """First fit into actual byte ranges; all DMA-visible buffers align to 8."""
    size=align8(size)
    if reverse:
        end=capacity
        for lo,hi in sorted(live,reverse=True):
            if end-size>=hi:break
            end=min(end,lo//8*8)
        if end-size<128:raise ValueError('sram-capacity')
        return end-size
    pos=128
    for lo,hi in sorted(live):
        if pos+size<=lo:break
        pos=max(pos,align8(hi))
    if pos+size>capacity:raise ValueError('sram-capacity')
    return pos


def compile_strip_stack(program,start,stop,tile_h):
    # Import the shared, tested LUT implementation lazily (tool layer is on
    # the caller's path); no separately approximated activation arithmetic.
    from output_pipeline_fusion import activation_table,encode_fused,decode_fused
    if (start,stop) not in set(macro_stacks(program)):
        raise ValueError('not-contiguous-conv-activation-stack')
    block=block_program(program,start,stop)
    shape_in=block.tensors[block.inputs[0]].shape
    shape_out=block.tensors[block.outputs[0]].shape
    if not 1<=tile_h<=shape_out[2]:raise ValueError('invalid-tile-height')
    source_bytes=math.prod(shape_in);output_bytes=math.prod(shape_out)
    output_ext=align8(source_bytes)
    payload=bytearray(output_ext+align8(output_bytes))
    commands=[];transfers=[];runs=[];stripes=[];peak=128
    def put(data):
        pos=align8(len(payload));payload.extend(bytes(pos-len(payload)));payload.extend(data)
        return pos
    params=[]
    for k in range(0,len(block.layers),2):
        conv,act=block.layers[k:k+2]
        weights,parameters=_parameter_rows(block,conv)
        _,apost=_parameter_rows(block,act)
        lut=activation_table(2 if act.op=='Relu' else 8,apost)
        w=conv.parameters['weight'];zero=np.all(w==0,axis=(1,2,3))
        live_outputs=len(w)
        # Same exact zero-filter optimization as common compiler. Channel
        # grouping guarantees a tail on standard Conv; DW stays unmodified.
        if conv.attributes.get('group',1)==1:
            nonzero=np.flatnonzero(~zero)
            live_outputs=int(nonzero[-1])+1 if len(nonzero) else 0
        row_stride=align8(math.prod(w.shape[1:]))
        weights=weights[:live_outputs*row_stride]
        parameters=parameters[:live_outputs*16]
        codes=[]
        for ch in range(live_outputs,len(w)):
            raw=requantize(np.asarray([int(conv.parameters['corrected_bias'][ch])],dtype=np.int64),
                           int(conv.parameters['multiplier'][ch]),int(conv.parameters['shift'][ch]),
                           block.tensors[conv.output].quantization.zero_point)
            codes.append(lut[int(raw[0])&255])
        params.append(dict(weights=weights,params=parameters,lut=lut,
                           weight_ext=put(weights),params_ext=put(parameters),lut_ext=put(lut),
                           live_outputs=live_outputs,row_stride=row_stride,constant_codes=codes))
    def emit(op,flags=0,a=0,b=0,c=0):commands.append((op,flags,0,a,b,c))
    def dma(direction,external,sram,size,role):
        nonlocal peak
        if external%8 or sram%8:raise ValueError('dma-alignment')
        if not 0<size<=32768 or external+size>len(payload) or sram+size>32768:
            raise ValueError('dma-bounds')
        idx=len(commands);emit(1,int(direction=='to_sram'),external,sram,size);emit(3,2)
        transfers.append(dict(command=idx,direction=direction,ext=external,sram=sram,bytes=size,role=role))
        peak=max(peak,sram+size)
    for strip,y0 in enumerate(range(0,shape_out[2],tile_h)):
        y1=min(y0+tile_h,shape_out[2]);local,rects=local_program(block,y0,y1)
        lo,hi=rects[block.inputs[0]];input_plane=(hi-lo)*shape_in[3]
        output_plane=(y1-y0)*shape_out[3]
        # DMA cannot pack arbitrary unaligned per-channel intervals. Keep
        # rejection visible; a width/packing backend is needed for these.
        full_input=lo==0 and hi==shape_in[2]
        full_output=y0==0 and y1==shape_out[2]
        if ((not full_input and (input_plane%8 or lo*shape_in[3]%8 or (shape_in[2]*shape_in[3])%8)) or
                (not full_output and (output_plane%8 or y0*shape_out[3]%8 or (shape_out[2]*shape_out[3])%8))):
            raise ValueError('dma-stripe-alignment')
        source_sram=128
        if source_sram+shape_in[1]*input_plane>32768:raise ValueError('sram-capacity')
        if full_input:
            dma('to_sram',0,source_sram,shape_in[1]*input_plane,'source')
        else:
            for ch in range(shape_in[1]):
                dma('to_sram',ch*shape_in[2]*shape_in[3]+lo*shape_in[3],
                    source_sram+ch*input_plane,input_plane,'source')
        current=source_sram
        for k in range(0,len(block.layers),2):
            conv,act=local.layers[k:k+2];row=params[k//2]
            ish=local.tensors[conv.inputs[0]].shape;osh=local.tensors[conv.output].shape
            insize=math.prod(ish);outsize=math.prod(osh)
            output=allocate(outsize,[(current,current+insize)],reverse=current<16384)
            live=[(current,current+insize),(output,output+outsize)]
            if row['live_outputs']:
                wbase=allocate(len(row['weights']),live);live.append((wbase,wbase+len(row['weights'])))
                pbase=allocate(len(row['params']),live);live.append((pbase,pbase+len(row['params'])))
                table=allocate(256,live);live.append((table,table+256))
                kh,kw=conv.parameters['weight'].shape[2:]
                sh,sw=conv.attributes.get('strides',[1,1]);pt,pl,pb,pr=conv.attributes.get('pads',[0]*4)
                depthwise=conv.attributes.get('group',1)!=1
                if depthwise and conv.attributes['group']!=ish[1]:raise ValueError('grouping-backend-gap')
                d=Descriptor(6 if depthwise else 4,input=current,output=output,weight=wbase,params=pbase,
                    count=kh*kw*(1 if depthwise else ish[1]),outputs=row['live_outputs']*osh[2]*osh[3],
                    row_stride=row['row_stride'],next_pc=64,kernel_h=kh,kernel_w=kw,stride_h=sh,stride_w=sw,
                    pad_top=pt,pad_bottom=pb,pad_left=pl,pad_right=pr,input_h=ish[2],input_w=ish[3],
                    input_c=ish[1],output_c=row['live_outputs'])
                d.validate();encoded=encode_fused(d,table)
                if decode_fused(encoded)!=(d,table):raise ValueError('descriptor roundtrip')
                dext=put(encoded+Descriptor(0).encode())
                dma('to_sram',dext,0,128,'descriptor')
                dma('to_sram',row['weight_ext'],wbase,len(row['weights']),'weight')
                dma('to_sram',row['params_ext'],pbase,len(row['params']),'parameter')
                dma('to_sram',row['lut_ext'],table,256,'activation_table')
                ri=len(commands);emit(2,0,0,32768<<16);emit(3,3)
                runs.append(dict(command=ri,strip=strip,layer=start+k,local_layer=k,
                    descriptor_hex=encoded.hex(),input=current,output=output,input_bytes=insize,
                    dynamic_output_bytes=d.outputs,lut_hex=row['lut'].hex(),weight_hex=row['weights'].hex(),
                    params_hex=row['params'].hex(),live_ranges=live))
            if row['constant_codes']:
                dynamic_bytes=row['live_outputs']*osh[2]*osh[3]
                if (output+dynamic_bytes)%8:raise ValueError('constant-tail-alignment')
                fill=b''.join(bytes([c])*(osh[2]*osh[3]) for c in row['constant_codes'])
                dma('to_sram',put(fill),output+dynamic_bytes,len(fill),'exact_constant_tail')
            current=output;peak=max(peak,output+outsize)
        if full_output:
            dma('from_sram',output_ext,current,shape_out[1]*output_plane,'output')
        else:
            for ch in range(shape_out[1]):
                dma('from_sram',output_ext+ch*shape_out[2]*shape_out[3]+y0*shape_out[3],
                    current+ch*output_plane,output_plane,'output')
        stripes.append(dict(index=strip,output_rows=[y0,y1],input_rows=[lo,hi],
                            rectangles={n:list(r) for n,r in rects.items()}))
    emit(0);code=b''.join(CMD.pack(*c) for c in commands)
    if len(code)>32768:raise ValueError('command-capacity')
    if len(payload)>8*1024*1024:raise ValueError('external-capacity')
    return code,bytes(payload),dict(schema=1,scope='exact full-width mode1 depth-first strips; mode2 equivalent for one X tile',
        complete_defines_baseline=False,start=start,stop=stop,tile_h=tile_h,tile_w=shape_out[3],
        cache_mode=1,input_external_base=0,output_external_base=output_ext,
        input_bytes=source_bytes,output_bytes=output_bytes,code_sha256=sha(code),payload_sha256=sha(payload),
        command_bytes=len(code),payload_bytes=len(payload),peak_sram_address=peak,
        runs=runs,transfers=transfers,stripes=stripes,
        common_optimizations=['channel grouping','dead-channel compaction','constant-filter tails','exact activation epilogue'],
        unsupported=['width tiles','vertical overlap caching','channel/reduction tile stacks','DMA/compute overlap inside stack'])


def replay_strip_stack(program,start,stop,source,code,payload,record):
    """Replay bytes with independent integer local-oracles and memory checks."""
    from output_pipeline_fusion import decode_fused
    if (sha(code),sha(payload))!=(record['code_sha256'],record['payload_sha256']):
        raise ValueError('artifact digest mismatch')
    block=block_program(program,start,stop)
    expected=evaluate(block,{block.inputs[0]:source})[block.outputs[0]]
    locals_=[]
    for row in record['stripes']:
        local,rects=local_program(block,*row['output_rows']);lo,hi=rects[block.inputs[0]]
        x=source[:,:,lo:hi,:].copy()
        locals_.append((local,evaluate(local,{local.inputs[0]:x})))
    ext=bytearray(payload);ext[:source.nbytes]=source.tobytes();sram=bytearray(32768)
    runs={row['command']:row for row in record['runs']};seen=[]
    for i in range(len(code)//16):
        op,flags,res,a,b,c=CMD.unpack_from(code,i*16)
        if res:raise ValueError('reserved command')
        if op==0:
            if i!=len(code)//16-1:raise ValueError('early halt')
            break
        if op==3:continue
        if op==1:
            if flags:sram[b:b+c]=ext[a:a+c]
            else:ext[a:a+c]=sram[b:b+c]
            continue
        if op!=2 or a!=0 or b!=32768<<16:raise ValueError('unexpected engine command')
        r=runs[i];seen.append(i);d,t=decode_fused(bytes(sram[:64]))
        if bytes(sram[:64]).hex()!=r['descriptor_hex']:raise ValueError('descriptor changed')
        local,values=locals_[r['strip']];conv=local.layers[r['local_layer']];act=local.layers[r['local_layer']+1]
        inp=values[conv.inputs[0]].tobytes()
        if bytes(sram[d.input:d.input+len(inp)])!=inp:raise ValueError('operand differs from integer oracle')
        for base,key in ((d.weight,'weight_hex'),(d.params,'params_hex'),(t,'lut_hex')):
            val=bytes.fromhex(r[key])
            if bytes(sram[base:base+len(val)])!=val:raise ValueError(f'{key} overwritten')
        lut=bytes.fromhex(r['lut_hex']);raw=values[conv.output].tobytes()[:d.outputs]
        out=bytes(lut[v] for v in raw)
        if out!=values[act.output].tobytes()[:d.outputs]:raise ValueError('epilogue arithmetic mismatch')
        sram[d.output:d.output+len(out)]=out
    base=record['output_external_base']
    actual=bytes(ext[base:base+expected.nbytes])
    if actual!=expected.tobytes():raise ValueError('strip result differs from full block integer oracle')
    return dict(status='passed',engine_runs=len(seen),stripes=len(locals_),output_sha256=sha(actual))
