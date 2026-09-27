#!/usr/bin/env python3
"""Executable arithmetic/layout prototypes, not routed hardware speed claims.

These screens precede large RTL/ABI changes. Every modeled operator is compared
with the independent INT8 oracle. Operation/transfer counts exclude controller
costs and must not be presented as measured cycles or expected board speedup.
"""
import copy
import json
import math
from pathlib import Path

import numpy as np

from variants import ROOT, check_frozen, sha
from run_boardless import load_model
from integer_reference import evaluate, rounded
from run_screening import save_json

BASE=ROOT/'work/phase6/experiments-v1/algorithms'


def lane_conv(program, layer, x, pixels, channels, padded_layout=False):
    p=layer.parameters;a=layer.attributes;w=p['weight'].astype(np.int64)
    iq=program.tensors[layer.inputs[0]].quantization;oq=program.tensors[layer.output].quantization
    shape=program.tensors[layer.output].shape
    groups=a.get('group',1);depthwise=groups!=1
    if depthwise and not (groups==x.shape[1]==len(w) and channels==1):raise ValueError('depthwise multiplier')
    kh,kw=w.shape[2:];sh,sw=a.get('strides',[1,1]);pt,pl,pb,pr=a.get('pads',[0,0,0,0])
    if a.get('dilations',[1,1])!=[1,1]:raise ValueError('unsupported dilation')
    plane=x.shape[2]*x.shape[3];physical_plane=(plane+7)//8*8 if padded_layout else plane
    raw=np.full((x.shape[1],physical_plane),iq.zero_point,np.int8)
    raw[:,:plane]=x.reshape(x.shape[1],plane)
    flat=raw.ravel();out=np.empty(shape,np.int8)
    used=slots=word_reads=0
    for c0 in range(0,len(w),channels):
        cs=min(channels,len(w)-c0)
        for pixel0 in range(0,shape[2]*shape[3],pixels):
            ps=min(pixels,shape[2]*shape[3]-pixel0)
            indices=np.arange(pixel0,pixel0+ps);yy=indices//shape[3];xx=indices%shape[3]
            accum=np.broadcast_to(p['bias'][c0:c0+cs],(ps,cs)).astype(np.int64).copy()
            for ci in range(1 if depthwise else x.shape[1]):
                input_channel=c0 if depthwise else ci
                for ky in range(kh):
                    for kx in range(kw):
                        iy=yy*sh-pt+ky;ix=xx*sw-pl+kx
                        valid=(iy>=0)&(iy<x.shape[2])&(ix>=0)&(ix<x.shape[3])
                        addresses=input_channel*physical_plane+iy*x.shape[3]+ix
                        values=np.full(ps,iq.zero_point,np.int64)
                        values[valid]=flat[addresses[valid]]
                        # One vector broadcast across output channels: eight
                        # products for 8x1 or 4x2. No hidden extra multipliers.
                        accum+=(values-iq.zero_point)[:,None]*w[c0:c0+cs,ci,ky,kx][None,:]
                        used+=ps*cs;slots+=pixels*channels
                        word_reads+=len(set((addresses[valid]//8).tolist()))
            for c in range(cs):
                out.reshape(len(w),-1)[c0+c,pixel0:pixel0+ps]=rounded(
                    accum[:,c]*int(p['multiplier'][c0+c]),p['shift'][c0+c],oq.zero_point)
    return out,dict(useful_mac_products=used,scheduled_lane_slots=slots,lane_utilization=used/slots,
        activation_word_reads_without_cache=word_reads,physical_plane=physical_plane,
        added_input_bytes=(physical_plane-plane)*x.shape[1])


def projected_weights(program, mixed):
    changed=copy.deepcopy(program)
    indices=[i for i,l in enumerate(program.layers) if l.op in ('Conv','Gemm')]
    details=[]
    for i in indices:
        layer=changed.layers[i];original=layer.parameters['weight']
        # A 15-value per-channel codebook stored as signed INT4 indices. The
        # existing INT8 datapath would need a decoder; this is not native INT4
        # MAC hardware and claims no automatic throughput improvement.
        if mixed and i in (indices[0],indices[-1]):continue
        w=original.reshape(len(original),-1).astype(np.int64)
        maxima=np.max(np.abs(w),axis=1)
        scales=np.maximum(maxima/7,1/7)
        q=np.clip(np.rint(w/scales[:,None]),-7,7).astype(np.int8)
        codebook=np.clip(np.rint(np.arange(-7,8)[None,:]*scales[:,None]),-128,127).astype(np.int8)
        decoded=np.take_along_axis(codebook,(q.astype(np.int16)+7),axis=1).reshape(original.shape)
        layer.parameters['weight']=decoded
        details.append(dict(layer=i,int8_bytes=original.size,packed_bytes=(q.size+1)//2,
                            codebook_bytes=codebook.size,changed_weights=int(np.count_nonzero(decoded!=original)),
                            max_integer_weight_error=int(np.max(np.abs(decoded.astype(np.int16)-original.astype(np.int16))))))
    return changed,details


def packed_multiply_check():
    tested=0
    for x in range(-128,128):
        for a in range(-8,8):
            for b in range(-8,8):
                # Two signed INT4 weights separated by a complete 12-bit
                # product lane fit one signed 18-bit operand, not a 9-bit one.
                packed=a+(b<<12)
                assert -(1<<17)<=packed<(1<<17)
                product=x*packed
                low=((product+2048)&4095)-2048
                high=(product-low)//4096
                assert (low,high)==(x*a,x*b)
                tested+=1
    return dict(status='passed-exhaustive-integer-identity',triples=tested,
        required_packed_weight_width=17,activation_width=8,
        limitation='18-bit multiplier operand plus sign correction; not two products in one existing 9x9 lane. Gowin mapping/area not established.')


def run():
    check_frozen();BASE.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',physical_board=False,scope=__doc__,
                               source_sha256=sha(Path(__file__)),models={})
    report['packed_int4']=packed_multiply_check()
    for name in ('kws','vww'):
        program,x,_,pins=load_model(name);row=dict(pins=pins,operators=[],precision=[])
        variants={mode:projected_weights(program,mode=='mixed-int4-codebook') for mode in ('all-int4-codebook','mixed-int4-codebook')}
        for sample,value in (('pinned',x),('seeded-stress',np.random.default_rng(617).integers(-128,128,x.shape,dtype=np.int8))):
            oracle=evaluate(program,{program.inputs[0]:value})
            for i,layer in enumerate(program.layers):
                if layer.op!='Conv':continue
                depthwise=layer.attributes.get('group',1)!=1
                pointwise=layer.parameters['weight'].shape[2:]==(1,1)
                if not(depthwise or pointwise):continue
                modes=[('depthwise-8x1',8,1,False)] if depthwise else [
                    ('pointwise-8x1',8,1,False),('pointwise-8x1-padded',8,1,True),('pointwise-4x2-padded',4,2,True)]
                for mode,pixels,channels,padded in modes:
                    output,counts=lane_conv(program,layer,oracle[layer.inputs[0]],pixels,channels,padded)
                    if not np.array_equal(output,oracle[layer.output]):raise ValueError(f'{name}/{sample}/{i}/{mode} mismatch')
                    row['operators'].append(dict(sample=sample,layer=i,mode=mode,status='passed-arithmetic',output_bytes=output.size,**counts))
            expected=oracle[program.outputs[0]]
            for mode,(changed,weights) in variants.items():
                output=evaluate(changed,{program.inputs[0]:value})[program.outputs[0]]
                row['precision'].append(dict(mode=mode,sample=sample,weights=weights,
                    changed_logit_bytes=int(np.count_nonzero(output!=expected)),
                    max_logit_code_error=int(np.max(np.abs(output.astype(np.int16)-expected.astype(np.int16)))),
                    same_prediction=bool(np.argmax(output)==np.argmax(expected)),
                    scope='two-fixture perturbation screen only; no accuracy estimate, training, pruning, distillation or FPGA implementation'))
        report['models'][name]=row;save_json(BASE/'report.json',report);print(name,'algorithm screens passed',flush=True)
    report['status']='passed-prototype-arithmetic';save_json(BASE/'report.json',report);check_frozen()


if __name__=='__main__':run()
