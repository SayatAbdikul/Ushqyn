#!/usr/bin/env python3
"""Inventory strongest programs and screen exact identity-epilogue elimination.

Frozen descriptors, models and hardware are never edited. The candidate is
only legal when all256 signed-byte input codes map to themselves. The actual
models currently have no such epilogues, which is an explicit negative result.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'compiler'),str(ROOT/'tools/phase6')]
from hardware_v2 import Descriptor
from output_pipeline_fusion import decode_fused,encode_fused
from matched_defines_baseline import CMD,NATIVE,sha,save
from matched_defines_finalize import validate_fixture
from matched_b1b2_generic import native_identity

BASE=ROOT/'work/phase6/engine-candidate-compiler-v1'
SOURCE=ROOT/'work/phase6/matched-baselines-v1/b3-final/report.json'


def bypass_identity_epilogue(raw,table):
    """Pure, exhaustive identity test; return the exact original if ineligible."""
    d,address=decode_fused(raw)
    if len(table)!=256:
        raise ValueError('activation table must contain all256 byte codes')
    if table!=bytes(range(256)):
        return raw,False
    candidate=d.encode()
    if Descriptor.decode(candidate)!=d:
        raise ValueError('unfused geometry changed')
    return candidate,True


def inventory(directory):
    commands=(directory/'commands.bin').read_bytes()
    external=bytearray((directory/'payload.bin').read_bytes())
    external_known=bytearray(b'\1')*len(external)
    initial=(directory/'input.bin').read_bytes();external[:len(initial)]=initial
    sram=bytearray(32768);known=bytearray(32768)
    rows=[];counters=Counter()
    for offset in range(0,len(commands),16):
        op,flags,res,a,b,c=CMD.unpack_from(commands,offset)
        counters[f'command_{op}']+=1
        if op==1:
            counters['dma_bytes']+=c
            if flags:
                sram[b:b+c]=external[a:a+c];known[b:b+c]=external_known[a:a+c]
            else:
                external[a:a+c]=sram[b:b+c];external_known[a:a+c]=known[b:b+c]
        elif op==2:
            pc=a;visited=set()
            while True:
                if pc in visited or pc+64>len(sram) or not all(known[pc:pc+64]):
                    raise ValueError('descriptor chain is not immutable and known')
                visited.add(pc);raw=bytes(sram[pc:pc+64])
                fused=bool(struct.unpack_from('<H',raw,6)[0]&1)
                d,lut=decode_fused(raw) if fused else (Descriptor.decode(raw),None)
                if d.opcode==0:break
                item=dict(command_index=offset//16,descriptor_pc=pc,descriptor=asdict(d),fused=fused)
                counters['descriptors']+=1
                if d.opcode in (1,4,6):
                    counters['macs']+=d.count*d.outputs
                    item['macs']=d.count*d.outputs
                if d.opcode in (4,6,5,7):
                    oh=(d.input_h+d.pad_top+d.pad_bottom-d.kernel_h)//d.stride_h+1
                    ow=(d.input_w+d.pad_left+d.pad_right-d.kernel_w)//d.stride_w+1
                    item.update(output_h=oh,output_w=ow,input_plane=d.input_h*d.input_w,
                                input_plane_mod8=(d.input_h*d.input_w)%8)
                    if d.opcode==6 and d.kernel_h==d.kernel_w==3 and d.stride_h==d.stride_w==1:
                        item['vector_pixels_issued']=d.output_c*oh*((ow+7)//8)*8
                        item['active_pixel_slots']=d.outputs
                        counters['dw_active_pixels']+=d.outputs
                        counters['dw_vector_slots']+=item['vector_pixels_issued']
                    if d.opcode==4 and d.kernel_h==d.kernel_w==d.stride_h==d.stride_w==1:
                        item['vector_pixels_issued']=d.output_c*((oh*ow+7)//8)*8
                        item['active_pixel_slots']=d.outputs
                        counters['pw_active_pixels']+=d.outputs
                        counters['pw_vector_slots']+=item['vector_pixels_issued']
                if fused:
                    if not all(known[lut:lut+256]):raise ValueError('unknown activation LUT bytes')
                    table=bytes(sram[lut:lut+256]);candidate,eligible=bypass_identity_epilogue(raw,table)
                    same=sum(a==b for a,b in zip(table,range(256)))
                    item['epilogue']=dict(table_sram=lut,table_sha256=sha(table),identity_codes=same,
                        total_codes=256,eligible=eligible,unchanged=raw==candidate,
                        descriptor_after_sha256=sha(candidate))
                    counters['fused_descriptors']+=1;counters['identity_epilogues']+=int(eligible)
                # Computed data must not be treated as immutable metadata later.
                known[d.output:d.output+d.outputs]=bytes(d.outputs)
                pc=d.next_pc
        elif op not in (0,3):raise ValueError('unknown command opcode')
    for key in ('dw','pw'):
        if counters[key+'_vector_slots']:
            counters[key+'_pixel_slot_utilization']=counters[key+'_active_pixels']/counters[key+'_vector_slots']
    return dict(counters=counters,descriptors=rows)


def microtile_bound(model, reference):
    """Optimistic bound for inserting8-pixel PW pack/compute/unpack stages.

    All existing activation-gather cycles are credited as removable, even
    though a real microtile still needs first reads. COPY bound counts only
    eight descriptor reads (16 cycles) and one MAC+write cycle per copied byte.
    New fused descriptors necessarily refill their LUT and clear their cache;
    weight-cache invalidation adds at least one WAIT per newly read weight word.
    No parameter setup, gather remainder, unaligned COPY preservation, command
    overhead, descriptor DMA, extra tail arithmetic or off-chip cost is charged.
    """
    directory=ROOT/'work/phase6/engine-profile-v1/final'
    proof=directory/'report.json'
    manifest=json.loads(proof.read_text())
    if manifest['status']!='passed':raise ValueError('profile is incomplete')
    profile_path=directory/f'{model}-s0-enriched.json'
    r=json.loads(profile_path.read_text())
    original=next(n for n in reference['native'] if n['stall_seed']==0)
    if r['original_native_result']!=original:
        raise ValueError('profile is not the strongest exact reference')
    raw=directory/f'{model}-s0.json.profile.json'
    if sha(raw)!=r['raw_profile_sha256']:raise ValueError('raw profile changed')
    rows=[]
    for index,row in enumerate(r['descriptors']):
        if row['opcode']!=4:continue
        d,lut=decode_fused(bytes.fromhex(row['descriptor_hex']))
        if not (d.kernel_h==d.kernel_w==d.stride_h==d.stride_w==1):continue
        pixels=d.input_h*d.input_w;tiles=(pixels+7)//8
        if tiles<=1:continue
        copies=(d.input_c+d.output_c)*tiles
        copied_bytes=(d.input_c+d.output_c)*pixels
        copy_cycles=16*copies+2*copied_bytes
        extra_descriptors=tiles-1
        setup=extra_descriptors*(256+256+32+32)
        extra_weight_wait=extra_descriptors*d.output_c*((d.input_c+7)//8)
        gather=row['categories']['activation_gather_cache']
        bound=gather-copy_cycles-setup-extra_weight_wait
        rows.append(dict(profile_descriptor_index=index,run=row['run'],input_channels=d.input_c,
            output_channels=d.output_c,spatial_pixels=pixels,microtiles=tiles,
            original_engine_cycles=row['engine_cycles'],original_activation_gather_cycles=gather,
            copy_descriptors=copies,copy_bytes=copied_bytes,copy_lower_bound_cycles=copy_cycles,
            extra_epilogue_cache_setup_cycles=setup,extra_weight_wait_lower_bound_cycles=extra_weight_wait,
            optimistic_net_saving_upper_bound_cycles=bound,
            dma_gather_minimum_commands=copies,command_store_limit=2048,
            dma_commands_alone_fit=copies<2048))
    return dict(profile_evidence=[dict(file=str(p.relative_to(ROOT)),sha256=sha(p))
                                  for p in (proof,profile_path,raw)],
        mechanism='one8-pixel dense tile per PW descriptor, retained in the word cache across output channels',
        scope='insertion into strongest dense-layout path; producer and consumers retain existing tensor layout',
        assumptions='zero residual input-gather cost; aligned perfect COPY packing; no added arithmetic or metadata transfer cost; all other work unchanged',
        rows=rows,
        positive_upper_bound_cycles=sum(max(0,x['optimistic_net_saving_upper_bound_cycles']) for x in rows),
        all_microtiles_upper_bound_cycles=sum(x['optimistic_net_saving_upper_bound_cycles'] for x in rows),
        all_dma_minimum_commands=sum(x['dma_gather_minimum_commands'] for x in rows),
        conclusion='no positive KWS insertion bound' if model=='kws' else 'dominant aligned-plane PWs reject; some smaller PWs retain an optimistic ceiling, not a measured gain',
        limitation='does not rule out a different persistent tile layout spanning producers/consumers or added hardware gather/loop support')


def run(output=BASE):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    source=json.loads(SOURCE.read_text())
    if source['status']!='passed':raise ValueError('strongest baseline incomplete')
    pins=dict(source['compiler_sources']);pins[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    for path,digest in pins.items():
        if sha(ROOT/path)!=digest:raise ValueError(f'source changed: {path}')
    # Pure lowering guard test: one positive identity and one nonidentity case.
    d=Descriptor(4,input=512,output=1024,weight=2048,params=4096,count=1,
                 outputs=8,row_stride=8,next_pc=64,input_c=1,output_c=1,input_w=8)
    raw=encode_fused(d,8192);candidate,eligible=bypass_identity_epilogue(raw,bytes(range(256)))
    assert eligible and candidate==d.encode()
    nonidentity=bytes([1])+bytes(range(1,256))
    candidate,eligible=bypass_identity_epilogue(raw,nonidentity)
    assert not eligible and candidate==raw
    report=dict(schema=1,status='running',physical_board=False,compiler_sources=pins,native=native_identity(),
        source_report=dict(file=str(SOURCE.relative_to(ROOT)),sha256=sha(SOURCE)),models={},
        candidate=dict(name='exhaustively proven identity epilogue elimination',
            exactness='all256 signed-byte codes are checked; only identity maps may bypass fused lookup',
            pure_lowering_guard_tests='passed identity and nonidentity cases',
            hardware_change=False,graph_or_arithmetic_change=False))
    for model in ('kws','vww'):
        selected=source['selected'][model]
        for sample in ('pinned','stress'):validate_fixture(selected[sample])
        directory=ROOT/selected['pinned']['directory']
        inv=inventory(directory)
        if inv['counters']['identity_epilogues']:
            raise ValueError('unexpected positive candidate needs executable full-model lowering/native verification')
        inv['pointwise_microtile_bound']=microtile_bound(model,selected['pinned'])
        inv.update(reference=selected,graph_sha256=source['models'][model]['graph_sha256'],
            disposition='no eligible descriptor; candidate would be byte-identical; no new native run',
            certified_engine_work_reduction=0)
        report['models'][model]=inv
    report['decision']=dict(status='negative',
        reason='neither strongest program has an identity epilogue; removing any actual table would change some INT8 codes',
        layout_screen=[
          dict(idea='8-pixel pointwise cache reuse',disposition='bounded rejection for both-workload insertion',
            reason='KWS COPY/setup lower bound exceeds all removable gather work; DMA alternative exceeds command capacity before full-model overhead; VWW partial headroom is not both-workload gain'),
          dict(idea='global H/W transposition',disposition='not prototyped',
            reason='KWS25x5 may improve narrow-row DW utilization, but VWW spatial planes are square; no credible both-workload headroom; input-layout work must be charged'),
          dict(idea='uniform aligned input-base coloring',disposition='negative existing cache-trace control',
            evidence='work/phase6/hypothesis-cost-v1/report.json',
            reason='uniform base shifts permute sets and preserve conflict equality for current one-input dense descriptors'),
          dict(idea='padded spatial/channel planes',disposition='not repeated',
            reason='frozen descriptor ABI has dense pitches; pack/unpack or extra MACs must be counted, and prior padded-stride work already tested this route')],
        interpretation='no new general compiler/data-layout mechanism with supported material benefit on both workloads was found in this bounded screen',
        limits='negative for screened exact transformation and fixed ABI; not a proof that all compiler optimizations are exhausted')
    for path,digest in pins.items():
        if sha(ROOT/path)!=digest:raise ValueError('source changed during inventory')
    report['status']='passed-negative-screen';save(output/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=BASE)
    run(parser.parse_args().output)
