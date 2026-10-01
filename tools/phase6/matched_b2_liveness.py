#!/usr/bin/env python3
"""Audit missed full-producer retention across partial consumer tiles.

This is a geometric opportunity report, not a speedup estimate. It checks
actual emitted B2 stage placements and reserves the complete producer while
trying first-fit consumer output/parameter placement with epilogue headroom.
"""
import argparse
import json
import math
import struct
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'compiler'),str(ROOT/'tools/phase6')]
from matched_b1b2_generic import matched_model,save,sha


def aligned(value):return (value+7)&~7


def fit(source,stage):
    occupied=[(0,128),(source['sram'],source['sram']+aligned(source['bytes']))]
    sizes=[stage['output']['bytes']]+[t['bytes'] for t in stage['loads'] if t['role']=='parameter']+[256]
    placed=[]
    for size in sizes:
        size=aligned(size);base=128
        for lo,hi in sorted(occupied):
            if base+size<=lo:break
            base=max(base,hi)
        if base+size>32768:return None
        occupied.append((base,base+size));placed.append(dict(base=base,bytes=size))
    return placed


def analyze(program,schedule):
    rows=[];stages=schedule['stages']
    for n,producer in enumerate(stages):
        layer=program.layers[producer['layer']]
        size=math.prod(program.tensors[layer.output].shape)
        store=producer['store']
        if producer['first_element']!=0 or producer['output']['bytes']!=size or store is None:
            continue
        candidates=[]
        for s in stages[n+1:]:
            consumer=program.layers[s['layer']]
            if consumer.inputs[0]!=layer.output:continue
            if s['inplace']:continue
            load=next((t for t in s['loads'] if t['role']=='input'),None)
            if load is not None and store['ext']<=load['ext'] and load['ext']+load['bytes']<=store['ext']+size:
                candidates.append((s,load))
        if len(candidates)<2 or all(t['bytes']==size for _,t in candidates):continue
        placements=[fit(producer['output'],s) for s,_ in candidates]
        rows.append(dict(producer_layer=producer['layer'],
            consumer_layer=candidates[0][0]['layer'],producer_bytes=size,
            consumer_tiles=len(candidates),input_load_bytes=sum(t['bytes'] for _,t in candidates),
            potential_removed_dma_bytes=size+sum(t['bytes'] for _,t in candidates),
            potential_removed_dma_commands=1+len(candidates),
            fixed_ram_removed_dma_busy_cycles=3*((size+7)//8+sum((t['bytes']+7)//8 for _,t in candidates))+1+len(candidates),
            all_consumer_placements_fit=all(p is not None for p in placements),
            source_sram=producer['output']['sram'],consumer_placements=placements,
            consumer_simultaneous_bytes=[128+aligned(size)+aligned(s['output']['bytes'])+
                sum(aligned(t['bytes']) for t in s['loads'] if t['role']=='parameter')+256
                for s,_ in candidates],
            proof_scope='source range retained; each consumer output/parameter/table fits; later consumers and following tensor lifetimes still require compiler+replay'))
    return rows


def run(source,output):
    data=json.loads(source.read_text())
    result=dict(status='analysis',physical_board=False,source_report_sha256=sha(source),
        analyzer_source_sha256=sha(Path(__file__)),
        scope='missed full-producer across partial-consumer SRAM retention; geometric feasibility only',models={})
    for model,record in data['models'].items():
        _,_,program,_,_=matched_model(model);rows={}
        for name,candidate in record['candidates'].items():
            if candidate.get('status')!='passed' or candidate['policy']!='B2':continue
            schedule=json.loads((ROOT/candidate['directory']/'unfused/schedule.json').read_text())
            opportunities=analyze(program,schedule)
            code=(ROOT/candidate['directory']/'commands.bin').read_bytes()
            transfers=[struct.unpack_from('<BBHIII',code,i) for i in range(0,len(code),16)
                       if code[i]==1]
            serial_formula=sum(3*((t[5]+7)//8)+1 for t in transfers)
            rows[name]=dict(opportunities=opportunities,
                total_potential_removed_bytes=sum(r['potential_removed_dma_bytes'] for r in opportunities),
                fitting_potential_removed_bytes=sum(r['potential_removed_dma_bytes'] for r in opportunities if r['all_consumer_placements_fit']),
                timed_native_cycles=candidate['native']['elapsed_cycles'],
                engine_busy_cycles=candidate['native']['engine_cycles'],
                dma_busy_cycles=candidate['native']['dma_cycles'],
                overlap_cycles=candidate['native']['overlap_cycles'],
                fixed_ram_serial_dma_formula_cycles=serial_formula,
                fixed_ram_formula_verified=serial_formula==candidate['native']['dma_cycles']
                    if not candidate['prefetch'] else None,
                formula_scope='fixed native RAM and serialized DMA only; not board SDRAM latency')
        result['models'][model]=dict(candidates=rows,selected=record.get('selected',{}).get('B2',{}).get('candidate'))
    save(output,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'work/phase6/matched-baselines-v1/b1b2-final-v1/report.json')
    parser.add_argument('--output',type=Path,default=ROOT/'work/phase6/matched-baselines-v1/b2-liveness-opportunities.json')
    args=parser.parse_args();r=run(args.source,args.output)
    print(json.dumps({m:{'selected':v['selected'],'candidate_count':len(v['candidates']),
        'maximum_fitting_saved_bytes':max((c['fitting_potential_removed_bytes'] for c in v['candidates'].values()),default=0),
        'selected_candidate':v['candidates'].get(v['selected'])} for m,v in r['models'].items()},indent=2))
