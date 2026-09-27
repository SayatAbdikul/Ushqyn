#!/usr/bin/env python3
"""Analyze immutable next-stage DMA loads that fit beyond the active SRAM live range.

This first gate leaves descriptors, models, and commands unchanged. It
identifies transfers that the hardware sequencer can legally overlap without
moving a live operand or relying on size alone.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

from variants import ROOT, check_frozen, sha
from run_screening import save_json, verify_files
from run_boardless import load_model
from integer_reference import evaluate
from scheduler.resident_verify import replay_resident

BASE=ROOT/'work/phase6/constant-sibling-v1/fixtures'
OUT=ROOT/'work/phase6/prefetch-tail-v1/analysis.json'
OUTPUT=ROOT/'work/phase6/prefetch-tail-v1'


def commands(blob):
    if len(blob)%16:raise ValueError('partial command')
    return [struct.unpack_from('<BBHIII',blob,16*i) for i in range(len(blob)//16)]


def analyze_fixture(fixture):
    model=fixture.name.split('-')[0]
    schedule=json.loads((fixture/'schedule.json').read_text())
    code=commands((fixture/'commands.bin').read_bytes())
    immutable={(t['ext'],t['sram'],t['bytes']):t['role']
               for s in schedule['stages'] for t in s['loads']
               if t['role'] in ('descriptor','parameter') and t['direction']=='to_sram'}
    runs=[i for i,c in enumerate(code) if c[0]==2]
    candidates=[];all_immutable=[]
    for run_index,next_run in zip(runs,runs[1:]):
        current=code[run_index]
        live_base=current[4]&65535;live_end=current[4]>>16
        if code[run_index+1][0]!=3 or not(code[run_index+1][1]&1):
            raise ValueError('engine run without immediate wait')
        for index in range(run_index+2,next_run):
            op,flags,reserved,ext,sram,nbytes=code[index]
            if op!=1 or flags!=1 or (ext,sram,nbytes) not in immutable:
                continue
            if index+1>=next_run or code[index+1][:2]!=(3,2):
                continue
            role=immutable[(ext,sram,nbytes)]
            record=dict(current_run_command=run_index,next_run_command=next_run,
                dma_command=index,role=role,ext=ext,sram=sram,bytes=nbytes,
                current_live=[live_base,live_end],
                in_spare_tail=sram>=live_end and sram+nbytes<=32768,
                in_spare_head=sram+nbytes<=live_base)
            intervening=[j for j in range(run_index+1,next_run) if j!=index and code[j][0]==1
                         and code[j][1]==1 and sram<code[j][4]+code[j][5]
                         and code[j][4]<sram+nbytes]
            record['overlapping_intervening_writes']=intervening
            all_immutable.append(record)
            if (record['in_spare_tail'] or record['in_spare_head']) and not intervening:
                candidates.append(record)
    return dict(model=model,fixture=str(fixture.relative_to(ROOT)),
        fixture_commands_sha256=sha(fixture/'commands.bin'),
        fixture_schedule_sha256=sha(fixture/'schedule.json'),
        total_immutable_next_load_bytes=sum(x['bytes'] for x in all_immutable),
        immediately_legal_prefetch_bytes=sum(x['bytes'] for x in candidates),
        immediately_legal_count=len(candidates),
        by_role={r:sum(x['bytes'] for x in candidates if x['role']==r)
                 for r in ('descriptor','parameter')},
        candidates=candidates)


def run():
    check_frozen()
    models={m:analyze_fixture(BASE/f'{m}-pinned-constant-sibling-timed')
            for m in ('kws','vww')}
    report=dict(status='analysis',physical_board=False,models=models,
        scope='byte-identical existing addresses and immutable descriptor/parameter transfers; no relocation')
    OUT.parent.mkdir(parents=True,exist_ok=True)
    save_json(OUT,report)
    print(json.dumps({m:{k:r[k] for k in ('total_immutable_next_load_bytes',
        'immediately_legal_prefetch_bytes','immediately_legal_count','by_role')}
        for m,r in models.items()},indent=2))
    return report


def reorder(code, selected):
    runs=[i for i,c in enumerate(code) if c[0]==2]
    chosen={x['dma_command'] for x in selected}
    if any(i+1>=len(code) or code[i+1][:2]!=(3,2) for i in chosen):
        raise ValueError('selected DMA has no WAIT_DMA')
    order=[];cursor=0
    for here,nxt in zip(runs,runs[1:]):
        order.extend(range(cursor,here+1))
        moved=[k for i in sorted(chosen) if here<i<nxt for k in (i,i+1)]
        order.extend(moved)
        move_set=set(moved)
        order.extend(i for i in range(here+1,nxt) if i not in move_set)
        cursor=nxt
    order.extend(range(cursor,len(code)))
    if sorted(order)!=list(range(len(code))):
        raise ValueError('command reordering lost or duplicated commands')
    return [code[i] for i in order],{old:new for new,old in enumerate(order)}


def prepare():
    check_frozen()
    baseline_manifest=ROOT/'work/phase6/constant-sibling-v1/fixtures.json'
    source=json.loads(baseline_manifest.read_text())
    if source['status']!='passed-replay':raise ValueError('baseline replay missing')
    records=[]
    for entry in source['fixtures']:
        name=entry['name'];model=entry['model'];sample=entry['sample']
        fixture=BASE/name
        verify_files(fixture,entry['files'])
        analysis=analyze_fixture(fixture)
        old=commands((fixture/'commands.bin').read_bytes())
        changed,mapping=reorder(old,analysis['candidates'])
        code=b''.join(struct.pack('<BBHIII',*cmd) for cmd in changed)
        original=json.loads((fixture/'schedule.json').read_text())
        if not all(str(i) in original['run_contracts'] for i,c in enumerate(old) if c[0]==2):
            raise ValueError('uncontracted baseline RUN')
        contracts={str(mapping[int(k)]):v for k,v in original['run_contracts'].items()}
        constants={str(mapping[int(k)]):v for k,v in original.get('constant_contracts',{}).items()}
        schedule=dict(original,run_contracts=contracts,constant_contracts=constants,
            program_sha256=hashlib.sha256(code).hexdigest(),
            catalogue='exact immutable tail DMA prefetch with byte-identical descriptors',
            tail_prefetch=dict(bytes=analysis['immediately_legal_prefetch_bytes'],
                               transfers=analysis['immediately_legal_count'],
                               moved_old_command_indices=[x['dma_command'] for x in analysis['candidates']]))
        program,pinned,_,_=load_model(model)
        x=(pinned if sample=='pinned' else
           np.random.default_rng(6073).integers(-128,128,pinned.shape,dtype=np.int8))
        oracle=evaluate(program,{program.inputs[0]:x})
        payload=(fixture/'payload.bin').read_bytes()
        result=replay_resident(program,code,payload,{program.inputs[0]:x},
            run_contracts=contracts,constant_contracts=constants,
            final_output=schedule['final_output'],
            snapshot_regions=schedule['snapshot_regions'],oracle=oracle)
        dest=OUTPUT/'fixtures'/name.replace('constant-sibling','constant-sibling-tail-prefetch')
        dest.mkdir(parents=True,exist_ok=True)
        for filename in entry['files']:
            data=(fixture/filename).read_bytes()
            if filename=='commands.bin':data=code
            if filename=='schedule.json':data=(json.dumps(schedule,indent=2,sort_keys=True)+'\n').encode()
            (dest/filename).write_bytes(data)
        records.append(dict(name=dest.name,model=model,sample=sample,
                            source_fixture=name,verification=result,
                            prefetch=analysis['immediately_legal_prefetch_bytes'],
                            files={p:sha(dest/p) for p in entry['files']}))
    manifest=dict(schema=1,status='passed-replay',physical_board=False,
        source_manifest_sha256=sha(baseline_manifest),
        source_payloads_identical=True,source_descriptors_identical=True,
        fixtures=records)
    save_json(OUTPUT/'fixtures.json',manifest)
    check_frozen()
    print(json.dumps({r['name']:r['prefetch'] for r in records},indent=2))
    return manifest


def native():
    check_frozen()
    manifest_path=OUTPUT/'fixtures.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest['status']!='passed-replay':raise ValueError('prefetch replay missing')
    executable=ROOT/'work/phase6/experiments-v1/all-exact/native/Vv2_tiled_host_bridge'
    baseline=json.loads((ROOT/'work/phase6/constant-sibling-v1/native-all-exact/report.json').read_text())
    if baseline['status']!='passed':raise ValueError('baseline native proof missing')
    target=OUTPUT/'native-all-exact';target.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',physical_board=False,
        executable_sha256=sha(executable),fixture_manifest_sha256=sha(manifest_path),results=[])
    for item in manifest['fixtures']:
        fixture=OUTPUT/'fixtures'/item['name']
        verify_files(fixture,item['files'])
        for seed in (0,6063):
            path=target/f'{item["name"]}-s{seed}.json'
            subprocess.run([str(executable),str(fixture),str(seed),str(path)],check=True)
            row=json.loads(path.read_text())
            row.update(fixture=item['name'],stall_seed=seed,prefetch_bytes=item['prefetch'])
            report['results'].append(row)
            save_json(target/'report.json',report)
    report['status']='passed'
    comparisons={}
    for model in ('kws','vww'):
        for seed in (0,6063):
            cand=next(r for r in report['results'] if r['fixture']==f'{model}-pinned-constant-sibling-tail-prefetch-timed' and r['stall_seed']==seed)
            base=next(r for r in baseline['results'] if r['fixture']==f'{model}-pinned-constant-sibling-timed' and r['stall_seed']==seed)
            comparisons[f'{model}-s{seed}']=dict(baseline_cycles=base['elapsed_cycles'],
                candidate_cycles=cand['elapsed_cycles'],speedup=base['elapsed_cycles']/cand['elapsed_cycles'],
                overlapped_cycles=cand['overlap_cycles'],prefetch_bytes=cand['prefetch_bytes'])
    report['comparisons']=comparisons
    save_json(target/'report.json',report)
    check_frozen()
    print(json.dumps(comparisons,indent=2,sort_keys=True))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('analyze','prepare','native'),nargs='?',default='analyze')
    {'analyze':run,'prepare':prepare,'native':native}[parser.parse_args().stage]()
