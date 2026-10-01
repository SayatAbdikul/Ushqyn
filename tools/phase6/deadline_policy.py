#!/usr/bin/env python3
"""Budgeted interruption points with native timing and explicit resident costs.

This is a utility/feasibility experiment, not a novelty claim. Original,
uniformly split and selectively split streams receive the same sparse point
optimizer, conservative live-granule checkpoints and resident controller model.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import resource
import struct
import subprocess
import sys
import time

from deadline_screen import (ROOT, REFERENCE, PARENT_SHA, MEM, sha, save,
    commands, pack, analyze_fixture, segments, Descriptor)
from deadline_pause import word_intervals

BASE = ROOT/'work/phase6/deadline-policy-v1'
SCREEN = ROOT/'work/phase6/deadline-screen-v1'
MODELS = ('kws','vww','ad')
VARIANTS = ('baseline','split25000','split50000','selective-long25000')
SEEDS = (0,6063)


def freeze_protocol():
    """Write before policy/native candidate evaluation; deadlines use originals."""
    p = json.loads((SCREEN/'profiles.json').read_text())
    costs={m:p[m]['baseline']['0']['native_cycles'] for m in MODELS}
    cases=[]
    for background in MODELS:
        for urgent in MODELS:
            if urgent == background: continue
            cb,cu=costs[background],costs[urgent]
            for i in range(1,25):
                fraction=i/25
                for slack in (1.10,1.25,1.50,2.00):
                    release=round(cb*fraction)
                    rows=[(background,0,4*cb+2*cu,0),
                          (urgent,release,release+math.ceil(cu*slack),1)]
                    cases.append(dict(name=f'one-{background}-{urgent}-f{i}-d{slack}',
                        background=background,urgent=urgent,kind='one',releases=rows))
            for utilization in (0.35,0.55,0.70):
                pb=math.ceil(2*cb/utilization);pu=math.ceil(2*cu/utilization)
                horizon=4*max(pb,pu)
                for phase in (0,0.37,0.73):
                    for slack in (1.10,1.25,1.50,2.00):
                        rows=[];ident=0
                        for m,period,offset,relative in (
                            (background,pb,0,pb),
                            (urgent,pu,round(cb*phase),math.ceil(cu*slack))):
                            for release in range(offset,horizon,period):
                                rows.append((m,release,release+relative,ident));ident+=1
                        cases.append(dict(name=f'periodic-{background}-{urgent}-u{utilization}-p{phase}-d{slack}',
                            background=background,urgent=urgent,kind='periodic',utilization=utilization,releases=rows))
            for count in (2,3):
                for spacing in (1.1,1.5,2.0):
                    rows=[(background,0,4*cb+4*cu,0)]
                    for j in range(count):
                        r=round(cb*0.25+j*cu*spacing)
                        rows.append((urgent,r,r+math.ceil(cu*1.5),j+1))
                    cases.append(dict(name=f'burst-{background}-{urgent}-n{count}-s{spacing}',
                        background=background,urgent=urgent,kind='burst',releases=rows))
    protocol=dict(status='frozen-before-new-policy-measurements',costs=costs,cases=cases,
        criteria='Compare misses, lateness, responses and checkpoint overhead across the complete frozen suite; do not tune on it.',
        policy_selection='Minimize maximum fixed-RAM service interval plus conservative save latency, under resident32KiB code+metadata; ties minimize checkpointcode bytes. Optimize at8cycles/8B and128fixeddispatch cycles.',
        checkpoint_rule='Save and restore every8Bgranule intersecting futurelive SRAM, including backed bytes; all controls same.',
        controller_model='One suspended background job; urgentjobs nonpreemptive and orderedEDF. Repeatedjobs reuse immutablepinnedinput/program after their preceding instance completes.',
        cost_grid=dict(word_cycles=[4,8,16,32],fixed_dispatch_cycles=[128,512]),
        memory_modes={'0':'exact native fixedRAMtrace','6063':'exact native sampledstalledRAMtrace'},
        source_sha256={n:sha(SCREEN/n) for n in ('profiles.json','evidence.json')})
    path=BASE/'protocol.json'
    protocol=json.loads(json.dumps(protocol))
    if path.exists():
        old=json.loads(path.read_text())
        if old!=protocol:raise ValueError('frozen protocol changed')
    else:save(path,protocol)
    return protocol


def native_identity():
    identity=json.loads((SCREEN/'native-build/identity.json').read_text())
    exe=SCREEN/'native-build/Vv2_tiled_host_bridge'
    if identity['engine_sha256']!=PARENT_SHA or sha(exe)!=identity['executable_sha256']:
        raise ValueError('selected native executable changed')
    for name,digest in identity['sources_sha256'].items():
        if sha(name)!=digest:raise ValueError('native source changed '+name)
    return exe,identity


def selective_fixture(source,dest,analysis,trace):
    """Only split RUNs whose original interval exceeds75k; target25k."""
    dest.mkdir(parents=True,exist_ok=True)
    for line in (source/'checks.txt').read_text().splitlines():
        n=line.split()[1];(dest/n).write_bytes((source/n).read_bytes())
    for n in ('input.bin','output.bin','checks.txt'):(dest/n).write_bytes((source/n).read_bytes())
    payload=bytearray((source/'payload.bin').read_bytes());rows=[];changes=[]
    original=analysis['commands']
    for index,row in enumerate(original):
        if row[0]!=2:rows.append(row);continue
        info=analysis['descriptors'][index];d=Descriptor(**info['descriptor'])
        if d.opcode==1:channels,plane=d.outputs,1
        elif d.opcode in (4,6):channels,plane=d.output_c,d.outputs//d.output_c
        else:rows.append(row);continue
        end=next(j for j in range(index+1,len(original)) if original[j][0]==3 and original[j][1]&1)
        duration=trace[end+1]['elapsed']-trace[index]['elapsed']
        if duration<=75000:rows.append(row);continue
        group=max(1,min(channels,int(25000*channels/duration)))
        if d.opcode==6:
            alignment=8//math.gcd(d.input_h*d.input_w,8)
            group=max(alignment,group//alignment*alignment)
        if group>=channels:rows.append(row);continue
        rows.append((3,3,0,0,0,0));groups=[]
        for first in range(0,channels,group):
            count=min(group,channels-first);q=copy.copy(d)
            q.output+=first*plane;q.outputs=count*plane
            q.weight+=first*d.row_stride;q.params+=first*16
            if q.opcode!=1:q.output_c=count
            if q.opcode==6:q.input_c=count;q.input+=first*d.input_h*d.input_w
            q.validate();raw=bytearray(q.encode());raw[6:8]=info['fusion'].to_bytes(2,'little')
            payload.extend(bytes((-len(payload))%8));address=len(payload)
            payload.extend(raw+Descriptor(0).encode())
            rows.extend([(1,1,0,address,row[3],128),(3,2,0,0,0,0),row])
            if first+count<channels:rows.append((3,3,0,0,0,0))
            groups.append(count)
        changes.append(dict(original_command=index,original_native_cycles=duration,groups=groups))
    (dest/'commands.bin').write_bytes(pack(rows));(dest/'payload.bin').write_bytes(payload)
    if len(rows)*16>MEM:raise ValueError('candidate program capacity')
    save(dest/'changes.json',dict(target_cycles=25000,split_only_above_original_cycles=75000,changes=changes))


def prepare_profiles():
    original=json.loads((SCREEN/'profiles.json').read_text());evidence=json.loads((SCREEN/'evidence.json').read_text())
    exe,identity=native_identity();profiles=copy.deepcopy(original);fixtures={};native=[]
    for m in MODELS:
        source=Path(evidence[m]['source_directory']);fixtures[m]={'baseline':str(source)}
        for name,digest in evidence[m]['files_sha256'].items():
            if sha(source/name)!=digest:raise ValueError('source fixture changed')
        for v in ('split25000','split50000'):fixtures[m][v]=str(SCREEN/'fixtures'/f'{m}-{v}')
        dest=BASE/'fixtures'/f'{m}-selective-long25000'
        analysis=analyze_fixture(source)
        trace=json.loads((SCREEN/'native'/f'{m}-baseline-s0.json.trace.json').read_text())
        selective_fixture(source,dest,analysis,trace)
        fixtures[m]['selective-long25000']=str(dest);profiles[m]['selective-long25000']={}
        pa=analyze_fixture(dest)
        for seed in SEEDS:
            report=BASE/'native'/f'{m}-selective-long25000-s{seed}.json';report.parent.mkdir(exist_ok=True)
            if m=='ad' and sha(dest/'commands.bin')==sha(source/'commands.bin') and sha(dest/'payload.bin')==sha(source/'payload.bin'):
                # AD contains no eligible RUN: exact identical originalfixture evidence.
                profiles[m]['selective-long25000'][str(seed)]=copy.deepcopy(original[m]['baseline'][str(seed)])
                native.append(dict(model=m,seed=seed,status='reused-identical-baseline',commands_sha256=sha(dest/'commands.bin')))
                continue
            cp=subprocess.run([str(exe),str(dest),str(seed),str(report)],cwd=ROOT,capture_output=True,text=True,check=True,timeout=180)
            result=json.loads(report.read_text());tt=json.loads(Path(str(report)+'.trace.json').read_text())
            if result['status']!='passed' or len(tt)!=len(pa['commands'])+1:raise ValueError('candidate native failure')
            ss=segments(pa,tt);profiles[m]['selective-long25000'][str(seed)]={
                'segments':ss,'program_bytes':pa['program_bytes'],'native_cycles':result['elapsed_cycles']}
            native.append(dict(model=m,seed=seed,result=result,stdout=cp.stdout.strip(),
                commands_sha256=sha(dest/'commands.bin'),payload_sha256=sha(dest/'payload.bin')))
            print(m,seed,result['elapsed_cycles'],'nativepassed',flush=True)
    save(BASE/'profiles.json',profiles);save(BASE/'fixtures.json',fixtures)
    save(BASE/'native-evidence.json',dict(identity=identity,runs=native))
    return profiles,fixtures


def sites(profile):
    out=[];elapsed=0
    for segment in profile['segments']:
        elapsed+=segment['cycles'];spans=word_intervals(segment.get('live_intervals',[]))
        live=sum(n for _,n in spans)
        if live!=segment['live_dma_bytes']:raise ValueError('granule span mismatch')
        # Per-site two precompiled DMA+WAIT+HALT programs and16B metadata.
        code=2*(2*len(spans)+1)*16
        out.append(dict(next_index=segment['next_index'],elapsed=elapsed,
            live_dma_bytes=live,spans=spans,checkpoint_code_bytes=code,
            resident_site_bytes=code+16))
    # Completion needs no save/restore table/program: the controller simply HALTs.
    out[-1].update(resident_site_bytes=0,checkpoint_code_bytes=0,live_dma_bytes=0,spans=[])
    return out


def plan_at_cap(ss,budget,cap,word=8):
    """Exact shortest path for minimum residentbytes given blocking surrogatecap."""
    count=len(ss);dp=[math.inf]*count;pred=[None]*count
    for j,s in enumerate(ss):
        end=s['elapsed'];savecost=word*(s['live_dma_bytes']//8)+32*len(s['spans'])+16 if j<count-1 else 0
        if end+savecost<=cap:dp[j]=s['resident_site_bytes'];pred[j]=-1
        for i in range(j-1,-1,-1):
            if end-ss[i]['elapsed']+savecost>cap:break
            proposal=dp[i]+s['resident_site_bytes']
            if proposal<dp[j]:dp[j]=proposal;pred[j]=i
    if dp[-1]>budget:return None
    chain=[];at=count-1
    while at!=-1:
        chain.append(at);at=pred[at]
    chain.reverse()
    return dict(site_indices=chain[:-1],resident_checkpoint_bytes=int(dp[-1]),blocking_surrogate_cap_cycles=cap)


def optimize_points(profile,urgent_program_bytes):
    # Two resident stress-only clobbercommands are reserved in every pair.
    urgent_program_bytes+=32
    ss=sites(profile);budget=MEM-profile['program_bytes']-urgent_program_bytes-16
    if budget<0:return dict(feasible=False,reason='jobprograms+header exceed32KiB',base_program_bytes=profile['program_bytes']+urgent_program_bytes+16)
    lo,hi=0,profile['native_cycles']
    while lo<hi:
        mid=(lo+hi)//2
        if plan_at_cap(ss,budget,mid):hi=mid
        else:lo=mid+1
    selected=plan_at_cap(ss,budget,lo)
    pareto=[]
    for factor in (1,1.25,1.5,2):
        r=plan_at_cap(ss,budget,math.ceil(lo*factor))
        if r:
            r.update(site_count=len(r['site_indices']),factor=factor);pareto.append(r)
    selected.update(feasible=True,budget_bytes=budget,site_count=len(selected['site_indices']),
        total_resident_bytes=profile['program_bytes']+urgent_program_bytes+16+selected['resident_checkpoint_bytes'],
        metadata_bytes=16*(1+len(selected['site_indices'])),
        max_live_dma_bytes=max([ss[i]['live_dma_bytes'] for i in selected['site_indices']] or [0]),
        checkpoint_program_bytes=sum(ss[i]['checkpoint_code_bytes'] for i in selected['site_indices']),pareto=pareto,
        selected_next_indices=[ss[i]['next_index'] for i in selected['site_indices']])
    return selected


def coarsen(profile,selected):
    permitted=set(selected['selected_next_indices']);result=[];duration=0
    for i,s in enumerate(profile['segments']):
        duration+=s['cycles']
        if s['next_index'] in permitted or i==len(profile['segments'])-1:
            q=dict(s,cycles=duration,spans=word_intervals(s.get('live_intervals',[])))
            result.append(q);duration=0
    if duration or sum(s['cycles'] for s in result)!=profile['native_cycles']:raise ValueError('coarsening conservation')
    return result


def simulate(bg,urgent,case,word,fixed,selected_next_indices=None,scan_record_cycles=5):
    """Same resident controller: one pausedbackground, EDF amongurgentjobs."""
    jobs=[dict(model=m,release=r,deadline=d,id=i,index=0,completion=None) for m,r,d,i in case['releases']]
    jobs.sort(key=lambda j:(j['release'],j['id']));now=0;cursor=0;ready=[];owner=None
    overhead=0;switches=0;save_bytes=0;restore_bytes=0;max_active_background=0
    bgname=case['background'];service=0;serialized_background=False;scan_cycles=0;scan_attempts=0;scan_matches=0
    table_positions=({pc:i+1 for i,pc in enumerate(selected_next_indices)} if selected_next_indices is not None else None)
    while cursor<len(jobs) or ready:
        while cursor<len(jobs) and jobs[cursor]['release']<=now:ready.append(jobs[cursor]);cursor+=1
        if not ready:now=jobs[cursor]['release'];continue
        bgjobs=[j for j in ready if j['model']==bgname]
        max_active_background=max(max_active_background,len(bgjobs))
        urgentjobs=[j for j in ready if j['model']!=bgname]
        # New backgroundinstances queue behind pausedactiveinstance (onecontext).
        activebg=next((j for j in bgjobs if j['index']),None)
        if activebg and len(bgjobs)>1:serialized_background=True
        if urgentjobs:chosen=min(urgentjobs,key=lambda j:(j['deadline'],j['release'],j['id']))
        else:chosen=activebg or min(bgjobs,key=lambda j:(j['deadline'],j['release'],j['id']))
        if (table_positions is not None and owner is not None and owner['completion'] is None and
            owner['model']==bgname and owner['index'] and bg[owner['index']-1]['next_index'] not in table_positions):
            chosen=owner  # A pending request remains latched until a listed WAIT.
        if chosen is not owner:
            cost=fixed
            if owner is not None and owner['completion'] is None:
                if owner['model']!=bgname:raise ValueError('urgentjob mustbe nonpreemptive')
                q=bg[owner['index']-1];n=q['live_dma_bytes'];save_bytes+=n
                cost+=n//8*word+32*len(q['spans'])+16
            if chosen['index']:
                q=bg[chosen['index']-1];n=q['live_dma_bytes'];restore_bytes+=n
                cost+=n//8*word+32*len(q['spans'])+16
            overhead+=cost;now+=cost;switches+=1
        owner=chosen
        if chosen['model']==bgname:q=bg[chosen['index']];duration=q['cycles'];chosen['index']+=1;done=chosen['index']==len(bg)
        else:duration=urgent['native_cycles'];chosen['index']=1;done=True
        now+=duration;service+=duration
        if done:chosen['completion']=now;ready.remove(chosen)
        elif (table_positions is not None and chosen['model']==bgname and
              (any(j['model']!=bgname for j in ready) or
               any(j['model']!=bgname and j['release']<=now for j in jobs[cursor:]))):
            # SRAM has an independent always-ready synchronous program port.
            # Header/record each traverse exactly five metadata FSM states.
            # At a match, original ADVANCE is skipped: charging all five states
            # is a conservative one-cycle overestimate of net branch latency.
            pc=bg[chosen['index']-1]['next_index'];position=table_positions.get(pc)
            extra=scan_record_cycles*(1+(position if position is not None else len(table_positions)))
            now+=extra;overhead+=extra;scan_cycles+=extra;scan_attempts+=1;scan_matches+=position is not None
    misses=sum(j['completion']>j['deadline'] for j in jobs)
    expected=sum(sum(s['cycles'] for s in bg) if j['model']==bgname else urgent['native_cycles'] for j in jobs)
    if expected!=service or any(j['completion'] is None for j in jobs):raise ValueError('schedule conservation')
    return dict(misses=misses,jobs=len(jobs),finish_cycles=now,switches=switches,overhead_cycles=overhead,
        save_bytes=save_bytes,restore_bytes=restore_bytes,service_cycles=service,
        max_lateness_cycles=max(j['completion']-j['deadline'] for j in jobs),
        max_response_cycles={m:max(j['completion']-j['release'] for j in jobs if j['model']==m) for m in (bgname,case['urgent'])},
        one_background_context_max_queued_jobs=max_active_background,background_queue_serialized=serialized_background,
        metadata_scan_cycles=scan_cycles,metadata_scan_attempts=scan_attempts,metadata_scan_matches=scan_matches,
        completions=[[j['id'],j['completion']] for j in jobs])


def pairwise_summary(records):
    grouped={}
    for r in records:
        k=(r['case'],r['seed'],r['word_cycles'],r['fixed_cycles']);grouped.setdefault(k,{})[r['variant']]=r['result']
    comparisons={}
    for control in ('baseline','split25000','split50000'):
        pairs=[(g[control],g['selective-long25000']) for g in grouped.values() if control in g and 'selective-long25000' in g]
        comparisons[control]=dict(cases=len(pairs),selective_fewer_misses=sum(b['misses']<a['misses'] for a,b in pairs),
            selective_more_misses=sum(b['misses']>a['misses'] for a,b in pairs),same_misses=sum(b['misses']==a['misses'] for a,b in pairs),
            selective_zero_miss_rescues=sum(a['misses']>0 and b['misses']==0 for a,b in pairs),
            control_zero_miss_rescues=sum(b['misses']>0 and a['misses']==0 for a,b in pairs))
    return comparisons


def calibrate(profiles,protocol):
    """Frozen suite replay with exact metadata-state latency at each drainedWAIT."""
    plans=json.loads((BASE/'plans.json').read_text());records=[]
    for case in protocol['cases']:
        b,u=case['background'],case['urgent'];key=f'{b}-{u}'
        for seed in SEEDS:
            for word in protocol['cost_grid']['word_cycles']:
                for fixed in protocol['cost_grid']['fixed_dispatch_cycles']:
                    for variant in VARIANTS:
                        plan=plans[key][variant]
                        if not plan['feasible']:continue
                        profile=profiles[b][variant][str(seed)]
                        bg=[dict(s,spans=word_intervals(s.get('live_intervals',[]))) for s in profile['segments']]
                        result=simulate(bg,profiles[u]['baseline'][str(seed)],case,word,fixed,
                            selected_next_indices=plan['selected_next_indices'])
                        records.append(dict(case=case['name'],kind=case['kind'],background=b,urgent=u,seed=seed,
                            word_cycles=word,fixed_cycles=fixed,variant=variant,result=result))
    save(BASE/'scan-calibrated-results.json',records)
    summary={}
    for variant in VARIANTS:
        rr=[r['result'] for r in records if r['variant']==variant]
        summary[variant]=dict(cases=len(rr),total_misses=sum(r['misses'] for r in rr),zero_miss_cases=sum(r['misses']==0 for r in rr),
            max_metadata_scan_cycles=max(r['metadata_scan_cycles'] for r in rr),
            total_metadata_scan_cycles=sum(r['metadata_scan_cycles'] for r in rr),
            mean_overhead_cycles=sum(r['overhead_cycles'] for r in rr)/len(rr))
    source=ROOT/'work/phase6/deadline-resident-v1/tile_sequencer.sv'
    text=source.read_text()
    expected=('META_FETCH0: if(memory_ready)state<=META_READ0;',
              'META_READ0: if(memory_rvalid)begin metadata_low<=memory_rdata;state<=META_FETCH1;end',
              'META_FETCH1: if(memory_ready)state<=META_READ1;',
              'META_READ1: if(memory_rvalid)begin metadata_high<=memory_rdata;state<=META_EXECUTE;end')
    if any(site not in text for site in expected):raise ValueError('metadata timing source changed')
    memory=ROOT/'rtl/v2/scratchpad.sv'
    if "assign ready=1'b1;" not in memory.read_text() or 'rvalid<=req && !wr;' not in memory.read_text():raise ValueError('programport latency changed')
    checked=audit_completion_records(records,protocol,profiles)
    report=dict(status='completed-frozen-fine-grained-scan-replay',records=len(records),cases=len(protocol['cases']),
        pairwise=pairwise_summary(records),summary=summary,completion_reconstruction_checks=checked,
        program_record_scan_cycles=5,scan_match_net_latency_overestimate_cycles=1,
        assumptions=['Every original drained WAIT remains present in the replay; pending urgent requests scan the immutable fulltable or prefix to firstmatch.',
            'All policies use the same resident metadata FSM and live-granule checkpoint rule.',
            'Metadata FSM states/programport latency are source-derived and corroborated by controllernative scan counters.',
            'Model includes scan cycles but retains explicit DMA/dispatch sensitivity costs, isolated native memorytraces and a one-background-context queue.',
            'This is not a native replay of the whole job arrival grid or a physical deadline guarantee.'],
        source_sha256={str(source):sha(source),str(memory):sha(memory)},
        protocol_sha256=sha(BASE/'protocol.json'),plans_sha256=sha(BASE/'plans.json'),profiles_sha256=sha(BASE/'profiles.json'),
        raw_records_sha256=sha(BASE/'scan-calibrated-results.json'),driver_sha256=sha(__file__),python_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save(BASE/'scan-calibrated-report.json',report)
    main=json.loads((BASE/'report.json').read_text());main['scan_calibrated']=report;main['driver_sha256']=sha(__file__)
    main['python_max_rss_bytes']=max(main['python_max_rss_bytes'],report['python_max_rss_bytes']);save(BASE/'report.json',main)
    with (BASE/'README.md').open('a') as f:
        f.write('\nA follow-up fine-grained replay preserves every drained WAIT and charges five cycles per metadata header/record '
            'only while a request is pending, including full unsuccessful table scans. It preserves the same frozen candidates '
            'and 828 arrival cases. Matching scans conservatively overcharge net branching by one cycle. See scan-calibrated-report.json '
            'and scan-calibrated-results.json. DMA and fixed dispatch remain explicit sensitivity costs; this adds the actual lookup mechanism '
            'without claiming native validation of all schedules.\n\n'+json.dumps(report['pairwise'],indent=2)+'\n')
    return report


def audit_calibration(profiles,protocol):
    """Preserve native lookup counters and reconstruct the frozen raw summaries."""
    plans=json.loads((BASE/'plans.json').read_text())
    records=json.loads((BASE/'scan-calibrated-results.json').read_text())
    report=json.loads((BASE/'scan-calibrated-report.json').read_text())
    if pairwise_summary(records)!=report['pairwise']:raise ValueError('calibrated pairwise summary')
    for name in ('protocol','plans','profiles'):
        if sha(BASE/(name+'.json'))!=report[name+'_sha256']:raise ValueError('calibration selection identity')
    checked=audit_completion_records(records,protocol,profiles)
    witnesses=[];dest=BASE/'scan-native-witnesses';dest.mkdir(exist_ok=True)
    for variant in VARIANTS:
        for seed in SEEDS:
            name=f'vww-ad-{variant}-s{seed}-m0.json'
            source=ROOT/'work/phase6/deadline-resident-v1/native'/name
            native=json.loads(source.read_text())
            (dest/name).write_bytes(source.read_bytes())
            profile=profiles['vww'][variant][str(seed)]
            bg=[dict(s,spans=word_intervals(s.get('live_intervals',[]))) for s in profile['segments']]
            release=int(0.8*2014205)
            case=dict(background='vww',urgent='ad',releases=[('vww',0,9999999,0),('ad',release,release+353303,1)])
            modeled=simulate(bg,profiles['ad']['baseline'][str(seed)],case,4,0,
                selected_next_indices=plans['vww-ad'][variant]['selected_next_indices'])
            if modeled['metadata_scan_cycles']!=native['metadata_scan_cycles']:raise ValueError('native scan calibration mismatch')
            witnesses.append(dict(variant=variant,seed=seed,native_scan_cycles=native['metadata_scan_cycles'],
                modeled_scan_cycles=modeled['metadata_scan_cycles'],modeled_scan_attempts=modeled['metadata_scan_attempts'],
                preserved_native_report=str(dest/name),sha256=sha(dest/name),native_report_origin=str(source)))
    scancheck=sum(r['result']['metadata_scan_cycles']%5==0 for r in records)
    if scancheck!=len(records):raise ValueError('metadata five-state granularity')
    audit=dict(status='passed-independent-raw-completion-and-native-scan-audit',
        completion_reconstruction_checks=checked,native_scan_counter_matches=witnesses,
        five_state_granularity_checks=scancheck,raw_records_sha256=sha(BASE/'scan-calibrated-results.json'),
        policy_selection_sha256=sha(BASE/'selection-identity.json'),driver_sha256=sha(__file__))
    save(BASE/'scan-calibrated-audit.json',audit)
    report['native_scan_corroboration']=witnesses;report['audit']=audit;report['driver_sha256']=sha(__file__)
    save(BASE/'scan-calibrated-report.json',report)
    main=json.loads((BASE/'report.json').read_text());main['scan_calibrated']=report;main['driver_sha256']=sha(__file__)
    main['python_max_rss_bytes']=max(main['python_max_rss_bytes'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save(BASE/'report.json',main)
    return audit


def combined_fixture(bgname,urgentname,variant,plan,fixtures,profile):
    """Real residentpair + all sparsecheckpointprograms + compactmetadata."""
    bgsrc=Path(fixtures[bgname][variant]);usrc=Path(fixtures[urgentname]['baseline'])
    dest=BASE/'resident'/f'{bgname}-{urgentname}-{variant}';dest.mkdir(parents=True,exist_ok=True)
    payload=bytearray((bgsrc/'payload.bin').read_bytes());inp=(bgsrc/'input.bin').read_bytes();payload[:len(inp)]=inp
    payload.extend(bytes((-len(payload))%8));ubase=len(payload)
    upayload=bytearray((usrc/'payload.bin').read_bytes());inp=(usrc/'input.bin').read_bytes();upayload[:len(inp)]=inp
    payload.extend(upayload);payload.extend(bytes((-len(payload))%8));contextbase=len(payload);payload.extend(bytes(MEM))
    clobberbase=len(payload);payload.extend(b'\xa5'*MEM)
    brows=commands((bgsrc/'commands.bin').read_bytes());urows=commands((usrc/'commands.bin').read_bytes())
    urows=[r if r[0]!=1 else tuple(list(r[:3])+[r[3]+ubase]+list(r[4:])) for r in urows]
    urgententry=len(brows);rows=brows+[(1,1,0,clobberbase,0,MEM),(3,2,0,0,0,0)]+urows;records=[];ss=sites(profile)
    for i in plan['site_indices']:
        s=ss[i];saveentry=len(rows)
        for base,length in s['spans']:rows.extend([(1,0,0,contextbase+base,base,length),(3,2,0,0,0,0)])
        rows.append((0,0,0,0,0,0));restoreentry=len(rows)
        for base,length in s['spans']:rows.extend([(1,1,0,contextbase+base,base,length),(3,2,0,0,0,0)])
        rows.append((0,0,0,0,0,0));records.append(dict(next_index=s['next_index'],global_next_index=s['next_index'],save_entry=saveentry,restore_entry=restoreentry,job_id=0,spans=s['spans'],live_dma_bytes=s['live_dma_bytes']))
    program=pack(rows);offset=len(program);metadata=struct.pack('<IIII',urgententry,len(records),0,0)
    metadata+=b''.join(struct.pack('<IIII',r['global_next_index'],r['save_entry'],r['restore_entry'],r['job_id']) for r in records)
    program+=metadata
    if len(program)!=plan['total_resident_bytes'] or len(program)>MEM:raise ValueError('resident capacityaccounting')
    (dest/'commands.bin').write_bytes(program);(dest/'metadata.bin').write_bytes(metadata);(dest/'payload.bin').write_bytes(payload)
    checks=[]
    for label,source,external in ((bgname,bgsrc,0),(urgentname,usrc,ubase)):
        # Keep every source oracle (classificationusually one finaloutput).
        for k,line in enumerate((source/'checks.txt').read_text().splitlines()):
            address,name=line.split();new=f'{label}-{k}-{name}';(dest/new).write_bytes((source/name).read_bytes());checks.append(f'{int(address)+external} {new}')
    (dest/'checks.txt').write_text('\n'.join(checks)+'\n')
    manifest=dict(status='resident-program-feasible',background=bgname,urgent=urgentname,variant=variant,
        metadata_layout='header<IIII:urgent_entry,count,background_job_id,reserved>; record<IIII:global_nextPC,save_entry,restore_entry,job_id>',
        metadata_offset_bytes=offset,metadata_entry_word=offset//8,metadata_bytes=len(metadata),
        urgent_entry=urgententry,actual_urgent_entry=urgententry+2,background_entry=0,context_external_base=contextbase,urgent_external_base=ubase,
        clobber_external_base=clobberbase,clobber_stress_program_bytes=32,
        resident_program_bytes=len(program),payload_bytes=len(payload),checkpoints=records,
        fixture_sha256={n:sha(dest/n) for n in ('commands.bin','metadata.bin','payload.bin','checks.txt')},
        model_program_commands={bgname:len(brows),urgentname:len(urows)},source_sha256={str(s/n):sha(s/n) for s in (bgsrc,usrc) for n in ('commands.bin','payload.bin','checks.txt')})
    save(dest/'policy-manifest.json',manifest)
    return manifest,str(dest)


def audit_resident(folder):
    """Reconstruct metadata/DMA programs from serialized bytes independently."""
    folder=Path(folder);manifest=json.loads((folder/'policy-manifest.json').read_text())
    raw=(folder/'commands.bin').read_bytes();metadata=(folder/'metadata.bin').read_bytes()
    if len(raw)>MEM or len(raw)!=manifest['resident_program_bytes']:raise ValueError('capacity')
    offset=manifest['metadata_offset_bytes']
    if offset%16 or raw[offset:]!=metadata:raise ValueError('metadata offset')
    header=struct.unpack('<IIII',metadata[:16])
    if header!=(manifest['urgent_entry'],len(manifest['checkpoints']),0,0):raise ValueError('header')
    rows=list(struct.iter_unpack('<BBHIII',raw[:offset]));checks=0
    for i,r in enumerate(manifest['checkpoints']):
        if struct.unpack('<IIII',metadata[16+i*16:32+i*16])!=(r['global_next_index'],r['save_entry'],r['restore_entry'],0):raise ValueError('table entry')
        if not 0<r['global_next_index']<manifest['model_program_commands'][manifest['background']]:raise ValueError('resume index')
        for entry,restore in ((r['save_entry'],0),(r['restore_entry'],1)):
            expected=[]
            for base,length in r['spans']:
                if base%8 or length%8 or not 0<=base<base+length<=MEM:raise ValueError('granule span')
                expected.extend([(1,restore,0,manifest['context_external_base']+base,base,length),(3,2,0,0,0,0)])
            expected.append((0,0,0,0,0,0))
            if rows[entry:entry+len(expected)]!=expected:raise ValueError('checkpoint program')
            checks+=1
        if sum(length for _,length in r['spans'])!=r['live_dma_bytes']:raise ValueError('live bytecount')
    for name,digest in manifest['fixture_sha256'].items():
        if sha(folder/name)!=digest:raise ValueError('serialized identity')
    return dict(folder=str(folder),status='passed',resident_bytes=len(raw),checkpoint_programs_checked=checks,
        metadata_records_checked=len(manifest['checkpoints']))


def audit_small_dp():
    """Independent exhaustive subset oracle across synthetic unequal contexts."""
    cases=0
    for times in ((10,20,35,60),(7,31,32,70),(20,40,60,80)):
        for costs in ((48,80,112,0),(240,48,160,0)):
            ss=[dict(elapsed=t,live_dma_bytes=(i+1)*8,spans=[(0,(i+1)*8)],resident_site_bytes=c) for i,(t,c) in enumerate(zip(times,costs))]
            ss[-1].update(live_dma_bytes=0,spans=[])
            for cap in (25,50,75,100,150):
                for budget in (0,100,250,1000):
                    best=math.inf
                    for bits in itertools.product((False,True),repeat=len(ss)-1):
                        chain=[i for i,keep in enumerate(bits) if keep]+[len(ss)-1]
                        previous=0;valid=True;cost=0
                        for index in chain:
                            s=ss[index];extra=8*s['live_dma_bytes']//8+32*len(s['spans'])+16 if index<len(ss)-1 else 0
                            if s['elapsed']-previous+extra>cap:valid=False
                            previous=s['elapsed'];cost+=s['resident_site_bytes']
                        if valid:best=min(best,cost)
                    actual=plan_at_cap(ss,budget,cap)
                    if (actual is None)!=(best>budget):raise ValueError('DP brute oracle feasibility')
                    if actual and actual['resident_checkpoint_bytes']!=best:raise ValueError('DP brute oracle cost')
                    cases+=1
    return cases


def audit_completion_records(records,protocol,profiles):
    """Derive deadline/response/service summaries from raw job completions."""
    cases={c['name']:c for c in protocol['cases']};checked=0
    for record in records:
        case=cases[record['case']];r=record['result'];done=dict(r['completions'])
        expected_ids={i for _,_,_,i in case['releases']}
        if set(done)!=expected_ids or len(done)!=len(r['completions']):raise ValueError('completion coverage')
        misses=0;lateness=[];response={case['background']:[],case['urgent']:[]};service=0
        for m,release,deadline,identity in case['releases']:
            end=done[identity]
            if end<release:raise ValueError('completion before release')
            misses+=end>deadline;lateness.append(end-deadline);response[m].append(end-release)
            variant=record['variant'] if m==case['background'] else 'baseline'
            service+=profiles[m][variant][str(record['seed'])]['native_cycles']
        if misses!=r['misses'] or max(lateness)!=r['max_lateness_cycles'] or service!=r['service_cycles']:raise ValueError('deadline/service summary')
        if {m:max(v) for m,v in response.items()}!=r['max_response_cycles']:raise ValueError('response summary')
        if max(done.values())!=r['finish_cycles']:raise ValueError('finish summary')
        checked+=1
    return checked


def evaluate(profiles,fixtures,protocol):
    plans={};pareto=[];resident=[]
    for b in MODELS:
        for u in MODELS:
            if b==u:continue
            key=f'{b}-{u}';plans[key]={}
            for v in VARIANTS:
                p=optimize_points(profiles[b][v]['0'],profiles[u]['baseline']['0']['program_bytes']);plans[key][v]=p
                pareto.append(dict(background=b,urgent=u,variant=v,plan=p))
                if (b,u)==('vww','ad') and p['feasible']:
                    manifest,path=combined_fixture(b,u,v,p,fixtures,profiles[b][v]['0']);resident.append(dict(path=path,manifest=manifest))
    save(BASE/'plans.json',plans);save(BASE/'resident-fixtures.json',resident)
    selection_identity=BASE/'selection-identity.json'
    frozen=dict(protocol_sha256=sha(BASE/'protocol.json'),profiles_sha256=sha(BASE/'profiles.json'),
        plans_sha256=sha(BASE/'plans.json'),fixtures_sha256=sha(BASE/'fixtures.json'),driver_sha256=sha(__file__))
    # The candidate choices/arrivals are now immutable before scheduling.
    # Driver refinements to auditing do not alter either frozen selection file.
    if selection_identity.exists():
        old=json.loads(selection_identity.read_text())
        for name in ('protocol_sha256','profiles_sha256','plans_sha256','fixtures_sha256'):
            if old[name]!=frozen[name]:raise ValueError('frozen selection changed '+name)
    else:save(selection_identity,frozen)
    records=[]
    for case in protocol['cases']:
        b,u=case['background'],case['urgent'];key=f'{b}-{u}'
        for seed in SEEDS:
            for word in protocol['cost_grid']['word_cycles']:
                for fixed in protocol['cost_grid']['fixed_dispatch_cycles']:
                    for v in VARIANTS:
                        p=plans[key][v]
                        if not p['feasible']:continue
                        bg=coarsen(profiles[b][v][str(seed)],p)
                        r=simulate(bg,profiles[u]['baseline'][str(seed)],case,word,fixed)
                        records.append(dict(case=case['name'],kind=case['kind'],background=b,urgent=u,seed=seed,word_cycles=word,fixed_cycles=fixed,variant=v,result=r))
    save(BASE/'results.json',records)
    grouped={}
    for r in records:
        k=(r['case'],r['seed'],r['word_cycles'],r['fixed_cycles']);grouped.setdefault(k,{})[r['variant']]=r['result']
    comparisons={}
    for v in VARIANTS[1:]:
        pairs=[(g['baseline'],g[v]) for g in grouped.values() if 'baseline'in g and v in g]
        comparisons[v]=dict(cases=len(pairs),zero_miss_rescues=sum(a['misses']>0 and b['misses']==0 for a,b in pairs),
            cases_fewer_misses=sum(b['misses']<a['misses'] for a,b in pairs),cases_more_misses=sum(b['misses']>a['misses'] for a,b in pairs),
            cases_same_misses=sum(b['misses']==a['misses'] for a,b in pairs),
            total_misses_baseline=sum(a['misses'] for a,b in pairs),total_misses_variant=sum(b['misses'] for a,b in pairs),
            mean_overhead_cycles_baseline=sum(a['overhead_cycles'] for a,b in pairs)/len(pairs),mean_overhead_cycles_variant=sum(b['overhead_cycles'] for a,b in pairs)/len(pairs))
    pairwise={}
    for control in ('split25000','split50000'):
        pairs=[(g[control],g['selective-long25000']) for g in grouped.values() if control in g and 'selective-long25000' in g]
        pairwise[control]=dict(cases=len(pairs),selective_fewer_misses=sum(b['misses']<a['misses'] for a,b in pairs),
            selective_more_misses=sum(b['misses']>a['misses'] for a,b in pairs),same_misses=sum(b['misses']==a['misses'] for a,b in pairs),
            selective_zero_miss_rescues=sum(a['misses']>0 and b['misses']==0 for a,b in pairs),
            uniform_zero_miss_rescues=sum(b['misses']>0 and a['misses']==0 for a,b in pairs))
    strata=[]
    for seed in SEEDS:
        for kind in ('one','periodic','burst'):
            for v in VARIANTS:
                rr=[r for r in records if r['seed']==seed and r['kind']==kind and r['variant']==v]
                strata.append(dict(seed=seed,kind=kind,variant=v,cases=len(rr),total_misses=sum(r['result']['misses'] for r in rr),
                    zero_miss_cases=sum(r['result']['misses']==0 for r in rr)))
    three=[]
    for v in VARIANTS:
        raw=sum(profiles[m][v]['0']['program_bytes'] for m in MODELS)
        full=raw+16+sum(sum(s['resident_site_bytes'] for s in sites(profiles[m][v]['0'])) for m in MODELS)
        three.append(dict(variant=v,job_program_bytes=raw,metadata_and_checkpoint_bytes_not_included=True,
            feasible_raw_programs=raw+16<=MEM,remaining_bytes_before_checkpoint_tables=MEM-raw-16,
            unrestricted_all_drained_sites_resident_bytes=full,unrestricted_all_drained_sites_fits=full<=MEM,
            unrestricted_all_drained_sites_conclusion='Rejected: all-site precompiled checkpointcode+metadata exceeds32KiB.' if full>MEM else 'Raw finitecatalogue fits.',
            conclusion='Requires separate joint threejoballocation and controller; no threejobperformanceclaim.'))
    checks=[audit_resident(r['path']) for r in resident]
    audit=dict(status='passed',independent_small_dp_cases=audit_small_dp(),resident_checks=checks,
        protocol_sha256=sha(BASE/'protocol.json'),selection_identity_sha256=sha(selection_identity),
        raw_records_sha256=sha(BASE/'results.json'),schedule_conservation_checks=len(records),
        raw_completion_reconstruction_checks=audit_completion_records(records,protocol,profiles))
    save(BASE/'audit.json',audit)
    return dict(comparisons=comparisons,selective_vs_uniform=pairwise,strata=strata,plans=pareto,three_job_capacity=three,audit=audit,
        schedule_records=len(records),frozen_cases=len(protocol['cases']),resident_fixtures=resident)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('run','evaluate','calibrate','audit'));args=parser.parse_args()
    BASE.mkdir(parents=True,exist_ok=True);started=time.monotonic();protocol=freeze_protocol()
    if args.stage=='run':profiles,fixtures=prepare_profiles()
    else:profiles=json.loads((BASE/'profiles.json').read_text());fixtures=json.loads((BASE/'fixtures.json').read_text())
    if args.stage=='calibrate':
        r=calibrate(profiles,protocol);print(json.dumps(dict(status=r['status'],pairwise=r['pairwise'],rss=r['python_max_rss_bytes']),indent=2));return
    if args.stage=='audit':
        r=audit_calibration(profiles,protocol);print(json.dumps(dict(status=r['status'],completion_checks=r['completion_reconstruction_checks'],native_scan_matches=len(r['native_scan_counter_matches'])),indent=2));return
    result=evaluate(profiles,fixtures,protocol)
    result.update(status='completed-native-streams-and-frozen-trace-model',engine_sha256=PARENT_SHA,
        driver_sha256=sha(__file__),elapsed_seconds=time.monotonic()-started,
        python_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,physical_board=False,
        limitations=['Scheduler uses replayed isolated native commandtimings and explicit checkpointcosts; sharedschedule itself is not nativeRTL.',
            'StalledRAMtrace is one sampledseed, not worstcaseanalysis.',
            'Native correctnessevidence checks new standalone selectivelysplit programs; residentcontrollercorrectness belongs to separate experiment.',
            'Fixed dispatch sensitivity128/512cycles does not yet reconstruct the resident RTL metadata scan cost at unlisted WAITs.',
            'Onebackgroundcontext/oneurgentprogram; threejobcapacity is rawcodecheckonly.',
            'The DP is optimal only in its finitecheckpoint catalogue and blocking+save surrogate; this does not exhaust policyspace.',
            'No area/Fmax/boardmeasurement; no claimed architecturalnovelty; ART and DERCA already establish preemptionpoint/dataflow selection.'])
    save(BASE/'report.json',result)
    (BASE/'README.md').write_text('# Budgeted resident interruption policy\n\n'
        'All policies receive the same resident context assumptions and conservative saving of live SRAM granules. '
        'The 828 arrival cases and their deadlines were frozen before candidate evaluation. The dynamic program minimizes '
        'a maximum blocking plus save-cost surrogate under the actual 32 KiB program and metadata capacity, '
        'without consulting scheduling outcomes. Its optimality is limited to this finite catalogue.\n\n'
        'The new selectively split programs pass exact native output checks in fixed RAM and one sampled stalled RAM mode. '
        'VWW takes 2,026,760 fixed-RAM cycles versus 2,014,205 for the original; KWS takes 1,120,565 versus 1,099,170. '
        'AD has no eligible long RUN and reuses an identical original program.\n\n'
        'The scheduling model covers all six ordered task pairs with one suspended background job and nonpreemptive urgent jobs. '
        'It evaluates 52,992 schedules: four variants across 828 cases, two native memory traces, four DMA word costs and '
        'two dispatch-cost allowances. These correlated cases are a sensitivity study, not deployment frequencies.\n\n'
        '| Stream | Zero-miss rescues over original | Cases with more misses than original |\n'
        '|---|---:|---:|\n| Uniform 25k | 996 | 155 |\n| Uniform 50k | 835 | 128 |\n| Selective long 25k | 729 | 121 |\n\n'
        'Against uniform 25k, the selective stream has fewer misses in 183 cases and more in 473; '
        'against uniform 50k, it has fewer in 161 and more in 325. The tested selective rule therefore does not establish '
        'an overall scheduling advantage over the strong controls. It uses less resident storage in the VWW/AD pair: '
        '26,896 bytes versus 32,720 for uniform 25k and 30,816 for uniform 50k. The original uses 21,872. '
        'All capacities include both model programs, sparse precompiled save/restore programs, metadata and a 32-byte '
        'stress-only clobber prefix. Artificial clobber latency is excluded from the scheduling model.\n\n'
        'Unrestricted precompiled checkpoints at every drained boundary exceed 32 KiB for all three-job variants '
        '(133,840 to 238,832 bytes). Raw model programs fit, but a budgeted three-job controller and joint allocation '
        'were not evaluated.\n\n'
        'See protocol.json, selection-identity.json, plans.json, native-evidence.json, results.json and report.json. '
        'The independent audit checks serialized metadata and checkpoint programs, an exhaustive small dynamic-program '
        'oracle, and deadline/response/service summaries reconstructed from every raw job completion record.\n\n'
        'The scheduling results use isolated native traces and explicit transport costs. They do not reproduce the resident '
        'RTL metadata scanning overhead, provide a worst-case memory bound, or establish physical deadlines. '
        'The resident controller experiment supplies separate interleaving and implementation evidence. '
        'ART and DERCA already cover preemption point placement and state/dataflow choices; architectural novelty remains unestablished.\n')
    print(json.dumps(dict(status=result['status'],comparisons=result['comparisons'],rss=result['python_max_rss_bytes']),indent=2),flush=True)


if __name__=='__main__':main()
