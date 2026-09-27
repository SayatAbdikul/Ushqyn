#!/usr/bin/env python3
"""Move immutable grouped-model parameter DMAs into the preceding RUN's spare SRAM tail.

The grouped source fixtures and routed engine remain immutable. Every moved
transfer retains its exact address and bytes; only command order changes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

import numpy as np

from variants import ROOT, check_frozen, sha
from run_screening import save_json, verify_files
from run_boardless import load_model
from scheduler.resident_verify import replay_resident
from followup_graph import group_channels, check_oracles
from prefetch_tail import analyze_fixture, commands, reorder

SOURCE=ROOT/'work/phase6/followup_graph'
OUTPUT=ROOT/'work/phase6/followup_graph_tail_prefetch'
LABEL='combined-spec-scalar-v1'
ENGINE=ROOT/'work/phase6/experiments-v1'/LABEL


def prepare():
    check_frozen()
    source_manifest=SOURCE/'fixtures.json'
    manifest=json.loads(source_manifest.read_text())
    if manifest['status']!='passed-replay':raise ValueError('grouped replay missing')
    records=[]
    for item in manifest['fixtures']:
        label=item['label'];model=item['model'];sample=item['sample']
        src=SOURCE/'fixtures'/label
        verify_files(src,item['files'])
        analysis=analyze_fixture(src)
        changed,mapping=reorder(commands((src/'commands.bin').read_bytes()),
                                analysis['candidates'])
        code=b''.join(struct.pack('<BBHIII',*c) for c in changed)
        schedule=json.loads((src/'schedule.json').read_text())
        runs={str(mapping[int(k)]):v for k,v in schedule['run_contracts'].items()}
        constants={str(mapping[int(k)]):v for k,v in schedule.get('constant_contracts',{}).items()}
        packs={str(mapping[int(k)]):v for k,v in schedule.get('pack_contracts',{}).items()}
        schedule.update(run_contracts=runs,constant_contracts=constants,pack_contracts=packs,
            program_sha256=hashlib.sha256(code).hexdigest(),
            catalogue='grouped exact model plus immutable tail parameter prefetch',
            tail_prefetch=dict(bytes=analysis['immediately_legal_prefetch_bytes'],
                               transfers=analysis['immediately_legal_count'],
                               moved_old_command_indices=[x['dma_command'] for x in analysis['candidates']]))
        original,pinned,_,_=load_model(model)
        grouped,mappings,_=group_channels(original)
        value=(pinned if sample=='pinned' else
            np.random.default_rng(6078).integers(-128,128,pinned.shape,dtype=np.int8))
        oracle=check_oracles(original,grouped,mappings,value)
        payload=(src/'payload.bin').read_bytes()
        replay=replay_resident(grouped,code,payload,{grouped.inputs[0]:value},
            run_contracts=runs,constant_contracts=constants,pack_contracts=packs,
            final_output=schedule['final_output'],
            snapshot_regions=schedule['snapshot_regions'],oracle=oracle)
        dest=OUTPUT/'fixtures'/label.replace('-grouped-','-grouped-tail-prefetch-')
        dest.mkdir(parents=True,exist_ok=True)
        for filename in item['files']:
            data=(src/filename).read_bytes()
            if filename=='commands.bin':data=code
            if filename=='schedule.json':data=(json.dumps(schedule,sort_keys=True,indent=2)+'\n').encode()
            (dest/filename).write_bytes(data)
        records.append(dict(name=dest.name,model=model,sample=sample,
                            verification=replay,
                            prefetch_bytes=analysis['immediately_legal_prefetch_bytes'],
                            prefetch_transfers=analysis['immediately_legal_count'],
                            source_fixture=label,
                            files={name:sha(dest/name) for name in item['files']}))
    result=dict(schema=1,status='passed-replay',physical_board=False,
                source_manifest_sha256=sha(source_manifest),
                source_payloads_identical=True,source_descriptors_identical=True,
                fixtures=records)
    save_json(OUTPUT/'fixtures.json',result)
    check_frozen()
    print(json.dumps({r['name']:r['prefetch_bytes'] for r in records},indent=2))
    return result


def native():
    check_frozen()
    path=OUTPUT/'fixtures.json'
    manifest=json.loads(path.read_text())
    if manifest['status']!='passed-replay':raise ValueError('missing grouped prefetch replay')
    engine_report=json.loads((ENGINE/'native/report.json').read_text())
    executable=ENGINE/'native/Vv2_tiled_host_bridge'
    if (engine_report['status']!='passed' or
            engine_report['sources'].get(str((ENGINE/'engine.sv').relative_to(ROOT)))!=sha(ENGINE/'engine.sv')):
        raise ValueError('combined engine identity mismatch')
    verify_files(ROOT,engine_report['sources'])
    target=OUTPUT/f'native-{LABEL}'
    target.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',label=LABEL,physical_board=False,
        executable_sha256=sha(executable),sources=engine_report['sources'],
        fixture_manifest_sha256=sha(path),results=[])
    for item in manifest['fixtures']:
        fixture=OUTPUT/'fixtures'/item['name']
        verify_files(fixture,item['files'])
        for seed in (0,6063):
            dest=target/f'{item["name"]}-s{seed}.json'
            subprocess.run([str(executable),str(fixture),str(seed),str(dest)],check=True)
            row=json.loads(dest.read_text())
            if row['status']!='passed':raise ValueError('native output mismatch')
            row.update(fixture=item['name'],stall_seed=seed,fixture_files=item['files'],
                       prefetch_bytes=item['prefetch_bytes'])
            report['results'].append(row)
            save_json(target/'report.json',report)
    comparisons={}
    for model in ('kws','vww'):
        base=SOURCE/'fixtures'/f'{model}-pinned-grouped-timed'
        for seed in (0,6063):
            dest=target/f'{model}-grouped-baseline-s{seed}.json'
            subprocess.run([str(executable),str(base),str(seed),str(dest)],check=True)
            baseline=json.loads(dest.read_text())
            cand=next(r for r in report['results'] if
                      r['fixture']==f'{model}-pinned-grouped-tail-prefetch-timed'
                      and r['stall_seed']==seed)
            comparisons[f'{model}-s{seed}']=dict(baseline_cycles=baseline['elapsed_cycles'],
                candidate_cycles=cand['elapsed_cycles'],
                speedup=baseline['elapsed_cycles']/cand['elapsed_cycles'],
                overlap_cycles=cand['overlap_cycles'],
                prefetch_bytes=cand['prefetch_bytes'])
    report.update(status='passed',comparisons=comparisons)
    save_json(target/'report.json',report)
    check_frozen()
    print(json.dumps(comparisons,indent=2,sort_keys=True))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','native'))
    {'prepare':prepare,'native':native}[parser.parse_args().stage]()
