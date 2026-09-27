#!/usr/bin/env python3
"""All-exact native gate for the isolated physical channel-stride candidate.

Checks every pinned/stress tensor before reporting fixed and stalled cycles.
This never uses a physical FPGA or replaces the selected baseline image.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

from variants import ROOT, check_frozen, sha
from run_screening import save_json, verify_files

BASE=ROOT/'work/phase6/padded-stride-v1'
REFERENCE=ROOT/'work/phase6/constant-sibling-v1'
ENGINE=ROOT/'rtl/phase6/padded_stride_engine.sv'
SOURCES=[ROOT/'rtl/v2'/name for name in ('target_pkg.sv','requantizer.sv')]+[
    ENGINE]+[ROOT/'rtl/v2'/name for name in ('scratchpad.sv','tile_dma.sv',
    'tiled_core.sv','command.sv','tile_sequencer.sv','tiled_host_bridge.sv')]
HARNESS=ROOT/'test/phase6/native.cpp'


def run(*,build=True):
    check_frozen()
    manifest_path=BASE/'fixtures.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest['status']!='passed-replay' or len(manifest['fixtures'])!=6:
        raise ValueError('six independent padded fixtures required')
    selected=json.loads((REFERENCE/'native-all-exact/report.json').read_text())
    if selected['status']!='passed':raise ValueError('selected native baseline missing')
    output=BASE/'native-padded-stride';output.mkdir(parents=True,exist_ok=True)
    executable=output/'Vv2_tiled_host_bridge'
    if build:
        with (output/'build.log').open('w') as log:
            subprocess.run(['verilator','--cc','--exe','--build','-j','2',
                '--public-flat-rw','-Wno-fatal','--top-module','v2_tiled_host_bridge',
                '--Mdir',str(output),*map(str,SOURCES),str(HARNESS)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    if not executable.exists():raise ValueError('candidate native executable absent')
    report=dict(status='running',physical_board=False,
        scope='full-model exact RTL simulation with external RAM model; no physical-board claim',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in SOURCES+[HARNESS]},
        executable_sha256=sha(executable),fixture_manifest_sha256=sha(manifest_path),
        reference_report_sha256=sha(REFERENCE/'native-all-exact/report.json'),results=[])
    save_json(output/'report.json',report)
    for fixture in manifest['fixtures']:
        directory=BASE/'fixtures'/fixture['name']
        verify_files(directory,fixture['files'])
        for seed in (0,6063):
            destination=output/f'{fixture["name"]}-s{seed}.json'
            start=time.monotonic()
            subprocess.run([str(executable),str(directory),str(seed),str(destination)],check=True)
            result=json.loads(destination.read_text())
            if result['status']!='passed':raise ValueError('candidate tensor mismatch')
            result.update(fixture=fixture['name'],fixture_files=fixture['files'],
                          simulation_seconds=time.monotonic()-start)
            report['results'].append(result)
            save_json(output/'report.json',report)
    baseline={(row['fixture'],row['stall_seed']):row for row in selected['results']}
    timing={}
    for model in ('kws','vww'):
        timing[model]={}
        for seed,label in ((0,'fixed'),(6063,'stalled')):
            old=baseline[(f'{model}-pinned-constant-sibling-timed',seed)]
            new=next(row for row in report['results'] if row['fixture']==
                f'{model}-pinned-constant-sibling-padded-timed' and row['stall_seed']==seed)
            before,after=old['elapsed_cycles'],new['elapsed_cycles']
            timing[model][label]=dict(baseline_cycles=before,candidate_cycles=after,
                                      speedup=before/after,saved_cycles=before-after)
    report['timing']=timing
    report['decision']=('native-cycle-go-to-route' if any(
        timing[model][state]['saved_cycles']>0 for model in timing for state in ('fixed','stalled'))
        else 'native-cycle-no-go')
    report['status']='passed';save_json(output/'report.json',report)
    check_frozen()
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-build',action='store_true')
    args=parser.parse_args()
    result=run(build=not args.no_build)
    print(json.dumps({k:result[k] for k in ('status','decision','timing')},sort_keys=True))
