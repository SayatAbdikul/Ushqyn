#!/usr/bin/env python3
"""Isolated line-cache invalidation/LUT-load overlap; no board or frozen edits."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'compiler'),str(ROOT/'tools/phase6')]
from matched_b1b2_generic import sha,NATIVE

BASE=ROOT/'work/phase6/engine-candidate-rtl-v1'
PARENT=ROOT/'work/phase6/pool-timing-v1/engine.sv'
PARENT_SHA='c347336ec58503f26abc47094289bd7926fcc5ed2515eca758a4b7dbc7f6d48a'
FINAL=ROOT/'work/phase6/matched-baselines-v1/b3-final/report.json'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def once(source,old,new):
    if source.count(old)!=1:raise ValueError('RTL patch site changed')
    return source.replace(old,new,1)


def source_files(engine,only=False):
    paths=[ROOT/'rtl/v2/target_pkg.sv',ROOT/'rtl/v2/requantizer.sv',engine]
    if not only:
        paths.extend(ROOT/'rtl/v2'/name for name in ('scratchpad.sv','tile_dma.sv','tiled_core.sv',
            'command.sv','tile_sequencer.sv','tiled_host_bridge.sv'))
    return paths


def prepare(output=BASE):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if sha(PARENT)!=PARENT_SHA:raise ValueError('selected engine changed')
    source=once(PARENT.read_text(),
        '    wire clearing_line=VALID_IN_BRAM&&state==LINE_CLEAR;\n'
        '    wire [LINE_BITS-1:0] line_write_index=clearing_line?line_clear_index:line_fill_index;',
        '''    // LUT and line-cache storage have distinct write ports. During the
    // mandatory 256-byte fused-table reload, clear each cache entry exactly
    // once. Larger caches retain the original sequential invalidation path.
    wire lut_line_clear=VALID_IN_BRAM&&(LINE_ENTRIES<=256)&&
        state==F_L_BYTES&&(32'(lut_index)<LINE_ENTRIES);
    wire clearing_line=VALID_IN_BRAM&&(state==LINE_CLEAR||lut_line_clear);
    wire [LINE_BITS-1:0] line_write_index=lut_line_clear?LINE_BITS'(lut_index):
        (clearing_line?line_clear_index:line_fill_index);''')
    source=once(source,
        '                        line_clear_index<=0;state<=VALID_IN_BRAM?LINE_CLEAR:GEOM1;',
        '''                        line_clear_index<=0;
                        state<=(VALID_IN_BRAM&&!(fused_activation&&LINE_ENTRIES<=256))?
                            LINE_CLEAR:GEOM1;''')
    engine=output/'engine.sv'
    if engine.exists() and engine.read_text()!=source:raise ValueError('candidate already differs')
    engine.write_text(source)
    # Exhaustive control-level check of parameterized entry coverage. Default
    # native tests below exercise the real 256-entry configuration.
    coverage=[]
    for entries in (32,64,128,192,256,512):
        touched=[i for i in range(256) if entries<=256 and i<entries]
        if entries<=256 and touched!=list(range(entries)):raise AssertionError('missing invalidation')
        if entries>256 and touched:raise AssertionError('larger-cache fallback changed')
        coverage.append(dict(entries=entries,overlapped_entries=len(touched),
            sequential_fallback=entries>256))
    record=dict(status='prepared',physical_board=False,parent=str(PARENT.relative_to(ROOT)),
        parent_sha256=sha(PARENT),engine_sha256=sha(engine),parameter_coverage=coverage,
        mechanism='invalidate line-cache BRAM while independent fused activation LUT BRAM loads; no arithmetic, ABI, layout or memory capacity change',
        expected_default_saved_engine_cycles_per_fused_descriptor=256,
        limitations=['Parameterized invalidation coverage is a control-model check; native/engine RTL tests use default256 entries.',
                     'No physical routing, resource measurement or board result is claimed.'])
    save(output/'identity.json',record)
    return output,engine


def edges(output=BASE):
    from cocotb.runner import get_runner
    output,engine=prepare(output);build=output/'edges'
    runner=get_runner('verilator')
    runner.build(verilog_sources=source_files(engine,True),hdl_toplevel='v2_engine',build_dir=build,
        build_args=['--timing','-Wno-fatal'],timescale=('1ns','1ps'))
    paths=[str(ROOT/'compiler'),str(ROOT/'test/phase6'),str(ROOT/'test/phase2')]+sys.path
    sys.path[:0]=paths
    runner.test(hdl_toplevel='v2_engine',test_module=['test_fused_activation','test_cache','test_pool_timing'],
        test_dir=ROOT/'test/phase6',build_dir=build,results_xml=str(build/'results.xml'),
        extra_env={'PYTHONPATH':os.pathsep.join(paths)})
    cases=ET.parse(build/'results.xml').findall('.//testcase')
    if len(cases)!=3 or any(c.find('failure') is not None or c.find('error') is not None for c in cases):
        raise ValueError('edge regressions failed')
    report=dict(status='passed',physical_board=False,tests=3,
        coverage='fused PW/DW tails, nonlinear LUT changes at same address, backpressure, abort/clear/reset, cache regressions and pool arithmetic',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_files(engine,True)+[
            ROOT/'test/phase6/test_fused_activation.py',ROOT/'test/phase6/test_cache.py',ROOT/'test/phase6/test_pool_timing.py']},
        results_sha256=sha(build/'results.xml'))
    save(build/'report.json',report);return report


def fused_count(directory):
    payload=(directory/'payload.bin').read_bytes();count=0
    for op,flags,_,a,b,n in struct.iter_unpack('<BBHIII',(directory/'commands.bin').read_bytes()):
        if op==1 and flags and b==0 and n==128 and payload[a+6]&1:count+=1
    return count


def measure(executable,directory,expected,seed,path):
    before={p.name:sha(p) for p in directory.iterdir() if p.is_file()}
    if before!=expected:raise ValueError('fixture files changed')
    exe_sha=sha(executable)
    with path.with_suffix('.log').open('w') as log:
        subprocess.run([str(executable),str(directory),str(seed),str(path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    if before!={p.name:sha(p) for p in directory.iterdir() if p.is_file()} or exe_sha!=sha(executable):
        raise ValueError('native inputs changed during execution')
    row=json.loads(path.read_text())
    if row['status']!='passed':raise ValueError('native failed')
    row.update(fixture_directory=str(directory.relative_to(ROOT)),fixture_files=before,executable_sha256=exe_sha)
    save(path,row);return row


def native(output=BASE):
    output,engine=prepare(output);build=output/'native';build.mkdir(exist_ok=True)
    harness=ROOT/'test/phase6/native.cpp';sources=source_files(engine)
    pins={str(p.relative_to(ROOT)):sha(p) for p in sources+[harness,Path(__file__)]}
    frozen=json.loads(FINAL.read_text())
    if frozen['status']!='passed':raise ValueError('matched final baseline incomplete')
    baseline_native=json.loads((NATIVE.parent/'report.json').read_text())
    if sha(NATIVE)!=baseline_native['executable_sha256']:raise ValueError('baseline native changed')
    with (build/'build.log').open('w') as log:
        subprocess.run(['verilator','--cc','--exe','--build','-j','2','--public-flat-rw','-Wno-fatal',
            '--top-module','v2_tiled_host_bridge','--Mdir',str(build),*map(str,sources),str(harness)],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    executable=build/'Vv2_tiled_host_bridge'
    report=dict(status='running',physical_board=False,source_sha256=pins,engine_sha256=sha(engine),
        baseline_engine_sha256=sha(PARENT),executable_sha256=sha(executable),baseline_executable_sha256=sha(NATIVE),
        final_baseline=dict(file=str(FINAL.relative_to(ROOT)),sha256=sha(FINAL)),results=[])
    save(build/'report.json',report)
    for model,selected in frozen['selected'].items():
        for sample in ('pinned','stress'):
            fixture=selected[sample];directory=ROOT/fixture['directory'];fused=fused_count(directory)
            for seed in (0,6063):
                a=measure(NATIVE,directory,fixture['files'],seed,build/f'{model}-{sample}-baseline-s{seed}.json')
                original=next(n for n in fixture['native'] if n['stall_seed']==seed)
                for key in ('elapsed_cycles','engine_cycles','dma_cycles','overlap_cycles','tensor_checks'):
                    if a[key]!=original[key]:raise ValueError('matched control differs from final baseline')
                b=measure(executable,directory,fixture['files'],seed,build/f'{model}-{sample}-candidate-s{seed}.json')
                row=dict(model=model,sample=sample,stall_seed=seed,fused_descriptors=fused,
                    expected_saved_engine_cycles=256*fused,baseline=a,candidate=b,
                    saved_elapsed_cycles=a['elapsed_cycles']-b['elapsed_cycles'],
                    saved_engine_cycles=a['engine_cycles']-b['engine_cycles'],
                    speedup=a['elapsed_cycles']/b['elapsed_cycles'])
                report['results'].append(row);save(build/'report.json',report)
                print(model,sample,seed,'saved',row['saved_elapsed_cycles'],'speedup',row['speedup'],flush=True)
    for name,expected in pins.items():
        if sha(ROOT/name)!=expected:raise ValueError('experiment source changed')
    report.update(status='passed',conclusion='bounded general setup-cycle improvement; no routing/board/novelty claim',
        routing_decision='skip because anticipated sub0.5% gain does not justify route/board expansion for this screen',
        feasibility='adds a small control/address mux; no added memory or multiplier. Actual CLS/Fmax unmeasured; baseline only273 CLS spare.')
    save(build/'report.json',report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','edges','native'])
    parser.add_argument('--output',type=Path,default=BASE)
    args=parser.parse_args();globals()[args.stage](args.output)
