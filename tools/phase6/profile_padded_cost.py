#!/usr/bin/env python3
"""Measure exact PACK state cycles before considering producer-native padding.

Uses the existing full-model fixture and native external-memory simulator,
with a separate Verilator build and a read-only instrumentation copy of the
native C++ harness. This does not program a board or modify selected RTL.
"""
import json
import math
import argparse
from pathlib import Path
import subprocess

from variants import ROOT, check_frozen, sha
from run_screening import save_json

ENGINE=ROOT/'rtl/phase6/padded_stride_engine.sv'
SOURCE=ROOT/'test/phase6/native_padded_profile.cpp'
BASE=ROOT/'work/phase6/producer-padded-cost'
BUILD=BASE/'native'
FIXTURES=ROOT/'work/phase6/padded-stride-v1/fixtures'
BASELINE_FIXTURES=ROOT/'work/phase6/constant-sibling-v1/fixtures'


def run(reuse=False):
    check_frozen()
    BUILD.mkdir(parents=True,exist_ok=True)
    inputs=[ROOT/'rtl/v2/target_pkg.sv',ROOT/'rtl/v2/requantizer.sv',ENGINE]
    inputs += [ROOT/'rtl/v2'/name for name in
               ('scratchpad.sv','tile_dma.sv','tiled_core.sv','command.sv',
                'tile_sequencer.sv','tiled_host_bridge.sv')]
    if not reuse:
        with (BUILD/'build.log').open('w') as log:
            subprocess.run(['verilator','--cc','--exe','--build','-j','2','--public-flat-rw',
                            '-Wno-fatal','--top-module','v2_tiled_host_bridge','--Mdir',str(BUILD),
                            *map(str,inputs),str(SOURCE)],cwd=ROOT,stdout=log,
                           stderr=subprocess.STDOUT,check=True)
    rows=[]
    for model in ('kws','vww'):
        fixture=FIXTURES/f'{model}-pinned-constant-sibling-padded-timed'
        for seed in (0,6063):
            dest=BUILD/f'{model}-s{seed}.json'
            if not reuse:
                subprocess.run([str(BUILD/'Vv2_tiled_host_bridge'),str(fixture),
                                str(seed),str(dest)],cwd=ROOT,check=True)
            row=json.loads(dest.read_text())
            if row['status']!='passed' or row['pack_entries']!=(4 if model=='kws' else 2):
                raise ValueError('incorrect exact PACK profile')
            rows.append(dict(model=model,**row))
    baseline_rows=[]
    for model in ('kws','vww'):
        fixture=BASELINE_FIXTURES/f'{model}-pinned-constant-sibling-timed'
        for seed in (0,6063):
            dest=BUILD/f'{model}-baseline-s{seed}.json'
            if not reuse:
                subprocess.run([str(BUILD/'Vv2_tiled_host_bridge'),str(fixture),
                                str(seed),str(dest)],cwd=ROOT,check=True)
            row=json.loads(dest.read_text())
            if row['status']!='passed' or row['pack_entries']!=0:
                raise ValueError('incorrect baseline profile')
            baseline_rows.append(dict(model=model,**row))
    baseline=json.loads((ROOT/'work/phase6/constant-sibling-v1/native-all-exact/report.json').read_text())
    candidate=json.loads((ROOT/'work/phase6/padded-stride-v1/native-padded-stride/report.json').read_text())
    comparisons={}
    for row in rows:
        model,seed=row['model'],row['stall_seed']
        base=next(x for x in baseline['results'] if x['fixture']==f'{model}-pinned-constant-sibling-timed' and x['stall_seed']==seed)
        cand=next(x for x in candidate['results'] if x['fixture']==f'{model}-pinned-constant-sibling-padded-timed' and x['stall_seed']==seed)
        base_profile=next(x for x in baseline_rows if x['model']==model and x['stall_seed']==seed)
        if (row['elapsed_cycles'],row['engine_cycles'],row['dma_cycles'])!=(cand['elapsed_cycles'],cand['engine_cycles'],cand['dma_cycles']):
            raise ValueError('instrumented run changed native timing')
        if (base_profile['elapsed_cycles'],base_profile['engine_cycles'],base_profile['dma_cycles'])!=(base['elapsed_cycles'],base['engine_cycles'],base['dma_cycles']):
            raise ValueError('unflagged experimental engine changed selected timing')
        # Optimistic: remove every PACK engine-state cycle and give back the
        # entire measured DMA difference. Future producer/scalar changes have
        # zero cost in this proxy. Give a deliberately generous additional
        # 2048 cycles per removed PACK for descriptor fetch and transition.
        dma_penalty=max(0,cand['dma_cycles']-base['dma_cycles'])
        optimistic= cand['elapsed_cycles']-row['pack_cycles']-dma_penalty
        generous=optimistic-2048*row['pack_entries']
        ultra=generous-row['pointwise_cross_cycles']
        comparisons[f'{model}-s{seed}']=dict(baseline_cycles=base['elapsed_cycles'],
            candidate_cycles=cand['elapsed_cycles'],pack_cycles=row['pack_cycles'],
            pack_entries=row['pack_entries'],dma_penalty_cycles=dma_penalty,
            baseline_cross_word_cycles=base_profile['cross_word_cycles'],
            baseline_cross_word_entries=base_profile['cross_word_entries'],
            candidate_remaining_cross_word_cycles=row['cross_word_cycles'],
            candidate_remaining_cross_word_entries=row['cross_word_entries'],
            baseline_pointwise_cross_cycles=base_profile['pointwise_cross_cycles'],
            baseline_pointwise_cross_entries=base_profile['pointwise_cross_entries'],
            candidate_remaining_pointwise_cross_cycles=row['pointwise_cross_cycles'],
            candidate_remaining_pointwise_cross_entries=row['pointwise_cross_entries'],
            optimistic_producer_native_cycles=optimistic,
            optimistic_speedup=base['elapsed_cycles']/optimistic,
            generous_extra_descriptor_allowance_cycles=2048*row['pack_entries'],
            generous_optimistic_cycles=generous,
            generous_optimistic_speedup=base['elapsed_cycles']/generous,
            ultra_optimistic_cycles=ultra,
            ultra_optimistic_speedup=base['elapsed_cycles']/ultra)
    geomean={f's{seed}':math.sqrt(comparisons[f'kws-s{seed}']['generous_optimistic_speedup']*
                                  comparisons[f'vww-s{seed}']['generous_optimistic_speedup'])
             for seed in (0,6063)}
    ultra_geomean={f's{seed}':math.sqrt(comparisons[f'kws-s{seed}']['ultra_optimistic_speedup']*
                                        comparisons[f'vww-s{seed}']['ultra_optimistic_speedup'])
                   for seed in (0,6063)}
    report=dict(status='passed',physical_board=False,
                source_sha256=sha(SOURCE),engine_sha256=sha(ENGINE),
                executable_sha256=sha(BUILD/'Vv2_tiled_host_bridge'),
                comparisons=comparisons,
                generous_geomean_speedup=geomean,
                ultra_optimistic_geomean_speedup=ultra_geomean,
                decision='stop-producer-native-padding' if max(ultra_geomean.values())<1.03 else 'prototype',
                limitation='Optimistic structural proxy, not a hardware speed measurement. It assumes zero cost to produce padded outputs and make the following in-place activation layout-aware, grants an arbitrary generous 2048-cycle allowance per PACK for descriptor overhead, then also removes every remaining pointwise PW_X2 state cycle including tiles that cannot fit padded SRAM.')
    save_json(BASE/'report.json',report)
    check_frozen()
    print(json.dumps(comparisons,indent=2,sort_keys=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse',action='store_true',help='analyze existing exact profile rows')
    run(parser.parse_args().reuse)
