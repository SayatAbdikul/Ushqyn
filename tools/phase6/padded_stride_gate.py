#!/usr/bin/env python3
"""Executable feasibility gate for pointwise padded physical channel planes.

Counts are derived from the exact selected schedule after constant-filter and
sibling retention. They are structural SRAM transactions, not measured speedup.
The gate refuses to call software prototype read counts a routed result.
"""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'compiler'),str(ROOT/'tools/phase6')]
from hardware_v2 import Descriptor
from run_screening import save_json
from variants import check_frozen, sha

BASE=ROOT/'work/phase6/constant-sibling-v1'
OUTPUT=ROOT/'work/phase6/padded-stride-gate/analysis.json'


def cost(stage):
    d=Descriptor.decode(bytes.fromhex(stage['descriptor_hex']))
    if not(d.opcode==4 and d.kernel_h==d.kernel_w==1 and d.stride_h==d.stride_w==1):
        return None
    logical=d.input_h*d.input_w
    physical=(logical+7)//8*8
    if logical==physical:return None
    parts=stage.get('constant_filter_parts')
    live_channels=(sum(p['channels'] for p in parts if not p['constant'])
                   if parts is not None else d.output_c)
    if not 0<live_channels<=d.output_c:raise ValueError('invalid live pointwise channels')
    groups=range(0,logical,8)
    cross_per_output=sum(((channel*logical+pixel)&7)+min(8,logical-pixel)>8
                         for channel in range(d.input_c) for pixel in groups)
    pack_reads=sum(1+(((channel*logical+pixel)&7)+min(8,logical-pixel)>8)
                   for channel in range(d.input_c) for pixel in groups)
    pack_words=d.input_c*((logical+7)//8)
    cross_reads=cross_per_output*live_channels
    return dict(layer=stage['layer'],first_element=stage['first_element'],
        logical_plane_bytes=logical,physical_plane_bytes=physical,
        input_channels=d.input_c,output_channels=d.output_c,
        executed_output_channels=live_channels,
        padded_input_extra_bytes=(physical-logical)*d.input_c,
        current_live_sram_bytes=stage['live'][1],
        current_tile_fits_after_simple_expansion=stage['live'][1]+(physical-logical)*d.input_c<=32768,
        exact_current_cross_word_read_events=cross_reads,
        required_pack_destination_words=pack_words,
        minimum_pack_source_reads=pack_reads,
        minimum_pack_sram_port_operations=pack_reads+pack_words,
        break_even_pack_cycles_per_word=cross_reads/pack_words)


def run():
    check_frozen();models={}
    native=json.loads((BASE/'native-all-exact/report.json').read_text())
    if native['status']!='passed':raise ValueError('selected RTL native proof missing')
    for model in ('kws','vww'):
        fixture=BASE/'fixtures'/f'{model}-pinned-constant-sibling-timed'
        schedule=json.loads((fixture/'schedule.json').read_text())
        if sha(fixture/'commands.bin')!=schedule['program_sha256']:
            raise ValueError('schedule command identity mismatch')
        rows=[r for r in native['results'] if r['fixture']==fixture.name]
        if sorted(r['stall_seed'] for r in rows)!=[0,6063] or any(r['status']!='passed' for r in rows):
            raise ValueError('native timing evidence missing')
        stages=[x for s in schedule['stages'] if (x:=cost(s)) is not None]
        cross=sum(s['exact_current_cross_word_read_events'] for s in stages)
        words=sum(s['required_pack_destination_words'] for s in stages)
        min_ops=sum(s['minimum_pack_sram_port_operations'] for s in stages)
        base=next(r['elapsed_cycles'] for r in rows if r['stall_seed']==0)
        models[model]=dict(stages=stages,stages_with_padding=len(stages),
            exact_current_cross_word_read_events=cross,
            pack_destination_words_if_repeated_per_tile=words,
            minimum_pack_sram_port_operations=min_ops,
            average_break_even_pack_cycles_per_word=cross/words,
            native_fixed_cycles=base,
            one_cycle_per_cross_proxy_after_minimum_pack_ops=base-cross+min_ops,
            one_cycle_per_cross_proxy_improvement_pct=(cross-min_ops)/base*100,
            overfull_current_tile_count=sum(not s['current_tile_fits_after_simple_expansion'] for s in stages))
    report=dict(status='feasibility-gate-only',physical_board=False,
        selected_fixture_manifest_sha256=sha(BASE/'fixtures.json'),
        selected_native_report_sha256=sha(BASE/'native-all-exact/report.json'),
        source_sha256=sha(Path(__file__)),models=models,
        decision='word-parallel PACK prototype only; byte-serial repacking is not justified for VWW',
        requirements=['64-byte descriptor opt-in flags with physical stride validation',
                      'packed-input construction in SRAM with separate logical/physical coordinates',
                      'retile any stage exceeding 32768 SRAM bytes',
                      'independent exact tensor and byte-provenance replay',
                      'actual RTL fixed/stalled cycles and routed area/timing before board use'],
        limitation='cross-word read events are exact for current PW descriptor traversal, but the cycle proxy ignores SRAM stalls, prefetch overlap, packing controller cost and timing/area changes')
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    save_json(OUTPUT,report)
    print(json.dumps({k:{n:v[n] for n in ('exact_current_cross_word_read_events',
        'pack_destination_words_if_repeated_per_tile','minimum_pack_sram_port_operations',
        'average_break_even_pack_cycles_per_word','overfull_current_tile_count')}
        for k,v in models.items()},sort_keys=True))
    check_frozen()
    return report


if __name__=='__main__':run()
