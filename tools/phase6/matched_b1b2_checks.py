#!/usr/bin/env python3
"""Focused data-lifetime regressions for matched B1/B2 scheduling."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'compiler'),str(ROOT/'tools/phase6')]
from chain_resident import compile_chain
from channel_compaction import check_oracles
from matched_b1b2_generic import matched_model, prefetch_candidates, save
from scheduler.matched_b1b2_tiling import compile_tiled
from scheduler.resident_verify import replay_resident
import struct


def packed(rows):
    return b''.join(struct.pack('<BBHIII',*r) for r in rows)


def run():
    original,x,program,maps,_ = matched_model('kws')
    oracle = check_oracles(original,program,maps,x)
    plan = compile_tiled(program,preferred_capacity=16384)
    code,payload,schedule = compile_chain(program,True,reuse_sibling_inputs=True,
        prepared_plans={False:plan,True:plan})
    producer = next(s for s in schedule['stages'] if s['layer']==2)
    consumers = [s for s in schedule['stages'] if s['layer']==3]
    assert len(consumers)>1 and consumers[0]['output']['bytes']<producer['output']['bytes']
    assert producer['store'] is not None, 'full producer must survive partial consumer tiles'
    replay = replay_resident(program,code,payload,{program.inputs[0]:x},
        run_contracts=schedule['run_contracts'],final_output=schedule['final_output'],
        snapshot_regions=schedule['snapshot_regions'],oracle=oracle)
    # A spare-half input may be prefetched only when no intermediate SDRAM
    # store produces its bytes. A store reading that SRAM half also blocks it.
    current=(2,0,0,0,1024<<16,0);wait=(3,3,0,0,0,0)
    load=(1,1,0,4096,16384,128);dma_wait=(3,2,0,0,0,0)
    nxt=(2,0,0,16384,16384|(17408<<16),0);halt=(0,0,0,0,0,0)
    legal=[current,wait,load,dma_wait,nxt,wait,halt]
    assert len(prefetch_candidates(packed(legal)))==1
    source_write=(1,0,0,4096,128,128)
    assert prefetch_candidates(packed([current,wait,source_write,dma_wait,*legal[2:]]))==[]
    target_read=(1,0,0,8192,16384,128)
    assert prefetch_candidates(packed([current,wait,target_read,dma_wait,*legal[2:]]))==[]
    result=dict(status='passed',partial_consumer=dict(producer_layer=2,consumer_layer=3,
        producer_bytes=producer['output']['bytes'],consumer_tiles=len(consumers),
        first_consumer_bytes=consumers[0]['output']['bytes'],producer_store_preserved=True,
        replay=replay),prefetch_dependency_cases=3)
    save(ROOT/'work/phase6/matched-baselines-v1/b1b2-regressions.json',result)
    return result


if __name__=='__main__':
    print(json.dumps(run(),sort_keys=True))
