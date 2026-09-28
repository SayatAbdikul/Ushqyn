#!/usr/bin/env python3
"""Audit full-model KWS spatial-fusion opportunities on the frozen ABI.

The existing KWS program already retains every intermediate activation in
SRAM. This checks whether the currently supported per-channel DMA/COPY strip
recipe can compose a nontrivial spatial partition without changing the ABI.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'compiler'),str(ROOT/'tools/phase6')]

from run_boardless import load_model
from followup_graph import group_channels
from channel_compaction import compact_channels
from hardware_v2 import Descriptor

OUT=ROOT/'work/phase6/matched-baselines-v1/b3-kws-audit/report.json'
FIXTURE=ROOT/'work/phase6/matched-baselines-v1/b3/fixtures/kws-pinned-timed'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    original,_,_,model_sources=load_model('kws')
    grouped,_,_=group_channels(original)
    graph,_,_,_=compact_channels(grouped)
    schedule=json.loads((FIXTURE/'schedule.json').read_text())
    layers=list(range(3,19,2))
    assert all(graph.layers[index].op=='Conv' and graph.layers[index+1].op=='Relu'
               for index in layers)
    geometry=[]
    for index in layers:
        layer=graph.layers[index]
        input_shape=graph.tensors[layer.inputs[0]].shape
        output_shape=graph.tensors[layer.output].shape
        geometry.append(dict(layer=index,op='depthwise' if layer.attributes.get('group',1)!=1
                             else 'pointwise',input_shape=input_shape,output_shape=output_shape))
        assert input_shape==output_shape==(1,64,25,5)
    stages={stage['layer']:stage for stage in schedule['stages']}
    retained=[]
    for index in (1,*layers):
        stage=stages[index]
        assert stage['output']['bytes']==8000 and stage['store'] is None
        retained.append(dict(layer=index,output_sram=stage['output']['sram'],
                             output_bytes=stage['output']['bytes'],live=stage['live'],
                             activation_loads=[load for load in stage['loads']
                                               if load['role']=='input']))
    assert all(not row['activation_loads'] for row in retained[1:])
    # The current exact strip composers write compact per-channel output and
    # use per-channel DMA or COPY to merge it into a full NCHW plane. Both
    # read-side operations require an eight-byte aligned source address.
    # For width 5, each strip's channel stride is 5*h, aligned only if h%8=0.
    candidates=[dict(height=h,channel_bytes=5*h,
                     channel_source_aligned=(5*h)%8==0)
                for h in range(1,25)]
    aligned_heights=[row['height'] for row in candidates if row['channel_source_aligned']]
    exact_two_partitions=[(first,25-first) for first in range(1,25)
                          if first in aligned_heights and 25-first in aligned_heights]
    assert aligned_heights==[8,16,24] and not exact_two_partitions
    # Descriptor.validate enforces aligned source bases for COPY and Conv,
    # while outputs have no programmable per-channel stride.
    Descriptor(3,input=8,output=1,count=5,outputs=5,next_pc=64).validate()
    try:
        Descriptor(3,input=9,output=1,count=5,outputs=5,next_pc=64).validate()
        raise AssertionError('unaligned COPY source unexpectedly valid')
    except ValueError:
        pass
    report=dict(schema=1,status='bounded-abi-barrier',physical_board=False,
        scope='KWS current direct per-channel compact-strip lowering on 27 MHz eight-lane ABI',
        model_source_sha256=model_sources,
        source_sha256={str(path.relative_to(ROOT)):sha(path) for path in (
            Path(__file__),ROOT/'compiler/hardware_v2.py',
            ROOT/'compiler/scheduler/defines_verify.py',
            ROOT/'tools/phase6/matched_defines_adapt.py')},
        fixture=str(FIXTURE.relative_to(ROOT)),
        fixture_files_sha256={name:sha(FIXTURE/name) for name in
            ('commands.bin','payload.bin','input.bin','output.bin','schedule.json')},
        graph_layers=geometry,existing_resident_outputs=retained,
        existing_max_live_sram_exclusive=max(row['live'][1] for row in retained),
        existing_intermediate_external_stores=0,
        existing_intermediate_activation_reloads=0,
        compact_channel_plane_bytes=125,
        strip_candidates=candidates,
        direct_per_channel_aligned_heights=aligned_heights,
        exact_two_partitions_with_aligned_channel_sources=exact_two_partitions,
        all_aligned_partition_impossible='every positive strip height must be divisible by 8 but total height is 25',
        lowering_barrier='current DMA/COPY requires aligned read bases; compact NCHW strip channel stride is 5*h; no descriptor output channel stride or byte-shift gather exists',
        qualification='this excludes the current direct per-channel strip composer, not every conceivable multi-pass repacking algorithm or a changed RTL ABI',
        decision='retain full-tensor layerwise/liveness KWS endpoint; it already eliminates all intermediate external traffic and uses one RUN per fused Conv+activation')
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],aligned_heights=aligned_heights,
                          max_live_sram=report['existing_max_live_sram_exclusive'],
                          intermediate_external_stores=0),sort_keys=True))


if __name__=='__main__':
    main()
