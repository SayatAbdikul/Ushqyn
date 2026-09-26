#!/usr/bin/env python3
"""Execute the author's pinned halo/cache geometry code on both primary models.

This closes a geometry cross-check, not B03 or a DeFiNES latency reproduction.
Download the pinned tarball into work/phase6/defines-source before running.
No package installation or active Phase 5 source/environment change is needed.
"""
import hashlib
import json
import sys
from variants import ROOT, sha, check_frozen
from run_boardless import load_model
from scheduler.defines_adapter import segment_workload
from scheduler.spatial import supported_segments

REVISION='7097d6090dc22321e44ce91434e7cc23b065864f'
source=ROOT/f'work/phase6/defines-source/DeFiNES-{REVISION}'
pins=json.loads((ROOT/'docs/research/evidence/phase0/prior-code.json').read_text())
for entry in pins['files']:
    if sha(source/entry['path'])!=entry['sha256']: raise ValueError('changed DeFiNES source')
sys.path.insert(0,str(source))
from classes.stages.DepthFirstStage import backpropagate_tilesize
from classes.workload.dnn_workload import DNNWorkload

check_frozen();records=[]
for name in ('kws','vww'):
    program,_,_,model_pins=load_model(name)
    for start,stop in supported_segments(program):
        workload=segment_workload(program,start,stop)
        for tile in (4,8):
            shape=program.tensors[program.layers[stop-1].output].shape
            tw,th=min(tile,shape[3]),min(tile,shape[2])
            for horizontal,vertical in ((False,False),(True,False),(True,True)):
                graph=DNNWorkload(workload)
                backpropagate_tilesize(graph,tw,th,horizontal,horizontal,vertical,vertical)
                # Independent in-regime recurrence (no spatial border clipping).
                width,height=tw,th
                for i in reversed(range(start,stop)):
                    layer=program.layers[i];node=graph.get_node_with_id(i)
                    assert (node.loop_dim_size['OX'],node.loop_dim_size['OY'])==(width,height)
                    if layer.op=='Conv':
                        kh,kw=layer.parameters['weight'].shape[2:]
                        sh,sw=layer.attributes.get('strides',[1,1]);dh,dw=layer.attributes.get('dilations',[1,1])
                        iw=(width-1)*sw+(kw-1)*dw+1;ih=(height-1)*sh+(kh-1)*dh+1
                        width=iw-max(0,iw-width*sw) if horizontal else iw
                        height=ih-max(0,ih-height*sh) if vertical else ih
                inp=graph.get_node_with_id(-1)
                assert (inp.loop_dim_size['OX'],inp.loop_dim_size['OY'])==(width,height)
                records.append({'model':name,'start':start,'stop':stop,'requested_tile_limit':tile,'output_tile':[th,tw],
                                'horizontal_cache':horizontal,'vertical_cache':vertical,'new_input_tile':[height,width],
                                'status':'matched-independent-recurrence','model_pins':model_pins})
    print(name+' upstream DeFiNES geometry cross-check passed',flush=True)
report={'schema':1,'status':'passed','physical_board':False,'revision':REVISION,
        'scope':'unmodified upstream in-regime halo/cache geometry; not boundaries, arithmetic, physical fit, latency or full B3',
        'upstream_sources':pins,'source_tar_sha256':sha(ROOT/'work/phase6/defines-source.tar.gz'),
        'upstream_source_sha256':{str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*.py'))},
        'adapter_sha256':sha(ROOT/'compiler/scheduler/defines_adapter.py'),'cases':records}
(ROOT/'work/phase6/optimization/defines-geometry.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
check_frozen()
