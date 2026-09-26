#!/usr/bin/env python3
"""Optimize actual bytecode with overlap, retaining current schedules as fallbacks."""
import argparse
import json
import platform
import sys
import numpy as np
from variants import ROOT, sha, check_frozen
from run_boardless import load_model
from integer_reference import evaluate
from scheduler.current_abi import CostModel
from scheduler.resident import compile_resident
from scheduler.resident_search import optimize_resident
from scheduler.resident_verify import replay_resident

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--spatial',action='store_true',help='Use the spatial pointwise engine cost model')
args=parser.parse_args();family='spatial-search' if args.spatial else 'search'
label_family='spatial-searched' if args.spatial else 'searched'
base=ROOT/'work/phase6/optimization';manifest=base/'fixtures.json'
record=json.loads(manifest.read_text())
record['fixtures']=[f for f in record['fixtures'] if not (f.get('searched') and f.get('spatial_search',False)==args.spatial)]
cost_path=ROOT/'docs/research/evidence/phase4/physical-sequence-costs.json'
cost=CostModel(json.loads(cost_path.read_text()));check_frozen()
for name in ('kws','vww'):
    program,x,_,pins=load_model(name)
    _,report=optimize_resident(program,cost.dma,spatial_pw=args.spatial)
    report.update(physical_board=False,host={'platform':platform.platform(),'python':sys.version.split()[0]},
                  dma_calibration_sha256=sha(cost_path),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
                    [*sorted((ROOT/'compiler/scheduler').glob('*.py')),ROOT/'rtl/phase6'/('spatial_engine.sv' if args.spatial else 'engine.sv')]})
    selected=report['selected']
    for sample,value in [('pinned',x),('stress',np.random.default_rng(6062).integers(-128,128,x.shape,dtype=np.int8))]:
        oracle=evaluate(program,{program.inputs[0]:value})
        initial=oracle[program.layers[0].output] if program.layers[0].op=='Transpose' else value
        for snapshots in ((True,False) if sample=='pinned' else (True,)):
            commands,payload,s=compile_resident(program,fused=selected['fused'],prefer_half=False,
                 tile_choices={i:i in selected['half_layers'] for i in range(len(program.layers))},
                 overlap=selected['overlap'],snapshots=snapshots)
            verification=replay_resident(program,commands,payload,{program.inputs[0]:value},
                run_contracts=s['run_contracts'],final_output=s['final_output'],snapshot_regions=s['snapshot_regions'],oracle=oracle)
            label=f'{name}-{sample}-{label_family}-'+('check' if snapshots else 'timed');path=base/'fixtures'/label
            path.mkdir(parents=True,exist_ok=True)
            (path/'commands.bin').write_bytes(commands);(path/'payload.bin').write_bytes(payload)
            (path/'input.bin').write_bytes(initial.tobytes());(path/'output.bin').write_bytes(oracle[program.outputs[0]].tobytes())
            checks=[f'{s["final_output"]["ext"]} output.bin']
            for index,r in s['snapshot_regions'].items():
                fn=f'layer-{index}.bin';(path/fn).write_bytes(oracle[program.layers[index].output].tobytes());checks.append(f'{r["ext"]} {fn}')
            (path/'checks.txt').write_text('\n'.join(checks)+'\n');(path/'schedule.json').write_text(json.dumps(s,indent=2,sort_keys=True)+'\n')
            record['fixtures'].append({'name':label,'model':name,'sample':sample,'snapshots':snapshots,'searched':True,
                'spatial_search':args.spatial,'pins':pins,'verification':verification,'files':{p.name:sha(p) for p in sorted(path.iterdir())}})
    (base/f'{name}-{family}.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(name,report['objective'],report['bound_gap_fraction'],report['wall_seconds'],flush=True)
manifest.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n');check_frozen()
