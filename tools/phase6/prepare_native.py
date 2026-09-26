#!/usr/bin/env python3
"""Pin full-model native simulation stimuli and independent layer oracles."""
import json
import sys
import numpy as np
from variants import ROOT, sha, check_frozen
from run_boardless import load_model
from integer_reference import evaluate
from phase4_compile import compile_tiled
from phase4_sequence import compile_sequence


def prepare():
    check_frozen()
    rows=[]
    for name in ('kws','vww'):
        program,x,manifest,pins=load_model(name)
        for sample,value in [('pinned',x),('stress',np.random.default_rng(6062).integers(-128,128,x.shape,dtype=np.int8))]:
            oracle=evaluate(program,{program.inputs[0]:value})
            initial=oracle[program.layers[0].output] if program.layers[0].op=='Transpose' else value
            for half in (False,True):
                plan,image=compile_tiled(program,prefer_half=half)
                for overlap in (False,True):
                    # Snapshots validate every computed layer; separate timed
                    # fixtures have no diagnostic snapshot DMA overhead.
                    for snapshots in ((True,False) if sample=='pinned' else (True,)):
                        label=f'{name}-{sample}-'+('half' if half else 'full')+'-'+('overlap' if overlap else 'serial')+'-'+('check' if snapshots else 'timed')
                        path=ROOT/'work/phase6/optimization/fixtures'/label
                        path.mkdir(parents=True,exist_ok=True)
                        commands,payload,schedule=compile_sequence(plan,image,overlap=overlap,snapshots=snapshots)
                        (path/'commands.bin').write_bytes(commands);(path/'payload.bin').write_bytes(payload)
                        (path/'input.bin').write_bytes(initial.tobytes())
                        checks=[]
                        (path/'output.bin').write_bytes(oracle[program.outputs[0]].tobytes())
                        checks.append(f'{schedule["final_output"]["ext"]} output.bin')
                        for index,region in schedule['snapshot_regions'].items():
                            fn=f'layer-{index}.bin';(path/fn).write_bytes(oracle[program.layers[index].output].tobytes())
                            checks.append(f'{region["ext"]} {fn}')
                        (path/'checks.txt').write_text('\n'.join(checks)+'\n')
                        (path/'schedule.json').write_text(json.dumps(schedule,sort_keys=True,indent=2)+'\n')
                        row={'name':label,'model':name,'sample':sample,'half':half,'overlap':overlap,'snapshots':snapshots,
                             'pins':pins,'files':{p.name:sha(p) for p in sorted(path.iterdir())}}
                        rows.append(row)
        print(name+' fixtures ready',flush=True)
    result={'schema':1,'scope':'independent integer oracle; snapshots check every materialized computed layer','fixtures':rows}
    dest=ROOT/'work/phase6/optimization/fixtures.json';dest.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    check_frozen()

if __name__=='__main__': prepare()
