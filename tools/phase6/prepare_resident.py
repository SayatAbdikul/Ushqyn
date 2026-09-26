#!/usr/bin/env python3
"""Produce independently checked executable SRAM-retention candidates."""
import json
import numpy as np
from variants import ROOT, sha, check_frozen
from run_boardless import load_model
from integer_reference import evaluate
from scheduler.resident import compile_resident
from scheduler.resident_verify import replay_resident

def prepare():
    check_frozen();base=ROOT/'work/phase6/optimization'
    manifest=base/'fixtures.json';record=json.loads(manifest.read_text())
    rows=[r for r in record['fixtures'] if '-resident-' not in r['name']]
    for name in ('kws','vww'):
        program,x,_,pins=load_model(name)
        for sample,value in [('pinned',x),('stress',np.random.default_rng(6062).integers(-128,128,x.shape,dtype=np.int8))]:
            oracle=evaluate(program,{program.inputs[0]:value})
            initial=oracle[program.layers[0].output] if program.layers[0].op=='Transpose' else value
            for half in (False,True):
                for overlap in (False,True):
                    for snapshots in ((True,False) if sample=='pinned' else (True,)):
                        label=f'{name}-{sample}-resident-'+('half' if half else 'full')+'-'+('overlap' if overlap else 'serial')+'-'+('check' if snapshots else 'timed')
                        path=base/'fixtures'/label;path.mkdir(parents=True,exist_ok=True)
                        commands,payload,schedule=compile_resident(program,prefer_half=half,overlap=overlap,snapshots=snapshots)
                        verification=replay_resident(program,commands,payload,{program.inputs[0]:value},
                            run_contracts=schedule['run_contracts'],final_output=schedule['final_output'],
                            snapshot_regions=schedule['snapshot_regions'],oracle=oracle)
                        (path/'commands.bin').write_bytes(commands);(path/'payload.bin').write_bytes(payload)
                        (path/'input.bin').write_bytes(initial.tobytes());(path/'output.bin').write_bytes(oracle[program.outputs[0]].tobytes())
                        checks=[f'{schedule["final_output"]["ext"]} output.bin']
                        for index,region in schedule['snapshot_regions'].items():
                            fn=f'layer-{index}.bin';(path/fn).write_bytes(oracle[program.layers[index].output].tobytes());checks.append(f'{region["ext"]} {fn}')
                        (path/'checks.txt').write_text('\n'.join(checks)+'\n')
                        (path/'schedule.json').write_text(json.dumps(schedule,sort_keys=True,indent=2)+'\n')
                        rows.append({'name':label,'model':name,'sample':sample,'half':half,'overlap':overlap,'snapshots':snapshots,
                                     'resident':True,'pins':pins,'verification':verification,'files':{p.name:sha(p) for p in sorted(path.iterdir())}})
        print(name+' resident candidates verified',flush=True)
    record['fixtures']=rows;manifest.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n');check_frozen()

if __name__=='__main__': prepare()
