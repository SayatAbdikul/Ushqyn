#!/usr/bin/env python3
"""Revalidate the existing three-pair schedule with common constant folding.

The command programs are immutable prior artifacts. Only the declared stress
input and its independent integer-oracle tensor checks are regenerated. Both
seeds of the selected pooled RTL executable are run for every exported fixture.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'compiler'),str(ROOT/'tools/phase6')]
from channel_compaction import check_oracles,compact_channels
from followup_graph import group_channels
from run_boardless import load_model
from novelty_constants import model as constant_model,verify_sample
from matched_campaign import hardware_contract,verify,relative,sha,NATIVE_SHA

BASE=ROOT/'work/phase6/matched-baselines-v1/current'
REFERENCE=ROOT/'work/phase6/pair7-fusion-board-v1/physical-short-v1'
FOLDED=ROOT/'work/phase6/novelty-constants-v1/threepair'
NATIVE=ROOT/'work/phase6/pool-timing-v1/native/Vv2_tiled_host_bridge'


def save(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def graph_identity(program):
    """Hash typed shapes, quantizers, topology and every exact parameter byte."""
    from dataclasses import asdict
    tensors={n:asdict(t) for n,t in sorted(program.tensors.items())}
    layers=[]
    for layer in program.layers:
        row=asdict(layer)
        row['parameters']={k:dict(dtype=str(v.dtype),shape=list(v.shape),
            sha256=__import__('hashlib').sha256(v.tobytes()).hexdigest())
            if isinstance(v,np.ndarray) else v for k,v in layer.parameters.items()}
        layers.append(row)
    payload=json.dumps(dict(tensors=tensors,layers=layers,inputs=program.inputs,
                            outputs=program.outputs),sort_keys=True,separators=(',',':')).encode()
    return __import__('hashlib').sha256(payload).hexdigest()


def run(output=BASE):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('choose fresh evidence output')
    hardware=hardware_contract()
    previous=json.loads((REFERENCE/'plan.json').read_text())
    folded=json.loads((FOLDED/'report.json').read_text())
    if (folded['status']!='passed-native' or folded['pooled_native_executable_sha256']!=NATIVE_SHA):
        raise ValueError('constant-folded three-pair native proof missing')
    verify(ROOT,folded['source_sha256'])
    output.mkdir(parents=True)
    sources=dict(folded['source_sha256'])
    for path in (Path(__file__),FOLDED/'report.json',REFERENCE/'plan.json',
                 ROOT/'tools/phase6/matched_campaign.py',
                 ROOT/'tools/phase6/channel_compaction.py',
                 ROOT/'tools/phase6/followup_graph.py',ROOT/'tools/phase6/run_boardless.py',
                 ROOT/'compiler/integer_reference.py',ROOT/'compiler/static_pipeline.py',
                 ROOT/'compiler/quantization.py'):
        sources[relative(path)]=sha(path)
    report=dict(schema=1,status='running',physical_board=False,native_sha256=NATIVE_SHA,
        hardware_sha256=hardware['image']['sha256'],sources=sources,models={},
        scope='existing three-pair schedule plus previously exact symbolic constants; same original models; new seed6157 stress oracle')
    save(output/'report.json',report)
    for model in ('kws','vww'):
        if model=='kws':
            original,pinned,_,model_sources=load_model(model)
            grouped,group_maps,_=group_channels(original)
            program,kept,_,_=compact_channels(grouped)
            maps={n:tuple(group_maps[n][i] for i in selected) for n,selected in kept.items()}
        else:
            original,pinned,grouped,folded_program,program,maps,group_maps,_,_,certificate,model_sources=constant_model()
        report['models'][model]=dict(graph_sha256=graph_identity(program),
                                    model_sources=model_sources,fixtures={})
        sources.update(model_sources)
        for sample in ('pinned','stress'):
            if model=='kws':
                label=previous['fixture_names'][model][sample]
                before=ROOT/'work/phase6/strip-fusion-pair7-v1/full/fixtures'/label
                files=previous['fixtures'][label]['files']
            else:
                label=f'vww-{sample}-known-compacted-strip3-7-11-'+('timed' if sample=='pinned' else 'check')
                before=FOLDED/'fixtures'/label
                files=folded['fixtures'][label]['files']
            verify(before,files)
            directory=output/'fixtures'/f'{model}-{sample}'
            directory.mkdir(parents=True)
            for name in files:shutil.copy2(before/name,directory/name)
            value=pinned if sample=='pinned' else np.random.default_rng(6157).integers(-128,128,pinned.shape,dtype=np.int8)
            oracle=(check_oracles(original,program,maps,value) if model=='kws' else
                    verify_sample(original,folded_program,program,maps,group_maps,certificate,value))
            initial=oracle[program.layers[0].output] if program.layers[0].op in ('Transpose','Reshape') else value
            (directory/'input.bin').write_bytes(initial.tobytes())
            (directory/'output.bin').write_bytes(oracle[program.outputs[0]].tobytes())
            schedule=json.loads((directory/'schedule.json').read_text())
            for i,region in schedule['snapshot_regions'].items():
                data=oracle[program.layers[int(i)].output].tobytes()
                if len(data)!=region['bytes']:raise ValueError('snapshot shape mismatch')
                (directory/f'layer-{i}.bin').write_bytes(data)
            file_pins={name:sha(directory/name) for name in files}
            native=[]
            for seed in (0,6063):
                record_path=output/f'{model}-{sample}-native-s{seed}.json'
                with record_path.with_suffix('.log').open('x') as log:
                    subprocess.run([str(NATIVE),str(directory),str(seed),str(record_path)],
                        check=True,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                row=json.loads(record_path.read_text())
                if row['status']!='passed':raise ValueError('selected schedule native mismatch')
                verify(directory,file_pins)
                row.update(executable_sha256=NATIVE_SHA,fixture_files=file_pins)
                save(record_path,row);native.append(row)
            report['models'][model]['fixtures'][sample]=dict(directory=relative(directory),files=file_pins,
                native=native,replay=dict(status='passed',
                    scope='unchanged command/payload from pinned prior replay proof, new original integer-oracle input and all declared layer checks',
                    prior_directory=relative(before),prior_files=files),
                command_identity_unchanged=file_pins['commands.bin']==files['commands.bin'],
                payload_identity_unchanged=file_pins['payload.bin']==files['payload.bin'])
            save(output/'report.json',report)
            print(model,sample,[r['elapsed_cycles'] for r in native],flush=True)
    verify(ROOT,sources)
    report['status']='passed'
    save(output/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE)
    args=parser.parse_args()
    print(run(args.output)['status'])
