#!/usr/bin/env python3
"""Execute frozen dense/factor models on the unchanged selected native engine.

All materialized layers are checked in diagnostic runs. Separate timed runs
exclude layer-snapshot DMA. The external RAM is an abstract fixed/stalled model;
these results do not establish physical SDRAM, board latency, accuracy or energy.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import struct
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'work/phase6/representation-native-v1'
sys.path[:0] = [str(ROOT/'compiler'),str(ROOT/'tools/phase6')]
from integer_reference import evaluate
from phase4_compile import compile_tiled
from phase4_sequence import compile_sequence
from program_image import load_image
from run_boardless import load_model

ENGINE_SHA = '9f934dcc27ae891e4c7b8b84bcc4b65e82d903492c1867572784fbe1cd9484ee'
NATIVE_DIR = ROOT/'work/phase6/deadline-screen-v1/native-build'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):
            h.update(part)
    return h.hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def native_identity():
    path = NATIVE_DIR/'identity.json'
    identity = json.loads(path.read_text())
    exe = NATIVE_DIR/'Vv2_tiled_host_bridge'
    if identity['engine_sha256'] != ENGINE_SHA or sha(exe) != identity['executable_sha256']:
        raise ValueError('selected native executable identity mismatch')
    for source,digest in identity['sources_sha256'].items():
        if sha(Path(source)) != digest:
            raise ValueError('native source changed: '+source)
    harness = ROOT/'work/phase6/deadline-screen-v1/trace.cpp'
    if sha(harness) != identity['harness_sha256']:
        raise ValueError('native harness changed')
    return exe,identity,sha(path)


def programs(name):
    factor_path = ROOT/f'work/phase6/representation-screen-v1/{name}/selected-int8.uq2'
    screen_path = ROOT/f'work/phase6/representation-screen-v1/{name}/report.json'
    screen = json.loads(screen_path.read_text())
    if sha(factor_path) != screen['export']['sha256']:
        raise ValueError('frozen factor export changed')
    factor = load_image(factor_path.read_bytes())
    if name == 'vww':
        dense,x,_,pins = load_model(name)
    else:
        dense_path = ROOT/'work/phase6/parallel-ad-v1/ad.uq2'
        dense = load_image(dense_path.read_bytes())
        input_path = ROOT/'work/phase6/parallel-ad-v1/native/frame-0/input.bin'
        x = np.frombuffer(input_path.read_bytes(),np.int8).reshape(dense.tensors[dense.inputs[0]].shape)
        pins = {str(p.relative_to(ROOT)):sha(p) for p in (dense_path,input_path)}
    if dense.inputs != factor.inputs or dense.outputs != factor.outputs:
        raise ValueError('factor external graph interface differs')
    for n in dense.inputs+dense.outputs:
        if dense.tensors[n] != factor.tensors[n]:
            raise ValueError('factor input/output tensor contract differs')
    pins.update({str(factor_path.relative_to(ROOT)):sha(factor_path),
                 str(screen_path.relative_to(ROOT)):sha(screen_path)})
    return {'dense':dense,'factor':factor},x,pins,screen


def geometry(program,plan,image):
    macs = weight_codes = weight_rows = params = 0
    for layer in program.layers:
        if layer.op in ('Conv','Gemm'):
            w = layer.parameters['weight']
            reduction = math.prod(w.shape[1:])
            macs += math.prod(program.tensors[layer.output].shape)*reduction
            weight_codes += w.size
            weight_rows += w.shape[0]*((reduction+7)//8*8)
            params += w.shape[0]*16
        elif layer.parameters:
            params += 16
    peak = max(t['scratch_bytes'] for layer in plan['layers'] for t in layer['tiles'])
    if peak > 32768 or len(image)>8*1024*1024:
        raise ValueError('planned memory capacity exceeded')
    return {'layers':len(program.layers),'materialized_layers':sum(bool(l['tiles']) for l in plan['layers']),
            'tiles':sum(len(l['tiles']) for l in plan['layers']),
            'useful_macs':macs,'logical_int8_weight_bytes':weight_codes,
            'aligned_weight_row_bytes':weight_rows,'hardware_parameter_table_bytes':params,
            'parameter_image_bytes':len(image),'activation_slot_bytes':plan['activation_slot_bytes'],
            'peak_data_scratch_bytes':peak,
            'layer_geometry':[{'index':l['index'],'kind':l['kind'],'tiles':len(l['tiles']),
                               'output_bytes':math.prod(program.tensors[program.layers[l['index']].output].shape),
                               'peak_scratch_bytes':max((t['scratch_bytes'] for t in l['tiles']),default=0)}
                              for l in plan['layers']]}


def prepare(program,plan,image,value,folder,snapshots):
    oracle = evaluate(program,{program.inputs[0]:value})
    first = program.layers[0]
    initial = oracle[first.output] if first.op=='Transpose' else value
    command,payload,schedule = compile_sequence(plan,image,overlap=True,snapshots=snapshots)
    if len(command)>32768 or len(payload)>8*1024*1024:
        raise ValueError('command or external image capacity exceeded')
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'commands.bin').write_bytes(command)
    (folder/'payload.bin').write_bytes(payload)
    (folder/'input.bin').write_bytes(initial.tobytes())
    (folder/'output.bin').write_bytes(oracle[program.outputs[0]].tobytes())
    checks = [f'{schedule["final_output"]["ext"]} output.bin']
    for i,region in schedule['snapshot_regions'].items():
        name = f'layer-{i}.bin'
        output = oracle[program.layers[i].output]
        if output.size != region['bytes']:
            raise ValueError('snapshot shape/region differs')
        (folder/name).write_bytes(output.tobytes())
        checks.append(f'{region["ext"]} {name}')
    if snapshots and len(schedule['snapshot_regions']) != sum(bool(l['tiles']) for l in plan['layers']):
        raise ValueError('materialized layer missing diagnostic snapshot')
    (folder/'checks.txt').write_text('\n'.join(checks)+'\n')
    save(folder/'schedule.json',schedule)
    encoded = list(struct.iter_unpack('<BBHIII',command))
    transfers = [r for r in encoded if r[0]==1]
    peak_live = max(t['live'][1] for t in schedule['tiles'])
    if peak_live > 32768:
        raise ValueError('live SRAM interval outside physical capacity')
    metadata = {'snapshots':snapshots,'command_count':len(encoded),'program_bytes':len(command),
                'payload_bytes':len(payload),'peak_sram_address':peak_live,
                'external_address_highwater':max([len(payload),*(r['ext']+r['bytes'] for r in schedule['snapshot_regions'].values())]),
                'diagnostic_tensor_checks':len(checks),'diagnostic_snapshot_layers':list(schedule['snapshot_regions']),
                'nonmaterialized_layers':[l['index'] for l in plan['layers'] if not l['tiles']],
                'scheduled_dma_bytes':sum(r[5] for r in transfers),
                'scheduled_dma_to_sram_bytes':sum(r[5] for r in transfers if r[1]),
                'scheduled_dma_to_external_bytes':sum(r[5] for r in transfers if not r[1]),
                'oracle_output_sha256':sha(folder/'output.bin'),
                'fixture_files_sha256':{p.name:sha(p) for p in folder.iterdir() if p.is_file()}}
    save(folder/'fixture.json',metadata)
    return metadata


def execute(exe,folder,dest,seed,expected):
    start = time.perf_counter()
    invocation = subprocess.run([str(exe),str(folder),str(seed),str(dest)],cwd=ROOT,
                                capture_output=True,text=True,timeout=180)
    if invocation.returncode:
        failure = {'status':'failed','returncode':invocation.returncode,
                   'stdout':invocation.stdout[-2000:],'stderr':invocation.stderr[-2000:]}
        save(dest,failure)
        raise RuntimeError('selected native rejected fixture: '+str(folder)+' '+invocation.stderr)
    result = json.loads(dest.read_text())
    trace_path = Path(str(dest)+'.trace.json')
    trace = json.loads(trace_path.read_text())
    if result['status']!='passed' or result['tensor_checks']!=expected['diagnostic_tensor_checks']:
        raise ValueError('native output checks incomplete')
    if len(trace)!=expected['command_count']+1 or trace[-1]['elapsed']!=result['elapsed_cycles']:
        raise ValueError('native command trace incomplete')
    result.update(simulation_seconds=time.perf_counter()-start,
                  native_trace_sha256=sha(trace_path),native_report_sha256=sha(dest))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=('vww','ad'),default=['vww','ad'])
    args = parser.parse_args()
    BASE.mkdir(parents=True,exist_ok=True)
    start = time.perf_counter()
    exe,identity,identity_sha = native_identity()
    compiler_files = [ROOT/'compiler'/n for n in ('phase4_compile.py','phase4_sequence.py','phase4_tiling.py',
                                                 'hardware_v2.py','integer_reference.py','quantization.py','program_image.py')]
    compiler_pins = {str(p.relative_to(ROOT)):sha(p) for p in compiler_files}
    report = {'status':'running','scope':__doc__,'source_sha256':sha(Path(__file__)),
              'native_identity':identity,'native_identity_sha256':identity_sha,
              'native_executable':str(exe),'compiler_source_sha256':compiler_pins,'models':{},
              'threads':1,'native_build_jobs':0,'native_build_reused':True,
              'physical_board':False,'selected_engine_changed':False}
    for name in args.models:
        print('begin',name,flush=True)
        programs_by_name,real,pins,screen = programs(name)
        model = {'source_pins':pins,'variants':{},'comparisons':[],
                 'factor_screen_selected':screen['selection']['selected']}
        stress = np.random.default_rng(6062).integers(-128,128,real.shape,dtype=np.int8)
        for variant,program in programs_by_name.items():
            plan,image = compile_tiled(program)
            record = {'geometry':geometry(program,plan,image),'fixtures':{},'runs':[]}
            for sample,value in (('pinned',real),('stress',stress)):
                for mode,snapshots in (('check',True),('timed',False)):
                    label = f'{name}-{variant}-{sample}-{mode}'
                    folder = BASE/'fixtures'/label
                    metadata = prepare(program,plan,image,value,folder,snapshots)
                    record['fixtures'][f'{sample}/{mode}'] = metadata
                    for seed in (0,6063):
                        dest = BASE/'native'/f'{label}-s{seed}.json'
                        dest.parent.mkdir(exist_ok=True)
                        result = execute(exe,folder,dest,seed,metadata)
                        result.update(sample=sample,mode=mode,seed=seed,fixture=str(folder.relative_to(ROOT)))
                        record['runs'].append(result)
                        print(label,seed,result['elapsed_cycles'],'cycles',flush=True)
            model['variants'][variant] = record
            report['models'][name] = model
            save(BASE/'report.json',report)
        for sample in ('pinned','stress'):
            for seed in (0,6063):
                dense = next(r for r in model['variants']['dense']['runs'] if (r['sample'],r['mode'],r['seed'])==(sample,'timed',seed))
                factor = next(r for r in model['variants']['factor']['runs'] if (r['sample'],r['mode'],r['seed'])==(sample,'timed',seed))
                model['comparisons'].append({'sample':sample,'seed':seed,
                    'dense_elapsed_cycles':dense['elapsed_cycles'],'factor_elapsed_cycles':factor['elapsed_cycles'],
                    'elapsed_cycle_reduction_fraction':(dense['elapsed_cycles']-factor['elapsed_cycles'])/dense['elapsed_cycles'],
                    'dense_engine_cycles':dense['engine_cycles'],'factor_engine_cycles':factor['engine_cycles'],
                    'engine_cycle_reduction_fraction':(dense['engine_cycles']-factor['engine_cycles'])/dense['engine_cycles'],
                    'dense_dma_cycles':dense['dma_cycles'],'factor_dma_cycles':factor['dma_cycles']})
        save(BASE/name/'report.json',model)
        save(BASE/'report.json',report)
    for path,digest in compiler_pins.items():
        if sha(ROOT/path)!=digest:
            raise ValueError('compiler changed during native run')
    native_identity()
    for model in report['models'].values():
        for path,digest in model['source_pins'].items():
            if sha(ROOT/path)!=digest:
                raise ValueError('model pin changed during native run')
    report.update(status='passed-native-full-model-comparison',elapsed_seconds=time.perf_counter()-start,
                  peak_python_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                  peak_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                  physical_resources='Unchanged selected engine/host RTL; no new synthesis or FPGA bitstream needed for the numerical model change. Board qualification remains unmeasured.')
    save(BASE/'report.json',report)


if __name__=='__main__':main()
