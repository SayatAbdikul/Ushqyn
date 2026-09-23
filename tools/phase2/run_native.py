#!/usr/bin/env python3
"""Run 1000 strict protocol-driven MLP jobs quickly in native Verilator."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,default=ROOT/'work/phase2/mlp');a=p.parse_args();a.fixture=a.fixture.resolve()
meta=json.loads((a.fixture/'board.json').read_text());f=np.load(a.fixture/'checks.npz',allow_pickle=False);n=len(f['inputs'])
(a.fixture/'native-inputs.bin').write_bytes(f['inputs'].tobytes());(a.fixture/'native-outputs.bin').write_bytes(f['outputs'].tobytes())
lines=[f"{meta['inputs']['input']} {meta['outputs']['fc3']} {meta['used_bytes']} {n} {meta['macs']}"]
for i,(name,address) in enumerate(meta['tensors'].items()):
    key='layer_'+name
    if key not in f:continue
    data=f[key][:3];filename=f'native-layer-{i}.bin';(a.fixture/filename).write_bytes(data.tobytes());lines.append(f'{address} {data[0].size} {filename}')
(a.fixture/'native.txt').write_text('\n'.join(lines)+'\n')
target=json.loads((ROOT/'hardware/targets/tang_nano_20k_v2.json').read_text());build=ROOT/'work/phase2/native';build.mkdir(parents=True,exist_ok=True)
subprocess.run(['verilator','--cc','--exe','--build','-j','4','--top-module','v2_system','--Mdir',str(build),'-CFLAGS','-std=c++17']+[str(ROOT/s) for s in target['sources']]+[str(ROOT/'test/phase2/mlp_native.cpp')],check=True)
subprocess.run([str(build/'Vv2_system'),str(a.fixture)],check=True)
r=a.fixture/'native-rtl-results.json';data=json.loads(r.read_text());data.update(image_sha256=hashlib.sha256((a.fixture/'board.bin').read_bytes()).hexdigest(),fixture_sha256=hashlib.sha256((a.fixture/'checks.npz').read_bytes()).hexdigest(),sources_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in target['sources']},counter_columns=['elapsed','compute_cycles','wait_cycles','control_cycles','useful_macs','read_bytes','write_bytes','layer_count','protocol_errors']);r.write_text(json.dumps(data,indent=2)+'\n')
