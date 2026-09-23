#!/usr/bin/env python3
"""Physical gate: load once, 1000 inputs, exact output/readback and counter checks."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from host import Client,RESET
p=argparse.ArgumentParser();p.add_argument('--port',required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--jobs',type=int,default=1000);a=p.parse_args()
import serial
meta=json.loads((a.fixture/'board.json').read_text());blob=(a.fixture/'board.bin').read_bytes();f=np.load(a.fixture/'checks.npz',allow_pickle=False)
if a.jobs<1 or a.jobs>len(f['inputs']):raise ValueError('job count exceeds pinned inputs')
if hashlib.sha256(blob).hexdigest()!=meta['image_sha256']:raise ValueError('image digest mismatch')
records=[];begin=time.time()
with serial.Serial(a.port,115200,timeout=2,write_timeout=2) as uart:
    client=Client(uart)
    if client.capabilities()!=len(blob):raise ValueError('target SRAM size mismatch')
    client.exchange(RESET);client.write(0,blob)
    if client.read(0,len(blob))!=blob:raise ValueError('initial readback mismatch')
    for i in range(a.jobs):
        client.write(meta['inputs']['input'],f['inputs'][i].tobytes());r=client.run(meta['entry'])
        for name,address in meta['outputs'].items():
            if client.read(address,f['outputs'][i].size)!=f['outputs'][i].tobytes():raise AssertionError(f'job {i} integer output mismatch')
        if r['useful_macs']!=meta['macs'] or r['elapsed']!=r['compute_cycles']+r['wait_cycles']+r['control_cycles']:raise AssertionError('profiling counter mismatch')
        records.append(r)
        if (i+1)%100==0:print(f'{i+1}/{a.jobs} exact board jobs',flush=True)
report=dict(scope='physical board',port=a.port,jobs=a.jobs,integer_mismatches=0,image_sha256=meta['image_sha256'],fixture_sha256=hashlib.sha256((a.fixture/'checks.npz').read_bytes()).hexdigest(),wall_seconds=time.time()-begin,counters=records)
a.report.write_text(json.dumps(report,indent=2)+'\n')
