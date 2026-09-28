#!/usr/bin/env python3
"""Source-matched pinned/stress diagnostics for the isolated VWW x-tile cut.

Six compact layer-12 tile stores and the existing complete layer-14 temporary
tensor yield intermediate checks without row scatter or a new routed image.
The frozen timed candidate and comparison evidence are read-only.
"""
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'compiler'),str(ROOT/'tools/phase6')]

from integer_reference import evaluate
from scheduler.defines_verify import replay_defines
import matched_defines_x_tiles as x_tiles

BASE=ROOT/'work/phase6/matched-baselines-v1/b3-x-diagnostics-v1'
TIMED=ROOT/'work/phase6/matched-baselines-v1/b3-x-tiles-v2'
B4=ROOT/'work/phase6/strip-fusion-pair7-v1/full/fixtures'
NATIVE=ROOT/'work/phase6/pool-timing-v1/native/Vv2_tiled_host_bridge'
CMD=struct.Struct('<BBHIII')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def align8(value):return (value+7)&~7


def diagnostic_segment(code,payload,record):
    """Add one exact contiguous layer-12 snapshot DMA after each DW tile."""
    old=[CMD.unpack_from(code,i) for i in range(0,len(code),16)]
    new_payload=bytearray(payload)
    snap_base=align8(len(new_payload))
    new_payload.extend(bytes(snap_base-len(new_payload)))
    locations={}
    for y in (0,8,16):
        for x,width in ((0,16),(16,8)):
            locations[(y,x)]=len(new_payload)
            new_payload.extend(bytes(32*8*width))
    original_runs={row['command']:row for row in record['runs']}
    new=[];mapping={};new_transfers=[]
    for old_index,command in enumerate(old):
        mapping[old_index]=len(new)
        new.append(command)
        if old_index>0 and command==x_tiles.WAIT_ENGINE:
            run=original_runs.get(old_index-1)
            if (run is not None and run['kind']=='depthwise' and
                    run['first_channel']==24):
                y,x,_,width=run['tile']
                ext=locations[(y,x)]
                size=32*8*width
                index=len(new)
                new.append((1,0,0,ext,16160,size))
                new.append(x_tiles.WAIT_DMA)
                new_transfers.append(dict(command=index,direction='from_sram',
                    ext=ext,sram=16160,bytes=size,role='layer12_tile_snapshot',
                    tile=[y,x,8,width]))
    out=copy.deepcopy(record)
    out['runs']=[dict(row,command=mapping[row['command']]) for row in record['runs']]
    out['transfers']=[dict(row,command=mapping[row['command']])
                      for row in record['transfers']]+new_transfers
    out['transfers'].sort(key=lambda row:row['command'])
    out['commands']=len(new)
    packed=b''.join(CMD.pack(*row) for row in new)
    out['code_sha256']=hashlib.sha256(packed).hexdigest()
    out['payload_sha256']=hashlib.sha256(new_payload).hexdigest()
    out['diagnostic_layer12_tiles']={f'y{y}-x{x}':dict(ext=ext,bytes=32*8*(16 if x==0 else 8))
                                      for (y,x),ext in locations.items()}
    assert len(new_transfers)==6 and len(new)==record['commands']+12
    return packed,bytes(new_payload),out


def raw_input(graph,physical):
    first=graph.layers[0]
    assert first.op=='Transpose' and first.attributes['perm']==[0,3,1,2]
    shape=graph.tensors[first.output].shape
    return np.frombuffer(physical,np.int8).reshape(shape).transpose(
        np.argsort(first.attributes['perm'])).copy()


def main():
    assert not BASE.exists(), 'choose a fresh diagnostic output directory'
    timed_report=json.loads((TIMED/'report.json').read_text())
    assert timed_report['status']=='passed-native'
    compacted,block,_,model_sources=x_tiles.pair11.model_block()
    old_payload=(x_tiles.SOURCE/'payload.bin').read_bytes()
    segment,payload,base_record=x_tiles.build(block,old_payload)
    assert base_record['code_sha256']==timed_report['segment']['code_sha256']
    code,payload,record=diagnostic_segment(segment,payload,base_record)
    BASE.mkdir(parents=True)
    results={}
    for sample,reference in (
        ('pinned_check',B4/'vww-pinned-compacted-strip3-7-11-check'),
        ('stress_check',B4/'vww-stress-compacted-strip3-7-11-check')):
        value=raw_input(compacted,(reference/'input.bin').read_bytes())
        oracle=evaluate(compacted,{compacted.inputs[0]:value})
        assert oracle[compacted.outputs[0]].tobytes()==(reference/'output.bin').read_bytes()
        source=oracle[block.inputs[0]]
        layer12=oracle[compacted.layers[12].output]
        layer14=oracle[block.outputs[0]]
        segment_replay=x_tiles.replay_segment(block,source,layer14,code,payload,record)
        ref_schedule=json.loads((reference/'schedule.json').read_text())
        source_replay=replay_defines(compacted,(reference/'commands.bin').read_bytes(),
            (reference/'payload.bin').read_bytes(),{compacted.inputs[0]:value},
            run_contracts=ref_schedule['run_contracts'],
            constant_contracts=ref_schedule.get('constant_contracts'),
            pack_contracts=ref_schedule.get('pack_contracts'),
            final_output=ref_schedule['final_output'],
            snapshot_regions=ref_schedule['snapshot_regions'],oracle=oracle)
        assert segment_replay['status']==source_replay['status']=='passed'
        fixture=BASE/'fixtures'/sample
        splice=x_tiles.splice(x_tiles.SOURCE,code,payload,record,fixture)
        (fixture/'input.bin').write_bytes((reference/'input.bin').read_bytes())
        (fixture/'output.bin').write_bytes((reference/'output.bin').read_bytes())
        lines=[f"{ref_schedule['final_output']['ext']} output.bin"]
        layer14_file='layer14-full.bin'
        (fixture/layer14_file).write_bytes(layer14.tobytes())
        lines.append(f"{record['output_temp_ext']} {layer14_file}")
        for y in (0,8,16):
            for x,width in ((0,16),(16,8)):
                name=f'layer12-y{y}-x{x}.bin'
                (fixture/name).write_bytes(layer12[:,:,y:y+8,x:x+width].tobytes())
                lines.append(f"{record['diagnostic_layer12_tiles'][f'y{y}-x{x}']['ext']} {name}")
        (fixture/'checks.txt').write_text('\n'.join(lines)+'\n')
        schedule=json.loads((fixture/'schedule.json').read_text())
        schedule['diagnostic_checks_enabled']=True
        schedule['diagnostic_layer12_tiles']=record['diagnostic_layer12_tiles']
        schedule['diagnostic_layer14_full_ext']=record['output_temp_ext']
        (fixture/'schedule.json').write_text(json.dumps(schedule,sort_keys=True,indent=2)+'\n')
        native=[]
        for seed in (0,6063):
            path=BASE/f'{sample}-native-s{seed}.json'
            subprocess.run([str(NATIVE),str(fixture),str(seed),str(path)],cwd=ROOT,check=True)
            row=json.loads(path.read_text())
            assert row['status']=='passed' and row['tensor_checks']==8
            native.append(dict(row,report=str(path.relative_to(ROOT)),report_sha256=sha(path)))
        results[sample]=dict(directory=str(fixture.relative_to(ROOT)),
            files={p.name:sha(p) for p in fixture.iterdir() if p.is_file()},
            replay=dict(status='passed',source=source_replay,segment=segment_replay),
            splice=splice,native=native,reference_input_sha256=sha(reference/'input.bin'),
            reference_output_sha256=sha(reference/'output.bin'))
    report=dict(schema=1,status='passed-native',physical_board=False,
        scope='pinned/stress full-model x-tile diagnostics, six layer12 tiles and full layer14',
        timed_report=str((TIMED/'report.json').relative_to(ROOT)),
        timed_report_sha256=sha(TIMED/'report.json'),
        selected_native_executable_sha256=sha(NATIVE),
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
                       str(Path(x_tiles.__file__).relative_to(ROOT)):sha(x_tiles.__file__)},
        model_source_sha256=model_sources,
        segment=dict(record,code_sha256=hashlib.sha256(code).hexdigest(),
                     payload_sha256=hashlib.sha256(payload).hexdigest()),
        fixtures=results)
    (BASE/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],diagnostic_commands=record['commands'],
        fixture_commands=[value['splice']['command_count'] for value in results.values()],
        native_cycles={key:[row['elapsed_cycles'] for row in value['native']]
                       for key,value in results.items()}),sort_keys=True))


if __name__=='__main__':main()
