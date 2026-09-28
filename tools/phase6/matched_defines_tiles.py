#!/usr/bin/env python3
"""Independent extra DeFiNES-style full-width tile-height screen.

This explores layer-11..14 fusion with 4/6/8/12 output rows per strip on the
same fixed INT8 graph and 27 MHz eight-lane backend.  It does not modify the
frozen restricted-B3 catalogue or claim coverage of non-full-width tiles.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'compiler'), str(ROOT/'tools/phase6')]

from hardware_v2 import Descriptor
from integer_reference import evaluate
from phase4_compile import _parameter_rows
from scheduler.defines_verify import replay_defines
from output_pipeline_fusion import activation_table, decode_fused, encode_fused
from run_boardless import load_model
from followup_graph import group_channels
from channel_compaction import compact_channels
import strip_fusion_pair7 as pair7_builder
import strip_fusion_pair11 as pair11_builder
import matched_defines_adapt as b3

OUT = ROOT/'work/phase6/matched-baselines-v1/b3-spatial-v1'
NATIVE = ROOT/'work/phase6/pool-timing-v1/native/Vv2_tiled_host_bridge'
CMD = struct.Struct('<BBHIII')
HALT = (0,0,0,0,0,0)
WAIT_DMA = (3,2,0,0,0,0)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2)+'\n')


def align8(value: int) -> int:
    return (value+7)&~7


def strip_geometry(height: int, index: int) -> dict:
    assert height in (4,6,8,12) and 24%height==0 and 0<=index<24//height
    y0=index*height
    input_y0=max(0,y0-1)
    input_y1=min(24,y0+height+1)
    return dict(index=index, origin_y=y0, output_h=height, input_y0=input_y0,
                input_y1=input_y1, input_h=input_y1-input_y0,
                pad_top=int(y0==0), pad_bottom=int(y0+height==24),
                input_plane_bytes=(input_y1-input_y0)*24,
                output_plane_bytes=height*24)


def build(block, height: int) -> tuple[bytes, bytes, dict]:
    """Lower one full-width 11..14 depth-first cut at a chosen strip height."""
    payload=bytearray(18432)  # layer-10 source tensor, later in the model slot

    def put(data: bytes) -> int:
        pos=align8(len(payload));payload.extend(bytes(pos-len(payload)));payload.extend(data)
        return pos

    rows=[_parameter_rows(block,layer) for layer in block.layers]
    external=[]
    for weights,params in rows:
        external.append(dict(weight=put(weights) if weights else None,
                             params=put(params) if params else None,
                             weight_bytes=weights,params_bytes=params))
    tables=[activation_table(2,external[i]['params_bytes']) for i in (1,3)]
    table_external=[put(table) for table in tables]
    commands=[];transfers=[];runs=[]

    def emit(op: int, flags: int=0, a: int=0, b: int=0, c: int=0):
        commands.append((op,flags,0,a,b,c))

    def dma(direction: str, external_address: int, sram: int, size: int, role: str):
        assert external_address%8==sram%8==0 and 0<size<=32768
        assert external_address+size<=len(payload) and sram+size<=32768
        index=len(commands)
        emit(1,int(direction=='to_sram'),external_address,sram,size);commands.append(WAIT_DMA)
        transfers.append(dict(command=index,direction=direction,role=role,
                              ext=external_address,sram=sram,bytes=size))

    def run(layer: int, strip: int, descriptor: Descriptor, table_sram: int, table_id: int):
        descriptor.validate()
        encoded=encode_fused(descriptor,table_sram)
        assert decode_fused(encoded)==(descriptor,table_sram)
        desc_ext=put(encoded+Descriptor(0).encode())
        dma('to_sram',desc_ext,0,128,'descriptor')
        dma('to_sram',external[layer]['weight'],descriptor.weight,
            len(external[layer]['weight_bytes']),'weight')
        dma('to_sram',external[layer]['params'],descriptor.params,
            len(external[layer]['params_bytes']),'params')
        dma('to_sram',table_external[table_id],table_sram,256,'activation_table')
        index=len(commands);emit(2,0,0,32768<<16);emit(3,3)
        runs.append(dict(command=index,layer=layer,strip=strip,
                         descriptor_hex=encoded.hex(),table_sha256=sha(tables[table_id])))

    geometry=[strip_geometry(height,i) for i in range(24//height)]
    for strip,geo in enumerate(geometry):
        source_base=128 if strip%2==0 else 10112
        resident_base=10112 if strip%2==0 else 128
        input_plane=geo['input_plane_bytes'];output_plane=geo['output_plane_bytes']
        if strip==0:
            for channel in range(32):
                dma('to_sram',channel*576+geo['input_y0']*24,
                    source_base+channel*input_plane,input_plane,f'source_strip{strip}')
        dw_weight=19328 if strip%2==0 else 20096
        dw_params=dw_weight+512;dw_table=dw_params+512
        dw=Descriptor(6,input=source_base,output=resident_base,weight=dw_weight,
            params=dw_params,count=9,outputs=32*output_plane,row_stride=16,next_pc=64,
            kernel_h=3,kernel_w=3,pad_top=geo['pad_top'],
            pad_bottom=geo['pad_bottom'],pad_left=1,pad_right=1,
            input_h=geo['input_h'],input_w=24,input_c=32,output_c=32)
        run(0,strip,dw,dw_table,0)
        pw_weight=19328;pw_params=pw_weight+1024;pw_table=pw_params+512
        pw=Descriptor(4,input=resident_base,output=source_base,weight=pw_weight,
            params=pw_params,count=32,outputs=32*output_plane,row_stride=32,next_pc=64,
            input_h=height,input_w=24,input_c=32,output_c=32)
        run(2,strip,pw,pw_table,1)
        if strip+1<len(geometry):
            next_geo=geometry[strip+1]
            next_plane=next_geo['input_plane_bytes']
            for channel in range(32):
                dma('to_sram',channel*576+next_geo['input_y0']*24,
                    resident_base+channel*next_plane,next_plane,
                    f'prefetch_source_strip{strip+1}')
        for channel in range(32):
            dma('from_sram',channel*576+geo['origin_y']*24,
                source_base+channel*output_plane,output_plane,
                f'output_strip{strip}')
    commands.append(HALT)
    code=b''.join(CMD.pack(*command) for command in commands)
    assert len(commands)<=2048 and len(code)<=32768 and len(payload)<=8*1024*1024
    activation_loads=sum(t['bytes'] for t in transfers if t['role'].startswith(('source_strip','prefetch_source')))
    activation_stores=sum(t['bytes'] for t in transfers if t['role'].startswith('output_strip'))
    report=dict(schema=1,status='lowered',tile=[height,24],strips=len(geometry),geometry=geometry,
                command_count=len(commands),payload_bytes=len(payload),
                activation_load_bytes=activation_loads,
                activation_store_bytes=activation_stores,
                max_sram_address_exclusive=21376,code_sha256=sha(code),payload_sha256=sha(payload),
                transfers=transfers,runs=runs)
    return code,bytes(payload),report


def splice(source_folder: Path, target_folder: Path, block_code: bytes,
           block_payload: bytes, plan: dict) -> dict:
    """Replace the layerwise 11..14 segment while preserving other layers."""
    old_code=(source_folder/'commands.bin').read_bytes()
    old_payload=(source_folder/'payload.bin').read_bytes()
    old_schedule=json.loads((source_folder/'schedule.json').read_text())
    old=[CMD.unpack_from(old_code,i) for i in range(0,len(old_code),CMD.size)]
    stages=old_schedule['stages']
    first=next(stage for stage in stages if stage['layer']==11)
    following=next(stage for stage in stages if stage['layer']==15)

    def descriptor_index(stage):
        load=next(t for t in stage['loads'] if t['role']=='descriptor')
        command=(1,1,0,load['ext'],load['sram'],load['bytes'])
        hits=[i for i,item in enumerate(old) if item==command]
        assert len(hits)==1,(stage['layer'],hits)
        return hits[0]

    cut_first,cut_end=descriptor_index(first),descriptor_index(following)
    assert cut_first<cut_end and old[cut_first-1]==WAIT_DMA
    assert all(not cut_first<=int(key)<cut_end or 11<=value['layer']<=14
               for key,value in old_schedule['run_contracts'].items())
    activation_ext=next(item['ext'] for item in first['loads'] if item['role']=='input')
    assert activation_ext==36864
    removed=old[cut_first:cut_end]
    preserved=[]
    for load in following['loads']:
        if load['role']!='parameter':continue
        command=(1,1,0,load['ext'],load['sram'],load['bytes'])
        matches=[i for i,item in enumerate(removed) if item==command]
        if matches:
            assert len(matches)==1 and removed[matches[0]+1]==WAIT_DMA
            preserved.extend((command,WAIT_DMA))
    payload=bytearray(old_payload)
    appended=align8(len(payload));payload.extend(bytes(appended-len(payload)))
    payload.extend(block_payload[18432:])
    commands=[CMD.unpack_from(block_code,i) for i in range(0,len(block_code),CMD.size)]
    transfers={item['command']:item for item in plan['transfers']}
    runs={item['command']:item for item in plan['runs']}
    snapshots={int(key):value for key,value in old_schedule['snapshot_regions'].items()}
    inserted=[];new_runs=[]
    for i,command in enumerate(commands[:-1]):
        op,flags,reserved,a,b,c=command
        if op==1:
            transfer=transfers[i]
            if transfer['role'].startswith(('source_strip','prefetch_source','output_strip')):
                a+=activation_ext
            else:
                assert flags==1 and a>=18432
                a=appended+a-18432
            command=(op,flags,reserved,a,b,c)
        if op==2:new_runs.append((len(inserted),runs[i]))
        inserted.append(command)
        if op==3 and flags==3 and i-1 in runs:
            stage=runs[i-1]
            layer=12 if stage['layer']==0 else 14
            if layer in snapshots:
                descriptor,_=decode_fused(bytes.fromhex(stage['descriptor_hex']))
                height=plan['tile'][0];output_plane=height*24
                for channel in range(32):
                    external=snapshots[layer]['ext']+channel*576+stage['strip']*output_plane
                    sram=descriptor.output+channel*output_plane
                    assert external%8==sram%8==0
                    inserted.extend(((1,0,0,external,sram,output_plane),WAIT_DMA))
    inserted.extend(preserved)
    combined=old[:cut_first]+inserted+old[cut_end:]
    code=b''.join(CMD.pack(*item) for item in combined)
    assert len(combined)<=2048 and len(code)<=32768 and len(payload)<=8*1024*1024
    delta=len(inserted)-(cut_end-cut_first)
    schedule=copy.deepcopy(old_schedule)
    schedule['stages']=[stage for stage in stages if not 11<=stage['layer']<=14]
    for field in ('run_contracts','constant_contracts','pack_contracts'):
        schedule[field]={str(i if i<cut_first else i+delta):value
                         for key,value in old_schedule.get(field,{}).items()
                         if not cut_first<=(i:=int(key))<cut_end}
    height=plan['tile'][0]
    for offset,stage in new_runs:
        schedule['run_contracts'][str(cut_first+offset)]=dict(
            layer=11 if stage['layer']==0 else 13,
            fused_activation_layer=12 if stage['layer']==0 else 14,
            fused_table_sha256=stage['table_sha256'],
            strip_origin_y=stage['strip']*height,strip_height=height)
    schedule.update(command_count=len(combined),program_bytes=len(code),
                    program_sha256=sha(code),image_sha256=sha(payload),
                    catalogue=f'additional DeFiNES-style full-width 11-14 tile height {height}',
                    pair11_spatial_splice=dict(source_fixture=source_folder.name,
                        source_code_sha256=sha(old_code),source_payload_sha256=sha(old_payload),
                        cut_first=cut_first,cut_end=cut_end,inserted_commands=len(inserted),
                        tile=[height,24],strips=plan['strips']))
    target_folder.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source_folder,target_folder,dirs_exist_ok=True)
    (target_folder/'commands.bin').write_bytes(code)
    (target_folder/'payload.bin').write_bytes(payload)
    save(target_folder/'schedule.json',schedule)
    return dict(status='lowered',command_count=len(combined),payload_bytes=len(payload),
                code_sha256=sha(code),payload_sha256=sha(payload))


def compacted_vww():
    original,_,_,_=load_model('vww')
    grouped,_,_=group_channels(original)
    compacted,_,_,_=compact_channels(grouped)
    return compacted


def symbolic(fixture: Path, graph, label: str) -> dict:
    schedule=json.loads((fixture/'schedule.json').read_text())
    first=graph.layers[0]
    assert first.op=='Transpose'
    physical=graph.tensors[first.output].shape
    value=np.frombuffer((fixture/'input.bin').read_bytes(),np.int8).reshape(physical).transpose(
        np.argsort(first.attributes['perm']))
    oracle=evaluate(graph,{graph.inputs[0]:value})
    assert oracle[graph.outputs[0]].tobytes()==(fixture/'output.bin').read_bytes()
    result=replay_defines(graph,(fixture/'commands.bin').read_bytes(),
                          (fixture/'payload.bin').read_bytes(),{graph.inputs[0]:value},
                          run_contracts=schedule['run_contracts'],
                          final_output=schedule['final_output'],
                          snapshot_regions=schedule.get('snapshot_regions'),
                          constant_contracts=schedule.get('constant_contracts'),
                          pack_contracts=schedule.get('pack_contracts'),oracle=oracle)
    path=OUT/'replay'/f'{label}.json'
    save(path,dict(status='passed',model='vww',fixture=str(fixture.relative_to(ROOT)),
                   fixture_files_sha256={name:file_sha(fixture/name) for name in
                       ('commands.bin','payload.bin','input.bin','output.bin','schedule.json')},
                   oracle_output_sha256=sha(oracle[graph.outputs[0]].tobytes()),replay=result))
    return dict(status='passed',report=str(path.relative_to(ROOT)),report_sha256=file_sha(path),
                engine_runs=result['engine_runs'],dma_bytes=result['dma_bytes'])


def native(fixture: Path, label: str) -> list[dict]:
    result=[]
    for seed in (0,6063):
        report=OUT/'native'/f'{label}-s{seed}.json'
        report.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run([str(NATIVE),str(fixture),str(seed),str(report)],check=True,cwd=ROOT)
        raw=json.loads(report.read_text())
        assert raw['status']=='passed' and raw['tensor_checks']>=1
        result.append(dict(stall_seed=seed,status='passed',tensor_checks=raw['tensor_checks'],
                           elapsed_cycles=raw['elapsed_cycles'],dma_cycles=raw['dma_cycles'],
                           engine_cycles=raw['engine_cycles'],
                           report=str(report.relative_to(ROOT)),report_sha256=file_sha(report)))
    return result


def verify_fixture(fixture: Path, graph, label: str) -> dict:
    row=b3.checked_fixture(fixture)
    row['replay']=symbolic(fixture,graph,label)
    row['native']=native(fixture,label)
    row['worst_seed_cycles']=max(item['elapsed_cycles'] for item in row['native'])
    return row


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    graph=compacted_vww()
    _,block,_,_=pair11_builder.model_block()
    block7=pair7_builder.model_block()[1]
    code7,payload7,plan7=pair7_builder.build(block7)
    source_base=ROOT/'work/phase6/strip-fusion-vww-v1/full/fixtures'
    source_timed=ROOT/'work/phase6/matched-baselines-v1/b3/catalogue/pair-110'
    assert source_timed.exists()
    sources={'pinned_timed':source_timed}
    for role,name in (('pinned_check','vww-pinned-compacted-strip-check'),
                      ('stress_check','vww-stress-compacted-strip-check')):
        dest=OUT/'sources'/f'pair-110-{role}'
        b3.splice_pair7_without_pair11(source_base/name,dest,code7,payload7,plan7)
        sources[role]=dest
    source_rows={role:verify_fixture(path,graph,f'source-110-{role}') for role,path in sources.items()}
    candidates={}
    for height in (4,6,8,12):
        code,payload,plan=build(block,height)
        target=OUT/'fixtures'/f'vww-h{height}-pinned-timed'
        lower=splice(source_timed,target,code,payload,plan)
        row=verify_fixture(target,graph,f'vww-h{height}-pinned-timed')
        row.update(tile=[height,24],strips=plan['strips'],block_plan=plan,
                   lower=lower)
        candidates[str(height)]=row
        save(OUT/'catalogue.partial.json',dict(status='running',candidates=candidates))
    winner=min(candidates,key=lambda h:(candidates[h]['worst_seed_cycles'],int(h)))
    distinct=min((h for h in candidates if h!='12'),
                 key=lambda h:(candidates[h]['worst_seed_cycles'],int(h)))
    diagnostic={}
    for height in sorted(set((winner,distinct)),key=int):
        code,payload,plan=build(block,int(height))
        diagnostic[height]={}
        for role in ('pinned_check','stress_check'):
            dest=OUT/'fixtures'/f'vww-h{height}-{role.replace("_","-")}'
            splice(sources[role],dest,code,payload,plan)
            diagnostic[height][role]=verify_fixture(dest,graph,f'vww-h{height}-{role}')
    report=dict(schema=1,status='passed-native',physical_board=False,
                scope='additional full-width tile-height points for VWW fused 11-14 on common fixed INT8 graph/27 MHz engine',
                source_sha256={name:file_sha(ROOT/name) for name in (
                    'tools/phase6/matched_defines_tiles.py','tools/phase6/matched_defines_adapt.py',
                    'tools/phase6/strip_fusion_pair7.py','tools/phase6/strip_fusion_pair11.py',
                    'compiler/scheduler/defines_verify.py','compiler/hardware_v2.py',
                    'compiler/integer_reference.py')},
                model_source_sha256=load_model('vww')[3],
                native_executable_sha256=file_sha(NATIVE),
                engine_sha256=file_sha(ROOT/'work/phase6/pool-timing-v1/engine.sv'),
                source_fixture_sha256={role:{name:file_sha(path/name) for name in
                    ('commands.bin','payload.bin','input.bin','output.bin','schedule.json')}
                    for role,path in sources.items()},
                source_pair110=source_rows,
                selection_rule='minimum worst native elapsed cycles over seeds 0 and 6063; tie by height',
                selection=winner,distinct_new_height=distinct,candidates=candidates,
                diagnostic_fixtures=diagnostic,
                limitations=['only full-width tiles; arbitrary horizontal tile/cache modes remain unlowered',
                             'only one fixed four-layer fusion block 11-14; other depths/cuts not searched',
                             'native cycles are RTL simulation, not physical board latency'])
    save(OUT/'report.json',report)
    print(json.dumps(dict(status=report['status'],winner=winner,
                          cycles={h:[v['elapsed_cycles'] for v in row['native']]
                                  for h,row in candidates.items()},
                          distinct_new_height=distinct),indent=2))


if __name__=='__main__':
    main()
