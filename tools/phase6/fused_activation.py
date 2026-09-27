#!/usr/bin/env python3
"""Executable isolated Conv/DW exact activation epilogue experiment.

Reload the256-byte exact INT8 map on every tagged descriptor. The SRAM live
interval includes this map, so later safe-tail prefetch cannot overwrite it.
No command programs the FPGA. Only route24 invokes Gowin.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time
import numpy as np

import combined_spec as engine_runner
import combined_scalar_uart as integrated
import tail_mask as route_runner
from followup_graph import group_channels, check_oracles
from run_boardless import load_model
from output_pipeline_fusion import activation_table, encode_fused, decode_fused
from prefetch_tail import analyze_fixture, commands, reorder
from run_screening import save_json, verify_files
from variants import ROOT, check_frozen, sha
from hardware_v2 import Descriptor
from scheduler.fused_verify import replay_fused

BASE = ROOT/'work/phase6/experiments-v1/fused-activation-v1'
ENGINE = BASE/'engine.sv'
PARENT = ROOT/'work/phase6/experiments-v1/stream-mask-v1/engine.sv'
SOURCE = ROOT/'work/phase6/followup_graph'
FIXTURES = BASE/'fixtures'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('fusion patch site changed: '+old[:90])
    return text.replace(old, new, 1)


def generated_engine():
    src = PARENT.read_text()
    src = replace_once(src, 'LUT_WORD_WRITE,Q_STREAM} state_t;',
                       'LUT_WORD_WRITE,Q_STREAM,F_L_REQ,F_L_WAIT,F_L_BYTES} state_t;')
    src = replace_once(src, '    logic [7:0] lut_index,lut_value;', '''    logic [7:0] lut_index,lut_value;
    wire fused_activation = d[0][48];
    wire [23:0] fused_table_base = {6'b0,d[0][63:49],3'b0};
    wire fusion_abi_valid = d[0][63:48]==0 ||
        (fused_activation&&(op==OP_CONV||op==OP_DWCONV)&&fused_table_base<=24'd32512);
    wire output_state = state==Q_WAIT||state==Q_STREAM;
    logic fused_result_valid;
    wire fused_lookup_ready = !fused_result_valid||mem_ready;
    wire output_valid = fused_activation ? fused_result_valid : qout;
    wire [7:0] output_result = fused_activation ? lut_value : qresult;
    wire requant_output_ready = output_state&&
        (fused_activation?fused_lookup_ready:mem_ready);
    always_ff @(posedge clk or negedge rst_n)begin
        if(!rst_n)fused_result_valid<=0;
        else if(abort_run||clear_counters||!output_state)fused_result_valid<=0;
        else if(fused_activation&&fused_lookup_ready)fused_result_valid<=qout;
    end''')
    src = replace_once(src, '''        if(SCALAR_LUT&&state==LUT_WAIT&&qout)activation_lut[lut_index]<=qresult;
        lut_value<=activation_lut[lut_read_index];''', '''        if(state==F_L_BYTES)activation_lut[lut_index]<=xword[lut_index[2:0]*8+:8];
        else if(SCALAR_LUT&&state==LUT_WAIT&&qout)activation_lut[lut_index]<=qresult;
        if(fused_activation&&output_state)begin
            if(qout&&fused_lookup_ready)lut_value<=activation_lut[qresult];
        end else lut_value<=activation_lut[lut_read_index];''')
    src = replace_once(src, '            X_REQ:begin mem_req=1;', '''            F_L_REQ:begin
                mem_req=1;address_full={40'b0,fused_table_base}+{56'b0,lut_index[7:3],3'b0};
            end
            X_REQ:begin mem_req=1;''')
    src = replace_once(src, '(state==LUT_WORD_WRITE)||qout;mem_wr=1;',
                       '(state==LUT_WORD_WRITE)||output_valid;mem_wr=1;')
    src = replace_once(src, '((state==Q_WAIT||state==Q_STREAM)?qresult:result_byte)',
                       '((state==Q_WAIT||state==Q_STREAM)?output_result:result_byte)')
    src = src.replace('((state==Q_WAIT||state==Q_STREAM)&&mem_ready)', 'requant_output_ready')
    src = replace_once(src, "d[0][63:48]!=0)fail(8'd2);", "!fusion_abi_valid)fail(8'd2);")
    src = replace_once(src, '                    else state<=spatial?GEOM0:OTHER_CHECK;', '''                    else if(fused_activation)begin lut_index<=0;state<=F_L_REQ;end
                    else state<=spatial?GEOM0:OTHER_CHECK;''')
    src = replace_once(src, '                X_REQ:if(mem_ready)state<=X_WAIT;', '''                F_L_REQ:if(mem_ready)state<=F_L_WAIT;
                F_L_WAIT:if(mem_rvalid)begin xword<=mem_rdata;state<=F_L_BYTES;end
                F_L_BYTES:begin
                    if(lut_index==255)state<=GEOM0;
                    else begin
                        lut_index<=lut_index+1;
                        if(lut_index[2:0]==7)state<=F_L_REQ;
                    end
                end
                X_REQ:if(mem_ready)state<=X_WAIT;''')
    src = src.replace('if(qout&&mem_ready)', 'if(output_valid&&mem_ready)')
    src = replace_once(src, 'state==D_REQ||state==D_WAIT||state==P_REQ',
                       'state==F_L_REQ||state==F_L_WAIT||state==D_REQ||state==D_WAIT||state==P_REQ')
    return '// Exact INT8 epilogue with mandatory per-descriptor LUT reload.\n'+src


def prepare():
    check_frozen()
    BASE.mkdir(parents=True, exist_ok=True)
    src = generated_engine()
    if ENGINE.exists() and ENGINE.read_text() != src:
        raise ValueError('immutable fused activation engine changed')
    if not ENGINE.exists():
        ENGINE.write_text(src)
    identity = dict(label=BASE.name, parent=str(PARENT.relative_to(ROOT)),
        parent_sha256=sha(PARENT), engine_sha256=sha(ENGINE),
        parameters=json.loads((PARENT.parent/'identity.json').read_text())['parameters'],
        change='tagged exact activation epilogue; mandatory LUT reload per producer descriptor',
        physical_board=False)
    path=BASE/'identity.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old!=identity and dict(old,parameters=identity['parameters'])==identity:
            save_json(path,identity)
        elif old!=identity:
            raise ValueError('immutable fused activation identity changed')
    else:
        save_json(path, identity)
    engine_runner.BASE = BASE; engine_runner.ENGINE = ENGINE
    engine_runner.runner.BASE = BASE; engine_runner.runner.ENGINE = ENGINE
    integrated.BASE=BASE;integrated.ENGINE=ENGINE
    route_runner.BASE=BASE;route_runner.ENGINE=ENGINE
    return identity


def stage_chunks(code, schedule):
    """Validate the original serialized grouped bytecode and locate stages."""
    cursor = 0
    rows = []
    for stage in schedule['stages']:
        start = cursor
        parts = stage.get('constant_filter_parts')
        loads = stage['loads'] if parts is None else [x for x in stage['loads'] if x['role'] != 'descriptor']
        for load in loads:
            expected = (1, int(load['direction']=='to_sram'), 0, load['ext'], load['sram'], load['bytes'])
            assert code[cursor] == expected and code[cursor+1] == (3,2,0,0,0,0)
            cursor += 2
        if parts is None:
            assert code[cursor][0] == 2 and code[cursor+1] == (3,3,0,0,0,0)
            cursor += 2
        else:
            for part in parts:
                if part['constant']:
                    assert code[cursor][0] == 1 and code[cursor+1] == (3,2,0,0,0,0)
                    cursor += 2
            for part in parts:
                if not part['constant']:
                    assert code[cursor][0] == 1 and code[cursor+1] == (3,2,0,0,0,0)
                    assert code[cursor+2][0] == 2 and code[cursor+3] == (3,3,0,0,0,0)
                    cursor += 4
        body_end = cursor
        snapshot = None
        if schedule['snapshots_enabled']:
            snapshot = cursor
            assert code[cursor][0] == 1 and code[cursor][1] == 0
            assert code[cursor+1] == (3,2,0,0,0,0)
            cursor += 2
        store = None
        if stage['store'] is not None:
            store = cursor
            assert code[cursor][0] == 1 and code[cursor][1] == 0
            assert code[cursor+1] == (3,2,0,0,0,0)
            cursor += 2
        rows.append(dict(start=start, body_end=body_end, end=cursor,
                         snapshot=snapshot, store=store))
    assert cursor == len(code)-1 and code[cursor] == (0,0,0,0,0,0)
    return rows


def lower_fixture(src):
    schedule = json.loads((src/'schedule.json').read_text())
    code = commands((src/'commands.bin').read_bytes())
    payload = bytearray((src/'payload.bin').read_bytes())
    chunks = stage_chunks(code, schedule)
    stages = schedule['stages']
    fusions = {}; skipped = set(); fused_layers = set()
    for i, stage in enumerate(stages[:-1]):
        pd = Descriptor.decode(bytes.fromhex(stage['descriptor_hex']))
        act = stages[i+1]
        ad = Descriptor.decode(bytes.fromhex(act['descriptor_hex']))
        if pd.opcode not in (4,6) or ad.opcode not in (2,8):
            continue
        assert pd.output == ad.input == ad.output and pd.outputs == ad.outputs
        assert stage['store'] is None
        base = (stage['live'][1]+7)//8*8
        has_live = any(code[k][0]==2 for k in range(chunks[i]['start'], chunks[i]['body_end']))
        if has_live and base+256 > 32768:
            continue
        load = next(t for t in act['loads'] if t['role']=='parameter' and t['sram']==ad.params and t['bytes']==16)
        table = activation_table(ad.opcode, payload[load['ext']:load['ext']+16])
        ext = (len(payload)+7)//8*8; payload.extend(bytes(ext-len(payload))); payload.extend(table)
        fusions[i] = dict(table=table, base=base, ext=ext, has_live=has_live,
                          activation_layer=act['layer'])
        skipped.add(i+1); fused_layers.add(stage['layer'])

    changed = []; runs = {}; constants = {}; output_stages = []; maps = {}
    def append(old_index, command):
        maps[old_index] = len(changed)
        if str(old_index) in schedule['run_contracts']:
            runs[str(len(changed))] = copy.deepcopy(schedule['run_contracts'][str(old_index)])
        if str(old_index) in schedule.get('constant_contracts', {}):
            constants[str(len(changed))] = copy.deepcopy(schedule['constant_contracts'][str(old_index)])
        changed.append(command)

    for i, stage in enumerate(stages):
        chunk = chunks[i]
        if i in skipped:
            # Final activation snapshot/store remains observable at the same
            # external address, after the preceding fused producer completed.
            for k in range(chunk['body_end'], chunk['end']):
                append(k, code[k])
            continue
        f = fusions.get(i)
        out_stage = copy.deepcopy(stage)
        output_stages.append(out_stage)
        if f and f['has_live']:
            changed.extend([(1,1,0,f['ext'],f['base'],256),(3,2,0,0,0,0)])
            out_stage['loads'].append(dict(direction='to_sram', ext=f['ext'], sram=f['base'], bytes=256, role='parameter'))
            out_stage['live'][1] = f['base']+256
        for k in range(chunk['start'], chunk['end']):
            if chunk['snapshot'] is not None and stage['layer'] in fused_layers and k in (chunk['snapshot'],chunk['snapshot']+1):
                continue
            c = code[k]
            if f and c[0]==1 and c[1]==1:
                _,_,_,ext,sram,length = c
                const = schedule.get('constant_contracts', {}).get(str(k))
                if const is not None:
                    replacement = bytes(f['table'][b] for b in payload[ext:ext+length])
                elif length == 128 and sram == stage['pc']:
                    desc = Descriptor.decode(bytes(payload[ext:ext+64]))
                    replacement = encode_fused(desc,f['base'])+bytes(payload[ext+64:ext+128])
                else:
                    replacement = None
                if replacement is not None:
                    fresh = (len(payload)+7)//8*8; payload.extend(bytes(fresh-len(payload))); payload.extend(replacement)
                    c = (1,1,0,fresh,sram,length)
                    if const is None:
                        for load in out_stage['loads']:
                            if load['ext']==ext and load['sram']==sram and load['bytes']==length:
                                load['ext']=fresh
            if f and c[0]==2:
                low = c[4]&65535
                c = (c[0],c[1],c[2],c[3],low|((f['base']+256)<<16),c[5])
            append(k,c)
            if f and str(len(changed)-1) in runs:
                runs[str(len(changed)-1)]['fused_activation_layer'] = f['activation_layer']
                runs[str(len(changed)-1)]['fused_table_sha256'] = hashlib.sha256(f['table']).hexdigest()
            if f and str(len(changed)-1) in constants:
                constants[str(len(changed)-1)]['fused_activation_layer'] = f['activation_layer']
        if f:
            out_stage['fused_activation_layer'] = f['activation_layer']
    changed.append((0,0,0,0,0,0))
    packed = b''.join(struct.pack('<BBHIII',*c) for c in changed)
    result = dict(schedule, stages=output_stages, run_contracts=runs, constant_contracts=constants,
        snapshot_regions={str(k):v for k,v in schedule['snapshot_regions'].items() if int(k) not in fused_layers},
        command_count=len(changed), program_bytes=len(packed),
        program_sha256=hashlib.sha256(packed).hexdigest(),
        image_sha256=hashlib.sha256(payload).hexdigest(),
        fusion=dict(pairs=len(fusions), fallback_pairs=sum(
            Descriptor.decode(bytes.fromhex(s['descriptor_hex'])).opcode in (2,8)
            for j,s in enumerate(stages) if j not in skipped),
            hidden_pre_activation_layers=sorted(fused_layers),
            table_loads=sum(f['has_live'] for f in fusions.values()),
            table_reloaded_every_descriptor=True),
        catalogue='exact grouped producers with tagged INT8 activation epilogue')
    return packed, bytes(payload), result


def check_table_lifetimes(code, payload, schedule):
    """Replay immutable DMA bytes and all live-region overlap constraints."""
    sram = bytearray(32768); known = bytearray(32768)
    pending_dma = pending_run = None
    fused_runs = 0
    def finish_dma():
        nonlocal pending_dma
        if pending_dma:
            _,direction,_,ext,base,size = pending_dma
            if direction:
                sram[base:base+size] = payload[ext:ext+size]
                known[base:base+size] = bytes([1])*size
            pending_dma = None
    def finish_run():
        nonlocal pending_run
        if pending_run:
            _,_,_,pc,live,_ = pending_run
            raw = bytes(sram[pc:pc+64]); tag = struct.unpack_from('<H',raw,6)[0]
            desc = decode_fused(raw)[0] if tag else Descriptor.decode(raw)
            known[desc.output:desc.output+desc.outputs] = bytes(desc.outputs)
            pending_run = None
    for i,c in enumerate(commands(code)):
        op,flags,_,a,b,n = c
        if op==3:
            if flags&1:finish_run()
            if flags&2:finish_dma()
        elif op==1:
            assert pending_dma is None
            if pending_run:
                low=pending_run[4]&65535;high=pending_run[4]>>16
                assert b+n<=low or b>=high, 'DMA overwrites live fused table/operand'
            pending_dma=c
        elif op==2:
            assert pending_run is None
            low=b&65535;high=b>>16
            if pending_dma:
                base,size=pending_dma[4:6]
                assert base+size<=low or base>=high
            assert all(known[a:a+128])
            raw=bytes(sram[a:a+64]);tag=struct.unpack_from('<H',raw,6)[0]
            if tag:
                desc,table=decode_fused(raw)
                assert low<=table and table+256<=high and all(known[table:table+256])
                assert 'fused_activation_layer' in schedule['run_contracts'][str(i)]
                assert hashlib.sha256(sram[table:table+256]).hexdigest()==schedule['run_contracts'][str(i)]['fused_table_sha256']
                fused_runs+=1
            pending_run=c
        else:
            assert op==0 and pending_run is None and pending_dma is None
    return dict(status='passed', fused_runs=fused_runs,
                scope='immutable LUT readiness and full RUN/DMA live-region exclusion')


def fixtures():
    prepare()
    source_manifest=json.loads((SOURCE/'fixtures.json').read_text())
    assert source_manifest['status']=='passed-replay'
    records=[];models={};oracles={}
    for item in source_manifest['fixtures']:
        src=SOURCE/'fixtures'/item['label'];verify_files(src,item['files'])
        code,payload,schedule=lower_fixture(src)
        name=item['label'].replace('-grouped-','-grouped-fused-')
        dest=FIXTURES/name;dest.mkdir(parents=True,exist_ok=True)
        (dest/'commands.bin').write_bytes(code);(dest/'payload.bin').write_bytes(payload)
        save_json(dest/'schedule.json',schedule)
        # Recompute prefetch against the expanded live intervals, never reuse
        # the old candidate set that predates the fused lookup table.
        analysis=analyze_fixture(dest)
        changed,mapping=reorder(commands(code),analysis['candidates'])
        code=b''.join(struct.pack('<BBHIII',*c) for c in changed)
        schedule['run_contracts']={str(mapping[int(k)]):v for k,v in schedule['run_contracts'].items()}
        schedule['constant_contracts']={str(mapping[int(k)]):v for k,v in schedule['constant_contracts'].items()}
        schedule['program_sha256']=hashlib.sha256(code).hexdigest()
        schedule['tail_prefetch']=dict(bytes=analysis['immediately_legal_prefetch_bytes'],
                                       transfers=analysis['immediately_legal_count'])
        lifetimes=check_table_lifetimes(code,payload,schedule)
        model=item['model'];sample=item['sample']
        if model not in models:
            original,pinned,_,_=load_model(model)
            grouped,mappings,_=group_channels(original)
            models[model]=(original,pinned,grouped,mappings)
        original,pinned,grouped,mappings=models[model]
        value=pinned if sample=='pinned' else np.random.default_rng(6078).integers(-128,128,pinned.shape,dtype=np.int8)
        if (model,sample) not in oracles:
            oracles[(model,sample)]=check_oracles(original,grouped,mappings,value)
        verification=replay_fused(grouped,code,payload,{grouped.inputs[0]:value},
            run_contracts=schedule['run_contracts'],constant_contracts=schedule['constant_contracts'],
            final_output=schedule['final_output'],snapshot_regions=schedule['snapshot_regions'],
            oracle=oracles[(model,sample)])
        names=['input.bin','output.bin']+[f'layer-{k}.bin' for k in schedule['snapshot_regions']]
        for filename in names:(dest/filename).write_bytes((src/filename).read_bytes())
        checks=[f'{schedule["final_output"]["ext"]} output.bin']
        checks += [f'{region["ext"]} layer-{k}.bin' for k,region in schedule['snapshot_regions'].items()]
        (dest/'checks.txt').write_text('\n'.join(checks)+'\n')
        (dest/'commands.bin').write_bytes(code);save_json(dest/'schedule.json',schedule)
        names += ['checks.txt','commands.bin','payload.bin','schedule.json']
        records.append(dict(name=name,model=item['model'],sample=item['sample'],
            fusion=schedule['fusion'],tail_prefetch=schedule['tail_prefetch'],
            command_count=schedule['command_count'],lifetimes=lifetimes,verification=verification,
            files={name:sha(dest/name) for name in names}))
    report=dict(status='passed-replay',physical_board=False,
        source_manifest_sha256=sha(SOURCE/'fixtures.json'),fixtures=records,
        sources={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),
            ROOT/'compiler/scheduler/fused_verify.py',ROOT/'tools/phase6/output_pipeline_fusion.py')})
    save_json(BASE/'fixtures.json',report)
    return report


def native():
    manifest=json.loads((BASE/'fixtures.json').read_text())
    build=BASE/'native';build.mkdir(exist_ok=True)
    sources=engine_runner.runner.sources();harness=ROOT/'test/phase6/native.cpp'
    with (build/'build.log').open('w') as log:
        subprocess.run(['verilator','--cc','--exe','--build','-j','2','--public-flat-rw',
            '-Wno-fatal','--top-module','v2_tiled_host_bridge','--Mdir',str(build),
            *map(str,sources),str(harness)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    exe=build/'Vv2_tiled_host_bridge'
    report=dict(status='running',physical_board=False,label=BASE.name,
        executable_sha256=sha(exe),sources={str(p.relative_to(ROOT)):sha(p) for p in sources+[harness]},
        fixture_manifest_sha256=sha(BASE/'fixtures.json'),results=[])
    for item in manifest['fixtures']:
        dest=FIXTURES/item['name'];verify_files(dest,item['files'])
        for seed in (0,6063):
            output=build/f'{item["name"]}-s{seed}.json'
            subprocess.run([str(exe),str(dest),str(seed),str(output)],check=True)
            row=json.loads(output.read_text());assert row['status']=='passed'
            row.update(fixture=item['name'],fixture_files=item['files'])
            report['results'].append(row);save_json(build/'report.json',report)
    report['status']='passed';save_json(build/'report.json',report)
    return report


def prepare_route24():
    """Write the same immutable24MHz inputs route_runner will subsequently use."""
    original=ROOT/'work/phase6/experiments-v1/combined-spec-scalar-uart-v1'
    pll=(original/'pll.v').read_text().replace('.FBDIV_SEL(4), .IDIV_SEL(5)',
        '.FBDIV_SEL(7), .IDIV_SEL(8)').replace('22.5 MHz from 27 MHz','24 MHz from 27 MHz')
    host=(original/'host.sv').read_text().replace('22500000','24000000')
    old='work/phase6/experiments-v1/combined-spec-scalar-uart-v1/'
    script=(original/'build.tcl').read_text().replace(old+'pll.v',str((BASE/'pll24.v').relative_to(ROOT)))
    script=script.replace(old+'host.sv',str((BASE/'host24.sv').relative_to(ROOT)))
    for path,content in ((BASE/'pll24.v',pll),(BASE/'host24.sv',host),(BASE/'build24.tcl',script)):
        if path.exists() and path.read_text()!=content:
            raise ValueError('immutable fused route input changed')
        if not path.exists():path.write_text(content)
    result=dict(status='prepared',physical_board=False,core_clock_mhz=24,
        sources={str(p.relative_to(ROOT)):sha(p) for p in (ENGINE,BASE/'pll24.v',BASE/'host24.sv',BASE/'build24.tcl')})
    save_json(BASE/'route24-inputs.json',result)
    return result


def compare():
    """Measure unchanged grouped-prefetch fixtures on the same RTL binary."""
    native_report=json.loads((BASE/'native/report.json').read_text())
    assert native_report['status']=='passed'
    exe=BASE/'native/Vv2_tiled_host_bridge'
    assert sha(exe)==native_report['executable_sha256']
    source=ROOT/'work/phase6/followup_graph_tail_prefetch'
    manifest=json.loads((source/'fixtures.json').read_text())
    result=dict(status='passed',physical_board=False,executable_sha256=sha(exe),
                source_manifest_sha256=sha(source/'fixtures.json'),comparisons={})
    for model in ('kws','vww'):
        name=f'{model}-pinned-grouped-tail-prefetch-timed'
        fixture=next(x for x in manifest['fixtures'] if x['name']==name)
        directory=source/'fixtures'/name;verify_files(directory,fixture['files'])
        for seed in (0,6063):
            path=BASE/'native'/f'baseline-{model}-s{seed}.json'
            subprocess.run([str(exe),str(directory),str(seed),str(path)],check=True)
            baseline=json.loads(path.read_text())
            candidate=next(x for x in native_report['results'] if
                x['fixture']==f'{model}-pinned-grouped-fused-timed' and x['stall_seed']==seed)
            result['comparisons'][f'{model}-s{seed}']=dict(
                baseline_cycles=baseline['elapsed_cycles'],candidate_cycles=candidate['elapsed_cycles'],
                speedup=baseline['elapsed_cycles']/candidate['elapsed_cycles'],
                saved_engine_cycles=baseline['engine_cycles']-candidate['engine_cycles'],
                baseline_sha256=sha(path),baseline_fixture_files=fixture['files'])
    save_json(BASE/'comparison.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','fixtures','edges','engine','native',
        'stream-edges','scalar-edges','integration','route24-inputs','route24','compare'))
    stage=parser.parse_args().stage;identity=prepare()
    if stage=='fixtures':result=fixtures()
    elif stage=='edges':result=engine_runner.cocotb('edges','test_fused_activation')
    elif stage=='stream-edges':result=engine_runner.cocotb('stream-edges','test_output_pipeline')
    elif stage=='scalar-edges':result=engine_runner.cocotb('scalar-edges','test_experiment_edges')
    elif stage=='engine':
        engine_runner.runner.engine();result=json.loads((BASE/'engine/report.json').read_text())
    elif stage=='native':result=native()
    elif stage=='integration':result=integrated.integration()
    elif stage=='route24-inputs':result=prepare_route24()
    elif stage=='route24':result=route_runner.route_24()
    elif stage=='compare':result=compare()
    else:result=identity
    check_frozen()
    print(json.dumps(dict(stage=stage,status=result.get('status','prepared'),
        engine_sha256=identity['engine_sha256'])),flush=True)
