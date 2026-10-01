#!/usr/bin/env python3
"""Observe frozen native engine states without changing RTL, programs or stimuli.

Link an observational harness against the SAME generated model archive/runtime
objects as the selected native executable. Every exposed timing/output counter
must exactly match the original harness for each fixture and seed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'work/phase6/engine-profile-v1'
BUILD = ROOT / 'work/phase6/pool-timing-v1/native'
MODEL_REPORT = BUILD / 'report.json'
SELECTION = ROOT / 'work/phase6/matched-baselines-v1/b3-final/report.json'
ENGINE = ROOT / 'work/phase6/pool-timing-v1/engine.sv'
ORIGINAL = BUILD / 'Vv2_tiled_host_bridge'
NATIVE_SHA = 'b97fbf944827a5345670f3603e7f10368a897e6bcb19d16cc56905d8e06d16ec'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


OBSERVER = r'''
#define E(name) d.rootp->v2_tiled_host_bridge__DOT__core__DOT__engine__DOT__##name
#define H(name) d.rootp->v2_tiled_host_bridge__DOT__##name
struct Profile {
    unsigned command=0,pc=0,run=0,op=255;
    uint64_t first=0, states[64]={}, words[8]={};
    uint64_t req_blocked=0,req_blocked_nonprefetch=0,read_pending=0,dma_overlap=0,useful_macs=0;
    uint64_t scalar_hit=0,scalar_miss=0,scalar_pad=0,broadcast_reuse=0;
    uint64_t pw_hit=0,pw_first_word_reads=0,pw_cross_reads=0,pw_prefetch_hit_decisions=0,weight_hit=0,weight_miss=0;
    uint64_t parameter_hit=0,q_input_blocked=0,q_output_pending=0,q_write_blocked=0;
    uint64_t accepted_reads=0,accepted_writes=0;
};
static std::vector<Profile> profiles;
static unsigned previous_state=0,run_number=0,run_command=0;
static bool previous_busy=false;
static uint64_t observed_seq=0,observed_dma=0,observed_overlap=0;
static void observe(){
    if(!d.rst_n)return;
    if(E(start)){run_command=d.rootp->v2_tiled_host_bridge__DOT__sequencer__DOT__command_index;run_number++;}
    unsigned state=E(state);bool busy=E(busy),dma=H(dma_busy)||H(dma_pending);
    if(H(seq_busy)){
        observed_seq++;if(dma)observed_dma++;
        if(busy&&dma)observed_overlap++;
    }
    if(busy){
        require(H(seq_busy),"engine busy outside sequencer interval");
        if(!previous_busy||(state==1&&E(di)==0&&previous_state!=1)){
            Profile p;p.command=run_command;p.run=run_number;p.pc=E(pc);p.first=observed_seq-1;
            profiles.push_back(p);
        }
        require(!profiles.empty(),"missing descriptor profile");
        auto& p=profiles.back();require(state<64,"state range");p.states[state]++;
        if(state==3){p.op=E(op);for(unsigned i=0;i<8;i++)p.words[i]=E(d)[i];}
        if(dma)p.dma_overlap++;
        if(E(mem_req)&&!E(mem_ready)){p.req_blocked++;if(state!=35)p.req_blocked_nonprefetch++;}
        bool waiting=state==2||state==5||state==8||state==10||state==19||state==22||state==30||state==32||state==34||state==45;
        if(waiting&&!E(mem_rvalid))p.read_pending++;
        if(E(mem_req)&&E(mem_ready)){if(E(mem_wr))p.accepted_writes++;else p.accepted_reads++;}
        if(state==11&&(E(op)==1||E(op)==4||E(op)==6))p.useful_macs+=E(valid_lanes);
        if(state==35)p.useful_macs+=E(pw_take);
        if(state==17){
            if(E(broadcast_mode)&&(E(oc)&1))p.broadcast_reuse++;
            else if(E(col)+E(win_lane)<E(count)){
                if(!E(window_inside))p.scalar_pad++;
                else if(E(line_hit))p.scalar_hit++;else p.scalar_miss++;
            }
        }
        if(state==29&&!E(pw_prefetch_pending)&&E(pw_cache_hit))p.pw_hit++;
        if((state==29||(state==35&&!E(pw_next_cache_hit)))&&E(mem_req)&&E(mem_ready))p.pw_first_word_reads++;
        if((state==31||(state==35&&E(pw_next_cache_hit)&&E(pw_next_crosses)))&&E(mem_req)&&E(mem_ready))p.pw_cross_reads++;
        if(state==35&&E(col)+1<E(count)&&E(pw_next_cache_hit))p.pw_prefetch_hit_decisions++;
        if((state==9||state==33)&&E(cached_weight))p.weight_hit++;
        if((state==9||state==33)&&E(mem_req)&&E(mem_ready))p.weight_miss++;
        if(state==4&&E(parameter_hit))p.parameter_hit++;
        if(E(qvalid)&&!E(qready))p.q_input_blocked++;
        if(state==13||state==43){if(!E(output_valid))p.q_output_pending++;
            else if(!E(mem_ready))p.q_write_blocked++;}
    }
    previous_state=state;previous_busy=busy;
}
static void write_profiles(const std::string& filename){
    std::ofstream f(filename);f<<"{\"sequencer_cycles\":"<<observed_seq<<",\"dma_cycles\":"<<observed_dma<<",\"overlap_cycles\":"<<observed_overlap<<",\"descriptors\":[";
    bool first=true;for(const auto& p:profiles){if(!first)f<<',';first=false;
        f<<"{\"run\":"<<p.run<<",\"command\":"<<p.command<<",\"pc\":"<<p.pc<<",\"opcode\":"<<p.op<<",\"first_sequencer_cycle\":"<<p.first<<",\"state_cycles\":[";
        for(unsigned i=0;i<64;i++){if(i)f<<',';f<<p.states[i];}f<<"],\"descriptor_words\":[";
        for(unsigned i=0;i<8;i++){if(i)f<<',';f<<p.words[i];}f<<']';
#define FIELD(n) f<<",\""#n"\":"<<p.n
        FIELD(req_blocked);FIELD(req_blocked_nonprefetch);FIELD(read_pending);FIELD(dma_overlap);FIELD(useful_macs);
        FIELD(scalar_hit);FIELD(scalar_miss);FIELD(scalar_pad);FIELD(broadcast_reuse);
        FIELD(pw_hit);FIELD(pw_first_word_reads);FIELD(pw_cross_reads);FIELD(pw_prefetch_hit_decisions);FIELD(weight_hit);FIELD(weight_miss);
        FIELD(parameter_hit);FIELD(q_input_blocked);FIELD(q_output_pending);FIELD(q_write_blocked);
        FIELD(accepted_reads);FIELD(accepted_writes);f<<'}';
#undef FIELD
    }f<<"]}\n";
}
'''


def state_names():
    names = re.search(r'typedef enum logic \[5:0\] \{([^}]+)\} state_t;', ENGINE.read_text()).group(1).split(',')
    required={0:'IDLE',1:'D_REQ',3:'D_CHECK',11:'MAC',17:'WIN_PREP',29:'PW_X_REQ',
              31:'PW_X2_REQ',35:'PW_MAC',43:'Q_STREAM',45:'F_L_WAIT'}
    if any(names[i] != n for i,n in required.items()): raise ValueError('observer enum assumptions changed')
    return names


def build(output):
    original = read(MODEL_REPORT)
    if original['status']!='passed' or original['executable_sha256']!=NATIVE_SHA or sha(ORIGINAL)!=NATIVE_SHA:
        raise ValueError('original native identity changed')
    pins = dict(original['sources'])
    for p,d in pins.items():
        if sha(ROOT/p)!=d:raise ValueError('frozen input changed: '+p)
    headers=sorted(BUILD.glob('*.h'))
    objects=[BUILD/'Vv2_tiled_host_bridge__ALL.a',*[BUILD/n for n in ('verilated.o','verilated_dpi.o','verilated_threads.o')]]
    pins.update({str(p.relative_to(ROOT)):sha(p) for p in headers+objects+[ORIGINAL,MODEL_REPORT,SELECTION,Path(__file__)]})
    text=(ROOT/'test/phase6/native.cpp').read_text()
    text=text.replace('static int step(){',OBSERVER+'\nstatic int step(){',1)
    text=text.replace('d.eval();int out=d.tx_valid&&d.tx_ready?d.tx_data:-1;',
                      'd.eval();int out=d.tx_valid&&d.tx_ready?d.tx_data:-1;observe();',1)
    text=text.replace('    d.final();return 0;','    write_profiles(std::string(argv[3])+".profile.json");\n    d.final();return 0;',1)
    harness=output/'native_profile.cpp';harness.write_text(text)
    verilator_root=re.search(r'^VERILATOR_ROOT = (.+)$',(BUILD/'Vv2_tiled_host_bridge.mk').read_text(),re.M).group(1)
    include=Path(verilator_root)/'include'
    cmd=['c++','-O2','-std=gnu++17','-I'+str(BUILD),'-I'+str(include),'-I'+str(include/'vltstd'),
         '-DVERILATOR=1','-DVM_COVERAGE=0','-DVM_SC=0','-DVM_TIMING=0','-DVM_TRACE=0',
         '-Wl,-U,__Z15vl_time_stamp64v,-U,__Z13sc_time_stampv',str(harness),*map(str,objects),
         '-pthread','-o',str(output/'profile_native')]
    with (output/'build.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    return dict(input_files=pins,original_executable_sha256=NATIVE_SHA,profile_executable_sha256=sha(output/'profile_native'),
                profile_harness_sha256=sha(harness),compile_command=cmd,
                instrumentation='read-only public-flat fields, sampled after falling-edge eval before rising edge; original compiled model and RAM RNG unchanged')


def category(op,state):
    if op==3:return 'copy_operator'
    if op==0:return 'halt_descriptor'
    if state in ('MAC','PW_MAC'):return 'arithmetic_issue'
    if state in ('WIN_PREP','WIN_REQ','WIN_WAIT','PW_X_REQ','PW_X_WAIT','PW_X2_REQ','PW_X2_WAIT','X_REQ','X_WAIT','DW_PREP'):
        return 'activation_gather_cache'
    if state in ('W_REQ','W_WAIT','PW_W_REQ','PW_W_WAIT'):return 'weight_access_cache'
    if state in ('Q_SEND','Q_WAIT','Q_STREAM','WRITE_OUT','LUT_WRITE','LUT_WORD_WRITE'):return 'requant_activation_writeback'
    if state in ('P_REQ','P_WAIT','P_CHECK'):return 'parameter_setup'
    if state in ('F_L_REQ','F_L_WAIT','F_L_BYTES','LUT_BUILD','LUT_SEND','LUT_WAIT'):return 'activation_table_setup'
    if state.startswith('POOL'):return 'pool_reduce'
    if state=='LINE_CLEAR':return 'activation_cache_clear'
    return 'descriptor_geometry_control'


def enrich(profile,schedule,names):
    totals=Counter();states=Counter();signals=Counter();opcodes={};families={}
    for row in profile['descriptors']:
        row['descriptor_hex']=struct.pack('<8Q',*row.pop('descriptor_words')).hex()
        fields=struct.unpack('<IBBBB8I12H',bytes.fromhex(row['descriptor_hex']))
        geometry_names=('input','output','weight','params','count','outputs','row_stride','next_pc',
            'kernel_h','kernel_w','stride_h','stride_w','pad_top','pad_bottom','pad_left','pad_right',
            'input_h','input_w','input_c','output_c')
        row['descriptor']=dict(opcode=fields[2],flags=list(fields[3:5]),**dict(zip(geometry_names,fields[5:])))
        d=row['descriptor'];op=row['opcode']
        family=('pointwise_conv' if d['kernel_h']==d['kernel_w']==1 else 'general_conv') if op==4 else {
            0:'halt',1:'gemm',2:'relu',3:'copy',5:'maxpool',6:'depthwise_conv',7:'avgpool',8:'clip'}.get(op,'unknown')
        row['family']=family
        row['states']={names[i]:n for i,n in enumerate(row.pop('state_cycles')) if n}
        row['engine_cycles']=sum(row['states'].values())
        parts=Counter()
        for state,n in row['states'].items():parts[category(row['opcode'],state)]+=n
        row['categories']=dict(parts);totals.update(parts);states.update(row['states'])
        for key in ('req_blocked','req_blocked_nonprefetch','read_pending','dma_overlap','useful_macs','scalar_hit','scalar_miss','scalar_pad',
                    'broadcast_reuse','pw_hit','pw_first_word_reads','pw_cross_reads','pw_prefetch_hit_decisions','weight_hit','weight_miss','parameter_hit',
                    'q_input_blocked','q_output_pending','q_write_blocked','accepted_reads','accepted_writes'):
            signals[key]+=row[key]
        contract=schedule.get('run_contracts',{}).get(str(row['command']))
        if contract:row['compiler_run_contract']=contract
        for i,c in enumerate(schedule.get('components',[])):
            if c['first_command']<=row['command']<c['last_command']:
                row['component']=dict(index=i,start=c['start'],stop=c['stop'],configuration=c['configuration'])
        op=opcodes.setdefault(str(row['opcode']),dict(descriptors=0,engine_cycles=0,categories=Counter()))
        op['descriptors']+=1;op['engine_cycles']+=row['engine_cycles'];op['categories'].update(parts)
        group=families.setdefault(family,dict(descriptors=0,engine_cycles=0,states=Counter(),categories=Counter()))
        group['descriptors']+=1;group['engine_cycles']+=row['engine_cycles'];group['states'].update(row['states']);group['categories'].update(parts)
    engine=sum(totals.values());elapsed=profile['sequencer_cycles']
    profile.update(engine_cycles=engine,categories=dict(totals),state_cycles=dict(states),signal_counts=dict(signals),
        opcode_totals=opcodes,family_totals=families,category_engine_fraction={k:v/engine for k,v in totals.items()},
        category_device_fraction={k:v/elapsed for k,v in totals.items()},
        top_descriptors=sorted(range(len(profile['descriptors'])),key=lambda i:-profile['descriptors'][i]['engine_cycles'])[:12])
    return profile


def run(output=BASE):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('choose a fresh output directory')
    output.mkdir(parents=True)
    start=time.monotonic();names=state_names();identity=build(output);selection=read(SELECTION)
    if selection['status']!='passed' or selection['native_sha256']!=NATIVE_SHA:raise ValueError('selection identity')
    report=dict(status='running',physical_board=False,identity=identity,state_names=names,models={},
        scope='Selected strongest b3-final pinned KWS/VWW programs; unchanged native model and stimuli; seeds0/6063',
        measured='Every descriptor state occupancy and named signal event is directly observed in native RTL simulation.',
        inferred='Category names interpret measured states. State occupancy is not a removable-stall or attainable-speedup bound.',
        limitations=['Native external RAM model, not physical SDRAM; no board or energy measurement.',
            'Requantization, activation lookup and SRAM writeback overlap and share states; their separate arithmetic costs cannot be inferred by subtracting state counts.',
            'read_pending includes normal SRAM response latency; req_blocked includes speculative prefetch backpressure and is not all critical-path stall.',
            'Scalar and pointwise cache signal counts have their source-defined scopes; they are not aggregate cache hit-rate denominators.',
            'Pinned exact final outputs only; not a fresh full-dataset accuracy evaluation.'])
    for model,selected in selection['selected'].items():
        fixture=selected['pinned'];directory=ROOT/fixture['directory'];files=fixture['files']
        for name,digest in files.items():
            if sha(directory/name)!=digest:raise ValueError('fixture input changed')
        schedule=read(directory/'schedule.json');rows=[]
        for seed in (0,6063):
            target=output/f'{model}-s{seed}.json'
            with target.with_suffix('.log').open('w') as log:
                subprocess.run([str(output/'profile_native'),str(directory),str(seed),str(target)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            result=read(target);expected=next(n for n in fixture['native'] if n['stall_seed']==seed)
            for key in ('status','physical_board','stall_seed','tensor_checks','elapsed_cycles','engine_cycles','dma_cycles','overlap_cycles','total_simulated_cycles'):
                if result[key]!=expected[key]:raise ValueError('instrumentation changed native result: '+key)
            raw_path=Path(str(target)+'.profile.json');profile=enrich(read(raw_path),schedule,names)
            for actual,wanted in (('sequencer_cycles','elapsed_cycles'),('engine_cycles','engine_cycles'),('dma_cycles','dma_cycles'),('overlap_cycles','overlap_cycles')):
                if profile[actual]!=result[wanted]:raise ValueError('profile counter mismatch '+actual)
            for name,digest in files.items():
                if sha(directory/name)!=digest:raise ValueError('fixture changed during run')
            profile.update(stall_seed=seed,raw_profile_sha256=sha(raw_path),original_native_result=expected,
                           instrumented_native_result=result)
            save(output/f'{model}-s{seed}-enriched.json',profile)
            rows.append(dict(stall_seed=seed,path=str((output/f'{model}-s{seed}-enriched.json').relative_to(ROOT)),
                sha256=sha(output/f'{model}-s{seed}-enriched.json'),engine_cycles=profile['engine_cycles'],
                elapsed_cycles=profile['sequencer_cycles'],categories=profile['categories'],
                category_engine_fraction=profile['category_engine_fraction'],category_device_fraction=profile['category_device_fraction'],
                signal_counts=profile['signal_counts'],opcode_totals=profile['opcode_totals'],family_totals=profile['family_totals']))
            print(model,seed,'passed',json.dumps(rows[-1]['category_engine_fraction'],sort_keys=True),flush=True)
        report['models'][model]=dict(fixture_directory=fixture['directory'],fixture_files=files,graph_sha256=selection['models'][model]['graph_sha256'],profiles=rows)
        save(output/'report.json',report)
    for path,digest in identity['input_files'].items():
        if sha(ROOT/path)!=digest:raise ValueError('source/model changed during profiling: '+path)
    if sha(output/'profile_native')!=identity['profile_executable_sha256']:raise ValueError('profiler executable changed')
    report.update(status='passed',seconds=time.monotonic()-start)
    save(output/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE)
    args=parser.parse_args();run(args.output)
