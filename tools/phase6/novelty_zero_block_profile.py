#!/usr/bin/env python3
"""Read-only native per-descriptor timing of the frozen P6 host bridge.

The Verilated design and fixture are byte-identical to the selected baseline.
Only an isolated copy of its C++ harness is instrumented. Engine busy edges
provide exact run-cycle attribution, including descriptor setup and epilogue.
"""
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'compiler'),str(ROOT/'tools/phase6')]
from novelty_zero_block import FIXTURE_ROOT,FIXTURES,schedule_pointwise_runs

BUILD=ROOT/'work/phase6/novelty-zero-block-v1/native-profile'
BASE=ROOT/'work/phase6/pool-timing-v1/native'
HARNESS=ROOT/'test/phase6/native.cpp'
VROOT=Path('/opt/homebrew/Cellar/verilator/5.044/share/verilator')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def compile_profile():
    BUILD.mkdir(parents=True,exist_ok=True)
    original=HARNESS.read_text()
    marker='static Request hold;'
    assert original.count(marker)==1
    modified=original.replace(marker,marker+'''
static std::vector<uint64_t> engine_run_cycles;
static bool previous_engine_busy=false;
static uint64_t engine_run_started=0;
static uint64_t pw_iterations=0, logical_eligible=0, physical_eligible=0;
static uint64_t logical_input_fetch_cycles=0, physical_input_fetch_cycles=0;
static uint64_t logical_weight_fetch_cycles=0, physical_weight_fetch_cycles=0;
static uint64_t pending_pw_input=0, pending_pw_weight=0;
static uint64_t fused_next_raw=0,fused_next_greedy=0,fused_next_geometry=0;
static uint64_t fused_next_geometry_cache_hit=0;
static bool fused_previous_consumes_current=false;
static uint8_t sram_byte(uint32_t a){
    auto r=d.rootp;auto i=a>>3;
    switch(a&7){
'''+''.join(f'    case {lane}:return r->v2_tiled_host_bridge__DOT__core__DOT__memory__DOT__banks__BRA__{lane}__KET____DOT__ram[i];\n' for lane in range(8))+'''
    default:throw std::runtime_error("bad SRAM byte lane");
    }
}
''')
    marker='d.eval();int out=d.tx_valid&&d.tx_ready?d.tx_data:-1;'
    assert modified.count(marker)==1
    modified=modified.replace(marker,'''d.eval();int out=d.tx_valid&&d.tx_ready?d.tx_data:-1;
    auto root=d.rootp;
    if(root->v2_tiled_host_bridge__DOT__core__DOT__engine__DOT__spatial_pw&&
       root->v2_tiled_host_bridge__DOT__core__DOT__engine__DOT__busy){
        unsigned state=root->v2_tiled_host_bridge__DOT__core__DOT__engine__DOT__state;
        if(state>=29&&state<=32)pending_pw_input++;
        else if(state==33||state==34)pending_pw_weight++;
        else if(state==35){
            pw_iterations++;
            bool can_fuse_next=false;
            uint32_t col=root->v2_tiled_host_bridge__DOT__core__DOT__engine__DOT__col;
            uint32_t count=root->v2_tiled_host_bridge__DOT__core__DOT__engine__DOT__count;
            uint32_t next_addr=root->v2_tiled_host_bridge__DOT__core__DOT__engine__DOT__pw_next_input_addr;
            uint64_t weight_word=root->v2_tiled_host_bridge__DOT__core__DOT__engine__DOT__whex;
            uint8_t next_weight=uint8_t(weight_word>>(((col+1)&7u)*8));
            if(root->v2_tiled_host_bridge__DOT__core__DOT__engine__DOT__pw_take==8&&
               col+1<count&&(col&7u)!=7&&next_addr%8==0&&next_weight!=0){
                can_fuse_next=true;
                uint8_t zp=root->v2_tiled_host_bridge__DOT__core__DOT__engine__DOT__zx;
                for(unsigned k=0;k<8;k++)can_fuse_next&=sram_byte(next_addr+k)==zp;
            }
            if(can_fuse_next){
                fused_next_geometry++;
                if(root->v2_tiled_host_bridge__DOT__core__DOT__engine__DOT__pw_next_cache_hit)
                    fused_next_geometry_cache_hit++;
            }
            can_fuse_next&=root->v2_tiled_host_bridge__DOT__core__DOT__engine__DOT__pw_next_cache_hit;
            if(can_fuse_next)fused_next_raw++;
            if(fused_previous_consumes_current)fused_previous_consumes_current=false;
            else if(can_fuse_next){fused_next_greedy++;fused_previous_consumes_current=true;}
            if(root->v2_tiled_host_bridge__DOT__core__DOT__engine__DOT__pw_take==8&&
               root->v2_tiled_host_bridge__DOT__core__DOT__engine__DOT__pw_weight!=0){
                uint32_t addr=root->v2_tiled_host_bridge__DOT__core__DOT__engine__DOT__pw_input_addr;
                uint8_t zp=root->v2_tiled_host_bridge__DOT__core__DOT__engine__DOT__zx;
                bool logical=true,physical=true;
                for(unsigned k=0;k<8;k++)logical&=sram_byte(addr+k)==zp;
                if(logical){
                    logical_eligible++;
                    logical_input_fetch_cycles+=pending_pw_input;
                    logical_weight_fetch_cycles+=pending_pw_weight;
                    for(unsigned k=0;k<8;k++){
                        physical&=sram_byte((addr&~7u)+k)==zp;
                        physical&=sram_byte(((addr+7)&~7u)+k)==zp;
                    }
                    if(physical){physical_eligible++;
                        physical_input_fetch_cycles+=pending_pw_input;
                        physical_weight_fetch_cycles+=pending_pw_weight;}
                }
            }
            pending_pw_input=pending_pw_weight=0;
        }
    }
''')
    marker='d.clk=1;d.eval();cycles++;return out;'
    assert modified.count(marker)==1
    modified=modified.replace(marker,'''d.clk=1;d.eval();cycles++;
    bool busy=d.rootp->v2_tiled_host_bridge__DOT__core__DOT__engine_busy;
    if(busy&&!previous_engine_busy)engine_run_started=cycles;
    if(!busy&&previous_engine_busy)engine_run_cycles.push_back(cycles-engine_run_started);
    previous_engine_busy=busy;
    return out;''')
    marker=r'<<cycles<<"}\n";'
    assert modified.count(marker)==1
    modified=modified.replace(marker,'''<<cycles<<",\\\"engine_run_cycles\\\":[";
    for(size_t i=0;i<engine_run_cycles.size();++i){if(i)report<<",";report<<engine_run_cycles[i];}
    report<<"],\\\"pw_events\\\":{\\\"iterations\\\":"<<pw_iterations
          <<",\\\"logical_eligible\\\":"<<logical_eligible
          <<",\\\"physical_eligible\\\":"<<physical_eligible
          <<",\\\"logical_input_fetch_cycles\\\":"<<logical_input_fetch_cycles
          <<",\\\"physical_input_fetch_cycles\\\":"<<physical_input_fetch_cycles
          <<",\\\"logical_weight_fetch_cycles\\\":"<<logical_weight_fetch_cycles
          <<",\\\"physical_weight_fetch_cycles\\\":"<<physical_weight_fetch_cycles
          <<",\\\"fused_next_raw\\\":"<<fused_next_raw
          <<",\\\"fused_next_greedy\\\":"<<fused_next_greedy
          <<",\\\"fused_next_geometry\\\":"<<fused_next_geometry
          <<",\\\"fused_next_geometry_cache_hit\\\":"<<fused_next_geometry_cache_hit
          <<"}}\\n";''')
    source=BUILD/'native_profile.cpp'
    source.write_text(modified)
    exe=BUILD/'native_profile'
    command=['c++','-std=c++17','-O2','-pthread',
        '-I'+str(BASE),'-I'+str(VROOT/'include'),'-I'+str(VROOT/'include/vltstd'),
        str(source),str(BASE/'Vv2_tiled_host_bridge__ALL.a'),
        str(BASE/'verilated.o'),str(BASE/'verilated_dpi.o'),
        str(BASE/'verilated_threads.o'),'-Wl,-U,__Z13sc_time_stampv',
        '-o',str(exe)]
    subprocess.run(command,check=True,cwd=ROOT,stdout=(BUILD/'build.log').open('w'),stderr=subprocess.STDOUT)
    return exe,source

def main():
    exe,source=compile_profile()
    result=dict(schema=1,status='running',scope='unchanged selected RTL, instrumented native harness; pointwise descriptor cycles include setup and output epilogue',
                hashes=dict(harness=sha(HARNESS),instrumented_harness=sha(source),
                            selected_native=sha(BASE/'Vv2_tiled_host_bridge'),
                            generated_design_archive=sha(BASE/'Vv2_tiled_host_bridge__ALL.a')),
                models={})
    for model in ('kws','vww'):
        fixture=FIXTURE_ROOT/FIXTURES[model]
        pointwise=schedule_pointwise_runs(fixture)
        commands=(fixture/'commands.bin').read_bytes()
        op2=[i for i in range(len(commands)//16)
             if struct.unpack_from('<B',commands,16*i)[0]==2]
        selected={index for index,*_ in pointwise}
        assert selected.issubset(op2)
        rows=[]
        for seed in (0,6063):
            output=BUILD/f'{model}-s{seed}.json'
            subprocess.run([str(exe),str(fixture),str(seed),str(output)],check=True,
                           stdout=(BUILD/f'{model}-s{seed}.log').open('w'),stderr=subprocess.STDOUT)
            native=json.loads(output.read_text())
            assert native['status']=='passed' and len(native['engine_run_cycles'])==len(op2)
            opportunity=json.loads((ROOT/'work/phase6/novelty-zero-block-v1/report.json').read_text())
            expected=opportunity['models'][model]['samples']['pinned']['totals']
            assert native['pw_events']['iterations']==expected['executed_channel_output_block_iterations']
            assert native['pw_events']['logical_eligible']==expected['weighted_skippable_iterations']
            assert native['pw_events']['physical_eligible']==expected['physical_word_certified_weighted_skips']
            run_cycles=dict(zip(op2,native['engine_run_cycles']))
            pointwise_cycles=sum(run_cycles[i] for i in selected)
            rows.append(dict(seed=seed,pointwise_descriptor_cycles=pointwise_cycles,
                             all_engine_descriptor_cycles=sum(native['engine_run_cycles']),
                             reported_engine_cycles=native['engine_cycles'],
                             reported_elapsed_cycles=native['elapsed_cycles'],
                             pointwise_share_of_engine=pointwise_cycles/native['engine_cycles'],
                             pw_events=native['pw_events'],
                             pointwise_runs=[dict(command=i,layer=contract['layer'],cycles=run_cycles[i])
                                             for i,_,contract,_ in pointwise]))
        result['models'][model]=dict(fixture=FIXTURES[model],
            fixture_sha256={k:sha(fixture/k) for k in ('commands.bin','payload.bin','schedule.json')},
            seeds=rows)
        (BUILD/'report.partial.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    result['status']='passed'
    (BUILD/'report.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({m:[dict(seed=r['seed'],pointwise_descriptor_cycles=r['pointwise_descriptor_cycles'],
                             pointwise_share_of_engine=r['pointwise_share_of_engine']) for r in d['seeds']]
                      for m,d in result['models'].items()},indent=2))

if __name__=='__main__':main()
