#!/usr/bin/env python3
"""Isolated native drained-boundary pause/resume and live-SRAM checkpoint test.

Copies the existing sequencer/host, leaves the selected arithmetic engine
unchanged, and interleaves a real VWW inference with an urgent real AD job.
No production RTL or board image is changed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import struct
import subprocess
import sys
import time

from deadline_screen import (ROOT, REFERENCE, PARENT_SHA, MEM, sha, save,
                             analyze_fixture, commands, pack, intervals)

BASE = ROOT / 'work/phase6/deadline-pause-v1'
SCREEN = ROOT / 'work/phase6/deadline-screen-v1'


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('isolated patch site changed: '+old[:100])
    return text.replace(old, new, 1)


def prepare_rtl():
    BASE.mkdir(parents=True, exist_ok=True)
    seq = (ROOT/'rtl/v2/tile_sequencer.sv').read_text()
    seq = once(seq, '    input logic [10:0] start_index,', '''    input logic [10:0] start_index,
    input logic pause_req,
    output logic paused,
    output logic [11:0] resume_index,''')
    seq = once(seq, '            command_index<=0;low_word<=0;high_word<=0;live_base<=0;live_end<=32768;',
               '            command_index<=0;paused<=0;resume_index<=0;low_word<=0;high_word<=0;live_base<=0;live_end<=32768;')
    seq = once(seq, '                    busy<=1;error_code<=0;elapsed<=0;engine_cycles<=0;',
               '                    busy<=1;paused<=0;error_code<=0;elapsed<=0;engine_cycles<=0;')
    seq = once(seq, '''                            else state<=ADVANCE;
                        end
                        default: fail(8'd10);''', '''                            else if(pause_req && !engine_busy && !dma_busy && command_index<2047)begin
                                // Pause only after the WAIT completed and both units drained.
                                // A new start at resume_index refetches the next command.
                                paused<=1;busy<=0;resume_index<=command_index+1;
                                command_index<=command_index+1;state<=IDLE;
                            end else state<=ADVANCE;
                        end
                        default: fail(8'd10);''')
    decoder = (ROOT/'rtl/v2/command.sv').read_text()
    decoder = once(decoder, '    input logic mmio_while_busy,',
                   '    input logic mmio_while_busy,\n    input logic seq_pause_while_busy,')
    decoder = once(decoder, '    logic [24:0] transfer_addr;', '''    // The only additional busy mutation is a CRC-validated one-byte pause request.
    // Operand/program memory and all other control writes remain protected.
    wire tiled_busy_pause_write = TILED_MEM_MAP && seq_pause_while_busy &&
        cmd==WRITE && address==24'h410000 && length==1 && payload[0]==8'h02;
    logic [24:0] transfer_addr;''')
    decoder = once(decoder, 'if(busy && !tiled_busy_mmio_write)respond(8\'d3,1);',
                   'if(busy && !tiled_busy_mmio_write && !tiled_busy_pause_write)respond(8\'d3,1);')
    host = (ROOT/'rtl/v2/tiled_host_bridge.sv').read_text()
    host = once(host, '        .mmio_while_busy(!seq_busy && memory_initialized && engine_busy &&',
                '        .seq_pause_while_busy(seq_busy && memory_initialized),\n        .mmio_while_busy(!seq_busy && memory_initialized && engine_busy &&')
    host = once(host, '    logic [11:0] seq_index;', '''    logic [11:0] seq_index, seq_resume_index;
    logic seq_pause_req, seq_paused;''')
    host = once(host, '        .abort_run(abort_command),\n        .busy(seq_busy)',
                '        .abort_run(abort_command),\n        .pause_req(seq_pause_req),.paused(seq_paused),.resume_index(seq_resume_index),\n        .busy(seq_busy)')
    host = once(host, '''        16'd0, seq_error, 7'd0,seq_busy};''',
                '''        7'd0,seq_pause_req,7'd0,seq_paused,seq_error,7'd0,seq_busy};''')
    host = once(host, '            seq_start<=0;seq_entry_index<=0;seq_reg_rvalid<=0;seq_reg_rdata<=0;',
                '            seq_start<=0;seq_entry_index<=0;seq_pause_req<=0;seq_reg_rvalid<=0;seq_reg_rdata<=0;')
    host = once(host, '''            seq_start<=0;
            seq_reg_rvalid''', '''            seq_start<=0;
            if(seq_paused)seq_pause_req<=0;
            // Isolated MMIO extension: bit1 of the existing START byte requests pause.
            // Control requests do not touch program or operand SRAM while busy.
            if(command_req && seq_control_region && command_wr &&
               command_addr[4:0]==0 && command_wstrb[0] && command_wdata[1] && seq_busy)
                seq_pause_req<=1;
            seq_reg_rvalid''')
    for name, text in (('command.sv', decoder), ('tile_sequencer.sv', seq), ('tiled_host_bridge.sv', host)):
        (BASE/name).write_text('// Isolated deadline-pause-v1 extension; production source unchanged.\n'+text)


def word_intervals(live):
    words = 0
    for start, size in live:
        lo, hi = start//8, (start+size+7)//8
        words |= ((1 << (hi-lo))-1) << lo
    return [(first*8, size*8) for first, size in intervals(words)]


def prepare_fixture(variant):
    evidence = json.loads((SCREEN/'evidence.json').read_text())
    vww = (Path(evidence['vww']['source_directory']) if variant == 'descriptor' else
           SCREEN/'fixtures/vww-split25000')
    ad = SCREEN/'fixtures/ad-baseline'
    folder = BASE/'fixtures'/variant
    folder.mkdir(parents=True, exist_ok=True)
    payload = bytearray((vww/'payload.bin').read_bytes())
    vinput = (vww/'input.bin').read_bytes()
    payload[:len(vinput)] = vinput
    payload.extend(bytes((-len(payload))%8))
    adbase = len(payload)
    apayload = bytearray((ad/'payload.bin').read_bytes())
    ainput = (ad/'input.bin').read_bytes(); apayload[:len(ainput)] = ainput
    payload.extend(apayload); payload.extend(bytes((-len(payload))%8))
    contextbase = len(payload); payload.extend(bytes(MEM))
    clobberbase = len(payload); payload.extend(b'\xa5'*MEM)
    if len(payload) > 8*1024*1024:
        raise ValueError('private payload namespaces exceed SDRAM')
    vrows = commands((vww/'commands.bin').read_bytes())
    arows = commands((ad/'commands.bin').read_bytes())
    for i, r in enumerate(arows):
        if r[0] == 1:
            arows[i] = tuple(list(r[:3])+[r[3]+adbase]+list(r[4:]))
    adentry = len(vrows)
    program = pack(vrows+arows)
    temporaryentry = len(program)//16
    reserve = 4096
    if len(program)+reserve > MEM:
        raise ValueError('two resident job programs and temporary DMA program do not fit')
    (folder/'commands.bin').write_bytes(program)
    (folder/'payload.bin').write_bytes(payload)
    (folder/'vww-output.bin').write_bytes((vww/'output.bin').read_bytes())
    (folder/'ad-output.bin').write_bytes((ad/'output.bin').read_bytes())
    vout = int((vww/'checks.txt').read_text().splitlines()[0].split()[0])
    aout = int((ad/'checks.txt').read_text().splitlines()[0].split()[0])+adbase
    (folder/'checks.txt').write_text(f'{vout} vww-output.bin\n{aout} ad-output.bin\n')
    a = analyze_fixture(vww)
    lines = []
    for state in a['boundaries']:
        ranges = word_intervals(state['live_intervals'])
        if 16*(2*len(ranges)+1) > reserve:
            raise ValueError('checkpoint program too large')
        lines.append(' '.join(map(str, [state['next_index'], len(ranges)]+
                                 [value for interval in ranges for value in interval])))
    (folder/'live.txt').write_text('\n'.join(lines)+'\n')
    release = int(2014205*0.8)
    deadline = math.ceil(294419*1.2)
    (folder/'control.txt').write_text(' '.join(map(str, [adentry, temporaryentry,
        contextbase, clobberbase, release, deadline]))+'\n')
    record = {'variant': variant, 'vww_source': str(vww), 'ad_source': str(ad),
        'ad_external_base': adbase, 'context_external_base': contextbase,
        'clobber_external_base': clobberbase, 'ad_entry': adentry,
        'temporary_entry': temporaryentry, 'resident_program_bytes': len(program),
        'temporary_program_reserved_bytes': reserve, 'total_program_capacity_bytes': len(program)+reserve,
        'release_vww_elapsed_cycles': release, 'ad_relative_deadline_cycles': deadline,
        'vww_reference_native_cycles': 2014205 if variant == 'descriptor' else 2122886,
        'source_files_sha256': {str(p): sha(p) for d in (vww,ad) for p in
            (d/'commands.bin',d/'payload.bin',d/'input.bin',d/'output.bin')},
        'files_sha256': {n: sha(folder/n) for n in ('commands.bin','payload.bin',
            'checks.txt','control.txt','live.txt','vww-output.bin','ad-output.bin')}}
    save(folder/'identity.json', record)
    return folder


HARNESS_TAIL = r'''
static unsigned phase=0;
struct Span { unsigned phase; uint64_t start,end; unsigned elapsed,engine,dma,overlap; };
static std::vector<Span> spans;
static uint64_t accepted_cycle=0,paused_cycle=0;
static bool previous_pause_request=false,previous_paused=false;
static void observe_pause(){
    auto* r=d.rootp;
    bool request=r->v2_tiled_host_bridge__DOT__seq_pause_req;
    bool paused=r->v2_tiled_host_bridge__DOT__seq_paused;
    if(request&&!previous_pause_request)accepted_cycle=cycles;
    if(paused&&!previous_paused){
        require(!r->v2_tiled_host_bridge__DOT__seq_busy&&!d.engine_busy&&!d.dma_busy&&
                !r->v2_tiled_host_bridge__DOT__dma_pending&&!pending,
                "pause was not fully drained");
        paused_cycle=cycles;
    }
    previous_pause_request=request;previous_paused=paused;
}
static Bytes sram_snapshot(){
    Bytes out(32768);auto* r=d.rootp;
    for(unsigned i=0;i<4096;i++){
        SRAM_SNAPSHOT_ASSIGNMENTS
    }
    return out;
}
static void entry(unsigned pc){
    command(3,0x41001c,{uint8_t(pc),uint8_t(pc>>8)});
}
static Bytes run_sequence(unsigned pc,unsigned tag){
    phase=tag;entry(pc);uint64_t start=cycles;
    command(3,0x410000,{1});
    unsigned ticks=0;while(d.rootp->v2_tiled_host_bridge__DOT__seq_busy&&ticks++<200000000)step();
    require(ticks<200000000,"sequence timeout");
    auto regs=command(2,0x410000,{},32);
    require(regs.size()==32&&regs[0]==0&&regs[1]==0,"sequence failed, error "+std::to_string(regs.at(1)));
    spans.push_back({tag,start,cycles,u32(regs,8),u32(regs,12),u32(regs,16),u32(regs,20)});
    return regs;
}
static void temporary_program(unsigned entrypc,const Bytes& program){
    require(entrypc*16+program.size()<=32768,"temporary program capacity");
    for(unsigned i=0;i<program.size();i+=64)
        command(3,0x500000+entrypc*16+i,Bytes(program.begin()+i,
            program.begin()+std::min(unsigned(program.size()),i+64)));
}
static void append32(Bytes& bytes,uint32_t v){for(unsigned k=0;k<4;k++)bytes.push_back(v>>(k*8));}
static void emit(Bytes& program,unsigned op,unsigned flag,unsigned a=0,unsigned b=0,unsigned c=0){
    program.insert(program.end(),{uint8_t(op),uint8_t(flag),0,0});
    append32(program,a);append32(program,b);append32(program,c);
}
static Bytes context_program(const std::vector<std::pair<unsigned,unsigned>>& ranges,
                             unsigned contextbase,bool restore){
    Bytes program;
    for(auto [base,length]:ranges){emit(program,1,restore,contextbase+base,base,length);emit(program,3,2);}
    emit(program,0,0);return program;
}
int main(int argc,char** argv){try{
    Verilated::commandArgs(argc,argv);require(argc==5,"fixture directory, seed, report, mode required");
    std::string dir=argv[1];unsigned seed=std::stoul(argv[2]),mode=std::stoul(argv[4]);
    stalls=seed!=0;rng.seed(seed);
    auto payload=file(dir+"/payload.bin"),program=file(dir+"/commands.bin");
    require(payload.size()<=memory.size(),"payload capacity");std::copy(payload.begin(),payload.end(),memory.begin());
    unsigned adentry,tempentry,contextbase,clobberbase,release,relative_deadline;
    std::ifstream cf(dir+"/control.txt");cf>>adentry>>tempentry>>contextbase>>clobberbase>>release>>relative_deadline;
    require(bool(cf),"invalid control manifest");
    std::map<unsigned,std::vector<std::pair<unsigned,unsigned>>> live;
    std::ifstream lf(dir+"/live.txt");unsigned pc,n;
    while(lf>>pc>>n){auto& r=live[pc];for(unsigned i=0;i<n;i++){unsigned b,c;lf>>b>>c;r.emplace_back(b,c);}}
    require(!live.empty(),"empty boundary liveness");
    d.rst_n=0;d.rx_valid=0;d.tx_ready=0;d.memory_initialized=1;d.memory_port_busy=0;
    for(int k=0;k<4;k++)step();d.rst_n=1;for(int k=0;k<4;k++)step();
    for(unsigned i=0;i<program.size();i+=64)
        command(3,0x500000+i,Bytes(program.begin()+i,program.begin()+std::min(unsigned(program.size()),i+64)));
    uint64_t release_cycle=0,ad_done=0,vww_start=0,vww_done=0,save_native=0,restore_native=0,clobber_native=0;
    unsigned resume=0,saved_bytes=0,mismatches=0,pause_wait=0,busy_guard_checks=0;
    if(mode==3){
        auto regs=run_sequence(0,1);
        save_native=u32(regs,8);
    }else{
        phase=1;entry(0);vww_start=cycles;command(3,0x410000,{1});
        for(unsigned address:{0x500000u,0u,0x410000u}){
            bool rejected=false;
            try{command(3,address,{0});}
            catch(const std::runtime_error& e){rejected=std::string(e.what())=="command rejected 3";}
            require(rejected,"busy mutation guard failed");busy_guard_checks++;
        }
        unsigned ticks=0;
        while(d.rootp->v2_tiled_host_bridge__DOT__seq_elapsed<release&&
              d.rootp->v2_tiled_host_bridge__DOT__seq_busy&&ticks++<200000000)step();
        require(ticks<200000000&&d.rootp->v2_tiled_host_bridge__DOT__seq_busy,"release occurred after VWW completion");
        release_cycle=cycles;command(3,0x410000,{2});
        ticks=0;while(d.rootp->v2_tiled_host_bridge__DOT__seq_busy&&ticks++<200000000)step();
        require(ticks<200000000,"pause timeout");
        auto regs=command(2,0x410000,{},32);
        require(regs[1]==0&&regs[2]==1,"pause status/error");
        resume=u32(regs,24);require(live.count(resume),"pause is not a declared drained boundary");
        require(resume==d.rootp->v2_tiled_host_bridge__DOT__seq_resume_index,"resume PC not exposed consistently");
        require(accepted_cycle&&paused_cycle>=accepted_cycle,"missing pause edges");
        pause_wait=paused_cycle-accepted_cycle;
        spans.push_back({1,vww_start,cycles,u32(regs,8),u32(regs,12),u32(regs,16),u32(regs,20)});
        auto before=sram_snapshot();auto ranges=live[resume];
        for(auto [base,length]:ranges)saved_bytes+=length;
        temporary_program(tempentry,context_program(ranges,contextbase,false));
        regs=run_sequence(tempentry,2);save_native=u32(regs,8);
        if(mode==1||mode==2){
            Bytes clear;emit(clear,1,1,clobberbase,0,32768);emit(clear,3,2);emit(clear,0,0);
            temporary_program(tempentry,clear);regs=run_sequence(tempentry,3);clobber_native=u32(regs,8);
            auto cleared=sram_snapshot();require(std::all_of(cleared.begin(),cleared.end(),[](auto v){return v==0xa5;}),"clobber DMA did not overwrite scratchpad");
        }
        run_sequence(adentry,4);ad_done=cycles;
        if(mode!=2){
            temporary_program(tempentry,context_program(ranges,contextbase,true));
            regs=run_sequence(tempentry,5);restore_native=u32(regs,8);
            auto after=sram_snapshot();
            for(auto [base,length]:ranges)for(unsigned i=base;i<base+length;i++)mismatches+=before[i]!=after[i];
            require(mismatches==0,"live scratchpad restore changed bytes");
        }
        run_sequence(resume,6);vww_done=cycles;
    }
    std::ifstream checks(dir+"/checks.txt");unsigned address,nchecks=0;std::string name;
    while(checks>>address>>name){
        if(mode==3&&name!="vww-output.bin")continue;
        auto expected=file(dir+"/"+name);require(address+expected.size()<=memory.size(),"check range");
        for(unsigned i=0;i<expected.size();i++)require(memory[address+i]==expected[i],"output mismatch "+name+" byte "+std::to_string(i));nchecks++;
    }
    require(nchecks==(mode==3?1:2),"missing exact output checks");
    std::ofstream report(argv[3]);
    report<<"{\"status\":\"passed\",\"physical_board\":false,\"seed\":"<<seed<<",\"mode\":"<<mode
          <<",\"tensor_checks\":"<<nchecks<<",\"busy_guard_checks\":"<<busy_guard_checks<<",\"pause_resume_index\":"<<resume<<",\"saved_live_dma_bytes\":"<<saved_bytes
          <<",\"restored_byte_mismatches\":"<<mismatches<<",\"release_cycle\":"<<release_cycle
          <<",\"accepted_cycle\":"<<accepted_cycle<<",\"paused_cycle\":"<<paused_cycle
          <<",\"pause_wait_cycles\":"<<pause_wait<<",\"ad_response_global_native_cycles\":"<<(ad_done?ad_done-release_cycle:0)
          <<",\"ad_relative_deadline_cycles\":"<<relative_deadline<<",\"ad_deadline_missed\":"<<(ad_done&&ad_done-release_cycle>relative_deadline?"true":"false")
          <<",\"vww_completion_global_native_cycles\":"<<(vww_done?vww_done-vww_start:0)
          <<",\"save_sequence_native_cycles\":"<<save_native<<",\"restore_sequence_native_cycles\":"<<restore_native
          <<",\"clobber_sequence_native_cycles\":"<<clobber_native<<",\"program_bytes\":"<<program.size()<<",\"spans\":[";
    for(unsigned i=0;i<spans.size();i++){
        if(i)report<<",";auto s=spans[i];
        report<<"{\"phase\":"<<s.phase<<",\"start\":"<<s.start<<",\"end\":"<<s.end<<",\"elapsed\":"<<s.elapsed
              <<",\"engine\":"<<s.engine<<",\"dma\":"<<s.dma<<",\"overlap\":"<<s.overlap<<"}";
    }
    report<<"]}\n";
    std::cout<<dir<<" mode "<<mode<<" seed "<<seed<<" exact outputs passed"<<std::endl;
    d.final();return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
'''


def build():
    prepare_rtl()
    selected = REFERENCE/'work/phase6/engine-candidate-rtl-v2'
    evidence = json.loads((selected/'native/report.json').read_text())
    if evidence['status'] != 'passed' or sha(selected/'engine.sv') != PARENT_SHA:
        raise ValueError('selected engine identity')
    original = (ROOT/'test/phase6/native.cpp').read_text()
    harness = original[:original.index('int main(')]
    harness = harness.replace('#include <random>', '#include <random>\n#include <map>')
    harness = harness.replace('static int step(){', 'static void observe_pause();\nstatic int step(){')
    harness = harness.replace('d.clk=1;d.eval();cycles++;return out;',
                              'd.clk=1;d.eval();cycles++;observe_pause();return out;')
    assignments = '\n'.join(f'        out[i*8+{lane}]=r->v2_tiled_host_bridge__DOT__core__DOT__memory__DOT__banks__BRA__{lane}__KET____DOT__ram[i];'
                            for lane in range(8))
    harness += HARNESS_TAIL.replace('SRAM_SNAPSHOT_ASSIGNMENTS', assignments)
    path = BASE/'native.cpp'; path.write_text(harness)
    sources = [ROOT/'rtl/v2'/n for n in ('target_pkg.sv','requantizer.sv')]
    sources += [selected/'engine.sv']
    sources += [ROOT/'rtl/v2'/n for n in ('scratchpad.sv','tile_dma.sv','tiled_core.sv')]
    sources += [BASE/'command.sv',BASE/'tile_sequencer.sv',BASE/'tiled_host_bridge.sv']
    for p in sources:
        if p.parent == BASE:
            continue
        key = ('work/phase6/engine-candidate-rtl-v2/engine.sv' if p.name=='engine.sv' else str(p.relative_to(ROOT)))
        if sha(p) != evidence['source_sha256'][key]:
            raise ValueError('shared native source changed: '+key)
    builddir = BASE/'native-build'; builddir.mkdir(exist_ok=True)
    with (builddir/'build.log').open('w') as log:
        subprocess.run(['verilator','--cc','--exe','--build','-j','2','--public-flat-rw',
            '-Wno-fatal','--top-module','v2_tiled_host_bridge','--Mdir',str(builddir),
            *map(str,sources),str(path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            timeout=300,check=True)
    exe = builddir/'Vv2_tiled_host_bridge'
    save(builddir/'identity.json', {'status':'passed-build','engine_sha256':PARENT_SHA,
        'selected_native_report_sha256':sha(selected/'native/report.json'),
        'source_sha256':{str(p):sha(p) for p in sources},'harness_sha256':sha(path),
        'executable_sha256':sha(exe)})
    return exe


def run():
    start = time.monotonic(); exe = build(); results=[]
    for variant in ('descriptor','selective-vww25k'):
        folder = prepare_fixture(variant)
        for seed in (0,6063):
            for mode in (3,0,1):
                dest = BASE/'native'/f'{variant}-s{seed}-m{mode}.json'
                dest.parent.mkdir(exist_ok=True)
                invocation = subprocess.run([str(exe),str(folder),str(seed),str(dest),str(mode)],
                    cwd=ROOT,capture_output=True,text=True,timeout=120)
                if invocation.returncode:
                    save(BASE/'failure.json',{'variant':variant,'seed':seed,'mode':mode,
                        'returncode':invocation.returncode,'stderr':invocation.stderr[-3000:]})
                    raise ValueError('native pause test failed: '+invocation.stderr)
                result=json.loads(dest.read_text()); result.update(variant=variant)
                if mode==3 and seed==0 and result['save_sequence_native_cycles'] != json.loads(
                    (folder/'identity.json').read_text())['vww_reference_native_cycles']:
                    raise ValueError('pause-disabled copy changed native cycle count')
                results.append(result)
                print(variant,seed,mode,'AD response',result['ad_response_global_native_cycles'],
                      'miss',result['ad_deadline_missed'],flush=True)
    folder=BASE/'fixtures/descriptor'; dest=BASE/'native/negative-no-restore.json'
    invocation=subprocess.run([str(exe),str(folder),'0',str(dest),'2'],cwd=ROOT,
                              capture_output=True,text=True,timeout=120)
    if invocation.returncode==0:
        raise ValueError('scratchpad restore negative control did not fail')
    negative={'status':'passed-negative-control','returncode':invocation.returncode,
              'stderr':invocation.stderr[-3000:],'scope':'clobber and AD overwrite without restoring live SRAM'}
    save(dest,negative)
    report={'status':'passed-native-pause','physical_board':False,'engine_sha256':PARENT_SHA,
        'native_runs':results,'negative_control':negative,'elapsed_seconds':time.monotonic()-start,
        'python_max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'driver_sha256':sha(Path(__file__)),
        'scope':'isolated drained-WAIT pause/resume, real two-job exact outputs, real context DMA, abstract external RAM',
        'limitations':['No board qualification, physical SDRAM timing, routed area, or worst-case deadline proof.',
            'Native host transports each UART byte in one cycle; global native response includes those cycles but does not model the physical750kbaud link.',
            'All live SRAM DMA granules are saved/restored. Immutable-source reload optimization is not implemented.',
            'A single urgent AD job is run to completion after pausing VWW. This is a two-job controller witness, not a general online scheduler.',
            'Mode1 adds artificial full scratchpad clobber as a correctness stress test; mode0 uses actual AD writes only.']}
    save(BASE/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','run'))
    args=parser.parse_args()
    if args.stage=='prepare':
        prepare_rtl()
        for variant in ('descriptor','selective-vww25k'):prepare_fixture(variant)
        print('prepared isolated pause copies and two real private-namespace job fixtures')
    else:
        r=run();print(json.dumps({k:r[k] for k in ('status','elapsed_seconds','python_max_rss_bytes')},indent=2))
