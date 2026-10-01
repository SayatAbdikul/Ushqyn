#!/usr/bin/env python3
"""Isolated autonomous resident two-job checkpoint controller feasibility test.

An urgent request is admitted once, then resident program/metadata SRAM drives
save, urgent inference, restore and background resumption without host service.
Arithmetic engine and production RTL are unchanged. This is an engineering
baseline; joint program/state feasibility must be compared with prior art.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name]='1'
import argparse
import json
import math
from pathlib import Path
import re
import resource
import shutil
import struct
import subprocess
import sys
import time

from deadline_screen import ROOT,REFERENCE,PARENT_SHA,sha,save

BASE=ROOT/'work/phase6/deadline-resident-v1'
POLICY=ROOT/'work/phase6/deadline-policy-v1'
GOWIN=Path('/Applications/GowinIDE.app/Contents/Resources/Gowin_EDA/IDE')


def once(text,old,new):
    if text.count(old)!=1:
        raise ValueError('isolated source patch changed: '+old[:100])
    return text.replace(old,new,1)


def prepare_rtl():
    BASE.mkdir(parents=True,exist_ok=True)
    seq=(ROOT/'rtl/v2/tile_sequencer.sv').read_text()
    seq=once(seq,'    input logic [10:0] start_index,','''    input logic [10:0] start_index,
    input logic urgent_req,
    input logic [10:0] metadata_index,
    output logic urgent_accept, urgent_pending,
    output logic [1:0] context_phase,
    output logic [11:0] resume_index,''')
    seq=once(seq,'typedef enum logic [3:0] {IDLE, FETCH0, READ0, FETCH1, READ1,\n                              EXECUTE, ADVANCE, DRAIN}',
             '''typedef enum logic [4:0] {IDLE, FETCH0, READ0, FETCH1, READ1,
                              EXECUTE, ADVANCE, DRAIN,
                              META_FETCH0, META_READ0, META_FETCH1, META_READ1, META_EXECUTE}''')
    seq=once(seq,'    wire fetching = state == FETCH0 || state == FETCH1;', '''    // Existing single program-SRAM port also reads immutable resident metadata.
    logic [10:0] metadata_cursor, urgent_entry, restore_entry;
    logic [11:0] metadata_remaining, lookup_next;
    logic metadata_header,metadata_fallback;
    logic [63:0] metadata_low, metadata_high;
    wire metadata_fetch = state==META_FETCH0 || state==META_FETCH1;
    wire fetching = state==FETCH0 || state==FETCH1 || metadata_fetch;
    wire [10:0] fetch_index = metadata_fetch ? metadata_cursor : command_index[10:0];
    wire fetch_high = state==FETCH1 || state==META_FETCH1;
    assign urgent_accept = busy && context_phase==0 && !urgent_pending && state!=DRAIN;''')
    seq=once(seq,'.addr(busy ? {9\'b0,command_index[10:0],(state==FETCH1),3\'b0} :',
                  '.addr(busy ? {9\'b0,fetch_index,fetch_high,3\'b0} :')
    seq=once(seq,'            command_index<=0;low_word<=0;high_word<=0;live_base<=0;live_end<=32768;',
             '''            command_index<=0;low_word<=0;high_word<=0;live_base<=0;live_end<=32768;
            urgent_pending<=0;context_phase<=0;resume_index<=0;
            metadata_cursor<=0;urgent_entry<=0;restore_entry<=0;metadata_remaining<=0;
            lookup_next<=0;metadata_header<=0;metadata_fallback<=0;metadata_low<=0;metadata_high<=0;''')
    seq=once(seq,'            engine_start<=0;dma_start<=0;abort_units<=0;',
             '''            engine_start<=0;dma_start<=0;abort_units<=0;
            if(urgent_req && urgent_accept)urgent_pending<=1;''')
    seq=once(seq,'                    busy<=1;error_code<=0;elapsed<=0;engine_cycles<=0;',
             '                    busy<=1;urgent_pending<=0;metadata_fallback<=0;context_phase<=0;error_code<=0;elapsed<=0;engine_cycles<=0;')
    seq=once(seq,'''                    state<=FETCH0;
                end
                FETCH0:''', '''                    state<=FETCH0;
                end else if(urgent_req && !engine_busy && !dma_busy)begin
                    // CRC-admitted notification may reach the host bus just
                    // after background HALT. Complete it autonomously too.
                    busy<=1;error_code<=0;elapsed<=0;engine_cycles<=0;dma_cycles<=0;overlap_cycles<=0;
                    urgent_pending<=1;metadata_fallback<=1;metadata_header<=1;
                    metadata_cursor<=metadata_index;state<=META_FETCH0;
                end
                FETCH0:''')
    seq=once(seq,'                            else begin busy<=0;state<=IDLE;end', '''                            else case(context_phase)
                                1: begin // SAVE HALT -> urgent resident program.
                                    command_index<={1'b0,urgent_entry};context_phase<=2;state<=FETCH0;
                                end
                                2: begin // Urgent HALT -> restore or completed-background exit.
                                    if(metadata_fallback)begin
                                        busy<=0;context_phase<=0;metadata_fallback<=0;state<=IDLE;
                                    end else begin
                                        command_index<={1'b0,restore_entry};context_phase<=3;state<=FETCH0;
                                    end
                                end
                                3: begin // Restore HALT -> compiler-approved original nextPC.
                                    command_index<=resume_index;context_phase<=0;state<=FETCH0;
                                end
                                default: if(urgent_pending || (urgent_req && urgent_accept))begin
                                    // Accepted late requests cannot be silently discarded.
                                    metadata_fallback<=1;metadata_header<=1;
                                    metadata_cursor<=metadata_index;state<=META_FETCH0;
                                end else begin busy<=0;urgent_pending<=0;state<=IDLE;end
                            endcase''')
    seq=once(seq,'''                            else state<=ADVANCE;
                        end
                        default: fail(8'd10);''', '''                            else if(urgent_pending && context_phase==0 && !engine_busy && !dma_busy && command_index<2047)begin
                                // Only a drained WAIT may consult sparse legal checkpoints.
                                lookup_next<=command_index+1;metadata_cursor<=metadata_index;metadata_fallback<=0;
                                metadata_header<=1;state<=META_FETCH0;
                            end else state<=ADVANCE;
                        end
                        default: fail(8'd10);''')
    seq=once(seq,'                DRAIN: if (!engine_busy && !dma_busy)', '''                META_FETCH0: if(memory_ready)state<=META_READ0;
                META_READ0: if(memory_rvalid)begin metadata_low<=memory_rdata;state<=META_FETCH1;end
                META_FETCH1: if(memory_ready)state<=META_READ1;
                META_READ1: if(memory_rvalid)begin metadata_high<=memory_rdata;state<=META_EXECUTE;end
                META_EXECUTE: begin
                    if(metadata_header)begin
                        // Header <urgentEntry,count,job0,reserved0>; records follow.
                        if(metadata_low[31:0]>=2048 || metadata_low[63:32]==0 ||
                           metadata_low[63:32]>2047 || metadata_high!=0 ||
                           {21'd0,metadata_cursor}+metadata_low[63:32]>=2048)fail(8'd12);
                        else begin
                            urgent_entry<=metadata_low[10:0];metadata_remaining<=metadata_low[43:32];
                            if(metadata_fallback)begin
                                command_index<={1'b0,metadata_low[10:0]};context_phase<=2;urgent_pending<=0;state<=FETCH0;
                            end else begin
                                metadata_cursor<=metadata_cursor+1;metadata_header<=0;state<=META_FETCH0;
                            end
                        end
                    end else if(metadata_low[31:0]>=2048 || metadata_low[63:32]>=2048 ||
                                metadata_high[31:0]>=2048 || metadata_high[63:32]!=0)fail(8'd12);
                    else if(metadata_low[31:0]=={20'd0,lookup_next})begin
                        resume_index<=lookup_next;restore_entry<=metadata_high[10:0];
                        command_index<={1'b0,metadata_low[42:32]};
                        context_phase<=1;urgent_pending<=0;state<=FETCH0;
                    end else if(metadata_remaining==1)state<=ADVANCE;
                    else begin
                        metadata_remaining<=metadata_remaining-1;metadata_cursor<=metadata_cursor+1;
                        state<=META_FETCH0;
                    end
                end
                DRAIN: if (!engine_busy && !dma_busy)''')
    (BASE/'tile_sequencer.sv').write_text('// Isolated resident controller; production unchanged.\n'+seq)
    for command_name,host_name,out_command,out_host in (
        ('rtl/v2/command.sv','rtl/v2/tiled_host_bridge.sv','command.sv','tiled_host_bridge.sv'),
        ('rtl/phase6/uart_burst_command.sv','rtl/phase6/uart_burst_bridge.sv','uart_burst_command.sv','uart_burst_bridge.sv')):
        command=(ROOT/command_name).read_text()
        command=once(command,'    input logic mmio_while_busy,','    input logic mmio_while_busy,\n    input logic seq_urgent_while_busy,')
        command=once(command,'    logic [24:0] transfer_addr;', '''    // Admission only: a CRC-validated one-byte urgent notification.
    wire tiled_busy_urgent_write = TILED_MEM_MAP && seq_urgent_while_busy &&
        cmd==WRITE && address==24'h410000 && length==1 && payload[0]==8'h02;
    logic [24:0] transfer_addr;''')
        command=once(command,"if(busy && !tiled_busy_mmio_write)respond(8'd3,1);",
                     "if(busy && !tiled_busy_mmio_write && !tiled_busy_urgent_write)respond(8'd3,1);")
        host=(ROOT/host_name).read_text()
        host=once(host,'        .mmio_while_busy(!seq_busy && memory_initialized && engine_busy &&',
                   '        .seq_urgent_while_busy(seq_urgent_accept && memory_initialized),\n        .mmio_while_busy(!seq_busy && memory_initialized && engine_busy &&')
        host=once(host,'    logic [11:0] seq_index;', '''    logic [11:0] seq_index,seq_resume_index;
    logic seq_urgent_req,seq_urgent_accept,seq_urgent_pending;
    logic [1:0] seq_context_phase;
    logic [10:0] seq_metadata_index;''')
        host=once(host,'        .abort_run(abort_command),\n        .busy(seq_busy)', '''        .abort_run(abort_command),
        .urgent_req(seq_urgent_req),.metadata_index(seq_metadata_index),
        .urgent_accept(seq_urgent_accept),.urgent_pending(seq_urgent_pending),
        .context_phase(seq_context_phase),.resume_index(seq_resume_index),
        .busy(seq_busy)''')
        host=once(host,"wire [255:0] seq_registers = {21'd0,seq_entry_index, 20'd0,seq_index, seq_overlap,",
                  "wire [255:0] seq_registers = {5'd0,seq_metadata_index,5'd0,seq_entry_index, 20'd0,seq_index, seq_overlap,")
        host=once(host,"        16'd0, seq_error, 7'd0,seq_busy};",
                  "        13'd0,seq_urgent_pending,seq_context_phase, seq_error, 7'd0,seq_busy};")
        host=once(host,'            seq_start<=0;seq_entry_index<=0;seq_reg_rvalid<=0;seq_reg_rdata<=0;',
                  '            seq_start<=0;seq_entry_index<=0;seq_metadata_index<=0;seq_urgent_req<=0;seq_reg_rvalid<=0;seq_reg_rdata<=0;')
        host=once(host,'''            seq_start<=0;
            seq_reg_rvalid''', '''            seq_start<=0;seq_urgent_req<=0;
            if(command_req && seq_control_region && command_wr && command_addr[4:0]==0 &&
               command_wstrb[0] && command_wdata[1] &&
               (seq_urgent_accept || (!seq_busy && !engine_busy && !dma_busy && !dma_pending)))seq_urgent_req<=1;
            seq_reg_rvalid''')
        host=once(host,'                        if (command_wstrb[5]) seq_entry_index[10:8]<=command_wdata[42:40];',
                  '''                        if (command_wstrb[5]) seq_entry_index[10:8]<=command_wdata[42:40];
                        if (command_wstrb[6]) seq_metadata_index[7:0]<=command_wdata[55:48];
                        if (command_wstrb[7]) seq_metadata_index[10:8]<=command_wdata[58:56];''')
        (BASE/out_command).write_text('// Isolated resident admission extension.\n'+command)
        (BASE/out_host).write_text('// Isolated resident controller host.\n'+host)
    save(BASE/'rtl-identity.json',{'engine_sha256':PARENT_SHA,'driver_sha256':sha(Path(__file__)),
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in BASE.glob('*.sv')}})


HARNESS_TAIL=r'''
static Bytes sram_snapshot(){
    Bytes out(32768);auto* r=d.rootp;
    for(unsigned i=0;i<4096;i++){
        SRAM_SNAPSHOT_ASSIGNMENTS
    }
    return out;
}
static unsigned prior_phase=0,uart_after_accept=0,resume_pc=0,contextbase=0,urgententry=0,clobberdonepc=0;
static uint64_t accepted_cycle=0,checkpoint_cycle=0,urgent_done_cycle=0,restore_done_cycle=0;
static bool active_service=false,clobber_checked=false,want_clobber=true,completion_fallback=false;
static unsigned metadata_cycles=0;
static Bytes before;
static std::map<unsigned,std::vector<std::pair<unsigned,unsigned>>> live;
static unsigned saved_bytes=0,restored_mismatches=0,context_mismatches=0;
static void observe_context(){
    auto* r=d.rootp;unsigned phase=r->v2_tiled_host_bridge__DOT__seq_context_phase;
    if(!accepted_cycle&&r->v2_tiled_host_bridge__DOT__seq_urgent_pending){accepted_cycle=cycles;active_service=true;}
    if(active_service&&d.rx_valid)uart_after_accept++;
    unsigned state=r->v2_tiled_host_bridge__DOT__sequencer__DOT__state;
    if(state>=8&&state<=12)metadata_cycles++;
    if(prior_phase==0&&phase==1){
        require(!d.engine_busy&&!d.dma_busy&&!r->v2_tiled_host_bridge__DOT__dma_pending&&!pending,"checkpoint was not drained");
        checkpoint_cycle=cycles;resume_pc=r->v2_tiled_host_bridge__DOT__seq_resume_index;
        require(live.count(resume_pc),"undeclared checkpoint PC");before=sram_snapshot();
        for(auto [base,length]:live[resume_pc])saved_bytes+=length;
    }
    if(prior_phase==1&&phase==2){
        for(auto [base,length]:live[resume_pc])for(unsigned i=base;i<base+length;i++)context_mismatches+=before[i]!=memory[contextbase+i];
        require(context_mismatches==0,"resident save DMA changed live bytes");
    }
    if(want_clobber&&phase==2&&!clobber_checked&&r->v2_tiled_host_bridge__DOT__seq_index==clobberdonepc){
        auto s=sram_snapshot();require(std::all_of(s.begin(),s.end(),[](auto v){return v==0xa5;}),"full SRAM destructive DMA clobber missing");
        clobber_checked=true;
    }
    if(prior_phase==2&&phase==3)urgent_done_cycle=cycles;
    if(prior_phase==0&&phase==2)completion_fallback=true;
    if(prior_phase==2&&phase==0)urgent_done_cycle=cycles;
    if(prior_phase==3&&phase==0){
        restore_done_cycle=cycles;auto after=sram_snapshot();
        for(auto [base,length]:live[resume_pc])for(unsigned i=base;i<base+length;i++)restored_mismatches+=before[i]!=after[i];
    }
    prior_phase=phase;
}
int main(int argc,char** argv){try{
    Verilated::commandArgs(argc,argv);require(argc==5||argc==6,"fixture,seed,report,mode[,release] required");
    std::string dir=argv[1];unsigned seed=std::stoul(argv[2]),mode=std::stoul(argv[4]);
    want_clobber=mode!=0;
    stalls=seed!=0;rng.seed(seed);
    auto payload=file(dir+"/payload.bin"),program=file(dir+"/commands.bin");
    require(payload.size()<=memory.size()&&program.size()<=32768,"resident memory capacity");
    std::copy(payload.begin(),payload.end(),memory.begin());
    unsigned entry,metadataindex,release,deadline;
    std::ifstream cf(dir+"/control.txt");cf>>entry>>metadataindex>>urgententry>>contextbase>>clobberdonepc>>release>>deadline;
    require(bool(cf),"invalid resident control metadata");
    if(argc==6)release=std::stoul(argv[5]);
    std::ifstream lf(dir+"/live.txt");unsigned pc,n;
    while(lf>>pc>>n){auto& spans=live[pc];for(unsigned i=0;i<n;i++){unsigned b,c;lf>>b>>c;spans.emplace_back(b,c);}}
    require(!live.empty(),"resident checkpoint metadata empty");
    if(mode==0){ // Same resident capacity, bypass artificial stress clobber only.
        for(unsigned k=0;k<4;k++)program[metadataindex*16+k]=clobberdonepc>>(k*8);
    }
    if(mode==2){ // Negative control: omit every restore DMA, preserving code capacity.
        unsigned count=u32(program,metadataindex*16+4);
        for(unsigned i=0;i<count;i++){
            unsigned restore=u32(program,(metadataindex+1+i)*16+8);
            std::fill(program.begin()+restore*16,program.begin()+restore*16+16,0);
        }
    }
    d.rst_n=0;d.rx_valid=0;d.tx_ready=0;d.memory_initialized=1;d.memory_port_busy=0;
    for(int k=0;k<4;k++)step();d.rst_n=1;for(int k=0;k<4;k++)step();
    for(unsigned i=0;i<program.size();i+=64)command(3,0x500000+i,Bytes(program.begin()+i,program.begin()+std::min(unsigned(program.size()),i+64)));
    command(3,0x41001c,{uint8_t(entry),uint8_t(entry>>8),uint8_t(metadataindex),uint8_t(metadataindex>>8)});
    uint64_t start=cycles;command(3,0x410000,{1});
    unsigned guard_checks=0;
    for(unsigned address:{0x500000u,0u,0x410000u,0x41001eu}){
        bool rejected=false;try{command(3,address,{0});}
        catch(const std::runtime_error& e){rejected=std::string(e.what())=="command rejected 3";}
        require(rejected,"unsafe busy write accepted");guard_checks++;
    }
    unsigned ticks=0;
    while(d.rootp->v2_tiled_host_bridge__DOT__seq_elapsed<release&&d.rootp->v2_tiled_host_bridge__DOT__seq_busy&&ticks++<20000000)step();
    require(ticks<20000000&&d.rootp->v2_tiled_host_bridge__DOT__seq_busy,"urgent release too late");
    uint64_t release_cycle=cycles;
    if(mode!=3)command(3,0x410000,{2});
    ticks=0;while(d.rootp->v2_tiled_host_bridge__DOT__seq_busy&&ticks++<20000000)step();
    require(ticks<20000000,"autonomous resident timeout");
    uint64_t completion=cycles;
    active_service=false;
    auto regs=command(2,0x410000,{},32);unsigned error=regs[1];
    unsigned mismatches=0,ad_mismatches=0,checks=0;
    std::ifstream checks_file(dir+"/checks.txt");unsigned address;std::string name;
    while(checks_file>>address>>name){
        if(mode==3&&name!="vww-output.bin")continue;
        auto expected=file(dir+"/"+name);unsigned local=0;
        for(unsigned i=0;i<expected.size();i++)local+=memory[address+i]!=expected[i];
        if(name=="ad-output.bin")ad_mismatches+=local;
        mismatches+=local;checks++;
    }
    require(uart_after_accept==0,"host UART was used after accepted urgent request");
    if(mode==3){require(error==0&&mismatches==0&&checks==1,"pause-disabled baseline mismatch");}
    else {
        require(accepted_cycle&&(checkpoint_cycle||completion_fallback)&&urgent_done_cycle&&(!want_clobber||clobber_checked),"resident phase coverage incomplete");
        require(ad_mismatches==0,"urgent AD output differs");
        if(mode==2)require(error||mismatches||restored_mismatches,"omitted restore negative failed to detect corruption");
        else require(error==0&&mismatches==0&&restored_mismatches==0&&checks==2,"resident exact output or restored context mismatch");
    }
    std::ofstream f(argv[3]);
    f<<"{\"status\":\"passed\",\"mode\":"<<mode<<",\"seed\":"<<seed<<",\"program_bytes\":"<<program.size()
     <<",\"busy_guard_checks\":"<<guard_checks<<",\"post_request_uart_rx_cycles\":"<<uart_after_accept
     <<",\"release_elapsed_target\":"<<release
     <<",\"release_cycle\":"<<release_cycle<<",\"accepted_cycle\":"<<accepted_cycle
     <<",\"checkpoint_cycle\":"<<checkpoint_cycle<<",\"urgent_done_cycle\":"<<urgent_done_cycle
     <<",\"restore_done_cycle\":"<<restore_done_cycle<<",\"response_cycles_from_accept\":"<<(urgent_done_cycle?urgent_done_cycle-accepted_cycle:0)
     <<",\"response_cycles_from_release\":"<<(urgent_done_cycle?urgent_done_cycle-release_cycle:0)
     <<",\"urgent_relative_deadline\":"<<deadline<<",\"urgent_deadline_missed\":"<<(urgent_done_cycle&&urgent_done_cycle-release_cycle>deadline?"true":"false")
     <<",\"vww_completion_cycles\":"<<completion-start<<",\"sequence_elapsed_cycles\":"<<u32(regs,8)
     <<",\"resume_pc\":"<<resume_pc<<",\"saved_live_bytes\":"<<saved_bytes
     <<",\"completion_fallback\":"<<(completion_fallback?"true":"false")
     <<",\"metadata_scan_cycles\":"<<metadata_cycles
     <<",\"save_context_mismatches\":"<<context_mismatches<<",\"restored_byte_mismatches\":"<<restored_mismatches
     <<",\"output_byte_mismatches\":"<<mismatches<<",\"sequence_error\":"<<error
     <<",\"tensor_checks\":"<<checks<<",\"clobber_checked\":"<<(clobber_checked?"true":"false")<<"}\n";
    d.final();return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
'''


def build_native():
    prepare_rtl()
    selected=REFERENCE/'work/phase6/engine-candidate-rtl-v2'
    original_identity=json.loads((selected/'native/report.json').read_text())
    if original_identity['status']!='passed' or sha(selected/'engine.sv')!=PARENT_SHA:
        raise ValueError('selected engine identity')
    original=(ROOT/'test/phase6/native.cpp').read_text()
    harness=original[:original.index('int main(')]
    harness=once(harness,'#include <random>','#include <random>\n#include <map>')
    harness=once(harness,'static int step(){','static void observe_context();\nstatic int step(){')
    harness=once(harness,'d.clk=1;d.eval();cycles++;return out;','d.clk=1;d.eval();cycles++;observe_context();return out;')
    assignments='\n'.join(f'        out[i*8+{lane}]=r->v2_tiled_host_bridge__DOT__core__DOT__memory__DOT__banks__BRA__{lane}__KET____DOT__ram[i];' for lane in range(8))
    harness+=HARNESS_TAIL.replace('SRAM_SNAPSHOT_ASSIGNMENTS',assignments)
    path=BASE/'native.cpp';path.write_text(harness)
    sources=[ROOT/'rtl/v2'/n for n in ('target_pkg.sv','requantizer.sv')]+[selected/'engine.sv']
    sources += [ROOT/'rtl/v2'/n for n in ('scratchpad.sv','tile_dma.sv','tiled_core.sv')]
    sources += [BASE/n for n in ('command.sv','tile_sequencer.sv','tiled_host_bridge.sv')]
    build=BASE/'native-build';build.mkdir(exist_ok=True)
    with (build/'build.log').open('w') as log:
        subprocess.run(['verilator','--cc','--exe','--build','-j','2','--public-flat-rw','-Wno-fatal',
            '--top-module','v2_tiled_host_bridge','--Mdir',str(build),*map(str,sources),str(path)],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    exe=build/'Vv2_tiled_host_bridge'
    save(build/'identity.json',{'engine_sha256':PARENT_SHA,'source_sha256':{str(p):sha(p) for p in sources},
                              'harness_sha256':sha(path),'executable_sha256':sha(exe)})
    return exe


def policy_fixture(source):
    manifest_path=source/'policy-manifest.json'
    manifest=json.loads(manifest_path.read_text())
    program=(source/'commands.bin').read_bytes()
    metadata_offset=manifest['metadata_offset_bytes']
    if len(program)>32768 or metadata_offset%16 or metadata_offset+len((source/'metadata.bin').read_bytes())>len(program):
        raise ValueError('resident code and metadata do not fit32KiB')
    folder=BASE/'fixtures'/source.name
    folder.mkdir(parents=True,exist_ok=True)
    for p in source.iterdir():
        if p.is_file():shutil.copyfile(p,folder/p.name)
    for filename,digest in manifest['fixture_sha256'].items():
        if sha(folder/filename)!=digest:raise ValueError('policy source fixture changed')
    lines=[]
    for line in (folder/'checks.txt').read_text().splitlines():
        address,name=line.split()
        renamed='vww-output.bin' if name.startswith('vww-') else 'ad-output.bin'
        shutil.copyfile(folder/name,folder/renamed)
        lines.append(address+' '+renamed)
    (folder/'checks.txt').write_text('\n'.join(lines)+'\n')
    (folder/'live.txt').write_text('\n'.join(' '.join(map(str,[c['global_next_index'],len(c['spans']),
        *(v for span in c['spans'] for v in span)])) for c in manifest['checkpoints'])+'\n')
    (folder/'control.txt').write_text(' '.join(map(str,[manifest['background_entry'],metadata_offset//16,
        manifest['urgent_entry'],manifest['context_external_base'],manifest['actual_urgent_entry'],
        int(2014205*.8),math.ceil(294419*1.2)]))+'\n')
    return folder,manifest


def run_native(fixtures):
    # Early capacity check precedes RTL build or simulation.
    prepared=[policy_fixture(Path(p)) for p in fixtures]
    save(BASE/'fit-gate.json',{'status':'passed-program-capacity','fixtures':[
        {'path':str(p),'program_bytes':(p/'commands.bin').stat().st_size,'metadata_offset':m['metadata_offset_bytes']} for p,m in prepared]})
    exe=build_native();results=[]
    for folder,manifest in prepared:
        for seed in (0,6063):
            for mode in (3,0,1,2):
                dest=BASE/'native'/f'{folder.name}-s{seed}-m{mode}.json';dest.parent.mkdir(exist_ok=True)
                invocation=subprocess.run([str(exe),str(folder),str(seed),str(dest),str(mode)],cwd=ROOT,
                    capture_output=True,text=True,timeout=180)
                if invocation.returncode:
                    save(BASE/'failure.json',{'fixture':str(folder),'seed':seed,'mode':mode,'stderr':invocation.stderr[-3000:]})
                    raise RuntimeError('autonomous native failed: '+invocation.stderr)
                row=json.loads(dest.read_text());row['variant']=folder.name;results.append(row)
                print(folder.name,seed,mode,'response',row['response_cycles_from_release'],'error',row['sequence_error'],flush=True)
    report={'status':'passed-native-resident-controller','engine_sha256':PARENT_SHA,'native_runs':results,
            'driver_sha256':sha(Path(__file__)),'native_identity_sha256':sha(BASE/'native-build/identity.json'),
            'python_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'physical_board':False,'physical':'pending','scope':__doc__}
    save(BASE/'report.json',report)
    return report


def run_arrivals(fixtures):
    """Verify early and completion-race admission on the same built controller."""
    identity=json.loads((BASE/'native-build/identity.json').read_text())
    exe=BASE/'native-build/Vv2_tiled_host_bridge'
    if sha(exe)!=identity['executable_sha256'] or sha(BASE/'native.cpp')!=identity['harness_sha256']:
        raise ValueError('arrival native identity changed')
    for path,digest in identity['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('arrival source changed')
    profiles=json.loads((POLICY/'profiles.json').read_text())
    results=[]
    for source in fixtures:
        manifest=json.loads((Path(source)/'policy-manifest.json').read_text())
        folder=BASE/'fixtures'/Path(source).name
        variant=manifest['variant']
        for seed in (0,6063):
            profile=profiles['vww'][variant][str(seed)]
            allowed={c['global_next_index'] for c in manifest['checkpoints']}
            elapsed=0;last_site=0
            for segment in profile['segments']:
                elapsed+=segment['cycles']
                if segment['next_index'] in allowed:last_site=elapsed
            if not 0<last_site<elapsed:raise ValueError('late arrival interval absent')
            # The stalled RAM consumes RNG values on every cycle, including
            # program preload. Use this exact resident harness's uninterrupted
            # run for the completion race, rather than another harness's trace.
            baseline_path=BASE/'native'/f'{folder.name}-s{seed}-m3.json'
            baseline=json.loads(baseline_path.read_text())
            if baseline['mode']!=3 or baseline['sequence_error'] or baseline['output_byte_mismatches']:
                raise ValueError('arrival uninterrupted control is invalid')
            matched_elapsed=baseline['sequence_elapsed_cycles']
            last_site=round(last_site*matched_elapsed/elapsed)
            elapsed=matched_elapsed
            positions={'early':elapsed//10,'middle':elapsed//2,
                       'after-last-site':last_site+(elapsed-last_site)//2,
                       'completion-race':elapsed-20}
            for label,release in positions.items():
                dest=BASE/'arrivals'/f'{folder.name}-s{seed}-{label}.json';dest.parent.mkdir(exist_ok=True)
                invocation=subprocess.run([str(exe),str(folder),str(seed),str(dest),'0',str(release)],
                    cwd=ROOT,capture_output=True,text=True,timeout=180)
                if invocation.returncode:
                    save(BASE/'arrival-failure.json',{'fixture':str(folder),'seed':seed,'label':label,
                         'release':release,'stderr':invocation.stderr[-3000:]})
                    raise RuntimeError('arrival native failed: '+invocation.stderr)
                row=json.loads(dest.read_text())
                row.update(variant=folder.name,label=label,last_allowed_site_elapsed=last_site,
                           last_allowed_site_timing='scaled standalone trace estimate; actual fallback is asserted',
                           uninterrupted_baseline_sha256=sha(baseline_path),
                           report_path=str(dest.relative_to(ROOT)),report_sha256=sha(dest))
                if label in ('after-last-site','completion-race') and not row['completion_fallback']:
                    raise ValueError('late completion fallback not exercised')
                results.append(row)
                print(folder.name,seed,label,row['response_cycles_from_release'],'fallback',row['completion_fallback'],flush=True)
    report={'status':'passed-native-arrival-coverage','runs':results,'native_identity_sha256':sha(BASE/'native-build/identity.json'),
            'driver_sha256':sha(Path(__file__)),'physical_board':False}
    save(BASE/'arrival-report.json',report)
    return report


def physical_recipe(mode):
    recipe=(REFERENCE/'work/phase6/engine-candidate-rtl-v2/build27.tcl').read_text()
    recipe=recipe.replace('set root {'+str(REFERENCE)+'}','set root {'+str(REFERENCE)+'}\nset resident {'+str(BASE)+'}')
    if mode=='candidate':
        for path in ('rtl/phase6/uart_burst_command.sv','rtl/phase6/uart_burst_bridge.sv','rtl/v2/tile_sequencer.sv'):
            recipe=once(recipe,'    '+path+'\n','')
        recipe=once(recipe,'add_file $ip','''add_file [file join $resident uart_burst_command.sv]
add_file [file join $resident uart_burst_bridge.sv]
add_file [file join $resident tile_sequencer.sv]
add_file $ip''')
    return recipe


def route(mode):
    folder=BASE/'physical'/mode;folder.mkdir(parents=True,exist_ok=True)
    recipe=folder/'build27.tcl';recipe.write_text(physical_recipe(mode))
    environment=dict(os.environ,DYLD_FRAMEWORK_PATH=str(GOWIN/'lib'),DYLD_LIBRARY_PATH=str(GOWIN/'lib'),
                     PHASE6_ENGINE=str(REFERENCE/'work/phase6/engine-candidate-rtl-v2/engine.sv'))
    start=time.perf_counter()
    with (folder/'route.log').open('w') as log:
        invocation=subprocess.run([str(GOWIN/'bin/gw_sh'),str(recipe)],cwd=folder,env=environment,
            stdout=log,stderr=subprocess.STDOUT,timeout=600)
    project=folder/'phase6_uart_burst/impl'
    pnr=project/'pnr/phase6_uart_burst.rpt.txt'
    timing=project/'pnr/phase6_uart_burst_tr_content.html'
    text=pnr.read_text(errors='replace') if pnr.exists() else ''
    ttext=timing.read_text(errors='replace') if timing.exists() else ''
    resources={}
    for key in ('Logic','Register','CLS','BSRAM','DSP'):
        match=re.search(rf'^\s*{key}\s*\|\s*([\d.]+)/([\d.]+)',text,re.M)
        if match:resources[key]={'used':float(match[1]),'available':float(match[2])}
    fmax=re.search(r'<td>27\.000\(MHz\)</td>\s*<td[^>]*>([\d.]+)\(MHz\)</td>',ttext)
    setup=re.search(r'Numbers of Setup Violated Endpoints</td>\s*<td[^>]*>(\d+)</td>',ttext)
    hold=re.search(r'Numbers of Hold Violated Endpoints</td>\s*<td[^>]*>(\d+)</td>',ttext)
    logtext=(folder/'route.log').read_text(errors='replace')
    over=re.search(r'number\((\d+)\((\d+) LUTs, (\d+) ALUs, (\d+) ROM16s, (\d+) SSRAMs\)\).*resource limit\((\d+)\)',logtext)
    passed=(invocation.returncode==0 and fmax and float(fmax[1])>=27 and setup and hold and int(setup[1])==int(hold[1])==0 and len(resources)==5)
    result={'status':'passed-route' if passed else 'failed-physical-gate','mode':mode,
            'returncode':invocation.returncode,'seconds':time.perf_counter()-start,'resources':resources,
            'routed_core_fmax_mhz':float(fmax[1]) if fmax else None,
            'setup_violations':int(setup[1]) if setup else None,'hold_violations':int(hold[1]) if hold else None,
            'synthesis_resource_failure':{'logic':int(over[1]),'limit':int(over[6])} if over else None,
            'engine_sha256':PARENT_SHA,'recipe_sha256':sha(recipe),'log_sha256':sha(folder/'route.log'),
            'routed_report_sha256':sha(pnr) if pnr.exists() else None,
            'peak_child_rss_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
    save(folder/'report.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','native','arrivals','route'))
    parser.add_argument('--fixtures',nargs='+',type=Path)
    parser.add_argument('--mode',choices=('baseline','candidate'),default='candidate')
    args=parser.parse_args()
    if args.stage=='prepare':prepare_rtl()
    elif args.stage in ('native','arrivals'):
        if not args.fixtures:raise ValueError('policy fixture paths required')
        (run_native if args.stage=='native' else run_arrivals)(args.fixtures)
    else:
        print(json.dumps(route(args.mode),indent=2))


if __name__=='__main__':main()
