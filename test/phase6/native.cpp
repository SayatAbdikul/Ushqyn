// Full command-sequencer/engine/SRAM/DMA simulation with a stalled RAM port.
// External RAM is a test model, not the electrical SDRAM controller.
#include "Vv2_tiled_host_bridge.h"
#include "Vv2_tiled_host_bridge___024root.h"
#include "verilated.h"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using Bytes=std::vector<uint8_t>;
static Vv2_tiled_host_bridge d;
static Bytes memory(8*1024*1024);
static std::mt19937 rng;
static bool stalls=false,pending=false,held=false;
static unsigned delay=0,sequence_id=0;
static uint64_t pending_data=0,cycles=0;
struct Request { uint32_t addr; uint64_t data; unsigned wr,mask; };
static Request hold;
static void require(bool b,const std::string& s){if(!b)throw std::runtime_error(s);}
static Bytes file(const std::string& s){std::ifstream f(s,std::ios::binary);require(bool(f),"missing "+s);return Bytes(std::istreambuf_iterator<char>(f),{});}
static uint32_t u32(const Bytes& b,unsigned i){return uint32_t(b.at(i))|uint32_t(b.at(i+1))<<8|uint32_t(b.at(i+2))<<16|uint32_t(b.at(i+3))<<24;}
static uint16_t crc(const Bytes& b,unsigned first,unsigned end){uint16_t c=65535;for(unsigned i=first;i<end;i++){c^=uint16_t(b[i])<<8;for(int k=0;k<8;k++)c=c&0x8000?(c<<1)^0x1021:c<<1;}return c;}
static int step(){
    d.clk=0; bool ready=!pending&&(!stalls||rng()%5!=0);
    d.ext_ready=ready;d.ext_rvalid=0;
    if(pending){if(delay==0){d.ext_rvalid=1;d.ext_rdata=pending_data;pending=false;}else delay--;}
    d.eval();int out=d.tx_valid&&d.tx_ready?d.tx_data:-1;
    Request r{d.ext_addr,d.ext_wdata,d.ext_wr,d.ext_wstrb};
    if(held)require(d.ext_req&&r.addr==hold.addr&&r.data==hold.data&&r.wr==hold.wr&&r.mask==hold.mask,"unstable external request");
    held=d.ext_req&&!ready;if(held)hold=r;
    if(d.ext_req&&ready){
        require(r.addr%8==0&&r.addr+8<=memory.size(),"external range/alignment");
        if(r.wr){for(unsigned k=0;k<8;k++)if(r.mask>>k&1)memory[r.addr+k]=r.data>>(k*8);}
        else{require(!pending,"overlapping RAM reads");pending=true;delay=stalls?rng()%5:0;pending_data=0;for(unsigned k=0;k<8;k++)pending_data|=uint64_t(memory[r.addr+k])<<(k*8);}
    }
    d.clk=1;d.eval();cycles++;return out;
}
static Bytes command(unsigned cmd,uint32_t addr=0,const Bytes& data={},unsigned length=0){
    if(cmd==3)length=data.size();unsigned seq=sequence_id++&255;
    Bytes b={0xa5,0x5a,2,uint8_t(cmd),uint8_t(seq),uint8_t(addr),uint8_t(addr>>8),uint8_t(addr>>16),uint8_t(length),uint8_t(length>>8)};
    b.insert(b.end(),data.begin(),data.end());auto c=crc(b,2,b.size());b.push_back(c);b.push_back(c>>8);
    d.tx_ready=0;d.rx_valid=1;for(auto byte:b){d.rx_data=byte;step();}d.rx_valid=0;d.tx_ready=1;
    Bytes reply;unsigned total=0;
    for(unsigned n=0;n<100000;n++){int byte=step();if(byte>=0)reply.push_back(byte);if(reply.size()==10)total=12+reply[8]+(unsigned(reply[9])<<8);if(total&&reply.size()==total)break;}
    require(total&&reply.size()==total,"protocol timeout");
    require(reply[0]==0xa5&&reply[1]==0x5a&&reply[2]==2&&reply[3]==(cmd|128)&&reply[4]==seq,"reply header");
    require(crc(reply,2,reply.size()-2)==(unsigned(reply[reply.size()-2])|(unsigned(reply.back())<<8)),"reply CRC");
    require(reply[10]==0,"command rejected "+std::to_string(reply[10]));
    return Bytes(reply.begin()+11,reply.end()-2);
}
int main(int argc,char** argv){try{
    Verilated::commandArgs(argc,argv);require(argc==4,"fixture directory, stall seed (0=fixed RAM), report path required");
    std::string dir=argv[1];unsigned seed=std::stoul(argv[2]);stalls=seed!=0;rng.seed(seed);
    auto payload=file(dir+"/payload.bin"),input=file(dir+"/input.bin"),commands=file(dir+"/commands.bin");
    require(payload.size()<=memory.size(),"payload exceeds external RAM");std::copy(payload.begin(),payload.end(),memory.begin());std::copy(input.begin(),input.end(),memory.begin());
    d.rst_n=0;d.rx_valid=0;d.tx_ready=0;d.memory_initialized=1;d.memory_port_busy=0;
    for(int k=0;k<4;k++)step();d.rst_n=1;for(int k=0;k<4;k++)step();
    for(unsigned i=0;i<commands.size();i+=64)command(3,0x500000+i,Bytes(commands.begin()+i,commands.begin()+std::min(unsigned(commands.size()),i+64)));
    command(3,0x410000,{1});
    unsigned ticks=0;while(d.rootp->v2_tiled_host_bridge__DOT__seq_busy&&ticks++<200000000)step();
    require(ticks<200000000,"sequence timeout");auto regs=command(2,0x410000,{},32);
    require(regs.size()==32&&regs[0]==0&&regs[1]==0,"sequence failed, error "+std::to_string(regs.at(1)));
    std::ifstream checks(dir+"/checks.txt");require(bool(checks),"missing checks");unsigned address,nchecks=0;std::string name;
    while(checks>>address>>name){auto expected=file(dir+"/"+name);require(address+expected.size()<=memory.size(),"check range");
        for(unsigned i=0;i<expected.size();i++)require(memory[address+i]==expected[i],"output mismatch "+name+" at byte "+std::to_string(i));nchecks++;}
    require(nchecks>0,"empty checks");
    std::ofstream report(argv[3]);report<<"{\"status\":\"passed\",\"physical_board\":false,\"stall_seed\":"<<seed<<",\"tensor_checks\":"<<nchecks<<",\"elapsed_cycles\":"<<u32(regs,8)<<",\"engine_cycles\":"<<u32(regs,12)<<",\"dma_cycles\":"<<u32(regs,16)<<",\"overlap_cycles\":"<<u32(regs,20)<<",\"total_simulated_cycles\":"<<cycles<<"}\n";
    std::cout<<argv[1]<<" seed "<<seed<<" passed: "<<u32(regs,8)<<" cycles, "<<nchecks<<" tensors"<<std::endl;
    d.final();return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
