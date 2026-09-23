// Protocol-driven SmallCNN regression of the exact board system hierarchy.
#include "Vv2_system.h"
#include "verilated.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <iterator>
using Bytes=std::vector<uint8_t>;
static Vv2_system dut;
static unsigned sequence_id=0;
static uint64_t cycles=0;
static void require(bool ok,const std::string& text){if(!ok)throw std::runtime_error(text);}
static Bytes file(const std::string& name){std::ifstream f(name,std::ios::binary);require(bool(f),"missing fixture "+name);return Bytes(std::istreambuf_iterator<char>(f),{});}
static uint16_t crc(const Bytes& b,size_t first,size_t end){uint16_t c=0xffff;for(size_t i=first;i<end;i++){c^=uint16_t(b[i])<<8;for(int k=0;k<8;k++)c=c&0x8000?(c<<1)^0x1021:c<<1;}return c;}
static uint32_t u32(const Bytes& b,size_t i){return uint32_t(b.at(i))|uint32_t(b.at(i+1))<<8|uint32_t(b.at(i+2))<<16|uint32_t(b.at(i+3))<<24;}
static int step(){dut.clk=0;dut.eval();int result=dut.tx_valid&&dut.tx_ready?dut.tx_data:-1;dut.clk=1;dut.eval();cycles++;return result;}
static Bytes command(uint8_t cmd,uint32_t addr=0,const Bytes& data={},unsigned length=0){
    if(cmd==3)length=data.size();unsigned seq=(sequence_id++)&255;
    Bytes packet={0xa5,0x5a,2,cmd,uint8_t(seq),uint8_t(addr),uint8_t(addr>>8),uint8_t(addr>>16),uint8_t(length),uint8_t(length>>8)};
    packet.insert(packet.end(),data.begin(),data.end());auto c=crc(packet,2,packet.size());packet.push_back(c&255);packet.push_back(c>>8);
    dut.tx_ready=0;dut.rx_valid=1;for(auto b:packet){dut.rx_data=b;step();}dut.rx_valid=0;dut.tx_ready=1;
    Bytes reply;unsigned total=0;
    for(unsigned n=0;n<100000;n++){int b=step();if(b>=0)reply.push_back(b);if(reply.size()==10)total=12+reply[8]+(unsigned(reply[9])<<8);if(total&&reply.size()==total)break;}
    require(total&&reply.size()==total,"response timeout");require(reply[0]==0xa5&&reply[1]==0x5a&&reply[2]==2&&reply[3]==(cmd|128)&&reply[4]==seq,"response header");
    require(reply[5]==uint8_t(addr)&&reply[6]==uint8_t(addr>>8)&&reply[7]==uint8_t(addr>>16),"response address");
    require(crc(reply,2,reply.size()-2)==(unsigned(reply[reply.size()-2])|(unsigned(reply.back())<<8)),"response CRC");require(reply[10]==0,"command rejected "+std::to_string(reply[10]));
    return Bytes(reply.begin()+11,reply.end()-2);
}
static void write(uint32_t addr,const Bytes& data){for(size_t i=0;i<data.size();i+=64){size_t end=std::min(i+64,data.size());command(3,addr+i,Bytes(data.begin()+i,data.begin()+end));}}
static Bytes read(uint32_t addr,unsigned length){Bytes b;for(unsigned i=0;i<length;i+=64){auto part=command(2,addr+i,{},std::min(64u,length-i));b.insert(b.end(),part.begin(),part.end());}return b;}
int main(int argc,char** argv){try{
    Verilated::commandArgs(argc,argv);require(argc==2,"fixture directory required");std::string dir=argv[1];
    std::ifstream config(dir+"/native.txt");unsigned input_addr,output_addr,used,jobs,macs,input_size,output_size,target_id;
    config>>input_addr>>output_addr>>used>>jobs>>macs>>input_size>>output_size>>target_id;require(bool(config),"fixture config");
    auto image=file(dir+"/board.bin"),inputs=file(dir+"/native-inputs.bin"),expected=file(dir+"/native-outputs.bin");
    require(inputs.size()==jobs*input_size&&expected.size()==jobs*output_size,"fixture sizes");
    struct Check{unsigned addr,size;Bytes data;};std::vector<Check> checks;
    std::vector<std::vector<uint32_t>> stage_counters;unsigned addr,size;std::string name;
    while(config>>addr>>size>>name)checks.push_back({addr,size,file(dir+"/"+name)});
    dut.rst_n=0;dut.rx_valid=0;dut.tx_ready=0;for(int i=0;i<4;i++)step();dut.rst_n=1;for(int i=0;i<4;i++)step();
    auto caps=command(1);require(caps.size()==10&&caps[0]==2&&caps[1]==2&&caps[2]==8&&
        (unsigned(caps[8])|(unsigned(caps[9])<<8))==target_id,"capabilities");
    write(0,image);require(read(0,image.size())==image,"whole SRAM readback");
    for(unsigned layer=0;layer<checks.size();layer++){
        write(0,image);
        write(input_addr,Bytes(inputs.begin(),inputs.begin()+input_size));
        unsigned halt=checks.size()*64;
        write(layer*64+36,{uint8_t(halt),uint8_t(halt>>8),uint8_t(halt>>16),uint8_t(halt>>24)});
        command(4);Bytes status;
        for(unsigned tries=0;tries<10000;tries++){status=command(5);if(!status[0])break;}
        require(!status[0]&&status[1]==0,"stage engine error "+std::to_string(status[1])+" at layer "+std::to_string(layer));
        std::vector<uint32_t> stage;for(unsigned k=0;k<9;k++)stage.push_back(u32(status,2+4*k));
        stage_counters.push_back(stage);
        auto actual=read(checks[layer].addr,checks[layer].size);
        if(actual!=checks[layer].data){
            for(unsigned j=0;j<actual.size();j++)if(actual[j]!=checks[layer].data[j]){
                std::cerr<<"first mismatch at layer "<<layer<<" byte "<<j<<" actual "<<int(int8_t(actual[j]))<<" expected "<<int(int8_t(checks[layer].data[j]))<<std::endl;
                break;
            }
            throw std::runtime_error("intermediate mismatch");
        }
        std::cout<<"exact layer "<<layer<<std::endl;
    }
    write(0,image);
    std::ofstream report(dir+"/native-rtl-results.json");report<<"{\"scope\":\"RTL simulation of the board system, not physical board\",\"jobs\":"<<jobs<<",\"integer_mismatches\":0,\"all_layer_jobs\":1,\"stage_checks\":"<<checks.size()<<",\"stage_counters\":[";
    for(unsigned layer=0;layer<stage_counters.size();layer++){
        if(layer)report<<',';report<<'[';
        for(unsigned k=0;k<9;k++){if(k)report<<',';report<<stage_counters[layer][k];}
        report<<']';
    }
    report<<"],\"counters\":[";
    for(unsigned i=0;i<jobs;i++){
        write(input_addr,Bytes(inputs.begin()+i*input_size,inputs.begin()+(i+1)*input_size));command(4);Bytes status;
        for(unsigned tries=0;tries<10000;tries++){status=command(5);require(status.size()==38,"status size");if(!status[0])break;}
        require(!status[0]&&status[1]==0,"engine error "+std::to_string(status[1])+" at job "+std::to_string(i));
        auto out=read(output_addr,output_size);require(out==Bytes(expected.begin()+i*output_size,expected.begin()+(i+1)*output_size),"integer output mismatch at job "+std::to_string(i));
        require(u32(status,18)==macs,"MAC counter");require(u32(status,2)==u32(status,6)+u32(status,10)+u32(status,14),"cycle reconciliation");
        // The physical reset-corrected baseline took 182,535 cycles and
        // 225,936 SRAM read bytes; both should improve with row reuse.
        require(u32(status,22)<225936&&u32(status,2)<182535,"line-buffer traffic/cycle regression");
        if(i)report<<',';report<<'[';for(unsigned k=0;k<9;k++){if(k)report<<',';report<<u32(status,2+4*k);}report<<']';
        if((i+1)%100==0)std::cout<<i+1<<" / "<<jobs<<" exact SmallCNN RTL jobs"<<std::endl;
    }
    report<<"],\"simulated_clock_cycles\":"<<cycles<<"}\n";dut.final();return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
