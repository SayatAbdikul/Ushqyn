"""Word-parallel in-place PACK and opt-in pointwise physical stride checks."""
import random
import struct

import cocotb
from cocotb.triggers import Timer

from hardware_v2 import Descriptor


def experimental(desc, opcode=None, flags=0):
    blob=bytearray(desc.encode())
    if opcode is not None:
        blob[5]=opcode
    blob[6]=flags&255
    blob[7]=(flags>>8)&255
    return bytes(blob)


class Harness:
    def __init__(self, dut):
        self.d=dut
        self.rng=random.Random(120981)
        self.memory=bytearray(32768)
        self.pending=None
        self.held=None
        self.reads=0
        self.writes=0

    async def step(self):
        d=self.d
        d.clk.value=0
        ready=self.pending is None and self.rng.randrange(4)!=0
        d.mem_ready.value=int(ready)
        d.mem_rvalid.value=0
        if self.pending is not None:
            delay,data=self.pending
            if delay==0:
                d.mem_rvalid.value=1
                d.mem_rdata.value=data
                self.pending=None
            else:
                self.pending=(delay-1,data)
        await Timer(5,units='ns')
        request=(int(d.mem_addr.value),int(d.mem_wr.value),
                 int(d.mem_wdata.value),int(d.mem_wstrb.value)) if int(d.mem_req.value) else None
        if self.held is not None:
            assert request==self.held,'request changed while scratchpad stalled'
        self.held=request if request is not None and not ready else None
        if request is not None and ready:
            address,wr,data,strobe=request
            assert address%8==0 and address+8<=len(self.memory),address
            if wr:
                for k in range(8):
                    if strobe>>k&1:
                        self.memory[address+k]=(data>>(8*k))&255
                        self.writes+=1
            else:
                assert self.pending is None
                self.pending=(self.rng.randrange(1,5),
                              int.from_bytes(self.memory[address:address+8],'little'))
                self.reads+=8
        d.clk.value=1
        await Timer(5,units='ns')

    async def reset(self):
        d=self.d
        d.clk.value=0
        d.rst_n.value=0
        d.start.value=0
        d.abort_run.value=0
        d.clear_counters.value=0
        d.start_pc.value=0
        d.mem_ready.value=0
        d.mem_rvalid.value=0
        d.mem_rdata.value=0
        self.pending=None
        self.held=None
        self.reads=self.writes=0
        await self.step()
        d.rst_n.value=1
        await self.step()
        self.reads=self.writes=0

    async def run(self, max_cycles=200000):
        self.d.start.value=1
        await self.step()
        self.d.start.value=0
        for _ in range(max_cycles):
            await self.step()
            if not int(self.d.busy.value):
                break
        else:
            raise AssertionError('engine timeout')
        assert int(self.d.read_bytes.value)==self.reads
        assert int(self.d.write_bytes.value)==self.writes
        assert int(self.d.elapsed.value)==(int(self.d.compute_cycles.value)+
                                           int(self.d.wait_cycles.value)+
                                           int(self.d.control_cycles.value))
        return int(self.d.error_code.value)


@cocotb.test()
async def pack_in_place_and_reject_invalid_descriptors(dut):
    h=Harness(dut)
    for height,width,channels in ((1,1,1),(1,5,3),(1,7,4),(2,4,3),
                                  (3,3,4),(3,5,2),(1,17,3),(8,8,4)):
        await h.reset()
        h.memory[:]=bytes([0xa5])*len(h.memory)
        plane=height*width
        physical=(plane+7)&~7
        base=1024
        source=bytes(h.rng.randrange(256) for _ in range(plane*channels))
        h.memory[base:base+len(source)]=source
        desc=Descriptor(3,input=base,output=base,count=plane,outputs=physical,
                        next_pc=64,input_h=height,input_w=width,
                        input_c=channels,output_c=channels)
        h.memory[:128]=experimental(desc,opcode=9)+Descriptor(0).encode()
        assert await h.run()==0
        expected=b''.join(source[c*plane:(c+1)*plane]+bytes(physical-plane)
                          for c in range(channels))
        assert h.memory[base:base+physical*channels]==expected
        assert h.writes==physical*channels

    invalid=[]
    base=1024
    common=dict(input=base,output=base,count=9,outputs=16,next_pc=64,
                input_h=3,input_w=3,input_c=2,output_c=2)
    invalid.append((Descriptor(3,**{**common,'outputs':9}),1))
    invalid.append((Descriptor(3,**{**common,'output':base+8}),1))
    invalid.append((Descriptor(3,**{**common,'row_stride':8}),1))
    invalid.append((Descriptor(3,**{**common,'input_h':4}),1))
    invalid.append((Descriptor(3,**{**common,'input_c':2048,'output_c':2048}),1))
    invalid.append((Descriptor(3,**common),2))
    for i,(desc,error) in enumerate(invalid):
        await h.reset()
        h.memory[:]=bytes(len(h.memory))
        h.memory[:128]=experimental(desc,opcode=9,flags=(1 if i==len(invalid)-1 else 0))+Descriptor(0).encode()
        before=bytes(h.memory[base:base+64])
        assert await h.run()==error
        assert h.memory[base:base+64]==before
        assert h.writes==0


@cocotb.test()
async def flagged_pointwise_matches_dense_legacy(dut):
    h=Harness(dut)
    for height,width,channels,out_channels in ((1,5,3,2),(3,3,4,3),(3,5,2,2),
                                               (2,4,3,2)):
        plane=height*width
        physical=(plane+7)&~7
        base=1024
        output=4096
        weight=8192
        params=24000
        stride=(channels+7)&~7
        data=[h.rng.randrange(-50,51) for _ in range(plane*channels)]
        weights=[[h.rng.randrange(-3,4) for _ in range(channels)]
                 for _ in range(out_channels)]
        expected=[]
        for oc in range(out_channels):
            for pos in range(plane):
                acc=sum(data[c*plane+pos]*weights[oc][c] for c in range(channels))
                expected.append(max(-128,min(127,acc)))
        results=[]
        for padded in (False,True):
            await h.reset()
            h.memory[:]=bytes(len(h.memory))
            h.memory[base:base+len(data)]=bytes(v&255 for v in data)
            for oc in range(out_channels):
                h.memory[weight+oc*stride:weight+oc*stride+channels]=bytes(v&255 for v in weights[oc])
                h.memory[params+oc*16:params+(oc+1)*16]=struct.pack(
                    '<iiBbbbbb',0,1<<30,30,0,0,-128,127,0)+b'\0\0'
            conv=Descriptor(4,input=base,output=output,weight=weight,params=params,
                            count=channels,outputs=out_channels*plane,row_stride=stride,
                            next_pc=128 if padded else 64,
                            input_h=height,input_w=width,input_c=channels,output_c=out_channels)
            if padded:
                pack=Descriptor(3,input=base,output=base,count=plane,outputs=physical,
                                next_pc=64,input_h=height,input_w=width,
                                input_c=channels,output_c=channels)
                h.memory[:192]=(experimental(pack,opcode=9)+
                                experimental(conv,flags=1)+Descriptor(0).encode())
            else:
                h.memory[:128]=conv.encode()+Descriptor(0).encode()
            assert await h.run()==0
            actual=[v if v<128 else v-256 for v in h.memory[output:output+len(expected)]]
            assert actual==expected
            results.append(actual)
        assert results[0]==results[1]

    # The stride flag is reserved for the accelerated, unpadded 1x1 path.
    await h.reset()
    h.memory[:]=bytes(len(h.memory))
    unsupported=Descriptor(4,input=1024,output=4096,weight=8192,params=24000,
                           count=3,outputs=5,row_stride=8,next_pc=64,
                           stride_w=2,input_h=1,input_w=9,input_c=3,output_c=1)
    h.memory[:128]=experimental(unsupported,flags=1)+Descriptor(0).encode()
    assert await h.run()==2
    assert h.writes==0
