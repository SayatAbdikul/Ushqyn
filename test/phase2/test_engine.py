import random,struct
import numpy as np
import cocotb
from cocotb.triggers import Timer
from hardware_v2 import Descriptor

@cocotb.test()
async def randomized_memory_latency_and_abort(d):
    rng=random.Random(20260910);memory=bytearray(32768)
    d.clk.value=0;d.rst_n.value=0;d.start.value=0;d.abort_run.value=0;d.clear_counters.value=0;d.start_pc.value=0;d.mem_ready.value=0;d.mem_rvalid.value=0;d.mem_rdata.value=0
    pending=None;held=None;read_count=0;write_count=0
    async def step():
        nonlocal pending,held,read_count,write_count
        d.clk.value=0
        ready=pending is None and rng.randrange(4)!=0
        d.mem_ready.value=ready;d.mem_rvalid.value=0
        if pending is not None:
            delay,data=pending
            if delay==0:d.mem_rvalid.value=1;d.mem_rdata.value=data;pending=None
            else:pending=(delay-1,data)
        await Timer(5,units='ns')
        request=(int(d.mem_addr.value),int(d.mem_wr.value),int(d.mem_wdata.value),int(d.mem_wstrb.value)) if int(d.mem_req.value) else None
        if held is not None and not int(d.abort_run.value):assert request==held,'request changed before acceptance'
        held=request if request is not None and not ready else None
        if request is not None and ready:
            address,wr,data,strobe=request;assert address%8==0 and address+8<=len(memory)
            if wr:
                for k in range(8):
                    if strobe>>k&1:memory[address+k]=(data>>(8*k))&255;write_count+=1
            else:
                assert pending is None;pending=(rng.randrange(1,6),int.from_bytes(memory[address:address+8],'little'));read_count+=8
        d.clk.value=1;await Timer(5,units='ns')
    await step();d.rst_n.value=1;await step()
    for trial in range(20):
        n=rng.choice([1,7,8,9,63,64,65,127,784]);rows=3;stride=(n+7)&~7;zx=rng.choice([-128,-117,0,127]);zy=rng.randrange(-128,128)
        x=np.array([rng.randrange(-128,128) for _ in range(n)],np.int8);w=np.array([[rng.randrange(-127,128) for _ in range(n)] for _ in range(rows)],np.int8)
        bias=[rng.randrange(-10000,10000) for _ in range(rows)];m=[rng.randrange(1,2**31) for _ in range(rows)];s=[rng.randrange(20,45) for _ in range(rows)]
        desc=Descriptor(1,input=512,output=2000,weight=4096,params=24000,count=n,outputs=rows,row_stride=stride,next_pc=64)
        memory[:128]=desc.encode()+Descriptor(0).encode();memory[512:512+n]=x.tobytes()
        for c in range(rows):
            memory[4096+c*stride:4096+c*stride+n]=w[c].tobytes()
            cb=bias[c]-zx*int(w[c].astype(np.int64).sum());memory[24000+c*16:24000+(c+1)*16]=struct.pack('<iiBbbbbb',cb,m[c],s[c],zy,zx,-128,127,0)+b'\0\0'
        read_count=write_count=0;d.start.value=1;await step();d.start.value=0
        for cycles in range(100000):
            await step()
            if not int(d.busy.value):break
        else:raise AssertionError('engine timeout')
        assert int(d.error_code.value)==0
        expected=[]
        for c in range(rows):
            a=bias[c]+sum((int(x[k])-zx)*int(w[c,k]) for k in range(n));v=a*m[c];q,r=divmod(abs(v),1<<s[c]);q+=2*r>=(1<<s[c]);expected.append(max(-128,min(127,(-q if v<0 else q)+zy)))
        assert memory[2000:2003]==np.array(expected,np.int8).tobytes()
        assert int(d.useful_macs.value)==n*rows
        assert int(d.read_bytes.value)==read_count and int(d.write_bytes.value)==write_count==rows
        assert int(d.elapsed.value)==int(d.compute_cycles.value)+int(d.wait_cycles.value)+int(d.control_cycles.value)
    # Abort during an outstanding read. Drain the response, then RUN fresh.
    d.start.value=1;await step();d.start.value=0
    for _ in range(4):await step()
    d.abort_run.value=1;await step();d.abort_run.value=0;held=None
    for _ in range(12):await step()
    assert not int(d.busy.value)
    d.clear_counters.value=1;await step();d.clear_counters.value=0
    assert int(d.elapsed.value)==0
