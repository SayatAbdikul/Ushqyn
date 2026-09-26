"""Adversarial cache capacity, tag, lifetime and per-channel parameter tests."""
import random
import struct
import numpy as np
import cocotb
from cocotb.triggers import Timer
from hardware_v2 import Descriptor


@cocotb.test()
async def channel_cache_tags_tails_and_invalidation(d):
    rng = random.Random(6061)
    memory = bytearray(32768)
    pending = held = None
    d.clk.value=0; d.rst_n.value=0; d.start.value=0; d.start_pc.value=0
    d.abort_run.value=0; d.clear_counters.value=0
    d.mem_ready.value=0; d.mem_rvalid.value=0; d.mem_rdata.value=0

    async def step():
        nonlocal pending, held
        d.clk.value=0
        ready=pending is None and rng.randrange(4)!=0
        d.mem_ready.value=ready; d.mem_rvalid.value=0
        if pending is not None:
            delay,data=pending
            if delay==0:
                d.mem_rvalid.value=1; d.mem_rdata.value=data; pending=None
            else: pending=delay-1,data
        await Timer(5,units='ns')
        req=(int(d.mem_addr.value),int(d.mem_wr.value),int(d.mem_wdata.value),int(d.mem_wstrb.value)) if int(d.mem_req.value) else None
        if held is not None and not int(d.abort_run.value) and int(d.rst_n.value):
            assert req==held, 'request changed during backpressure'
        held=req if req is not None and not ready else None
        if req is not None and ready:
            a,w,data,mask=req
            assert a%8==0 and a+8<=len(memory)
            if w:
                for k in range(8):
                    if mask>>k&1: memory[a+k]=(data>>(8*k))&255
            else:
                assert pending is None
                pending=rng.randrange(0,5),int.from_bytes(memory[a:a+8],'little')
        d.clk.value=1
        await Timer(5,units='ns')

    await step(); d.rst_n.value=1; await step()
    # Aligned/unaligned channel planes, each capacity boundary, odd output
    # channels and fallback when the number of channels exceeds cache size.
    for trial,(ic,ih,iw,oc) in enumerate([(1,1,17,3),(31,3,5,3),(32,1,9,4),
        (63,5,5,3),(64,5,5,4),(65,1,9,3),(127,1,9,3),(128,3,3,3),
        (129,1,9,2),(255,1,9,2),(256,1,9,2),(257,1,9,2)]):
        n=ic; stride=(n+7)&~7; plane=ih*iw; zx=[-128,-117,0,127][trial%4]
        x=np.array([rng.randrange(-128,128) for _ in range(ic*plane)],np.int8).reshape(ic,plane)
        w=np.array([rng.randrange(-127,128) for _ in range(oc*ic)],np.int8).reshape(oc,ic)
        desc=Descriptor(4,input=512,output=12000,weight=16000,params=24000,count=n,outputs=oc*plane,
            row_stride=stride,next_pc=64,kernel_h=1,kernel_w=1,input_h=ih,input_w=iw,input_c=ic,output_c=oc)
        memory[:128]=desc.encode()+Descriptor(0).encode()
        expected=[]
        biases=[]; params=[]
        for c in range(oc):
            bias=rng.randrange(-5000,5001); mult=rng.randrange(1<<25,1<<30); shift=34+c%3; zy=c-7
            biases.append(bias); params.append((mult,shift,zy))
            cb=bias-zx*int(w[c].astype(np.int64).sum())
            memory[16000+c*stride:16000+c*stride+ic]=w[c].tobytes()
            memory[24000+c*16:24000+(c+1)*16]=struct.pack('<iiBbbbbb',cb,mult,shift,zy,zx,-128,127,0)+b'\0\0'
        async def execute(value):
            memory[512:512+value.size]=value.tobytes()
            expected=[]
            for c in range(oc):
                mult,shift,zy=params[c]
                for p in range(plane):
                    a=biases[c]+sum((int(value[k,p])-zx)*int(w[c,k]) for k in range(ic))
                    v=a*mult; q,r=divmod(abs(v),1<<shift); q+=2*r>=1<<shift
                    expected.append(max(-128,min(127,(-q if v<0 else q)+zy)))
            d.start.value=1; await step(); d.start.value=0
            for _ in range(500000):
                await step()
                if not int(d.busy.value): break
            else: raise AssertionError('cache test timeout')
            assert int(d.error_code.value)==0
            actual=memory[12000:12000+len(expected)]
            assert actual==np.array(expected,np.int8).tobytes(), (trial,ic,ih,iw,oc)
            assert int(d.useful_macs.value)==ic*oc*plane
            assert int(d.elapsed.value)==sum(int(getattr(d,k).value) for k in ['compute_cycles','wait_cycles','control_cycles'])
        await execute(x)
        # Reuse identical SRAM addresses with new tensor values across RUNs.
        await execute(np.bitwise_not(x))
        if trial in (4,8,11):
            d.start.value=1; await step(); d.start.value=0
            for _ in range(100): await step()
            d.abort_run.value=1; await step(); d.abort_run.value=0; held=None
            for _ in range(16): await step()
            assert not int(d.busy.value)
            d.clear_counters.value=1; await step(); d.clear_counters.value=0
            await execute(x)
        if trial==6:
            d.rst_n.value=0; await step(); held=None; pending=None
            d.rst_n.value=1; await step(); await execute(x)
    # Preserve the original eight-channel overflow boundary: individual
    # products may temporarily cross INT32 while their eight-term sum fits.
    desc=Descriptor(4,input=512,output=12000,weight=16000,params=24000,count=8,outputs=8,
        row_stride=8,next_pc=64,input_h=1,input_w=8,input_c=8,output_c=1)
    memory[:128]=desc.encode()+Descriptor(0).encode()
    memory[512:576]=bytes([1])*64
    memory[24000:24016]=struct.pack('<iiBbbbbb',2147483600,1,31,0,0,-128,127,0)+b'\0\0'
    for weights,expected_error in (([127]*4+[-127]*4,0),([127]*8,5)):
        memory[16000:16008]=np.array(weights,np.int8).tobytes()
        d.start.value=1;await step();d.start.value=0
        for _ in range(10000):
            await step()
            if not int(d.busy.value):break
        else:raise AssertionError('overflow-boundary test timeout')
        assert int(d.error_code.value)==expected_error
        if not expected_error:assert memory[12000:12008]==bytes([1])*8
