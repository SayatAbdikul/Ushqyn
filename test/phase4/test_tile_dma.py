"""Abstract-port DMA test; no SDRAM-controller or power claims."""
import random

import cocotb
from cocotb.triggers import Timer


@cocotb.test()
async def tile_transfer_backpressure_bounds_and_abort(d):
    rng=random.Random(4024)
    sram=bytearray(32768)
    external=bytearray(8*1024*1024)
    pending={'sram':None,'ext':None}
    held={'sram':None,'ext':None}
    d.clk.value=0;d.rst_n.value=0;d.start.value=0;d.abort_run.value=0
    d.to_sram.value=0;d.sram_base.value=0;d.ext_base.value=0;d.length_bytes.value=0
    for name in ('sram','ext'):
        getattr(d,f'{name}_ready').value=0
        getattr(d,f'{name}_rvalid').value=0
        getattr(d,f'{name}_rdata').value=0

    async def step():
        d.clk.value=0
        ready={}
        for name in ('sram','ext'):
            ready[name]=pending[name] is None and rng.randrange(4)!=0
            getattr(d,f'{name}_ready').value=ready[name]
            getattr(d,f'{name}_rvalid').value=0
            if pending[name] is not None:
                delay,data=pending[name]
                if delay==0:
                    getattr(d,f'{name}_rvalid').value=1
                    getattr(d,f'{name}_rdata').value=data
                    pending[name]=None
                else:pending[name]=(delay-1,data)
        await Timer(5,units='ns')
        for name,array in (('sram',sram),('ext',external)):
            req=(int(getattr(d,f'{name}_addr').value),
                 int(getattr(d,f'{name}_wr').value),
                 int(getattr(d,f'{name}_wdata').value),
                 int(getattr(d,f'{name}_wstrb').value)) if int(getattr(d,f'{name}_req').value) else None
            if held[name] is not None and not int(d.abort_run.value):
                assert req==held[name],f'{name} request changed under backpressure'
            held[name]=req if req is not None and not ready[name] else None
            if req is not None and ready[name]:
                address,wr,data,mask=req
                assert address%8==0 and address+8<=len(array)
                if wr:
                    for lane in range(8):
                        if (mask>>lane)&1:array[address+lane]=(data>>(lane*8))&255
                else:
                    assert pending[name] is None
                    pending[name]=(rng.randrange(1,5),int.from_bytes(array[address:address+8],'little'))
        d.clk.value=1
        await Timer(5,units='ns')

    async def transfer(to_sram,sram_base,ext_base,length):
        d.to_sram.value=to_sram;d.sram_base.value=sram_base
        d.ext_base.value=ext_base;d.length_bytes.value=length
        d.start.value=1;await step();d.start.value=0
        for _ in range(20000):
            await step()
            if not int(d.busy.value):break
        else:raise AssertionError('DMA timeout')
        assert int(d.error_code.value)==0
        assert int(d.bytes_copied.value)==length
        assert int(d.physical_read_bytes.value)==8*((length+7)//8)

    await step();d.rst_n.value=1;await step()
    for trial in range(120):
        length=rng.choice((1,2,7,8,9,15,16,17,31,32,63,64,65,127,128,255))
        sram_base=rng.randrange(0,(32768-length)//8+1)*8
        ext_base=rng.choice((0,0xfff8,0x1fff8,0x3ffff8,0x7ffff8,
                             rng.randrange(0,(len(external)-length)//8+1)*8))
        if ext_base+length>len(external):ext_base=len(external)-((length+7)//8)*8
        payload=bytes(rng.randrange(256) for _ in range(length))
        if trial&1:
            external[ext_base:ext_base+length]=payload
            await transfer(1,sram_base,ext_base,length)
            assert sram[sram_base:sram_base+length]==payload
        else:
            sram[sram_base:sram_base+length]=payload
            await transfer(0,sram_base,ext_base,length)
            assert external[ext_base:ext_base+length]==payload
    # A transfer ending on the last byte verifies the full 24-bit address path.
    external[-16:]=bytes(range(16))
    await transfer(1,0,len(external)-16,16)
    assert sram[:16]==bytes(range(16))
    for sram_base,ext_base,length in ((1,0,8),(0,1,8),(32768,0,1),
                                      (0,len(external),1),(0,0,0),(0,0,32769)):
        d.sram_base.value=sram_base;d.ext_base.value=ext_base
        d.length_bytes.value=length;d.start.value=1
        await step();d.start.value=0
        assert not int(d.busy.value) and int(d.error_code.value)==1
    # Abort with an outstanding read; wait until its response is drained.
    d.to_sram.value=1;d.sram_base.value=0;d.ext_base.value=0
    d.length_bytes.value=64;d.start.value=1;await step();d.start.value=0
    for _ in range(20):
        await step()
        if pending['ext'] is not None:break
    else:raise AssertionError('no outstanding external read')
    pending['ext']=(8,pending['ext'][1])
    d.abort_run.value=1;await step();await step();d.abort_run.value=0
    assert int(d.busy.value), 'repeated abort must not bypass read-response drain'
    for _ in range(20):
        await step()
        if not int(d.busy.value):break
    assert not int(d.busy.value) and int(d.error_code.value)==2
    await transfer(1,0,len(external)-16,16)
    assert sram[:16]==bytes(range(16))
