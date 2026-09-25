"""Buffered write ordering, coherence, tails, reset and refresh starvation."""
import random
import cocotb
from cocotb.triggers import Timer


@cocotb.test()
async def burst_coherence_masks_boundaries_refresh_reset(d):
    rng=random.Random(250926)
    expected=bytearray(8388608)
    d.clk.value=0;d.clk_sdram.value=0;d.rst_n.value=0
    d.ext_req.value=0;d.ext_wr.value=0;d.ext_addr.value=0
    d.ext_wdata.value=0;d.ext_wstrb.value=0

    async def step():
        d.clk.value=0;await Timer(5,units='ns')
        ready=int(d.ext_ready.value);valid=int(d.ext_rvalid.value);data=int(d.ext_rdata.value)
        d.clk.value=1;await Timer(5,units='ns')
        return ready,valid,data

    async def transaction(address,write,data=0,mask=255):
        d.ext_req.value=1;d.ext_wr.value=write;d.ext_addr.value=address
        d.ext_wdata.value=data;d.ext_wstrb.value=mask
        for _ in range(1000):
            ready,_,_=await step()
            if ready: break
        else: raise AssertionError('request timeout')
        d.ext_req.value=0
        if write:
            for k in range(8):
                if mask&(1<<k):expected[address+k]=(data>>(8*k))&255
        else:
            for _ in range(1000):
                _,valid,result=await step()
                if valid:break
            else:raise AssertionError('read timeout')
            assert result==int.from_bytes(expected[address:address+8],'little'),hex(address)

    await step();d.rst_n.value=1
    for _ in range(4):await step()
    addresses=[0,8,56,64,1016,1024,0x1ffff8,0x200000,0x7ffff8]
    for address in addresses:
        await transaction(address,True,rng.getrandbits(64))
    for _ in range(200):
        address=rng.choice(addresses)
        if rng.randrange(3):await transaction(address,True,rng.getrandbits(64),rng.randrange(1,256))
        else:await transaction(address,False)
    for address in addresses:await transaction(address,False)
    # Same short line is written continuously, preventing idle/full-line
    # flush. Refresh must still force a drain and progress.
    before=int(d.controller.refresh_count.value)
    for _ in range(1800):await transaction(8,True,rng.getrandbits(64))
    assert int(d.controller.refresh_count.value)>before+3
    assert not int(d.refresh_deadline_missed.value)
    await transaction(8,False)
    # RESET discards uncommitted buffered data and invalidates cached reads.
    # Previously committed memory remains intact in this contract model.
    for _ in range(100):await step()
    d.rst_n.value=0;await step();d.rst_n.value=1
    for _ in range(5):await step()
    for address in addresses:await transaction(address,False)
