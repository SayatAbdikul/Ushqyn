"""Overflow and abort at a speculative PW read, including SRAM backpressure."""
import struct

import cocotb
from cocotb.triggers import Timer

from hardware_v2 import Descriptor


@cocotb.test()
async def overflow_abort_and_delayed_prefetch(d):
    memory = bytearray(32768)
    desc = Descriptor(4, input=512, output=12000, weight=16000, params=24000,
        count=16, outputs=8, row_stride=16, next_pc=64, input_h=1,
        input_w=8, input_c=16, output_c=1)
    memory[:128] = desc.encode()+Descriptor(0).encode()
    pending = None
    stale_after_failure = 0
    d.clk.value=0;d.rst_n.value=0;d.start.value=0;d.start_pc.value=0
    d.abort_run.value=0;d.clear_counters.value=0
    d.mem_ready.value=0;d.mem_rvalid.value=0;d.mem_rdata.value=0

    async def step(action=None):
        nonlocal pending, stale_after_failure
        d.clk.value=0;d.mem_rvalid.value=0
        had_pending=pending is not None
        if pending is not None:
            delay,data=pending
            if delay==0:
                d.mem_rvalid.value=1;d.mem_rdata.value=data;pending=None
                if int(d.busy.value):stale_after_failure+=1
            else:pending=(delay-1,data)
        d.mem_ready.value=0 if had_pending else 1
        await Timer(1,units='ns')
        # PW_MAC is enum 35. At col 7, the first 8-channel group can fail
        # while a read of the next input channel is already being offered.
        target=(int(d.state.value)==35 and int(d.col.value)==7 and
            int(d.pw_overflow.value)==1 and int(d.mem_req.value)==1)
        if action in ('overflow-stalled','abort-stalled') and target:
            d.mem_ready.value=0
        if action=='abort-stalled' and target:
            d.abort_run.value=1
        await Timer(4,units='ns')
        req=int(d.mem_req.value)
        ready=int(d.mem_ready.value)
        if target:
            assert req and not int(d.mem_wr.value)
            assert int(d.mem_addr.value)==576
        if req and ready:
            addr=int(d.mem_addr.value)
            assert addr%8==0 and addr+8<=len(memory)
            if int(d.mem_wr.value):
                data=int(d.mem_wdata.value);mask=int(d.mem_wstrb.value)
                for lane in range(8):
                    if mask>>lane&1:memory[addr+lane]=(data>>(8*lane))&255
            else:
                assert pending is None
                delay=6 if target and action=='overflow-accepted' else 1
                pending=(delay,int.from_bytes(memory[addr:addr+8],'little'))
        d.clk.value=1
        await Timer(5,units='ns')
        return target, bool(req and ready)

    await step();d.rst_n.value=1;await step()

    def overflow_image():
        memory[512:640]=bytes([1])*128
        memory[16000:16016]=bytes([127])*8+bytes(8)
        memory[24000:24016]=struct.pack('<iiBbbbbb',2147483600,1<<30,31,
                                           0,0,-128,127,0)+b'\0\0'

    def valid_image():
        # Distinct pixels expose a stale input word on a subsequent run.
        memory[512:640]=bytes(range(1,9))*16
        memory[16000:16016]=bytes([1])*16
        memory[24000:24016]=struct.pack('<iiBbbbbb',0,1<<30,31,
                                           0,0,-128,127,0)+b'\0\0'
        memory[12000:12008]=bytes([0xA5])*8

    async def launch():
        d.start.value=1;await step();d.start.value=0

    async def recover_and_check():
        valid_image()
        await launch()
        for _ in range(40000):
            await step()
            if not int(d.busy.value):break
        else:raise AssertionError('valid restart timed out')
        assert int(d.error_code.value)==0
        assert memory[12000:12008]==bytes(range(8,65,8))

    for action,expected_error in (('overflow-stalled',5),
                                  ('overflow-accepted',5),
                                  ('abort-stalled',6)):
        assert pending is None
        overflow_image()
        await launch()
        saw_target=saw_accept=False
        for _ in range(40000):
            hit,accepted=await step(action)
            if hit:
                saw_target=True;saw_accept=accepted
                break
        else:raise AssertionError('speculative overflow read not reached')
        assert not int(d.busy.value)
        assert int(d.error_code.value)==expected_error
        assert saw_target and saw_accept==(action=='overflow-accepted')
        if action=='overflow-accepted':assert pending is not None
        d.abort_run.value=0
        await recover_and_check()
    assert stale_after_failure>=1
