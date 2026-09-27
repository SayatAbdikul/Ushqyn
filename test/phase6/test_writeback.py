"""Output handshake regression: tails, in-place activation and cancelled writes."""
import struct

import cocotb
from cocotb.triggers import Timer
from hardware_v2 import Descriptor


@cocotb.test()
async def held_output_and_cancellation(d):
    memory = bytearray(32768)
    pending = held = None
    write_wait = writes = 0
    d.clk.value = 0; d.rst_n.value = 0; d.start.value = 0; d.start_pc.value = 0
    d.abort_run.value = 0; d.clear_counters.value = 0
    d.mem_ready.value = 0; d.mem_rvalid.value = 0; d.mem_rdata.value = 0

    async def step():
        nonlocal pending, held, write_wait, writes
        d.clk.value = 0
        d.mem_rvalid.value = pending is not None
        if pending is not None:
            d.mem_rdata.value = pending; pending = None
        d.mem_ready.value = 0
        await Timer(1, units='ns')
        request = ((int(d.mem_addr.value), int(d.mem_wr.value),
                    int(d.mem_wdata.value), int(d.mem_wstrb.value))
                   if int(d.mem_req.value) else None)
        cancelled = int(d.abort_run.value) or int(d.clear_counters.value) or not int(d.rst_n.value)
        if held is not None and not cancelled:
            assert request == held, 'output changed before the SRAM accepted it'
        ready = not cancelled
        if request and request[1] and ready:
            ready = write_wait >= 5
            write_wait = 0 if ready else write_wait + 1
        d.mem_ready.value = ready
        await Timer(4, units='ns')
        held = request if request is not None and not ready and not cancelled else None
        if request and ready:
            address, wr, data, mask = request
            assert address % 8 == 0 and address + 8 <= len(memory)
            if wr:
                assert mask and mask & (mask-1) == 0
                writes += 1
                for lane in range(8):
                    if mask >> lane & 1:
                        memory[address+lane] = data >> (8*lane) & 255
            else:
                pending = int.from_bytes(memory[address:address+8], 'little')
        d.clk.value = 1
        await Timer(5, units='ns')

    await step(); d.rst_n.value = 1; await step()
    for op in (2, 3, 8):
        for length in (1, 7, 8, 9, 17):
            for inplace in (False, True):
                destination = 512 if inplace else 2051  # Exercise unaligned writes.
                memory[:] = bytes(len(memory))
                desc = Descriptor(op, input=512, output=destination, params=24000,
                                  count=length, outputs=length, next_pc=64)
                memory[:128] = desc.encode() + Descriptor(0).encode()
                lower, upper, zx, zy = (-31, 127, -31, 5) if op == 2 else (-40, 60, -7, -9)
                memory[24000:24016] = struct.pack('<iiBbbbbb', 0, 3, 2, zy, zx,
                                                  lower, upper, 0) + b'\0\0'
                values = [(-128, -41, -31, -1, 0, 59, 60, 127)[i % 8] for i in range(length)]
                memory[512:512+length] = bytes(v & 255 for v in values)
                expected = []
                for value in values:
                    if op != 3:
                        product = (max(lower, min(upper, value))-zx)*3
                        quotient, remainder = divmod(abs(product), 4)
                        quotient += remainder >= 2
                        value = max(-128, min(127, (-quotient if product < 0 else quotient)+zy))
                    expected.append(value & 255)
                writes = 0
                d.start.value = 1; await step(); d.start.value = 0
                for _ in range(2000):
                    await step()
                    if not int(d.busy.value): break
                else: raise AssertionError('writeback timeout')
                assert int(d.error_code.value) == 0
                assert memory[destination:destination+length] == bytes(expected)
                assert writes == length == int(d.write_bytes.value)
                assert int(d.elapsed.value) == sum(int(getattr(d, k).value) for k in
                    ('compute_cycles', 'wait_cycles', 'control_cycles'))
    # Cancel a registered quantizer result while the destination stalls.
    for signal in ('abort_run', 'clear_counters', 'rst_n'):
        desc = Descriptor(2, input=512, output=2051, params=24000,
                          count=1, outputs=1, next_pc=64)
        memory[:128] = desc.encode() + Descriptor(0).encode()
        memory[512] = 42; memory[2051] = 99
        memory[24000:24016] = struct.pack('<iiBbbbbb', 0, 1, 0, 0, 0, 0, 127, 0) + b'\0\0'
        writes = write_wait = 0
        d.start.value = 1; await step(); d.start.value = 0
        for _ in range(200):
            await step()
            if held and held[1]: break
        else: raise AssertionError('no pending output to cancel')
        getattr(d, signal).value = 0 if signal == 'rst_n' else 1
        await step()
        getattr(d, signal).value = 1 if signal == 'rst_n' else 0
        held = pending = None; write_wait = 0
        for _ in range(10): await step()
        assert not int(d.busy.value) and writes == 0 and memory[2051] == 99
        d.clear_counters.value = 1; await step(); d.clear_counters.value = 0
        d.start.value = 1; await step(); d.start.value = 0
        for _ in range(200):
            await step()
            if not int(d.busy.value): break
        assert int(d.error_code.value) == 0 and writes == 1 and memory[2051] == 42
