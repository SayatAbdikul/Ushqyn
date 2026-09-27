"""Exhaustive vector-LUT activation codes under stalls and cancellation."""
import random
import struct

import cocotb
from cocotb.triggers import Timer
from hardware_v2 import Descriptor


@cocotb.test()
async def activation_tables_and_cancel(d):
    rng = random.Random(617)
    memory = bytearray(32768)
    pending = held = None
    writes = 0
    d.clk.value = 0; d.rst_n.value = 0; d.start.value = 0; d.start_pc.value = 0
    d.abort_run.value = 0; d.clear_counters.value = 0
    d.mem_ready.value = 0; d.mem_rvalid.value = 0; d.mem_rdata.value = 0

    async def step():
        nonlocal pending, held, writes
        d.clk.value = 0
        await Timer(1, units='ns')  # Settle newly driven reset/abort controls.
        cancelled = bool(int(d.abort_run.value) or int(d.clear_counters.value) or not int(d.rst_n.value))
        ready = pending is None and rng.randrange(3) != 0 and not cancelled
        d.mem_ready.value = ready; d.mem_rvalid.value = 0
        if pending is not None:
            delay, data = pending
            if delay == 0:
                d.mem_rvalid.value = 1; d.mem_rdata.value = data; pending = None
            else: pending = (delay-1, data)
        await Timer(4, units='ns')
        request = ((int(d.mem_addr.value), int(d.mem_wr.value), int(d.mem_wdata.value), int(d.mem_wstrb.value))
                   if int(d.mem_req.value) else None)
        if held is not None and not cancelled:
            assert request == held, 'request changed under backpressure'
        held = request if request and not ready and not cancelled else None
        if request and ready:
            address, wr, data, mask = request
            assert address % 8 == 0 and address + 8 <= len(memory)
            if wr:
                assert mask
                for lane in range(8):
                    if mask >> lane & 1: memory[address+lane] = data >> (8*lane) & 255
                writes += mask.bit_count()
            else: pending = (rng.randrange(4), int.from_bytes(memory[address:address+8], 'little'))
        d.clk.value = 1; await Timer(5, units='ns')

    def configure(op, length, destination, multiplier, shift, zx, zy, lower, upper):
        memory[:] = bytes(len(memory))
        desc = Descriptor(op, input=512, output=destination, params=24000,
                          count=length, outputs=length, next_pc=64)
        memory[:128] = desc.encode() + Descriptor(0).encode()
        memory[24000:24016] = struct.pack('<iiBbbbbb', 0, multiplier, shift, zy, zx,
                                          lower, upper, 0) + b'\0\0'
        data = bytes(i % 256 for i in range(length))
        memory[512:512+length] = data
        # Follow the scalar reference's byte-at-a-time semantics even for
        # partially overlapping buffers; previous writes can affect later reads.
        reference = bytearray(memory)
        for i in range(length):
            value = reference[512+i]
            value = value if value < 128 else value-256
            product = (max(lower, min(upper, value))-zx)*multiplier
            quotient, remainder = divmod(abs(product), 1 << shift)
            if shift: quotient += 2*remainder >= 1 << shift
            value = max(-128, min(127, (-quotient if product < 0 else quotient)+zy))
            reference[destination+i] = value & 255
        return bytes(reference[destination:destination+length])

    async def run(length, destination, expected):
        nonlocal writes
        writes = 0
        d.start.value = 1; await step(); d.start.value = 0
        for _ in range(30000):
            await step()
            if not int(d.busy.value): break
        else: raise AssertionError('activation timeout')
        assert int(d.error_code.value) == 0
        assert memory[destination:destination+length] == expected
        assert writes == length
        assert 0 < int(d.write_bytes.value) <= length
        assert int(d.elapsed.value) == sum(int(getattr(d, k).value) for k in
            ('compute_cycles', 'wait_cycles', 'control_cycles'))

    await step(); d.rst_n.value = 1; await step()
    configs = [(2, 3, 2, -31, 5, -31, 127),
               (2, 2147483647, 62, -128, 127, -128, 127),
               (8, 1, 0, 127, -128, -128, 127),
               (8, 2147483647, 31, -7, -9, -40, 60)]
    for op, m, shift, zx, zy, lower, upper in configs:
        for length, destination in ((511, 2051), (512, 512), (529, 2051), (529, 515)):
            expected = configure(op, length, destination, m, shift, zx, zy, lower, upper)
            await run(length, destination, expected)
    # Cancel both LUT construction and a held output, then change the mapping.
    for signal in ('abort_run', 'clear_counters', 'rst_n'):
        for location in ('build', 'write'):
            configure(2, 529, 2051, 3, 2, -31, 5, -31, 127)
            writes = 0; d.start.value = 1; await step(); d.start.value = 0
            for cycle in range(3000):
                await step()
                if (location == 'build' and cycle == 100) or (location == 'write' and held and held[1]): break
            else: raise AssertionError('no cancellation point')
            before = writes
            getattr(d, signal).value = 0 if signal == 'rst_n' else 1; await step()
            getattr(d, signal).value = 1 if signal == 'rst_n' else 0; held = None
            for _ in range(10): await step()
            assert not int(d.busy.value) and writes == before, (signal, location, int(d.busy.value), before, writes)
            expected = configure(8, 529, 2051, 7, 3, -9, 13, -60, 100)
            await run(529, 2051, expected)
