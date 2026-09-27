"""Independent pool reference at arithmetic limits, row boundaries and stalls."""
import random
import struct

import cocotb
from cocotb.triggers import Timer
from hardware_v2 import Descriptor


@cocotb.test()
async def pool_widths_and_balanced_chunks(d):
    rng = random.Random(8201)
    memory = bytearray(32768)
    pending = held = None
    d.clk.value = 0
    d.rst_n.value = 0
    d.start.value = 0
    d.start_pc.value = 0
    d.abort_run.value = 0
    d.clear_counters.value = 0
    d.mem_ready.value = 0
    d.mem_rvalid.value = 0
    d.mem_rdata.value = 0

    async def step():
        nonlocal pending, held
        d.clk.value = 0
        ready = pending is None and rng.randrange(4) != 0
        d.mem_ready.value = ready
        d.mem_rvalid.value = 0
        if pending is not None:
            delay, data = pending
            if delay == 0:
                d.mem_rvalid.value = 1
                d.mem_rdata.value = data
                pending = None
            else:
                pending = delay-1, data
        await Timer(5, units='ns')
        request = ((int(d.mem_addr.value), int(d.mem_wr.value),
                    int(d.mem_wdata.value), int(d.mem_wstrb.value))
                   if int(d.mem_req.value) else None)
        if held is not None:
            assert held == request, 'pool request changed under backpressure'
        held = request if request is not None and not ready else None
        if request is not None and ready:
            address, wr, data, mask = request
            assert address % 8 == 0 and address + 8 <= len(memory)
            if wr:
                for lane in range(8):
                    if mask >> lane & 1:
                        memory[address+lane] = (data >> (lane*8)) & 255
            else:
                assert pending is None
                pending = (rng.randrange(4),
                           int.from_bytes(memory[address:address+8], 'little'))
        d.clk.value = 1
        await Timer(5, units='ns')

    async def run(op, ih, iw, kh, kw, sh, sw, zx, zy, values, shift):
        nonlocal pending, held
        assert len(values) == ih*iw
        pending = held = None
        d.rst_n.value = 0
        await step()
        d.rst_n.value = 1
        await step()
        memory[:] = bytes(len(memory))
        oh = (ih-kh)//sh+1
        ow = (iw-kw)//sw+1
        desc = Descriptor(op, input=1024, output=22000, params=24000,
                          count=ih*iw, outputs=oh*ow, next_pc=64,
                          kernel_h=kh, kernel_w=kw,
                          stride_h=sh, stride_w=sw,
                          input_h=ih, input_w=iw)
        memory[:128] = desc.encode()+Descriptor(0).encode()
        memory[1024:1024+len(values)] = bytes(x & 255 for x in values)
        memory[24000:24016] = (struct.pack('<iiBbbbbb', 0, 1, shift, zy,
                                          zx, -128, 127, 0)+b'\0\0')
        expected = []
        for y in range(oh):
            for x in range(ow):
                window = [values[(y*sh+dy)*iw+x*sw+dx]
                          for dy in range(kh) for dx in range(kw)]
                centered = sum(sample-zx for sample in window) if op == 7 else max(window)-zx
                magnitude = (abs(centered)+(1 << (shift-1))) >> shift if shift else abs(centered)
                quantized = (-magnitude if centered < 0 else magnitude)+zy
                expected.append(max(-128, min(127, quantized)) & 255)
        d.start.value = 1
        await step()
        d.start.value = 0
        for _ in range(150000):
            await step()
            if not int(d.busy.value):
                break
        else:
            raise AssertionError('pool timeout')
        assert int(d.error_code.value) == 0
        actual = memory[22000:22000+len(expected)]
        assert actual == bytes(expected), (op, ih, iw, kh, kw, zx, zy,
                                           list(actual), expected)
        assert int(d.write_bytes.value) == len(expected)

    await run(7, 31, 31, 31, 31, 31, 31, -128, 5, [127]*961, 11)
    await run(7, 31, 31, 31, 31, 31, 31, 127, -7, [-128]*961, 11)
    await run(7, 31, 31, 31, 31, 31, 31, -11, 3,
              [127 if k%3 else -128 for k in range(961)], 11)
    await run(7, 7, 11, 3, 5, 2, 2, -117, 7,
              [rng.randrange(-128, 128) for _ in range(77)], 7)
    await run(7, 5, 9, 2, 4, 1, 2, 127, -3,
              [rng.randrange(-128, 128) for _ in range(45)], 6)
    await run(7, 3, 3, 3, 3, 1, 1, -9, 0,
              [-128, 127, 3, 25, -30, 1, -8, 127, -125], 0)
    await run(5, 5, 7, 3, 3, 2, 2, 13, -2,
              [rng.randrange(-128, 128) for _ in range(35)], 0)
