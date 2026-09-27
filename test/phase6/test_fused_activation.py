"""Fused exact LUT reload and epilogue across tails, stalls and restart."""
import random
import struct

import cocotb
from cocotb.triggers import Timer
from hardware_v2 import Descriptor


@cocotb.test()
async def fused_backpressure_reload_and_restart(d):
    memory = bytearray(32768)
    rng = random.Random(6819)
    pending = held = None
    writes = 0
    stall_writes = True
    d.clk.value = 0; d.rst_n.value = 0; d.start.value = 0; d.start_pc.value = 0
    d.abort_run.value = 0; d.clear_counters.value = 0
    d.mem_ready.value = 0; d.mem_rvalid.value = 0; d.mem_rdata.value = 0

    async def step():
        nonlocal pending, held, writes
        d.clk.value = 0
        d.mem_rvalid.value = pending is not None
        if pending is not None:
            d.mem_rdata.value = pending
            pending = None
        d.mem_ready.value = 0
        await Timer(1, units='ns')
        cancelled = bool(int(d.abort_run.value) or int(d.clear_counters.value)
                         or not int(d.rst_n.value))
        request = ((int(d.mem_addr.value), int(d.mem_wr.value),
                    int(d.mem_wdata.value), int(d.mem_wstrb.value))
                   if int(d.mem_req.value) else None)
        if held is not None and not cancelled:
            assert request == held, 'SIMD stream request changed under backpressure'
        ready = not cancelled and (not stall_writes or rng.randrange(4) == 0)
        d.mem_ready.value = ready
        await Timer(4, units='ns')
        held = request if request and not ready and not cancelled else None
        if request and ready:
            address, wr, data, mask = request
            assert address % 8 == 0 and address + 8 <= len(memory)
            if wr:
                assert mask and mask & (mask-1) == 0
                for lane in range(8):
                    if mask >> lane & 1:
                        memory[address+lane] = data >> (lane*8) & 255
                writes += 1
            else:
                pending = int.from_bytes(memory[address:address+8], 'little')
        d.clk.value = 1
        await Timer(5, units='ns')

    def configure(depthwise, width):
        memory[:] = bytes(len(memory))
        h, ic, oc = (3, 2, 2) if depthwise else (1, 9, 2)
        plane = h*width
        params = dict(input=512, output=12003, weight=16000, params=24000,
            count=9 if depthwise else ic, outputs=plane*oc, row_stride=16,
            next_pc=64, input_h=h, input_w=width, input_c=ic, output_c=oc)
        if depthwise:
            params.update(kernel_h=3, kernel_w=3, pad_top=1, pad_bottom=1,
                          pad_left=1, pad_right=1)
        desc = Descriptor(6 if depthwise else 4, **params)
        encoded = bytearray(desc.encode())
        struct.pack_into('<H', encoded, 6, ((30000//8)<<1)|1)
        memory[:128] = bytes(encoded)+Descriptor(0).encode()
        # A fresh nonlinear mapping at the same SRAM pointer on every run
        # exposes unsafe address-only LUT caching and incorrect issue order.
        table = list(range(256)); rng.shuffle(table)
        memory[30000:30256] = bytes(table)
        xs = [rng.randrange(-31, 32) for _ in range(ic*plane)]
        memory[512:512+len(xs)] = bytes(x & 255 for x in xs)
        expected = []
        for channel in range(oc):
            weights = [rng.randrange(-3, 4) for _ in range(9 if depthwise else ic)]
            memory[16000+channel*16:16000+channel*16+len(weights)] = bytes(w & 255 for w in weights)
            bias = 13-channel*19
            memory[24000+channel*16:24016+channel*16] = struct.pack(
                '<iiBbbbbb', bias, 3, 3, -2, 0, -128, 127, 0)+b'\0\0'
            for y in range(h):
                for x in range(width):
                    acc = bias
                    if depthwise:
                        for ky in range(3):
                            for kx in range(3):
                                iy, ix = y+ky-1, x+kx-1
                                if 0 <= iy < h and 0 <= ix < width:
                                    acc += xs[channel*plane+iy*width+ix]*weights[ky*3+kx]
                    else:
                        acc += sum(xs[k*plane+y*width+x]*weights[k] for k in range(ic))
                    product = acc*3
                    q = (abs(product)+4)//8
                    q = (-q if product < 0 else q)-2
                    expected.append(table[max(-128, min(127, q)) & 255])
        memory[12003:12003+len(expected)] = bytes([0xA5])*len(expected)
        return bytes(expected)

    async def launch_and_finish(expected):
        nonlocal writes
        writes = 0
        d.start.value = 1; await step(); d.start.value = 0
        for _ in range(40000):
            await step()
            if not int(d.busy.value):
                break
        else:
            raise AssertionError('SIMD stream timeout')
        assert int(d.error_code.value) == 0
        assert memory[12003:12003+len(expected)] == expected
        assert writes == len(expected) == int(d.write_bytes.value)
        assert int(d.elapsed.value) == sum(int(getattr(d, name).value) for name in
            ('compute_cycles', 'wait_cycles', 'control_cycles'))

    await step(); d.rst_n.value = 1; await step()
    for stall_writes in (False, True):
        for depthwise in (False, True):
            for width in (1, 5, 7, 8, 9, 13):
                await launch_and_finish(configure(depthwise, width))
    stall_writes = True
    for signal in ('abort_run', 'clear_counters', 'rst_n'):
        configure(False, 13)
        writes = 0
        d.start.value = 1; await step(); d.start.value = 0
        for _ in range(40000):
            await step()
            if held and held[1] and writes >= 2:
                break
        else:
            raise AssertionError('no held SIMD output')
        before = writes
        getattr(d, signal).value = 0 if signal == 'rst_n' else 1
        await step()
        getattr(d, signal).value = 1 if signal == 'rst_n' else 0
        pending = held = None
        for _ in range(10):
            await step()
        assert not int(d.busy.value) and writes == before
        await launch_and_finish(configure(True, 9))
