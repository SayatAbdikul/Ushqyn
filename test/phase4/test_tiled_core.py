"""Concurrent kernel and DMA through one real RTL scratchpad port."""

import random
import struct

import cocotb
from cocotb.triggers import Timer
from hardware_v2 import Descriptor


@cocotb.test()
async def engine_dma_arbitration_and_read_ownership(d):
    rng = random.Random(20260924)
    external = bytearray(8*1024*1024)
    pending_ext = None
    held_ext = None
    d.clk.value = 0; d.rst_n.value = 0
    d.engine_start.value = 0; d.engine_abort.value = 0; d.engine_start_pc.value = 0
    d.dma_start.value = 0; d.dma_abort.value = 0; d.dma_to_sram.value = 0
    d.dma_sram_base.value = 0; d.dma_ext_base.value = 0; d.dma_length.value = 0
    d.host_req.value = 0; d.host_wr.value = 0; d.host_addr.value = 0
    d.host_wdata.value = 0; d.host_wstrb.value = 0
    d.ext_ready.value = 0; d.ext_rvalid.value = 0; d.ext_rdata.value = 0

    async def step():
        nonlocal pending_ext, held_ext
        d.clk.value = 0
        ready = pending_ext is None and rng.randrange(4) != 0
        d.ext_ready.value = ready
        d.ext_rvalid.value = 0
        if pending_ext is not None:
            delay, data = pending_ext
            if delay == 0:
                d.ext_rvalid.value = 1
                d.ext_rdata.value = data
                pending_ext = None
            else:
                pending_ext = (delay-1, data)
        await Timer(5, units='ns')
        req = (int(d.ext_addr.value), int(d.ext_wr.value),
               int(d.ext_wdata.value), int(d.ext_wstrb.value)) if int(d.ext_req.value) else None
        if held_ext is not None:
            assert req == held_ext, 'external request changed under backpressure'
        held_ext = req if req is not None and not ready else None
        if req is not None and ready:
            addr, wr, data, mask = req
            assert addr % 8 == 0 and addr+8 <= len(external)
            if wr:
                for lane in range(8):
                    if mask & (1 << lane):
                        external[addr+lane] = (data >> (8*lane)) & 255
            else:
                assert pending_ext is None
                pending_ext = (rng.randrange(1, 5),
                               int.from_bytes(external[addr:addr+8], 'little'))
        d.clk.value = 1
        await Timer(5, units='ns')

    async def host_write(addr, payload):
        assert addr % 8 == 0
        for offset in range(0, len(payload), 8):
            chunk = payload[offset:offset+8]
            d.host_addr.value = addr+offset
            d.host_wdata.value = int.from_bytes(chunk.ljust(8, b'\0'), 'little')
            d.host_wstrb.value = (1 << len(chunk))-1
            d.host_wr.value = 1; d.host_req.value = 1
            await step()
            assert int(d.host_ready.value)
            d.host_req.value = 0

    async def host_read(addr, length):
        assert addr % 8 == 0
        result = bytearray()
        for offset in range(0, length, 8):
            d.host_addr.value = addr+offset
            d.host_wr.value = 0; d.host_req.value = 1
            await step()
            assert int(d.host_ready.value) and int(d.host_rvalid.value)
            result.extend(int(d.host_rdata.value).to_bytes(8, 'little'))
            d.host_req.value = 0
        return bytes(result[:length])

    async def concurrent_transfer(direction, sb, eb, payload):
        length = len(payload)
        if direction:
            external[eb:eb+length] = payload
        else:
            await host_write(sb, payload)
        d.dma_to_sram.value = direction; d.dma_sram_base.value = sb
        d.dma_ext_base.value = eb; d.dma_length.value = length
        d.engine_start.value = 1; d.dma_start.value = 1
        await step()
        d.engine_start.value = 0; d.dma_start.value = 0
        d.host_req.value = 1; d.host_wr.value = 0; d.host_addr.value = 0
        saw_engine = saw_dma = False
        overlap_cycles = 0
        contention_cycles = 0
        for _ in range(100000):
            await step()
            saw_engine |= bool(int(d.engine_done.value))
            saw_dma |= bool(int(d.dma_done.value))
            if int(d.engine_busy.value) and int(d.dma_busy.value):
                overlap_cycles += 1
            if int(d.ereq.value) and int(d.dreq.value):
                contention_cycles += 1
            if int(d.engine_busy.value) or int(d.dma_busy.value):
                assert not int(d.host_ready.value), 'host accessed live SRAM'
            if saw_engine and saw_dma and not int(d.engine_busy.value) and not int(d.dma_busy.value):
                break
        else:
            raise AssertionError('concurrent compute/DMA timeout')
        d.host_req.value = 0
        assert overlap_cycles > 10
        assert contention_cycles > 0, 'test never exercised arbitration'
        assert int(d.engine_error.value) == int(d.dma_error.value) == 0
        assert int(d.dma_bytes_copied.value) == length
        if direction:
            assert await host_read(sb, length) == payload
        else:
            assert external[eb:eb+length] == payload

    await step(); d.rst_n.value = 1; await step()
    desc = Descriptor(2, input=512, output=1024, params=256,
                      count=256, outputs=256, next_pc=64)
    desc.validate()
    await host_write(0, desc.encode()+Descriptor(0).encode())
    params = struct.pack('<iiBbbbbb2x', 0, 1 << 30, 30, 0, 0, 0, 127, 0)
    await host_write(256, params)
    await host_write(512, bytes(range(256)))
    await concurrent_transfer(0, 2048, len(external)-256,
                              bytes(rng.randrange(256) for _ in range(256)))
    expected = bytes(x if x < 128 else 0 for x in range(256))
    assert await host_read(1024, 256) == expected
    await concurrent_transfer(1, 4096, 0xfff8,
                              bytes(rng.randrange(256) for _ in range(257)))
    assert await host_read(1024, 256) == expected
