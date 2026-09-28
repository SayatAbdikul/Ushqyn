"""Adversarial ABI and generic-fallback checks for contract-hint engine."""
import struct

import cocotb
from cocotb.triggers import Timer

from hardware_v2 import Descriptor


@cocotb.test()
async def contract_hints_reject_bad_shapes_and_preserve_generic(dut):
    memory = bytearray(32768)
    pending = None
    writes = 0
    dut.clk.value = 0
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.abort_run.value = 0
    dut.clear_counters.value = 0
    dut.start_pc.value = 0
    dut.mem_ready.value = 0
    dut.mem_rvalid.value = 0
    dut.mem_rdata.value = 0

    async def step():
        nonlocal pending, writes
        dut.clk.value = 0
        dut.mem_ready.value = pending is None
        dut.mem_rvalid.value = pending is not None
        dut.mem_rdata.value = pending or 0
        pending = None
        await Timer(5, units='ns')
        if int(dut.mem_req.value) and int(dut.mem_ready.value):
            addr = int(dut.mem_addr.value)
            assert addr % 8 == 0 and addr + 8 <= len(memory)
            if int(dut.mem_wr.value):
                data = int(dut.mem_wdata.value)
                strobe = int(dut.mem_wstrb.value)
                for i in range(8):
                    if (strobe >> i) & 1:
                        memory[addr+i] = (data >> (8*i)) & 255
                        writes += 1
            else:
                pending = int.from_bytes(memory[addr:addr+8], 'little')
        dut.clk.value = 1
        await Timer(5, units='ns')

    async def run(desc, hint, expected_error):
        nonlocal pending, writes
        dut.rst_n.value = 0
        pending = None
        writes = 0
        memory[:] = bytes(len(memory))
        blob = bytearray(desc.encode())
        blob[7] |= hint
        memory[:128] = blob + Descriptor(0).encode()
        memory[512:515] = bytes((1, 2, 3))
        memory[2048] = 2
        memory[4096:4112] = struct.pack('<iiBbbbbb',
            0, 1 << 30, 30, 0, 0, -128, 127, 0) + b'\0\0'
        await step()
        dut.rst_n.value = 1
        await step()
        dut.start.value = 1
        await step()
        dut.start.value = 0
        for _ in range(1000):
            await step()
            if not int(dut.busy.value):
                break
        else:
            raise AssertionError('descriptor did not finish')
        assert int(dut.error_code.value) == expected_error
        if expected_error:
            assert writes == 0, 'malformed contract wrote output'
        return bytes(memory[1024:1027]), int(dut.elapsed.value)

    pw = Descriptor(4, input=512, output=1024, weight=2048, params=4096,
                    count=1, outputs=3, row_stride=8, next_pc=64,
                    input_h=1, input_w=3, input_c=1, output_c=1)
    dense, generic_cycles = await run(pw, 0, 0)
    fast, fast_cycles = await run(pw, 0x20, 0)
    assert dense == fast == bytes((2, 4, 6))
    assert generic_cycles >= fast_cycles

    conv3 = Descriptor(4, input=512, output=1024, weight=2048, params=4096,
                       count=9, outputs=1, row_stride=16, next_pc=64,
                       kernel_h=3, kernel_w=3, input_h=3, input_w=3,
                       input_c=1, output_c=1)
    dw3 = Descriptor(6, input=512, output=1024, weight=2048, params=4096,
                     count=9, outputs=1, row_stride=16, next_pc=64,
                     kernel_h=3, kernel_w=3, input_h=3, input_w=3,
                     input_c=1, output_c=1)
    await run(conv3, 0x20, 1)  # PW hint on 3x3 ordinary convolution.
    await run(dw3, 0x20, 2)    # PW hint on depthwise opcode.
    await run(pw, 0x40, 2)     # DW hint on pointwise opcode.
    await run(pw, 0x60, 2)     # Contradictory hints.
