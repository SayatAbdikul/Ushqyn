"""Malformed descriptor fields must still fail after geometry width narrowing."""
import copy

import cocotb
from cocotb.triggers import Timer

from hardware_v2 import Descriptor


@cocotb.test()
async def malformed_width_boundaries(dut):
    memory = bytearray(32768)
    dut.clk.value = 0
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.abort_run.value = 0
    dut.clear_counters.value = 0
    dut.start_pc.value = 0
    dut.mem_ready.value = 0
    dut.mem_rvalid.value = 0
    dut.mem_rdata.value = 0
    pending = None

    async def step():
        nonlocal pending
        dut.clk.value = 0
        dut.mem_ready.value = pending is None
        dut.mem_rvalid.value = pending is not None
        dut.mem_rdata.value = pending or 0
        pending = None
        await Timer(5, units='ns')
        if int(dut.mem_req.value) and int(dut.mem_ready.value):
            assert not int(dut.mem_wr.value), 'malformed descriptor wrote SRAM'
            address = int(dut.mem_addr.value)
            assert address % 8 == 0 and address + 8 <= len(memory)
            pending = int.from_bytes(memory[address:address+8], 'little')
        dut.clk.value = 1
        await Timer(5, units='ns')

    await step()
    dut.rst_n.value = 1
    await step()
    spatial = Descriptor(4, input=512, output=1024, weight=2048, params=4096,
                         count=1, outputs=1, row_stride=8, next_pc=64,
                         input_h=1, input_w=1, input_c=1, output_c=1)
    gemm = Descriptor(1, input=512, output=1024, weight=2048, params=4096,
                      count=1, outputs=1, row_stride=8, next_pc=64)
    cases = (
        ('input_h_256', spatial, 'input_h', 256),
        ('input_w_256', spatial, 'input_w', 256),
        ('kernel_h_32', spatial, 'kernel_h', 32),
        ('kernel_w_32', spatial, 'kernel_w', 32),
        ('input_c_1025', spatial, 'input_c', 1025),
        ('output_c_1025', spatial, 'output_c', 1025),
        ('stride_h_32', spatial, 'stride_h', 32),
        ('pad_top_31', spatial, 'pad_top', 31),
        ('spatial_weight_stride_65536', spatial, 'row_stride', 65536),
        ('gemm_weight_stride_65536', gemm, 'row_stride', 65536),
        ('gemm_outputs_32769', gemm, 'outputs', 32769),
    )
    for name, template, field, value in cases:
        desc = copy.copy(template)
        setattr(desc, field, value)
        memory[:128] = desc.encode() + Descriptor(0).encode()
        dut.start.value = 1
        await step()
        dut.start.value = 0
        for _ in range(32):
            await step()
            if not int(dut.busy.value):
                break
        else:
            raise AssertionError(f'{name}: descriptor did not reject')
        assert int(dut.error_code.value) == 1, f'{name}: incorrect rejection code'
