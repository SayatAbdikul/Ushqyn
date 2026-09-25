"""Pinned KWS/VWW kernel geometries against an independent scalar RTL oracle."""
import random
import json
import os
from pathlib import Path

import cocotb
import numpy as np
from cocotb.triggers import Timer

from hardware_v2 import Descriptor
from kernel_vectors import vectors
from holdout_vectors import vectors as holdout_vectors
from phase4_cost import engine_cycles


@cocotb.test()
async def pinned_audio_vision_kernels_with_stalls(d):
    rng = random.Random(240923)
    memory = bytearray(32768)
    profiles=[]
    pending = None
    held = None
    added_stall_cycles = 0
    d.clk.value = 0
    d.rst_n.value = 0
    d.start.value = 0
    d.abort_run.value = 0
    d.clear_counters.value = 0
    d.start_pc.value = 0
    d.mem_ready.value = 0
    d.mem_rvalid.value = 0
    d.mem_rdata.value = 0

    async def step():
        nonlocal pending, held, added_stall_cycles
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
                pending = (delay - 1, data)
        await Timer(5, units='ns')
        request = (int(d.mem_addr.value), int(d.mem_wr.value),
                   int(d.mem_wdata.value), int(d.mem_wstrb.value)) if int(d.mem_req.value) else None
        if request is not None and not ready and int(d.busy.value):
            added_stall_cycles += 1
        if held is not None:
            assert request == held, 'memory request changed under backpressure'
        held = request if request is not None and not ready else None
        if request is not None and ready:
            address, wr, data, mask = request
            assert address % 8 == 0 and address + 8 <= len(memory)
            if wr:
                for lane in range(8):
                    if (mask >> lane) & 1:
                        memory[address + lane] = (data >> (8*lane)) & 255
            else:
                assert pending is None
                delay=rng.randrange(1,5)
                added_stall_cycles += delay
                pending = (delay, int.from_bytes(memory[address:address+8], 'little'))
        d.clk.value = 1
        await Timer(5, units='ns')

    async def execute(desc, inputs, weights, params, expected, macs, label):
        nonlocal pending, held, added_stall_cycles
        memory[:] = bytes(len(memory))
        memory[:128] = desc.encode() + Descriptor(0).encode()
        memory[desc.input:desc.input+len(inputs)] = np.asarray(inputs, dtype=np.int8).tobytes()
        if weights is not None:
            for c, row in enumerate(weights):
                data = np.asarray(row, dtype=np.int8).tobytes()
                memory[desc.weight+c*desc.row_stride:desc.weight+c*desc.row_stride+len(data)] = data
        for c, data in enumerate(params):
            memory[desc.params+c*16:desc.params+(c+1)*16] = data
        pending = held = None
        added_stall_cycles = 0
        d.start.value = 1
        await step()
        d.start.value = 0
        for _ in range(500000):
            await step()
            if not int(d.busy.value):
                break
        else:
            raise AssertionError('kernel timeout')
        assert int(d.error_code.value) == 0
        actual = np.frombuffer(memory[desc.output:desc.output+len(expected)], np.int8)
        np.testing.assert_array_equal(actual, np.array(expected, np.int8))
        assert int(d.useful_macs.value) == macs
        assert int(d.elapsed.value) == (int(d.compute_cycles.value) +
                                         int(d.wait_cycles.value) + int(d.control_cycles.value))
        assert int(d.elapsed.value) == engine_cycles(desc) + added_stall_cycles
        profiles.append(dict(label=label,opcode=desc.opcode,inputs=len(inputs),
                             outputs=len(expected),useful_macs=int(d.useful_macs.value),
                             simulated_core_cycles=int(d.elapsed.value),
                             simulated_compute_cycles=int(d.compute_cycles.value),
                             simulated_wait_cycles=int(d.wait_cycles.value),
                             simulated_control_cycles=int(d.control_cycles.value),
                             physical_sram_read_bytes=int(d.read_bytes.value),
                             physical_sram_write_bytes=int(d.write_bytes.value)))

    await step()
    d.rst_n.value = 1
    await step()

    for vector in vectors():
        await execute(*vector)
    for vector in holdout_vectors():
        await execute(*vector)
    root=Path(os.environ['REPO_ROOT'])
    (root/'work/phase4/kernel-profiles.json').write_text(json.dumps(dict(
        evidence_type='randomized-stall RTL simulation; not board latency',
        random_seed=240923,profiles=profiles),indent=2)+'\n')
