"""Direct paired PW arithmetic, INT32 overflow, and spatial/channel tails."""
from __future__ import annotations

import random
import struct

import cocotb
from cocotb.triggers import Timer

from hardware_v2 import Descriptor


@cocotb.test()
async def paired_overflow_and_odd_tails(d):
    async def case(label: str, count: int, weights: list[list[int]], biases: list[int], expected_error: int) -> None:
        area = 9
        channels = 3
        stride = (count + 7) & ~7
        memory = bytearray(32768)
        rng = random.Random(6063)
        pending = None
        held = None
        desc = Descriptor(
            4, input=512, output=12000, weight=16000, params=24000,
            count=count, outputs=channels * area, row_stride=stride,
            next_pc=64, kernel_h=1, kernel_w=1,
            input_h=1, input_w=area, input_c=count, output_c=channels,
        )
        memory[:128] = desc.encode() + Descriptor(0).encode()
        memory[512:512 + count * area] = bytes([1]) * (count * area)
        for channel in range(channels):
            memory[16000 + channel * stride:16000 + channel * stride + count] = bytes(
                value & 255 for value in weights[channel]
            )
            memory[24000 + channel * 16:24000 + (channel + 1) * 16] = (
                struct.pack("<iiBbbbbb", biases[channel], 1 << 30, 31, 0, 0, -128, 127, 0)
                + b"\0\0"
            )
        d.clk.value = 0
        d.rst_n.value = 0
        d.start.value = 0
        d.start_pc.value = 0
        d.abort_run.value = 0
        d.clear_counters.value = 0
        d.mem_ready.value = 0
        d.mem_rvalid.value = 0
        d.mem_rdata.value = 0

        async def step() -> None:
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
                    pending = (delay - 1, data)
            await Timer(5, units="ns")
            request = (
                int(d.mem_addr.value), int(d.mem_wr.value),
                int(d.mem_wdata.value), int(d.mem_wstrb.value),
            ) if int(d.mem_req.value) else None
            if held is not None and int(d.rst_n.value):
                assert request == held, f"{label}: request changed under backpressure"
            held = request if request is not None and not ready else None
            if request is not None and ready:
                addr, wr, data, mask = request
                assert addr % 8 == 0 and addr + 8 <= len(memory)
                if wr:
                    for lane in range(8):
                        if (mask >> lane) & 1:
                            memory[addr + lane] = (data >> (8 * lane)) & 255
                else:
                    assert pending is None
                    pending = (rng.randrange(5), int.from_bytes(memory[addr:addr + 8], "little"))
            d.clk.value = 1
            await Timer(5, units="ns")

        await step()
        d.rst_n.value = 1
        await step()
        d.start.value = 1
        await step()
        d.start.value = 0
        for _ in range(30000):
            await step()
            if not int(d.busy.value):
                break
        else:
            raise AssertionError(f"{label}: engine timeout")
        assert int(d.error_code.value) == expected_error, (
            label, int(d.error_code.value), expected_error
        )
        if expected_error == 0:
            expected = bytearray()
            for channel in range(channels):
                signed_weights = weights[channel]
                total = biases[channel] + sum(signed_weights)
                product = total * (1 << 30)
                quotient, remainder = divmod(abs(product), 1 << 31)
                quotient += 2 * remainder >= 1 << 31
                rounded = -quotient if product < 0 else quotient
                expected.extend([(max(-128, min(127, rounded)) & 255)] * area)
            assert memory[12000:12000 + len(expected)] == expected, label
            assert int(d.useful_macs.value) == count * channels * area, label
        print(f"AUDIT {label}: error={expected_error}, count={count}, area={area}, channels={channels}")

    # A positive intermediate sum after four inputs is allowed to exceed
    # INT32 as long as the original eight-input boundary remains in range.
    await case(
        "cancel_then_odd_channel_tail", 8,
        [[127] * 4 + [-127] * 4, [1] * 8, [-1] * 8],
        [2147483600, 0, 0], 0,
    )
    await case(
        "even_channel_overflow", 8,
        [[127] * 8, [1] * 8, [-1] * 8],
        [2147483600, 0, 0], 5,
    )
    await case(
        "odd_channel_overflow", 8,
        [[127] * 4 + [-127] * 4, [127] * 8, [-1] * 8],
        [2147483600, 2147483600, 0], 5,
    )
    # The 8-input boundary must fail even when input nine would cancel it.
    await case(
        "odd_boundary_overflow_before_cancellation", 9,
        [[0] * 9, [127] + [0] * 7 + [-127], [0] * 9],
        [0, 2147483600, 0], 5,
    )
