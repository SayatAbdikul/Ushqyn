"""Exercise the isolated exact requantizer candidate across signed boundaries."""
import random

import cocotb
from cocotb.triggers import Timer


def reference(acc, mul, shift, zero_point):
    product = acc * mul
    magnitude = abs(product)
    rounded = magnitude if shift == 0 else (magnitude + (1 << (shift - 1))) >> shift
    scaled = (-rounded if product < 0 else rounded) + zero_point
    return max(-128, min(127, scaled))


@cocotb.test()
async def exact_signed_rounding_and_handshake(d):
    d.clk.value = 0
    d.rst_n.value = 0
    d.flush.value = 0
    d.in_valid.value = 0
    d.out_ready.value = 1
    d.accumulator.value = 0
    d.multiplier.value = 0
    d.shift.value = 0
    d.zero_point.value = 0

    async def tick():
        d.clk.value = 0
        await Timer(5, units="ns")
        d.clk.value = 1
        await Timer(5, units="ns")

    await tick()
    d.rst_n.value = 1
    await tick()

    accs = (-2147483648, -2147483647, -(1 << 30), -129, -1,
            0, 1, 127, 128, 1 << 30, 2147483647)
    multipliers = (-2147483648, -2147483647, -(1 << 30), -1,
                   0, 1, 1 << 30, 2147483647)
    zeros = (-128, -17, 0, 127)
    vectors = ((a, m, s, z) for s in range(63) for a in accs
               for m in multipliers for z in zeros)
    rng = random.Random(619)
    random_vectors = ((rng.randrange(-(1 << 31), 1 << 31),
                       rng.randrange(-(1 << 31), 1 << 31),
                       rng.randrange(63), rng.randrange(-128, 128))
                      for _ in range(5000))
    for acc, mul, shift, zero in (*vectors, *random_vectors):
        d.accumulator.value = acc
        d.multiplier.value = mul
        d.shift.value = shift
        d.zero_point.value = zero
        d.in_valid.value = 1
        assert int(d.in_ready.value) == 1
        await tick()
        got = int(d.result.value)
        if got >= 128:
            got -= 256
        assert int(d.out_valid.value) == 1
        assert got == reference(acc, mul, shift, zero), (acc, mul, shift, zero, got)

    d.out_ready.value = 0
    d.in_valid.value = 1
    d.accumulator.value = -123456
    d.multiplier.value = 78901
    d.shift.value = 24
    d.zero_point.value = -5
    held = int(d.result.value)
    for _ in range(5):
        await tick()
        assert int(d.in_ready.value) == 0
        assert int(d.out_valid.value) == 1
        assert int(d.result.value) == held
    d.flush.value = 1
    await tick()
    assert int(d.out_valid.value) == 0
