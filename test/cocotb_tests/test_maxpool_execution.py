"""
Cocotb test for `maxpool_execution` — Patch G (R2: NCHW layout).

Preloads a known NCHW int8 tensor into the wrapper's vector buffer (channel
outermost), runs maxpool with pool=2 stride=2, and asserts the destination
buffer matches `compiler.golden_model.maxpool` byte-for-byte.

Uses geometry that mirrors SmallCNN's conv1 → maxpool1 boundary
(channels=4, fmap=26×26) so the test exercises the same case the integration
test will see.
"""

import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../compiler'))
import golden_model


TILE_ELEMS = 32


async def _reset(dut):
    dut.rst.value = 1
    dut.start.value = 0
    dut.preload_en.value = 0
    dut.preload_buf_id.value = 0
    for i in range(TILE_ELEMS):
        dut.preload_data[i].value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def _preload(dut, buf_id, data):
    """Stream `data` into the wrapper's preload bus, TILE_ELEMS at a time."""
    dut.preload_en.value = 1
    dut.preload_buf_id.value = buf_id

    cycles = (len(data) + TILE_ELEMS - 1) // TILE_ELEMS
    for c in range(cycles):
        for i in range(TILE_ELEMS):
            idx = c * TILE_ELEMS + i
            v = int(data[idx]) if idx < len(data) else 0
            dut.preload_data[i].value = v
        await RisingEdge(dut.clk)
    dut.preload_en.value = 0
    await RisingEdge(dut.clk)


def _golden_maxpool_output(x_int8, channels, fmap_h, fmap_w, pool_size, stride):
    golden_model.buffers = {}
    golden_model.flag = 0
    golden_model.buffers[0] = list(np.asarray(x_int8, dtype=np.int8))
    golden_model.maxpool(
        dest=1, x=0,
        fmap_h=fmap_h, fmap_w=fmap_w, channels=channels,
        pool_size=pool_size, stride=stride
    )
    return np.array(golden_model.buffers[1], dtype=np.int8)


async def _run_maxpool_case(dut, *, channels, fmap_h, fmap_w, pool_size, stride, seed):
    """Drive one maxpool run end-to-end and return the dest-buffer contents."""
    rng = np.random.RandomState(seed)
    # NCHW int8 tensor, varied so the max-pool windows pick distinct elements.
    x_nchw = rng.randint(-128, 127, size=(channels, fmap_h, fmap_w),
                         dtype=np.int8)

    X_BUF, OUT_BUF = 2, 4

    await _reset(dut)
    await _preload(dut, X_BUF, x_nchw.flatten().tolist())

    dut.fmap_h.value      = fmap_h
    dut.fmap_w.value      = fmap_w
    dut.in_channels.value = channels
    dut.pool_size.value   = pool_size
    dut.stride_val.value  = stride
    dut.x_buffer_id.value = X_BUF
    dut.dest_buffer_id.value = OUT_BUF

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    timeout = 200_000
    while int(dut.done.value) == 0:
        await RisingEdge(dut.clk)
        timeout -= 1
        if timeout <= 0:
            raise cocotb.result.TestFailure("maxpool_execution timed out before DONE")

    out_h = (fmap_h - pool_size) // stride + 1
    out_w = (fmap_w - pool_size) // stride + 1
    n     = channels * out_h * out_w

    for _ in range(2):
        await RisingEdge(dut.clk)

    raw = int(dut.buffers.vector_buffer_inst.buffers[OUT_BUF].value)
    dut_out = []
    for addr in range(n):
        byte = (raw >> (addr * 8)) & 0xFF
        if byte >= 0x80:
            byte -= 0x100
        dut_out.append(byte)
    return np.array(dut_out, dtype=np.int8), x_nchw


@cocotb.test()
async def test_maxpool_4x14x14_geom(dut):
    """4-channel 14×14 NCHW input → 4×7×7. 4·196 = 784 bytes, fits the 8192-bit
    default vector-buffer width. Mirrors the SmallCNN conv1→pool1 layout
    family without overflowing the per-buffer width."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    actual, x_nchw = await _run_maxpool_case(
        dut, channels=4, fmap_h=14, fmap_w=14, pool_size=2, stride=2, seed=0
    )
    expected = _golden_maxpool_output(
        x_nchw.flatten(), channels=4, fmap_h=14, fmap_w=14,
        pool_size=2, stride=2,
    )
    np.testing.assert_array_equal(
        actual, expected,
        err_msg=f"NCHW maxpool 4×14×14 mismatch (first 16): "
                f"dut={actual[:16].tolist()} expected={expected[:16].tolist()}"
    )


@cocotb.test()
async def test_maxpool_8x11x11_geom(dut):
    """Geometry matching SmallCNN's conv2 output: 8×11×11, pool=2 stride=2 → 8×5×5."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    actual, x_nchw = await _run_maxpool_case(
        dut, channels=8, fmap_h=11, fmap_w=11, pool_size=2, stride=2, seed=42
    )
    expected = _golden_maxpool_output(
        x_nchw.flatten(), channels=8, fmap_h=11, fmap_w=11,
        pool_size=2, stride=2,
    )
    np.testing.assert_array_equal(
        actual, expected,
        err_msg=f"8×11×11 NCHW maxpool mismatch (first 16): "
                f"dut={actual[:16].tolist()} expected={expected[:16].tolist()}"
    )
