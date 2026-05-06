"""
Cocotb test for `conv2d_execution` — Patch F (R1: relu_flag plumbing).

Drives a small Conv geometry (1×1×4×4 input, 1 out_c, 3×3 kernel, stride=1, pad=1)
twice, once with relu_flag=0 and once with relu_flag=1, and asserts the
dest-buffer contents match `compiler.golden_model.conv2d` byte-for-byte.

The test is wired through the existing `conv2d_execution_tb_wrapper.sv`,
which exposes a preload bus for the buffer-controller behind the conv unit
plus the conv unit's geometry inputs. The Verilog top-module for this run
is therefore `conv2d_execution_tb_wrapper`, and the cocotb Makefile sets
TEST_TARGET=conv2d_execution to point at the right RTL file list.
"""

import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

# Make the compiler's golden_model importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../compiler'))
import golden_model
from helper_functions import quantize_int32_to_int8_rtl_exact


TILE_ELEMS = 32


async def _reset(dut):
    dut.rst.value = 1
    dut.start.value = 0
    dut.preload_en.value = 0
    dut.is_matrix.value = 0
    dut.preload_buf_id.value = 0
    dut.relu_flag.value = 0
    for i in range(TILE_ELEMS):
        dut.preload_data[i].value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def _preload(dut, buf_id, num_elements, data, is_matrix):
    """Stream `data` into the wrapper's buffer-preload port, TILE_ELEMS at a time."""
    dut.preload_en.value = 1
    dut.is_matrix.value = 1 if is_matrix else 0
    dut.preload_buf_id.value = buf_id

    cycles = (num_elements + TILE_ELEMS - 1) // TILE_ELEMS
    for c in range(cycles):
        for i in range(TILE_ELEMS):
            idx = c * TILE_ELEMS + i
            v = int(data[idx]) if idx < len(data) else 0
            dut.preload_data[i].value = v
        await RisingEdge(dut.clk)
    dut.preload_en.value = 0
    await RisingEdge(dut.clk)


def _golden_conv2d_output(x, w, b, fmap_h, fmap_w, in_c, out_c, kh, kw,
                          stride, pad, apply_relu):
    """Run the golden model on the same int8 tensors and return the int8 output."""
    golden_model.buffers = {}
    golden_model.flag = 0
    golden_model.pending_conv_config = {}
    golden_model.buffers[0] = list(np.asarray(x, dtype=np.int8))
    golden_model.buffers[1] = list(np.asarray(w, dtype=np.int8))
    golden_model.buffers[2] = list(np.asarray(b, dtype=np.int8))
    golden_model.conv2d(
        dest=3, w=1, x=0, b=2,
        fmap_h=fmap_h, fmap_w=fmap_w, in_c=in_c, out_c=out_c,
        kh=kh, kw=kw, stride=stride, pad=pad, apply_relu=apply_relu
    )
    return np.array(golden_model.buffers[3], dtype=np.int8)


async def _run_conv_case(dut, *, relu_flag):
    """Drive one Conv2D run end-to-end and return the dest-buffer contents."""
    # Geometry — small, deterministic case that exercises both signs of accumulator.
    fmap_h, fmap_w = 4, 4
    in_c, out_c   = 1, 1
    kh, kw        = 3, 3
    stride        = 1
    pad           = 1

    # Weights chosen so the int32 accumulator straddles zero — both relu_flag values
    # therefore differ in the negative half (relu_flag=1 clamps to 0).
    w_data = np.array([1, -2, 1, 1, 0, -1, 1, -2, 1], dtype=np.int8)
    x_data = np.array([
        [ 1,  2, -3,  4],
        [-5,  6,  7, -8],
        [ 9, 10, 11, 12],
        [13, 14, 15, 16],
    ], dtype=np.int8).flatten()
    b_data = np.array([2], dtype=np.int8)

    W_BUF, X_BUF, B_BUF, OUT_BUF = 1, 2, 3, 4

    # Reset
    await _reset(dut)

    # Preload buffers via the wrapper's preload bus.
    await _preload(dut, W_BUF, len(w_data), w_data.tolist(), is_matrix=True)
    await _preload(dut, X_BUF, len(x_data), x_data.tolist(), is_matrix=False)
    await _preload(dut, B_BUF, len(b_data), b_data.tolist(), is_matrix=False)

    # Drive geometry + start.
    dut.fmap_h.value       = fmap_h
    dut.fmap_w.value       = fmap_w
    dut.in_channels.value  = in_c
    dut.out_channels.value = out_c
    dut.kernel_h.value     = kh
    dut.kernel_w.value     = kw
    dut.stride_val.value   = stride
    dut.pad_val.value      = pad
    dut.relu_flag.value    = 1 if relu_flag else 0

    dut.w_buffer_id.value    = W_BUF
    dut.x_buffer_id.value    = X_BUF
    dut.b_buffer_id.value    = B_BUF
    dut.dest_buffer_id.value = OUT_BUF

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for done with a generous timeout.
    timeout = 50_000
    while int(dut.done.value) == 0:
        await RisingEdge(dut.clk)
        timeout -= 1
        if timeout <= 0:
            raise cocotb.result.TestFailure("conv2d_execution timed out before DONE")

    # Drain buffer at OUT_BUF by reading the vector_buffer_inst's storage array
    # directly. Each buffer is BUFFER_WIDTH bits wide (8192 by default), packed
    # 8 bits per element. We slice the int8 values out of the packed word.
    out_h = (fmap_h + 2 * pad - kh) // stride + 1
    out_w = (fmap_w + 2 * pad - kw) // stride + 1
    n     = out_c * out_h * out_w

    # Settle a couple cycles after `done` so the last writes are visible.
    for _ in range(2):
        await RisingEdge(dut.clk)

    raw = int(dut.buffers.vector_buffer_inst.buffers[OUT_BUF].value)
    dut_out = []
    for addr in range(n):
        byte = (raw >> (addr * 8)) & 0xFF
        if byte >= 0x80:
            byte -= 0x100
        dut_out.append(byte)
    return np.array(dut_out, dtype=np.int8), {
        "x": x_data, "w": w_data, "b": b_data,
        "fmap_h": fmap_h, "fmap_w": fmap_w,
        "in_c": in_c, "out_c": out_c,
        "kh": kh, "kw": kw, "stride": stride, "pad": pad,
    }


@cocotb.test()
async def test_conv2d_relu_flag_0(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    actual, geom = await _run_conv_case(dut, relu_flag=False)
    expected = _golden_conv2d_output(
        geom["x"], geom["w"], geom["b"],
        geom["fmap_h"], geom["fmap_w"], geom["in_c"], geom["out_c"],
        geom["kh"], geom["kw"], geom["stride"], geom["pad"],
        apply_relu=False,
    )
    np.testing.assert_array_equal(
        actual, expected,
        err_msg=f"relu_flag=0 mismatch: dut={actual.tolist()} expected={expected.tolist()}"
    )


@cocotb.test()
async def test_conv2d_relu_flag_1(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    actual, geom = await _run_conv_case(dut, relu_flag=True)
    expected = _golden_conv2d_output(
        geom["x"], geom["w"], geom["b"],
        geom["fmap_h"], geom["fmap_w"], geom["in_c"], geom["out_c"],
        geom["kh"], geom["kw"], geom["stride"], geom["pad"],
        apply_relu=True,
    )
    # Sanity: with apply_relu=True, no element should be negative.
    assert np.all(actual >= 0), \
        f"relu_flag=1 produced negatives: {actual.tolist()}"
    np.testing.assert_array_equal(
        actual, expected,
        err_msg=f"relu_flag=1 mismatch: dut={actual.tolist()} expected={expected.tolist()}"
    )
