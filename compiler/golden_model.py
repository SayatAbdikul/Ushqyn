""" 
Golden model of the accelerator.
Emulates all the instructions.

ISA opcode table:
  0x01  LOAD_V       – load vector from DRAM to buffer
  0x02  LOAD_M       – load matrix from DRAM to buffer
  0x03  STORE        – write buffer to DRAM
  0x04  GEMV         – matrix-vector multiply + int8 quantization
  0x05  RELU         – element-wise ReLU
  0x06  CONV2D_CFG   – configure conv2d geometry (precedes CONV2D_RUN)
  0x07  CONV2D_RUN   – execute conv2d using the pending geometry config
  0x08  MAXPOOL      – sliding-window max-pooling
"""
import os
import numpy as np
from dram import get_dram
from helper_functions import quantize_int32_to_int8, quantize_int32_to_int8_rtl_exact
from accelerator_config import AcceleratorConfig

# Set GOLDEN_DEBUG=1 in the env to enable per-op trace prints in gemv/conv2d/maxpool.
DEBUG_GOLDEN = os.environ.get("GOLDEN_DEBUG", "0") == "1"

# ── Global state ─────────────────────────────────────────────────────────────
buffers = {}
output_length = AcceleratorConfig.OUT_N
quantized_output_scale = 0.1
quantized_output_zero_point = 0
output_buffer = 0

# Holds geometry fields from the most recent CONV2D_CFG instruction.
# CONV2D_RUN reads from this dict.
pending_conv_config = {}


# ── Memory loading ────────────────────────────────────────────────────────────
def load_memory(dram_file, use_file=True):
    """Load memory from a hex file or from in-memory DRAM state.

    Args:
        dram_file: Path to the hex file to load from.
        use_file:  If True, read from file.  If False, use in-memory DRAM state.

    Returns:
        np.array of int8 values representing memory contents.
    """
    if not use_file:
        return get_dram()

    memory = []
    with open(dram_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                val = int(line, 16)
                memory.append(np.int8(np.uint8(val)))
    return np.array(memory, dtype=np.int8)


# ── Instruction decoder ────────────────────────────────────────────────────────
def i_decoder(instruction):
    opcode = instruction & 0x1F  # bits [4:0]

    if opcode == 1:  # LOAD_V
        dest = instruction >> 5  & 0x1F
        length = instruction >> 10 & 0x3FFFF  # 18 bits
        addr   = instruction >> 40 & 0xFFFFFF
        load_v(dest, addr, length)

    elif opcode == 2:  # LOAD_M
        dest = instruction >> 5  & 0x1F
        cols = instruction >> 10 & 0x3FF
        rows = instruction >> 20 & 0x3FF
        addr = instruction >> 40 & 0xFFFFFF
        load_m(dest, addr, rows, cols)

    elif opcode == 3:  # STORE
        buf_id = instruction >> 5  & 0x1F
        length = instruction >> 10 & 0x3FFFF
        addr   = instruction >> 40 & 0xFFFFFF
        store(buf_id, addr, length)

    elif opcode == 4:  # GEMV
        dest = instruction >> 5  & 0x1F
        cols = instruction >> 10 & 0x3FF
        rows = instruction >> 20 & 0x3FF
        b    = instruction >> 30 & 0x1F
        x    = instruction >> 35 & 0x1F
        w    = instruction >> 40 & 0x1F
        gemv(dest, w, x, b, rows, cols)

    elif opcode == 5:  # RELU
        dest   = instruction >> 5  & 0x1F
        x      = instruction >> 10 & 0x1F
        length = instruction >> 20 & 0x3FF  # 10-bit (≤1023 elements; FC outputs only)
        relu(dest, x, length)

    elif opcode == 6:  # CONV2D_CFG
        # Decodes geometry; does NOT modify buffers.  Next CONV2D_RUN will use this config.
        # Bit layout (matches assembler.py):
        #   [ 4: 0] opcode   [ 9: 5] dest
        #   [15:10] fmap_h   [21:16] fmap_w
        #   [27:22] in_c     [33:28] out_c
        #   [37:34] kh       [41:38] kw
        #   [44:42] stride   [47:45] pad
        global pending_conv_config
        pending_conv_config = {
            'dest'  : instruction >>  5 & 0x1F,
            'fmap_h': instruction >> 10 & 0x3F,
            'fmap_w': instruction >> 16 & 0x3F,
            'in_c'  : instruction >> 22 & 0x3F,
            'out_c' : instruction >> 28 & 0x3F,
            'kh'    : instruction >> 34 & 0x0F,
            'kw'    : instruction >> 38 & 0x0F,
            'stride': instruction >> 42 & 0x07,
            'pad'   : instruction >> 45 & 0x07,
        }

    elif opcode == 7:  # CONV2D_RUN
        # Bit layout:
        #   [ 4: 0] opcode  [ 9: 5] dest
        #   [14:10] x_id    [19:15] w_id
        #   [24:20] b_id    [25]    relu_flag
        dest      = instruction >>  5 & 0x1F
        x_id      = instruction >> 10 & 0x1F
        w_id      = instruction >> 15 & 0x1F
        b_id      = instruction >> 20 & 0x1F
        relu_flag = bool(instruction >> 25 & 0x01)
        # Geometry comes from the most recent CONV2D_CFG.
        cfg = pending_conv_config
        if DEBUG_GOLDEN:
            print(f"[DBG_CONV2D_RUN] dest={dest} x_id={x_id} w_id={w_id} "
                  f"b_id={b_id} relu={relu_flag} cfg={cfg}")
        conv2d(
            dest   = dest,
            w      = w_id,
            x      = x_id,
            b      = b_id,
            fmap_h = cfg['fmap_h'],
            fmap_w = cfg['fmap_w'],
            in_c   = cfg['in_c'],
            out_c  = cfg['out_c'],
            kh     = cfg['kh'],
            kw     = cfg['kw'],
            stride = cfg['stride'],
            pad    = cfg['pad'],
            apply_relu = relu_flag,
        )

    elif opcode == 8:  # MAXPOOL
        # Bit layout:
        #   [ 4: 0] opcode  [ 9: 5] dest   [14:10] x_id
        #   [17:15] pool_size  [20:18] stride
        #   [26:21] fmap_h  [32:27] fmap_w  [37:33] channels
        dest      = instruction >>  5 & 0x1F
        x_id      = instruction >> 10 & 0x1F
        pool_size = instruction >> 15 & 0x07
        stride    = instruction >> 18 & 0x07
        fmap_h    = instruction >> 21 & 0x3F
        fmap_w    = instruction >> 27 & 0x3F
        channels  = instruction >> 33 & 0x1F
        maxpool(dest, x_id, fmap_h, fmap_w, channels, pool_size, stride)

    else:
        return f"UNKNOWN_OPCODE: {opcode}"


# ── Buffer / DRAM instructions ────────────────────────────────────────────────
def load_v(dest, addr, length):
    """Load vector from memory to buffer."""
    buffers[dest] = memory[addr:addr + length]


def load_m(dest, addr, rows, cols):
    """Load matrix from DRAM into buffer `dest`.

    DRAM contract: weight regions are stored row-major with cols padded to
    a multiple of AcceleratorConfig.TILE_ELEMS. The instruction's `cols`
    field is the LOGICAL (unpadded) column count for the matrix; this
    function reads rows * pad(cols) bytes from DRAM, reshapes, and drops
    padding to keep only [:, :cols] in `buffers[dest]`.

    compile.py asymmetry (load-bearing for both paths):
      • FC weights — compile.py emits LOAD_M with cols ALREADY padded.
        load_m drops nothing; buffers[w] holds the padded matrix flat,
        and gemv() strides over it using stride=pad(cols).
      • Conv weights — compile.py emits LOAD_M with the UNPADDED cols
        (in_C * kH * kW). load_m drops the padding here; buffers[w]
        holds the unpadded matrix flat, which conv2d() reshapes as
        [out_C, in_C, kH, kW].

    Do not "fix" this asymmetry without simultaneously updating gemv()'s
    stride logic and conv2d()'s reshape.
    """
    global buffers
    TILE_WIDTH = AcceleratorConfig.TILE_ELEMS
    padded_cols = ((cols + TILE_WIDTH - 1) // TILE_WIDTH) * TILE_WIDTH
    # Memory region is padded, so read rows * padded_cols
    transfer_length = rows * padded_cols
    raw_data = memory[addr:addr + transfer_length]

    # Reshape and drop padding
    matrix = np.array(raw_data).reshape(rows, padded_cols)
    matrix = matrix[:, :cols]

    # Flatten it back into the buffer
    buffers[dest] = matrix.flatten().tolist()


def store(buf_id, addr, length):
    """Store buffer to memory."""
    for i in range(length):
        memory[addr + i] = buffers[buf_id][i]
    global output_buffer
    output_buffer = buf_id


# ── Compute instructions ───────────────────────────────────────────────────────
flag = 0

def gemv(dest, w, x, b, rows, cols):
    """Matrix-vector multiply with int8 output quantization.

    `cols` is the LOGICAL (unpadded) column count of the weight matrix
    — same convention as conv2d. The weight buffer `buffers[w]`, however,
    holds the matrix flat with cols PADDED to TILE_ELEMS columns: see
    load_m()'s docstring for the FC-vs-Conv asymmetry. We compute
    stride = pad(cols) and skip the trailing zero-padded columns by
    iterating j in [0, cols).
    """
    global flag
    buffers[dest] = [0] * rows

    TILE_WIDTH = AcceleratorConfig.TILE_ELEMS
    stride = ((cols + TILE_WIDTH - 1) // TILE_WIDTH) * TILE_WIDTH

    if DEBUG_GOLDEN and flag < 3:
        print(f"[DBG_GOLDEN] GEMV start: rows={rows}, cols={cols}")

    for i in range(rows):
        sum_val = np.int32(0)
        for j in range(cols):
            sum_val += np.int32(buffers[w][i * stride + j]) * np.int32(buffers[x][j])
        sum_val += np.int32(buffers[b][i])

        if DEBUG_GOLDEN and flag < 3 and i < 2:
            print(f"[DBG_GOLDEN] ACCUM row={i} bias={buffers[b][i]} final_sum={sum_val}")

        buffers[dest][i] = np.int32(sum_val)

    flag += 1

    max_abs = np.max(np.abs(buffers[dest]))
    if DEBUG_GOLDEN and flag <= 3:
        print(f"[DBG_GOLDEN] COMPUTE_SCALE: max_abs={max_abs}")

    buffers[dest] = quantize_int32_to_int8_rtl_exact(
        np.array(buffers[dest], dtype=np.int32),
        max_abs,
        quantized_output_zero_point
    )


def relu(dest, x, length):
    """Apply ReLU activation to the first `length` elements."""
    buffers[dest] = [max(0, val) for val in buffers[x][:length]]


def conv2d(dest, w, x, b, fmap_h, fmap_w, in_c, out_c, kh, kw, stride, pad,
           apply_relu=False):
    """Direct 2-D convolution reference (NCHW layout).

    Weight buffer layout : [out_c, in_c, kh, kw]  (row-major, flat in buffer —
                          UNPADDED, see load_m()'s docstring for why)
    Input  buffer layout : [in_c,  fmap_h, fmap_w] (row-major, flat in buffer)
    Output buffer layout : [out_c, out_h,  out_w]  (row-major, flat in buffer)

    Quantization: same RTL-exact path as GEMV (per-tensor max-abs scaling).
    ReLU is optionally applied *after* quantization when apply_relu=True.
    """
    # Reconstruct nd-arrays from flat buffers
    x_flat = np.array(buffers[x], dtype=np.int32)
    w_flat = np.array(buffers[w], dtype=np.int32)
    b_flat = np.array(buffers[b], dtype=np.int32)

    x_data = x_flat.reshape(in_c, fmap_h, fmap_w)
    w_data = w_flat.reshape(out_c, in_c, kh, kw)

    out_h = (fmap_h + 2 * pad - kh) // stride + 1
    out_w = (fmap_w + 2 * pad - kw) // stride + 1

    # Zero-pad the input if needed
    if pad > 0:
        x_padded = np.pad(x_data, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
    else:
        x_padded = x_data

    # Direct convolution
    output = np.zeros((out_c, out_h, out_w), dtype=np.int32)
    for oc in range(out_c):
        for oh in range(out_h):
            for ow in range(out_w):
                acc = np.int32(0)
                for ic in range(in_c):
                    for khi in range(kh):
                        for kwi in range(kw):
                            acc += (np.int32(w_data[oc, ic, khi, kwi]) *
                                    np.int32(x_padded[ic, oh * stride + khi, ow * stride + kwi]))
                output[oc, oh, ow] = acc + b_flat[oc]
                
                if DEBUG_GOLDEN:
                    # Targeted single-element trace for layer 1 at (oc=2, oh=9, ow=18)
                    if dest == 10 and oc == 2 and oh == 9 and ow == 18:
                        print(f"[DBG_GOLDEN_CONV] TARGET oc=2, oh=9, ow=18: accum={acc} bias={b_flat[oc]} final_sum={output[oc, oh, ow]}")
                        print(f"[DBG_GOLDEN_CONV] TARGET x_window = {x_padded[:, oh*stride:oh*stride+kh, ow*stride:ow*stride+kw].flatten()}")
                        print(f"[DBG_GOLDEN_CONV] TARGET w_window = {w_data[oc, :, :, :].flatten()}")

                    # Targeted single-element trace for layer 2 at (oc=0, oh=0, ow=0)
                    if dest == 11 and oc == 0 and oh == 0 and ow == 0:
                        x_win = x_padded[:, oh*stride:oh*stride+kh, ow*stride:ow*stride+kw].flatten()
                        w_win = w_data[oc, :, :, :].flatten()
                        acc0 = np.sum(np.int32(x_win[0:32]) * np.int32(w_win[0:32]))
                        acc1 = np.sum(np.int32(x_win[32:36]) * np.int32(w_win[32:36]))
                        print(f"[DBG_GOLDEN_CONV] LAYER2 oc=0, oh=0, ow=0: accum={acc} bias={b_flat[oc]} final_sum={output[oc, oh, ow]}")
                        print(f"[DBG_GOLDEN_CONV] LAYER2 acc0(0-31) = {acc0}, acc1(32-35) = {acc1}")

    # Per-tensor RTL-exact quantization (same pipeline as GEMV)
    max_abs  = int(np.max(np.abs(output)))
    if DEBUG_GOLDEN:
        max_idx = np.argmax(np.abs(output))
        print(f"[DBG_GOLDEN_CONV] COMPUTE_SCALE: max_abs={max_abs} at index={max_idx}")
    quantized = quantize_int32_to_int8_rtl_exact(
        output.flatten().astype(np.int32), max_abs, 0
    )

    if apply_relu:
        quantized = np.maximum(quantized, np.int8(0))

    if DEBUG_GOLDEN:
        print(f"[DBG_GOLDEN_CONV] dest={dest} output max={np.max(quantized)}, min={np.min(quantized)}, mean={np.mean(quantized):.2f}")
    buffers[dest] = quantized.tolist()


def maxpool(dest, x, fmap_h, fmap_w, channels, pool_size, stride):
    """Sliding-window max-pooling (NCHW layout).

    Input  buffer layout : [channels, fmap_h, fmap_w]
    Output buffer layout : [channels, out_h,  out_w]

    No quantization – operates on int8 values directly.
    """
    x_flat = np.array(buffers[x], dtype=np.int8)
    x_data = x_flat.reshape(channels, fmap_h, fmap_w)

    out_h = (fmap_h - pool_size) // stride + 1
    out_w = (fmap_w - pool_size) // stride + 1

    output = np.full((channels, out_h, out_w), fill_value=-128, dtype=np.int8)
    for c in range(channels):
        for oh in range(out_h):
            for ow in range(out_w):
                window = x_data[c,
                                oh * stride : oh * stride + pool_size,
                                ow * stride : ow * stride + pool_size]
                output[c, oh, ow] = np.max(window)

    if DEBUG_GOLDEN:
        print(f"[DBG_GOLDEN_POOL] dest={dest} output max={np.max(output)}, min={np.min(output)}")
    buffers[dest] = output.flatten().tolist()


# ── Program execution ─────────────────────────────────────────────────────────
def execute_program(hex_file, use_in_memory=False):
    """Execute the program from a hex file.

    Args:
        hex_file:      Path to the hex file containing program + data.
        use_in_memory: If True, use the in-memory DRAM state (skip file load).

    Returns:
        Output buffer slice of length OUT_N.
    """
    global buffers, output_buffer, flag, memory, pending_conv_config
    buffers = {}
    output_buffer = 0
    flag = 0
    pending_conv_config = {}

    with open(hex_file, 'r') as file:
        lines = [line.strip() for _, line in
                 zip(range(AcceleratorConfig.DRAM_ADDR_INPUTS), file)]
        instructions = [''.join(lines[i:i+8]) for i in range(0, len(lines), 8)]
        instructions = [int(instr, 16) for instr in instructions if instr]

    memory = load_memory(hex_file, use_file=not use_in_memory)

    for instruction in instructions:
        i_decoder(instruction)

    return buffers[output_buffer][0:output_length]
