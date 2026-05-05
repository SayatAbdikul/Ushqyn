""" dram.py - A module for managing DRAM in a custom architecture.
This module provides functions to read and write data to a simulated DRAM,
save initializers from an ONNX model, and manage memory operations.
It includes functions to handle quantization of tensors, write to DRAM,
and read from DRAM."""

import numpy as np
import onnx
from onnx import numpy_helper
from helper_functions import quantize_tensor_f32_int8
from top_sort import topological_sort
from accelerator_config import AcceleratorConfig

MEM_SIZE = AcceleratorConfig.MEM_SIZE  # Total memory size (Reduced to 60KB for FPGA fit)
dram = np.zeros(MEM_SIZE, dtype=np.int8)

def write_to_dram(array, start_addr):
    end_addr = start_addr + len(array)
    # Check for overflow but allow overwriting (warning optional or removed for repeated runs)
    if end_addr > len(dram):
        print(f"DRAM overflow: trying to write {len(array)} bytes at address {hex(start_addr)}")
        raise ValueError("DRAM overflow")
        
    dram[start_addr:end_addr] = array
    # print(f"Written {len(array)} bytes to DRAM at address {hex(start_addr)}")
    return end_addr  # Return next free address

def read_from_dram(start_addr, length):
    end_addr = start_addr + length
    if end_addr > len(dram):
        print(f"DRAM overflow: trying to read {length} bytes from address {hex(start_addr)}")
        raise ValueError("DRAM overflow")
    data = dram[start_addr:end_addr]
    # print(f"Read {length} bytes from DRAM at address {hex(start_addr)}")
    return np.array(dram[start_addr:end_addr], dtype=np.int8)

def get_dram():
    return dram.copy()  # Return a copy to avoid external modifications

def save_all_initializers_to_dram(model_path, dram_offsets):
    """Single-pass DRAM walker — the source of truth for initializer layout.

    Walks ONNX nodes in topological order and writes initializers per node
    type into a SHARED bias region so FC and Conv biases never overlap.
    Mirrors compile.py's emission order so bias_counter (compile.py) and
    bias_ptr (here) stay in lockstep.

    DRAM layout produced (offsets from dram_offsets):
        weights      → FC / Gemm / MatMul weights, padded to TILE_ELEMS cols
        conv_weights → Conv weights [out_C, in_C*kH*kW], padded to TILE_ELEMS cols
        biases       → ALL biases in topological order:
                       [conv1.bias | conv2.bias | … | fc.bias | …]

    Skips Reshape, Flatten, BatchNormalization (no DRAM footprint).

    Returns:
        (weight_map, bias_map, conv_weight_map) — each dict maps
        initialiser name → DRAM start address.
    """
    global dram
    dram = np.zeros(MEM_SIZE, dtype=np.int8)
    model = onnx.load(model_path)
    graph = model.graph

    weight_ptr      = dram_offsets["weights"]
    conv_weight_ptr = dram_offsets.get("conv_weights",
                                       AcceleratorConfig.DRAM_ADDR_CONV_WEIGHTS)
    bias_ptr        = dram_offsets["biases"]

    weight_map      = {}
    bias_map        = {}
    conv_weight_map = {}

    initializer_data = {init.name: init for init in graph.initializer}
    visited          = set()
    TILE_WIDTH       = AcceleratorConfig.TILE_ELEMS

    for node in topological_sort(graph):
        if node.op_type in ("Reshape", "Flatten", "BatchNormalization"):
            continue

        if node.op_type == "Conv":
            # Conv inputs are ordered [activation, weight, bias]; iterate so
            # weight is written before bias (matches compile.py emission).
            for input_name in node.input:
                if input_name not in initializer_data or input_name in visited:
                    continue
                visited.add(input_name)
                array = numpy_helper.to_array(initializer_data[input_name])
                scale = (np.max(np.abs(array)) / 127
                         if np.max(np.abs(array)) > 0 else 1.0)
                q_arr = np.clip(np.round(array / scale), -128, 127).astype(np.int8)

                if array.ndim > 1:        # conv weight [out_C, in_C, kH, kW]
                    out_c = q_arr.shape[0]
                    cols  = int(np.prod(q_arr.shape[1:]))
                    q_2d  = q_arr.reshape(out_c, cols)
                    pad_cols = (TILE_WIDTH - (cols % TILE_WIDTH)) % TILE_WIDTH
                    q_padded = (np.pad(q_2d, ((0, 0), (0, pad_cols)), mode='constant')
                                if pad_cols else q_2d)
                    conv_weight_map[input_name] = conv_weight_ptr
                    conv_weight_ptr = write_to_dram(q_padded.flatten(), conv_weight_ptr)
                else:                     # conv bias [out_C]
                    bias_map[input_name] = bias_ptr
                    bias_ptr = write_to_dram(q_arr.flatten(), bias_ptr)
            continue

        # FC / Gemm / MatMul / etc. — generic path
        for input_name in node.input:
            if input_name not in initializer_data or input_name in visited:
                continue
            visited.add(input_name)
            array = numpy_helper.to_array(initializer_data[input_name])
            scale = (np.max(np.abs(array)) / 127
                     if np.max(np.abs(array)) > 0 else 1.0)

            if array.ndim > 1:   # weight matrix [rows, cols]
                rows, cols = array.shape
                padded_cols = ((cols + TILE_WIDTH - 1) // TILE_WIDTH) * TILE_WIDTH
                padded = np.zeros((rows, padded_cols), dtype=np.int8)
                padded[:, :cols] = quantize_tensor_f32_int8(array, scale)
                if np.count_nonzero(padded[:, cols:]) > 0:
                    print(f"ERROR: padding non-zero for {input_name}")
                weight_map[input_name] = weight_ptr
                weight_ptr = write_to_dram(padded.flatten(), weight_ptr)
            else:                # bias vector
                q = quantize_tensor_f32_int8(array, scale).flatten()
                bias_map[input_name] = bias_ptr
                bias_ptr = write_to_dram(q, bias_ptr)

    return weight_map, bias_map, conv_weight_map


def save_initializers_to_dram(model_path, dram_offsets):
    """Backwards-compat wrapper around save_all_initializers_to_dram.

    Returns (weight_map, bias_map). For pure-MLP graphs the produced DRAM
    image is byte-identical to the previous implementation. For graphs
    containing Conv nodes, conv weights are now ALSO written (to the
    conv_weights region) — previously this function silently skipped them,
    which forced callers to also call save_conv_weights_to_dram and led
    to bias-region overlap. New code should call
    save_all_initializers_to_dram directly.
    """
    w, b, _ = save_all_initializers_to_dram(model_path, dram_offsets)
    return w, b


def save_conv_weights_to_dram(model_path, dram_offsets):
    """Backwards-compat wrapper. Returns (conv_weight_map, conv_bias_map).

    Idempotent if save_all_initializers_to_dram (or save_initializers_to_dram)
    already ran on the same model — the unified walker writes everything in
    one pass, so a second call here produces the same DRAM image.

    NOTE: this used to reset bias_ptr to dram_offsets["biases"] and overwrite
    whatever save_initializers_to_dram had previously written there. The
    unified walker now owns bias-region layout, so calling this function
    after save_initializers_to_dram is safe (no overlap).
    """
    model = onnx.load(model_path)
    conv_inputs = {n for node in model.graph.node
                   if node.op_type == "Conv" for n in node.input}
    _, bias_map, conv_weight_map = save_all_initializers_to_dram(model_path, dram_offsets)
    conv_bias_map = {k: v for k, v in bias_map.items() if k in conv_inputs}
    return conv_weight_map, conv_bias_map

def save_input_to_dram(input_tensor, addr):
    """Write a model input tensor to DRAM.
    
    Accepts a PyTorch tensor or numpy array of any shape (e.g. (28,28),
    (1,28,28), (1,1,28,28)).  Flattens to 1D, then quantizes floats to INT8
    using max-abs per-tensor scaling before writing.
    """
    import numpy as np
    if hasattr(input_tensor, 'numpy'):
        arr = input_tensor.numpy()
    else:
        arr = np.array(input_tensor)
    arr = arr.flatten().astype(np.float32)
    
    # Quantize float32 → int8 (match training normalization scale)
    max_abs = np.max(np.abs(arr))
    if max_abs > 0:
        scale = max_abs / 127.0
        q_arr = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
    else:
        q_arr = np.zeros(len(arr), dtype=np.int8)
    
    write_to_dram(q_arr, addr)

def save_dram_to_file(filename="dram.hex"):
    """Saves the current state of DRAM to a hex file."""
    counter = 0
    # May be commented out to avoid overwriting to file in this example on each input
    with open(filename, "w") as f:
        for byte in dram:
            # Convert signed int8 to unsigned for hex
            val = np.uint8(byte)
            f.write(f"{val:02X}\n")
            counter = counter + 1
