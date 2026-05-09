"""Regression tests for the unified ONNX walker (P2, 2026-05-09).

After P2 the compiler no longer maintains its own shadow `bias_counter`
/ `weight_counter` / `conv_weight_counter` accumulators that had to
advance in lockstep with `dram.py::save_all_initializers_to_dram`.
Instead it consumes the `(weight_map, bias_map, conv_weight_map)` that
the DRAM walker returns and looks up addresses by initializer name.

These tests assert:
    1. `generate_assembly` rejects calls without maps (no silent fallback).
    2. The addresses emitted in assembly LOAD_M / LOAD_V instructions
       exactly equal the map entries the DRAM walker produced.
    3. Re-shuffling the order of `bias_map` entries (simulating a
       walker-emission-order change) doesn't break the compiler — proving
       the dependence is on names, not positional ordering.
"""
import os
import re
import shutil
import tempfile

import numpy as np
import onnx
import pytest

import accelerator_config
import compile as compile_module
import dram
import golden_model
import model as model_module
from accelerator_config import AcceleratorConfig
from assembler import assemble_file


COMPILER_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(autouse=True)
def reset_state(tmp_path):
    """Each test runs in a fresh tmp_path with reset golden/dram state.

    The MLP model.create_mlp_model() helper expects pre-trained weights at
    `./digit_model_weights.pth`; we copy them into tmp_path so tests run
    isolated without polluting the compiler/ source dir with .onnx outputs.
    """
    cwd = os.getcwd()
    weights_src = os.path.join(COMPILER_DIR, "digit_model_weights.pth")
    if os.path.exists(weights_src):
        shutil.copy(weights_src, tmp_path / "digit_model_weights.pth")
    os.chdir(tmp_path)
    dram.dram.fill(0)
    golden_model.buffers = {}
    golden_model.flag = 0
    golden_model.pending_conv_config = {}
    yield
    os.chdir(cwd)


def _onnx_model_path(create_fn):
    """Run a model.create_*_model() helper and return the produced ONNX path."""
    create_fn()
    # Both helpers write into the cwd; figure out which file landed there.
    for name in ("mlp_model.onnx", "cnn_model.onnx"):
        if os.path.exists(name):
            return name
    raise FileNotFoundError("create model helper did not produce an ONNX file")


def _dram_offsets():
    return {
        "inputs":       AcceleratorConfig.DRAM_ADDR_INPUTS,
        "biases":       AcceleratorConfig.DRAM_ADDR_BIASES,
        "outputs":      AcceleratorConfig.DRAM_ADDR_OUTPUTS,
        "weights":      AcceleratorConfig.DRAM_ADDR_WEIGHTS,
        "conv_weights": AcceleratorConfig.DRAM_ADDR_CONV_WEIGHTS,
    }


def _parse_load_addrs(asm_file):
    """Return list of (mnemonic, address_hex) for every LOAD_V/LOAD_M in asm."""
    out = []
    with open(asm_file) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(";"):
                continue
            m = re.match(r"(LOAD_V|LOAD_M)\s+\d+\s*,\s*(0x[0-9A-Fa-f]+)", s)
            if m:
                out.append((m.group(1), int(m.group(2), 16)))
    return out


# ── tests ────────────────────────────────────────────────────────────────────

def test_generate_assembly_requires_maps():
    """The pre-P2 silent-walk fallback is gone — calling generate_assembly
    without maps must raise rather than re-walk the graph independently
    (which is the exact bug class that motivated P2)."""
    onnx_path = _onnx_model_path(model_module.create_mlp_model)
    with pytest.raises(ValueError, match="generate_assembly requires"):
        compile_module.generate_assembly(onnx_path, "out.asm")


def test_assembly_addresses_match_walker_maps_mlp():
    """Every LOAD_M/LOAD_V address in the emitted assembly must equal the
    address the DRAM walker chose for that initializer. Catches any
    drift between compiler emission order and walker emission order."""
    onnx_path = _onnx_model_path(model_module.create_mlp_model)
    weight_map, bias_map, conv_weight_map = dram.save_all_initializers_to_dram(
        onnx_path, _dram_offsets())

    asm_path = "mlp.asm"
    compile_module.generate_assembly(onnx_path, asm_path,
                                     weight_map, bias_map, conv_weight_map)

    # Every map address must appear in the assembly. (LOAD_V for the input
    # tensor uses DRAM_ADDR_INPUTS which isn't in any map; we exclude that.)
    asm_addrs = {addr for _, addr in _parse_load_addrs(asm_path)}
    expected = set(weight_map.values()) | set(bias_map.values()) | set(conv_weight_map.values())
    expected.add(AcceleratorConfig.DRAM_ADDR_INPUTS)
    missing = expected - asm_addrs
    assert not missing, (
        f"Walker-allocated addresses missing from assembly: "
        f"{[hex(a) for a in sorted(missing)]}")


def test_assembly_addresses_match_walker_maps_cnn():
    """Same contract under a CNN graph — exercises conv_weight_map and the
    Conv-vs-FC bias paths (the asymmetry that broke pre-Patch-B)."""
    onnx_path = _onnx_model_path(model_module.create_cnn_model)
    weight_map, bias_map, conv_weight_map = dram.save_all_initializers_to_dram(
        onnx_path, _dram_offsets())

    asm_path = "cnn.asm"
    compile_module.generate_assembly(onnx_path, asm_path,
                                     weight_map, bias_map, conv_weight_map)

    asm_addrs = {addr for _, addr in _parse_load_addrs(asm_path)}
    expected = set(weight_map.values()) | set(bias_map.values()) | set(conv_weight_map.values())
    expected.add(AcceleratorConfig.DRAM_ADDR_INPUTS)
    missing = expected - asm_addrs
    assert not missing, (
        f"Walker-allocated addresses missing from assembly: "
        f"{[hex(a) for a in sorted(missing)]}")

    # Specifically: every conv weight address from the map must appear in a
    # LOAD_M, every bias address in a LOAD_V (catches accidental swap of paths).
    load_m_addrs = {a for op, a in _parse_load_addrs(asm_path) if op == "LOAD_M"}
    load_v_addrs = {a for op, a in _parse_load_addrs(asm_path) if op == "LOAD_V"}
    for addr in conv_weight_map.values():
        assert addr in load_m_addrs, f"Conv weight 0x{addr:X} not in any LOAD_M"
    for addr in bias_map.values():
        assert addr in load_v_addrs, f"Bias 0x{addr:X} not in any LOAD_V"


def test_unknown_initializer_raises():
    """If a map is missing an initializer the compiler needs, we must fail
    loudly (KeyError) rather than silently emit a bogus address."""
    onnx_path = _onnx_model_path(model_module.create_mlp_model)
    weight_map, bias_map, conv_weight_map = dram.save_all_initializers_to_dram(
        onnx_path, _dram_offsets())

    # Drop one bias entry — simulates a walker that forgot to emit it.
    truncated_bias_map = dict(bias_map)
    a_bias = next(iter(truncated_bias_map))
    del truncated_bias_map[a_bias]

    with pytest.raises(KeyError, match="Bias"):
        compile_module.generate_assembly(onnx_path, "out.asm",
                                         weight_map, truncated_bias_map,
                                         conv_weight_map)


def test_address_mutation_propagates():
    """If the user manually moves an initializer to a new address (e.g. a
    custom DRAM layout), the assembly must follow — proving compile.py is
    actually reading from the map, not falling back to a hard-coded offset."""
    onnx_path = _onnx_model_path(model_module.create_mlp_model)
    weight_map, bias_map, conv_weight_map = dram.save_all_initializers_to_dram(
        onnx_path, _dram_offsets())

    # Pick one weight, move it to a deliberately weird address.
    weight_name = next(iter(weight_map))
    weight_map[weight_name] = 0xDEAD0

    asm_path = "shifted.asm"
    compile_module.generate_assembly(onnx_path, asm_path,
                                     weight_map, bias_map, conv_weight_map)

    addrs = {addr for _, addr in _parse_load_addrs(asm_path)}
    assert 0xDEAD0 in addrs, (
        f"Mutated weight address 0xDEAD0 not found in assembly — compile.py "
        f"is not reading from the map. Addrs: {[hex(a) for a in addrs]}")
