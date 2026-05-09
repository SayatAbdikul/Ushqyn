gtkwave dump.vcd
golden_buffer = golden_model.buffers[5]
def execute_program(hex_file):
differences = rtl_output - golden_output
# Ushqyn — Cocotb Golden Model Verification

## Summary

Single cocotb testbench that runs the Ushqyn RTL end-to-end and compares results against the Python golden model. The RTL is started once, runs through all instructions until the zero instruction is fetched, and its outputs are compared to the golden model using the same `dram.hex` contents.

## What This Test Does

1. Prepare `dram.hex` (instructions, weights, biases, inputs). The main test generates and assembles the model on the fly if needed and syncs the contents to all four RTL memories.
2. Apply reset and start a 100 MHz clock.
3. Pulse `start` once, let the RTL run until `done` asserts when the zero instruction is hit.
4. Read RTL outputs directly from the store memory (and also from `y[]` for debug).
5. Run the Python golden model (`execute_program`) on the same `dram.hex`.
6. Compare outputs element-by-element, accepting small quantization differences (max error ≤ 2).

## Files

- `test_golden_comparison.py` — cocotb tests:
  - `test_accelerator_mnist_dataset` (main end-to-end check on a small MNIST subset)
  - `test_single_instruction_load_v` (sanity check)
  - `test_reset_behavior` (reset check)
- `Makefile` — drives cocotb with Verilator; `make run_test` prepares `dram.hex` and runs the test.
- `run_test.py` — convenience wrapper (dependency check, generate `dram.hex` if missing, then run `make`).
- `run.sh` — quick wrapper around the Makefile.
- `requirements.txt` — Python dependencies for the cocotb testbench.

## Quick Start

```bash
cd test/cocotb_tests
python3 -m venv .venv && source .venv/bin/activate  # optional but recommended
pip install -r requirements.txt

# Recommended
python3 run_test.py --clean

# Or
make run_test

# Or
./run.sh
```

Outputs: `dump.vcd` (waveform) and `results.xml` in this directory. View waveforms with `gtkwave dump.vcd`.

## Execution Details

- Clock: 100 MHz (`clock_period = 10ns`).
- Start pulse: once at the beginning; `done` asserts when the zero instruction is fetched.
- Output readback: from RTL store memory at `0x20000` (10 bytes) plus `y[0:9]` for debug.
- MNIST subset: defaults to the first 10 test images for faster runs.

## Debugging Tips

- Key signals: `tinyml_accelerator_top.t_state`, `tinyml_accelerator_top.instr`, `tinyml_accelerator_top.done`, `execution_u.state`, memory interfaces.
- Enable `$display` traces in RTL files if deeper visibility is needed.
- Increase timeout in `execute_all(timeout_cycles=...)` if you change instruction count or clock speed.

## Maintenance Notes

- When RTL behavior changes, update `golden_model.py`, `assembler.py`, and `test_golden_comparison.py` to match.
- Keep memory addresses in sync with `compiler/compile.py` (`inputs=0x700`, `weights=0x10700`, `biases=0x13000`, `outputs=0x20000`).
- If you add new instructions, update both RTL and golden model decoding/execution paths and extend the test.
2. Enable RTL `$display` statements
3. Add intermediate checkpoints
4. Verify quantization scaling factors match
5. Check memory alignment and addresses

## ✅ Success Criteria

The test **PASSES** if:
- RTL completes all instructions without timeout
- Output values match golden model exactly, OR
- Max error ≤ 2 (acceptable quantization difference)

The test **FAILS** if:
- Timeout occurs (>200,000 cycles)
- Max error > 2
- File I/O errors
- Golden model execution fails

## 🙏 Acknowledgments

This verification framework demonstrates best practices for:
- Hardware-software co-verification
- Golden model methodology
- Cocotb testbench design
- Comprehensive documentation

---

**Created**: December 2025  
**Project**: Ushqyn  
**Purpose**: RTL Verification against Golden Model  
**Status**: Complete and tested
