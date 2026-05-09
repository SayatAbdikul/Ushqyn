gtkwave dump.vcd
# Ushqyn — Cocotb Golden Model Verification

Single cocotb testbench that runs the Ushqyn RTL end-to-end and compares against the Python golden model. The RTL is started once, runs through all instructions until the zero instruction is fetched, and the outputs are checked element-by-element with a small quantization tolerance.

## Overview

Main test: `test_golden_comparison.py`

- Generates/uses `dram.hex` (instructions, weights, biases, inputs) and syncs it into all RTL memories.
- Pulses `start` once, waits for `done` when the zero instruction is hit.
- Reads RTL outputs from the store memory at `0x20000` (10 bytes) and also from `y[]` for debug.
- Runs the Python golden model on the same `dram.hex` and compares outputs (accepts max error ≤ 2).
- Uses a small MNIST subset (first 10 test images) for faster runs.

## Prerequisites

- Verilator
  - macOS: `brew install verilator`
  - Ubuntu/Debian: `sudo apt-get install verilator`
- Python deps: from this directory you can use a venv (recommended)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start (from `test/cocotb_tests`)

**IMPORTANT: Activate the virtual environment first:**

```bash
source .venv/bin/activate
```

Then run tests:

```bash
# Run full accelerator tests
make run_test

# Run GEMV tests only
make TEST_TARGET=top_gemv run_test

# Or use the venv wrapper (no need to activate venv first)
./make_venv.sh run_test                          # Full accelerator
./make_venv.sh TEST_TARGET=top_gemv run_test     # GEMV tests only
```

Alternative quick commands:
```bash
python3 run_test.py --clean      # Full accelerator (with cleanup)
python3 run_test.py --gemv       # GEMV tests only
./run.sh                          # Full accelerator
./run_gemv_tests.sh              # GEMV tests
```

These will prepare `dram.hex` if missing, build with Verilator, run the cocotb test, and write `dump.vcd` and `results.xml` here.

## Manual Preparation (optional)

```bash
cd ../../compiler
python3 main.py   # generates dram.hex
cd ../test/cocotb_tests
make              # runs the testbench
```

## Test Files

`test_golden_comparison.py` — main testbench with:
  - `test_accelerator_mnist_dataset` (end-to-end, MNIST subset)
  - `test_single_instruction_load_v` (sanity)
  - `test_reset_behavior` (reset check)

- **`test_top_gemv.py`** — isolated GEMV module testbench with:
  - `test_top_gemv_small` (4×8 matrix, quick validation)
  - `test_top_gemv_medium` (16×32 matrix, scalability check)
  - `test_top_gemv_with_quantization_check` (2×4 fixed values, detailed verification)
  - Includes Python `GoldenGEMV` reference implementation
  - Validates quantization behavior matching RTL

- **`test_execution_unit.py` (NEW)** — modular execution unit testbench with:
   - `test_neural_network_complete` (full 784→12→32→10 network)
   - `test_single_load_v` (single LOAD_V instruction)
   - `test_single_gemv` (single GEMV operation)
   - Uses golden_model.py functions for verification
   - Tests complete neural network execution sequence from model_assembly.asm

- `Makefile` — cocotb + Verilator build; `make run_test` prepares and runs.
   - Now supports `TEST_TARGET=top_gemv` for isolated GEMV testing
   - Now supports `TEST_TARGET=execution_unit` for execution unit testing
   - `make TEST_TARGET=execution_unit run_test` runs execution unit tests
  - `make TEST_TARGET=top_gemv run_test` runs only GEMV tests
  - Default: full accelerator testbench

- `run_test.py` — convenience wrapper (dependency check, `dram.hex` generation, run make).
- `run.sh` — quick wrapper.

## Memory Map (must match compiler)

- Instructions: `0x00000`
- Input: `0x00700`
- Weights: `0x10700`
- Biases: `0x13000`
- Output: `0x20000` (10 bytes)

## Debugging

- Waveform: `dump.vcd` (view with `gtkwave dump.vcd`).
- Increase timeout in `execute_all(timeout_cycles=...)` if you change program length.
- Watch signals: `tinyml_accelerator_top.t_state`, `tinyml_accelerator_top.instr`, `tinyml_accelerator_top.done`, `execution_u.state`, memory interfaces.

## top_gemv Testbench (`test_top_gemv.py`)

### Running GEMV Tests

**IMPORTANT: Must activate venv or use wrapper script**

```bash
# Option 1: Activate venv first (recommended for repeated commands)
source .venv/bin/activate
make TEST_TARGET=top_gemv run_test

# Option 2: Use venv wrapper script (no activation needed)
./make_venv.sh TEST_TARGET=top_gemv run_test

# Option 3: Python convenience script
python3 run_test.py --gemv

# Option 4: Bash quick-start script  
./run_gemv_tests.sh
```

**Why venv activation is required:** The Makefile system needs the venv Python interpreter for proper dependency resolution during Verilator compilation.

### Test Cases Overview

| Test | Matrix Size | Purpose |
|------|------------|---------|
| `test_top_gemv_small` | 4×8 | Quick validation, basic correctness |
| `test_top_gemv_medium` | 16×32 | Larger problem, scalability check |
| `test_top_gemv_with_quantization_check` | 2×4 | Fixed values, detailed quantization verification |

### Golden Model for GEMV

The testbench includes `GoldenGEMV` class implementing:
- **GEMV Operation:** y = W @ x + bias (where W is rows×cols, x is cols, y is rows)
- **Computation:** int32 accumulation for accuracy
- **Quantization:** Automatic scale calculation and int32→int8 conversion
- **Matching RTL:** Uses same `quantize_int32_to_int8()` from `helper_functions.py`

### GEMV Test Flow

```
1. Generate test data
   ├─ Weights (int8, rows × cols)
   ├─ Input vector x (int8, cols)
   └─ Bias (int8, rows)

2. Run golden model
   ├─ Compute y_int32 = W @ x + bias
   ├─ Calculate scale (max_abs / 127.0)
   └─ Quantize to int8

3. Configure RTL
   ├─ Reset and start clock
   ├─ Set rows, cols parameters
   └─ Load x and bias vectors

4. Stream weights to RTL
   ├─ Wait for w_ready signal
   ├─ Feed weight tile row-by-row
   └─ Tile size: 32 (TILE_SIZE parameter)

5. Wait for computation (done signal)

6. Compare outputs
   ├─ Accept max_error ≤ 2 (quantization tolerance)
   └─ Report pass/fail
```

### Expected Behavior

- RTL should produce outputs matching golden model within rounding tolerance
- Quantization differences of 1-2 LSBs are acceptable
- Computation should complete within `timeout` cycles
- Signals should follow handshake protocol (w_ready/w_valid)

### Troubleshooting GEMV Tests

**Issue:** `w_ready` never asserts
- Check `top_gemv.sv` implementation
- Verify TILE_SIZE parameter (default 32)
- Ensure state machine is running

**Issue:** Timeout on done signal
- Check GEMV computation logic
- Verify multiplication and accumulation
- Inspect waveform: `gtkwave sim_build/dump.vcd`

**Issue:** Output mismatch >2 error
- Verify golden model matches RTL design
- Check quantization scale calculation
- Ensure int32 arithmetic is correct
- Add logging to debug intermediate values



## Extending

- When RTL behavior changes, update `compiler/golden_model.py`, `assembler.py`, and `test_golden_comparison.py` together.
- Keep address constants aligned with `compiler/compile.py`.
- If you add instructions, update decoding/execution in both RTL and golden model and extend the test.
