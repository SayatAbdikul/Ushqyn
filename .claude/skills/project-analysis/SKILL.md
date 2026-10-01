---
name: project-analysis
description: Analyze the current state of the Ushqyn project, audit docs, cross-reference implementation vs documentation, and propose improvements.
user-invocable: true
argument-hint: "[focus-area]"
---

# Skill: Ushqyn Project Analysis & Documentation Improvement

## Trigger Conditions

Use this skill when the user asks to:
- Analyze the current state of the project
- Review or audit the docs folder
- Propose documentation improvements or changes
- Understand what has been implemented vs. what is planned
- Sync documentation with actual code state
- Identify gaps between spec and implementation

---

## Step 1 — Read All Markdown Files in `docs/`

Begin by discovering and reading every `.md` file inside `docs/` and other key locations.

```bash
find docs/ test/ memory_tools/ rtl/execution_unit/ -name "*.md" | sort
```

Also read: `README.md` at project root.

Build a mental model of:
- What the project claims to implement
- What the current architecture is (modules, interfaces, data flow)
- What milestones or TODOs are tracked
- What testing strategy is documented
- What is explicitly marked as WIP, TODO, PLANNED, or FUTURE

---

## Step 1.5 — Check Config Generation and DRAM Hex Before Anything Else

This project requires two generated files that must exist before simulation:

```bash
# 1. Generate RTL config package
python3 generate_config.py   # → produces rtl/accelerator_config_pkg.sv

# 2. Compile model and generate DRAM hex (if not present)
cd compiler && python3 main.py   # → produces dram.hex
```

If either is missing, warn the user immediately — simulation will use wrong parameters or fail silently.

---

## Step 2 — Cross-Reference with the Actual Codebase

After reading the docs, scan the actual source tree:

```bash
# All RTL source files (two trees: simulation and FPGA-synthesis)
find rtl/ src/ -name "*.sv" | grep -v 'sim_build\|obj_dir\|venv' | sort

# Verilator C++ testbenches
find test/ -name "*_tb.cpp" | sort

# cocotb Python testbenches
find test/ -name "test_*.py" | sort

# Compiler/toolchain Python files
find compiler/ -name "*.py" | sort

# Check config generation and Makefile targets
find . -name "generate_config.py" -o -name "Makefile" | grep -v 'venv\|sim_build' | sort
```

Key things to verify:
- Does `rtl/accelerator_config_pkg.sv` exist? (generated, not committed)
- Does `compiler/dram.hex` exist? (generated, not committed)
- Are there `.sv` files in `rtl/fpga_modules/` that differ from `src/`? (they should almost match — `src/` is the synthesis source of truth)
- Which modules in `rtl/` have corresponding C++ tests in `test/`?
- Which modules have cocotb tests in `test/cocotb_tests/` or `test/heavy_test_fpga/`?
- Are there implemented modules that are **not documented**?
- Are there documented modules that are **not yet implemented**?

---

## Step 3 — Assess the Project Along These Dimensions

### 3.1 Architecture Completeness

The **real** module hierarchy is:

```
tinyml_accelerator_top.sv
├── fetch_unit.sv              — DRAM read, instruction fetch
├── i_decoder.sv               — decode (GEMV/LOAD_V/LOAD_M/STORE/RELU opcodes)
├── simple_memory.sv           — 64KB flat DRAM model
└── modular_execution_unit.sv  (fpga_modules: modular_execution.sv)
    ├── buffer_controller.sv   — buffer address/slot management
    ├── buffer_file.sv         — 32-entry register file for tile buffers
    ├── load_execution.sv      — LOAD_V / LOAD_M execution
    ├── store_execution.sv     — STORE execution
    ├── relu_execution.sv      — RELU (tile-streamed ReLU + pass-through)
    └── gemv_execution.sv      — GEMV orchestration (streams w/b/x tiles)
        └── gemv_unit_core.sv  — tile GEMV FSM with PE tile array
            ├── pe.sv (×TILE_SIZE) — 8-bit signed multiply (1-cycle latency)
            ├── Gowin_SDPB_32  — x-vector BSRAM (packed 4:1, 1-cycle read)
            ├── Gowin_SDPB_32  — accumulator BSRAM (1-cycle read latency)
            ├── scale_calculator.sv    — max-abs reciprocal scale
            └── quantizer_pipeline.sv  — INT32 → INT8 quantizer
```

Also check:
- `src/fpga_top.sv` — FPGA synthesis top (includes UART and BSRAM IP)
- `src/uart_rx.sv`, `src/uart_tx.sv` — UART hardware I/O (no AXI/APB)
- `src/fetch_unit_fpga.sv` — FPGA variant of fetch unit

### 3.2 Two Parallel RTL Trees

| Tree | Purpose | Top Module |
|---|---|---|
| `rtl/` + `rtl/fpga_modules/` | Simulation (Verilator/cocotb) | `tinyml_accelerator_top.sv` |
| `src/` | FPGA synthesis (Gowin EDA) | `src/fpga_top.sv` |

`rtl/fpga_modules/` must stay in sync with `src/`. The key difference is mock IP modules:
- `Gowin_RAM16SDP_Mock.sv` replaces Gowin LUTRAM for simulation (async read)
- `Gowin_SDPB_32.sv` simulates Gowin BSRAM (1-cycle registered read)

### 3.3 Implementation Status

For each component, state:
- Implemented and tested
- Implemented but untested or partially tested
- Documented but not implemented
- Unclear / status ambiguous

### 3.4 Test Coverage

Two main test ecosystems exist — check both:

| Test Location | Type | Notes |
|---|---|---|
| `test/*_tb.cpp` | Verilator C++ | Legacy unit tests for individual modules |
| `test/new_unit_tests/*.cpp` | Verilator C++ | Newer unit tests |
| `test/cocotb_tests/test_*.py` | cocotb | Older per-module cocotb tests, may be stale |
| `test/heavy_test/` | cocotb + golden model | Full MNIST test (non-FPGA RTL) |
| `test/heavy_test_fpga/` | cocotb + golden model | **Primary** full MNIST FPGA simulation test |

The golden model at `compiler/golden_model.py` models all instructions in Python and is compared against RTL output in the heavy tests.

### 3.5 Documentation Freshness

Check for:
- Stale sections mentioning architectures no longer present (e.g., systolic array, AXI, INT4)
- Missing docs for `compiler/` toolchain (assembler, DRAM layout, config generation)
- Missing docs for `src/` FPGA synthesis tree vs `rtl/` sim tree distinction
- Missing docs for Gowin IP mock strategy (why mocks exist, how they differ)
- Broken links, `TBD`, `TODO`, placeholder text

### 3.6 Missing Documentation

Common gaps in this project:
- `compiler/` pipeline is largely undocumented (what each `.py` does, how to run)
- `generate_config.py` and its effect on `accelerator_config_pkg.sv` is not documented
- UART loading flow (`memory_tools/`, `src/uart_rx.sv`) has a README but integration with FPGA flow may be incomplete
- No top-level "How to run a simulation" guide for new contributors

---

## Step 4 — Produce a Structured Report

```
# Project State Analysis — Ushqyn

## Summary
<2-4 sentence executive summary>

## Implementation Status Table
| Module | SV File (rtl/) | SV File (src/) | Verilator Test | cocotb Test | Doc Coverage | Status |
|--------|---------------|----------------|----------------|-------------|--------------|--------|
| ...    | ...           | ...            | ...            | ...         | ...          | ...    |

## documentation Issues Found
### Stale / Outdated Content
- <file>: <description>

### Missing Documentation
- <module or topic>: <what is missing>

## Gaps Between Docs and Implementation
### Documented but Not Implemented
### Implemented but Not Documented

## Proposed Changes
### High Priority
### Medium Priority
### Low Priority

## Suggested New Docs to Create
- `docs/<filename>.md` — <purpose and content outline>
```

---

## Step 5 — Offer to Apply Changes

After presenting the report, ask which proposed changes to act on. Offer to:

1. **Edit existing `.md` files** — fix stale content, fill TODOs, correct architecture descriptions
2. **Create new `.md` files** — compiler toolchain guide, simulation setup, FPGA vs sim tree explanation
3. **Generate a `STATUS.md`** — living document tracking implementation progress per module
4. **Add inline doc comments** — SystemVerilog module headers where missing

Do not make changes without explicit approval unless the user says "just do it".

---

## Context: Project Stack

| Layer | Technology |
|---|---|
| RTL Language | SystemVerilog (`.sv`) |
| Simulation | Verilator (C++ harness) + cocotb (Python) |
| Primary integration test | `test/heavy_test_fpga/` — cocotb + Verilator, MNIST digit recognition |
| Target FPGA | Gowin Tang Nano 20K |
| Gowin IP mocks | `Gowin_RAM16SDP_Mock.sv` (async-read LUTRAM), `Gowin_SDPB_32.sv` (registered-read BSRAM) |
| Compiler toolchain | Python in `compiler/` — generates `program.hex` + `dram.hex` |
| Config generation | `generate_config.py` → `rtl/accelerator_config_pkg.sv` |
| Golden model | `compiler/golden_model.py` — Python SW reference for correctness comparison |
| Hardware I/O | UART (`src/uart_rx.sv`, `src/uart_tx.sv`, `memory_tools/`) |
| Neural network | 3-layer MLP: 784→12→32→10, INT8 quantized weights |
| Dataflow | Tile-streaming: weights/biases/x-vectors streamed tile-by-tile from DRAM |

---

## Notes

- Never fabricate implementation status. If a `.sv` file exists but you haven't read it, say so and read it before reporting.
- `src/` is the FPGA synthesis source of truth. `rtl/fpga_modules/` mirrors it with simulation mocks.
- If `rtl/accelerator_config_pkg.sv` doesn't exist, tell the user to run `python3 generate_config.py` first.
- If docs are written in a language other than English, analyse them in that language but write the report in English unless the user specifies otherwise.
- The PE (`pe.sv`) introduces a **1-cycle multiply latency** — FSM states that register PE sums must account for this.
