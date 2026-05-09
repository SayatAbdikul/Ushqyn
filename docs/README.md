# Ushqyn Documentation

Comprehensive documentation for the Ushqyn hardware accelerator.

## Overview

Ushqyn (Ұшқын — Kazakh for *spark*) is a specialized hardware accelerator for neural-network inference with quantized 8-bit integer arithmetic. It implements a custom **8-instruction ISA** covering both MLP and small-CNN workloads. The simulation RTL has been validated bit-exactly against a Python golden model on a trained SmallCNN; the FPGA build runs a 3-layer MLP on MNIST on the Gowin GW2AR-18 (Tang Nano 20K) at 89 MHz.

## What's where

### Architecture documentation
- **[RTL_ARCHITECTURE.md](RTL_ARCHITECTURE.md)** — Detailed RTL architecture
  - 8-opcode ISA bit layouts
  - Module hierarchy and descriptions
  - FSM state diagrams for each execution unit
  - Quantization pipeline and tile budgeting
  - Optimization history (A1–A6, B1–B2 GEMV optimizations; F–J CNN bring-up)

### Visual diagrams
- **[diagrams/](diagrams/)** — System diagrams (architecture, hierarchy, GEMV pipeline, memory)

### Test documentation
- **[../test/TEST_GUIDE.md](../test/TEST_GUIDE.md)** — Test suite overview
- **[../test/TESTBENCH_COMPARISON.md](../test/TESTBENCH_COMPARISON.md)** — Verilator-C++ vs cocotb comparison
- **[../test/heavy_test/README.md](../test/heavy_test/README.md)** — Full-MNIST validation
- **[../test/heavy_test_fpga/README.md](../test/heavy_test_fpga/README.md)** — FPGA-tier simulation
- **[../test/cocotb_tests/README.md](../test/cocotb_tests/README.md)** — Per-unit cocotb tests

## Module summary

| Category | Modules | Purpose |
|---|---|---|
| **Top level** | `fpga_top` (FPGA) / `tinyml_accelerator_top` (sim) | System coordinator, fetch+decode+execute FSM |
| **Control** | `fetch_unit`, `i_decoder` | Instruction fetch (8 bytes / instruction) and 8-opcode decode |
| **Execution** | `modular_execution_unit` + 7 sub-modules (load, store, gemv, relu, **conv2d, maxpool**) | Per-opcode execution dispatch |
| **Memory** | `simple_memory` (unified) | 32 KB DRAM (Gowin_SP BRAM on FPGA, register array in sim) |
| **Buffers** | `buffer_controller` + `buffer_file` (vector + matrix) | Tile-indexed BSRAM-backed buffers |
| **Computation** | `gemv_unit_core` / `top_gemv` + `pe[]` | Tiled GEMV (and conv2d MAC array) |
| **Quantization** | `quantizer_pipeline`, `scale_calculator` | int32 → int8 with reciprocal-multiply rounding |
| **Arithmetic** | `wallace_32x32`, `compressor_3to2` | 32×32 multiplier (used by scale_calculator) |
| **Activation** | `relu`, fused-ReLU path in `conv2d_execution` | `max(0, x)` on int8 |
| **Data movement** | `load_v`, `load_m`, `store` | DRAM ↔ buffer transfers |

## Design hierarchy

```
Level 1: System
  └─ fpga_top (FPGA) / tinyml_accelerator_top (sim)

Level 2: Subsystems
  ├─ fetch_unit
  ├─ i_decoder            (8 opcodes; relu_flag latched on CONV2D_RUN)
  ├─ simple_memory        (unified 32 KB)
  └─ modular_execution_unit

Level 3: Execution modules (one per opcode)
  ├─ buffer_controller
  │   └─ buffer_file (vector × N) + buffer_file (matrix × M)
  ├─ load_execution                           ─→ LOAD_V (0x01), LOAD_M (0x02)
  │   ├─ load_v
  │   └─ load_m
  ├─ store_execution                          ─→ STORE (0x03)
  │   └─ store
  ├─ gemv_execution                           ─→ GEMV (0x04)
  │   └─ top_gemv (sim)  /  gemv_unit_core (FPGA)
  │       ├─ pe[0..TILE_ELEMS-1]
  │       ├─ Gowin_SDPB_32 (x-vector BSRAM, packed 4:1)
  │       ├─ Gowin_SDPB_32 (accumulator BSRAM, 32-bit)
  │       ├─ scale_calculator
  │       │   └─ wallace_32x32 → compressor_3to2
  │       └─ quantizer_pipeline
  ├─ relu_execution                           ─→ RELU (0x05)
  │   └─ relu
  ├─ conv2d_execution                         ─→ CONV2D_CFG (0x06) + CONV2D_RUN (0x07)
  │   ├─ pe[0..TILE_ELEMS-1]
  │   ├─ accum_ram[0..ACCUM_DEPTH-1] (int32, parameterized depth)
  │   ├─ scale_calculator
  │   └─ quantizer_pipeline
  └─ maxpool_execution                        ─→ MAXPOOL (0x08)

Level 4: IP blocks (FPGA path)
  ├─ Gowin_SP             (BRAM for unified DRAM)
  ├─ Gowin_SDPB_32        (dual-port BSRAM for x_mem and res_mem)
  └─ DSP blocks           (MULT36X36, MULTADDALU18X18)
```

## Two RTL trees

| Tree | Purpose | Top module | Workloads | IP / memory |
|---|---|---|---|---|
| `src/` | FPGA synthesis (Gowin EDA) | `fpga_top.sv` | **MLP only** | Gowin IP primitives, UART loader |
| `rtl/` + `rtl/execution_unit/` | Simulation (Verilator/cocotb), CNN-capable | `tinyml_accelerator_top.sv` | **MLP + CNN** | Mock IP, `$readmemh` memory |
| `rtl/fpga_modules/` | FPGA-targeted simulation mocks | shared with `src/` | MLP only today | Gowin BSRAM mocks |

**`src/` is the synthesis source of truth** for the MLP path. `rtl/fpga_modules/` mirrors it for simulation (Verilator/cocotb) with mocked Gowin BSRAM blocks. The CNN path lives only in `rtl/execution_unit/` — porting `conv2d_execution.sv` and `maxpool_execution.sv` to `rtl/fpga_modules/` (and through to `src/`) is open work.

### Simulation mocks
- `Gowin_RAM16SDP_Mock.sv` — Async-read LUTRAM
- `Gowin_SDPB_32.sv` (in `rtl/fpga_modules/`) — Synchronous-read BSRAM (1-cycle latency)

## Instruction Set

| Opcode | Mnemonic | Operands | Notes |
|---|---|---|---|
| `0x00` | NOP | — | Zero-word program terminator |
| `0x01` | LOAD_V | dest, addr, length | Length 18 bits (≤ 262143 elements) |
| `0x02` | LOAD_M | dest, addr, rows, cols | rows/cols 10 bits each (≤ 1023) |
| `0x03` | STORE | src, addr, length | Length 18 bits |
| `0x04` | GEMV | dest, w, x, b, rows, cols | int32 accum, per-tensor max-abs scale |
| `0x05` | RELU | dest, x, length | Standalone (10-bit length, ≤ 1023) |
| `0x06` | CONV2D_CFG | dest, fmap_h, fmap_w, in_c, out_c, kh, kw, stride, pad | Latches geometry; no execution |
| `0x07` | CONV2D_RUN | dest, x, w, b, relu_flag | Reads geometry from prior CONV2D_CFG |
| `0x08` | MAXPOOL | dest, x, fmap_h, fmap_w, channels, pool_size, stride | NCHW layout |

Per-opcode bit layouts are documented in [RTL_ARCHITECTURE.md](RTL_ARCHITECTURE.md) and implemented in `compiler/assembler.py` (encoder), `rtl/i_decoder.sv` (RTL decoder), and `compiler/golden_model.py::i_decoder` (golden decoder). All three must stay in sync — the disassembler in `compiler/disassembler.py` rounds out the four-way invariant.

## DRAM Memory Map

The design uses a unified 32 KB memory (FPGA: Gowin_SP BRAM; sim: register array via `$readmemh`):

| Region | Default address | Contents |
|---|---|---|
| Instructions | `0x0000` | Program code (8 bytes / instruction, fetched MSB-first) |
| Inputs | `0x00C0` | Input vector (loaded per image via LOAD_V to buffer 9) |
| Biases | `0x04C0` | All biases in topological order: conv1.bias → conv2.bias → … → fc.bias |
| Outputs | `0x08C0` | Inference results (written by STORE) |
| FC weights | `0x0940` | Gemm/MatMul weights, padded to TILE_ELEMS columns |
| Conv weights | `0x3000` | Conv weights `[out_C, in_C·kH·kW]`, padded to TILE_ELEMS columns |

Addresses are configured in `generate_config.py::DEFAULT_CONFIG` and emitted to both `rtl/accelerator_config_pkg.sv` and `compiler/accelerator_config.py`. The compiler generates `dram.hex` which is loaded via UART on FPGA or `$readmemh` in simulation.

## Key configuration parameters

| Parameter | FPGA / `rtl/fpga_modules/` | Simulation `rtl/execution_unit/` | Notes |
|---|---:|---:|---|
| `TILE_ELEMS` | 8 | 32 | Elements per tile |
| `TILE_WIDTH` | 64 | 256 | Bits per tile (= TILE_ELEMS × 8) |
| `ADDR_WIDTH` | 16 | 16 | Memory address bits → 64 KB DRAM |
| `MAX_ROWS` | 784 | 1024 | Max vector/matrix dimension |
| `VECTOR_BUFFER_WIDTH` | 8192 (1 KB / buffer) | 32768 (4 KB / buffer) | Bumped in Patch J for SmallCNN's 4×26×26 conv1 output |
| `MATRIX_BUFFER_WIDTH` | 32768 | 131072 | Wider in sim to fit MLP weights without sharding |
| `VECTOR_BUFFER_COUNT` | 8 (MLP path) / 16 (CNN path) | 16 | Compiler uses buffer ID 9 for input — needs ≥ 10 |
| `ACCUM_DEPTH` (conv2d) | n/a (no FPGA conv2d yet) | 4096 | New in Patch I; with `$error` overflow check |

## GEMV Pipeline (gemv_unit_core.sv)

The GEMV core implements a multi-stage pipelined FSM. See [RTL_ARCHITECTURE.md](RTL_ARCHITECTURE.md) for the full diagram.

### Phase 1 — data loading
1. **LOAD_X / STORE_X** — receive x-vector tiles, pack 4 int8 per 32-bit BSRAM word (B1)
2. **LOAD_BIAS / STORE_BIAS** — receive bias tiles, write to accumulator BSRAM

### Phase 2 — weight processing (per weight tile)
```
WAIT_TILE → WAIT_PE → SUM_PARTIAL → READ_ACCUM → PREP_ACCUM → ACCUMULATE → WAIT_NEXT
   │                                                                          │
   └─── (B2: next x-tile prefetched during PREP_ACCUM/ACCUMULATE) ────────────┘
```

### Phase 3 — post-processing
1. **FIND_MAX** — scan accumulator for max absolute value
2. **COMPUTE_SCALE** — `recip = (127 << 24) / max_abs` via iterative restoring divider
3. **QUANTIZE** — `int32 × recip` → `(prod + (1<<23)) >> 24` → clamp [-128, 127]
4. **OUTPUT_Y** — stream quantized results back as tiles

### Optimizations
| Tag | What | Effect |
|---|---|---|
| **B1** | x-vector BSRAM packed 4:1 | 2 reads per tile instead of 8 |
| **B2** | Next x-tile prefetched during accumulate pipeline | Zero-overhead tile transitions |
| **A2** | Pipelined adder tree (SUM_PARTIAL stage) | Halves logic depth |
| **A3** | Registered accumulator write-back | Breaks res_dout+sum carry chain |
| **A6** | x_mem from LUTRAM → BSRAM | Eliminates 6-level MUX cascade |

## Conv2D Pipeline (conv2d_execution.sv) — simulation RTL only

The conv2d execution unit performs full-tensor INT8 quantization with a BSRAM-backed accumulator (parameterized via `ACCUM_DEPTH`):

```
IDLE → INIT_CONV → LOOP_OC_INIT → TILE_LOOP_INIT → FETCH_W_TILE → WAIT_W_TILE
       → LOOP_OH_OW_INIT → FETCH_X_PIXEL (×patch elems) → WAIT_PE
       → ACCUMULATE → STORE_ACCUM → LOOP_OH_OW_NEXT → … (next pixel)
       → … (next out_c) → INIT_QUANT → MAX_PASS → START_SCALE → WAIT_SCALE
       → STREAM_QUANT (with optional fused ReLU clamp) → DONE
```

Key invariants enforced after the F+G+H+I+J patch series:
- **NCHW layout end-to-end**: `linear_idx = oc·H·W + oh·W + ow`. MaxPool reads/writes the same.
- **Fused ReLU on int8**: `(relu_flag && q < 0) ? 0 : q` in STREAM_QUANT, matching `golden_model.conv2d(apply_relu=True)`.
- **Bias = int8 added to int32 accumulator**: `acc + sext(bias[oc])`, addressed via `quant_oc` advancing every `out_h × out_w` elements.
- **Per-tensor max-abs quant**: `max_abs_reg = max(|biased_val|)` in MAX_PASS; same `scale_calculator + quantizer_pipeline` as GEMV.
- **Overflow assertion**: `out_h × out_w × out_channels > ACCUM_DEPTH` triggers `$error` in INIT_CONV.

## MaxPool (maxpool_execution.sv) — simulation RTL only

NCHW-layout sliding-window max-pool, operates directly on int8 (no rescaling):

```
IDLE → INIT → LOOP_INIT → WINDOW_FETCH → WINDOW_EVAL (×pool_size²)
       → EMIT_PIXEL → … (next ow → oh → c) → DONE
```

Read addr: `c × fmap_h × fmap_w + ih × fmap_w + iw`. Output written in `[c, oh, ow]` order — channel outermost, matching conv2d output and `golden_model.maxpool`.

## Quantization Pipeline

Bit-exact between hardware (`scale_calculator.sv` + `quantizer_pipeline.sv`) and software (`compiler/helper_functions.py::quantize_int32_to_int8_rtl_exact`):

1. **Calibration** — find `max_abs = max(|x|)` across all int32 accumulator values
2. **Scale calculation** — `recip = (127 << 24) / max_abs` via iterative restoring division (`scale_calculator.sv`)
3. **Quantization** — `prod = x × recip` (signed × unsigned, int64), then `rounded = (prod + (1<<23)) >> 24`, then clamp to `[-128, 127]` (`quantizer_pipeline.sv`)

## Compiler Toolchain

```
model.py            — MLP and SmallCNN PyTorch architectures
     │
compile.py          — ONNX → assembly (topological walk, emits 8-opcode instructions)
     │
assembler.py        — Assembly → 64-bit machine code
     │
dram.py             — save_all_initializers_to_dram: pack instructions + weights + biases
                      in topological order into a single dram.hex DRAM image
     │
golden_model.py     — Bit-exact Python reference (RTL contract)
                      execute_program(dram.hex) → output buffer
```

Configuration: `compiler/accelerator_config.py` (auto-generated from `generate_config.py`) defines TILE_ELEMS, DRAM addresses, and memory layout. The same generator emits `rtl/accelerator_config_pkg.sv` so the RTL and toolchain stay in lockstep.

## FPGA Workflow

```bash
# 1. Compile model
cd compiler && python3 main.py

# 2. Synthesize with Gowin EDA (src/ directory)

# 3. Program FPGA bitstream

# 4. Load DRAM via UART
cd memory_tools && ./uart_load_hex ../compiler/dram.hex /dev/ttyUSB1

# 5. Run inference (S1 button) and read results
./uart_read_max /dev/ttyUSB1
```

## Test Suites

| Test | Location | Framework | Workload | Purpose |
|---|---|---|---|---|
| **CNN per-unit** (NEW) | `test/cocotb_tests/test_conv2d_execution.py` | cocotb + Verilator | bit-exact conv2d vs golden | Patch F regression |
| **MaxPool per-unit** (NEW) | `test/cocotb_tests/test_maxpool_execution.py` | cocotb + Verilator | bit-exact NCHW maxpool vs golden | Patch G regression |
| **CNN end-to-end** | `compiler/test_cnn_golden.py` | pytest | LOAD_V count + FC bias DRAM + NumPy shadow chain | Compiler Patch B regression |
| **FPGA simulation** | `test/heavy_test_fpga/` | cocotb + Verilator | MLP, 20 images by default | Primary FPGA-tier test |
| **Full sim validation** | `test/heavy_test/` | cocotb + Verilator | MLP, 10K images | Production validation |
| **Component tests** | `test/cocotb_tests/` | cocotb | per-module | gemv, buffer, load, etc. |
| **C++ unit tests** | `test/new_unit_tests/` + `test/*_tb.cpp` | Verilator C++ | low-level | Standalone module smoke tests |

## Status (2026-05)

- **Compiler / Golden model**: bit-exact for both MLP and SmallCNN paths.
- **Simulation RTL (`rtl/execution_unit/`)**: bit-exact against golden on every MNIST image tested through SmallCNN; ~94–96 % accuracy on trained models, matching PyTorch float32 within sample noise.
- **FPGA build (`src/`)**: MLP at 89 MHz, 25,470 cycles/image, 95 % accuracy on 10 K images. CNN ops not yet ported.
- **Per-unit cocotb tests**: 4/4 PASS (`conv2d_execution` × 2, `maxpool_execution` × 2).

## Future Enhancements

- **CNN on FPGA**: port `conv2d_execution.sv` and `maxpool_execution.sv` into `rtl/fpga_modules/` with TILE_ELEMS=8 and a Gowin BSRAM-backed `accum_ram`. Estimated 2–4 days.
- **Quantization quality**: per-channel weight quant, asymmetric int8, calibration-based static scales — matter most on networks deeper than SmallCNN.
- **B3**: Stream weights directly from DRAM during GEMV (bypass buffer controller).
- **B5**: Wider tiles (TILE_ELEMS=16) for 2× throughput with minimal BSRAM increase.
- **Buffer-controller pipelining**: register the opcode-decode path to lift Fmax further.

---

| Version | Date | Description |
|---|---|---|
| 3.0 | May 2026 | CNN bring-up: 8-opcode ISA, conv2d/maxpool execution units, end-to-end RTL ↔ Golden bit-exact match (Patches A–D, F–J). |
| 2.0 | Mar 2026 | FPGA deployment, unified memory, B1/B2 GEMV optimizations |
| 1.0 | Dec 2025 | Initial documentation |
