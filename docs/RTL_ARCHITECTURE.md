# Ushqyn RTL Architecture Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Module Hierarchy](#module-hierarchy)
3. [Top-Level Architecture](#top-level-architecture)
4. [Execution Unit Architecture](#execution-unit-architecture)
5. [Memory Subsystem](#memory-subsystem)
6. [GEMV Core Pipeline](#gemv-core-pipeline)
7. [Quantization Pipeline](#quantization-pipeline)
8. [Module Descriptions](#module-descriptions)
9. [Signal Flow Diagrams](#signal-flow-diagrams)
10. [Optimizations](#optimizations)
11. [Conv2D Pipeline (`conv2d_execution.sv`)](#conv2d-pipeline-conv2d_executionsv)
12. [MaxPool Pipeline (`maxpool_execution.sv`)](#maxpool-pipeline-maxpool_executionsv)
13. [CNN Compiler-Fix Series (Patches A–J)](#cnn-compiler-fix-series-patches-aj)
14. [Two RTL Trees](#two-rtl-trees)
15. [FPGA Synthesis Results](#fpga-synthesis-results-gowin-gw2ar-18-mlp-only-path)

---

## System Overview

Ushqyn (Ұшқын — Kazakh for *spark*) is a hardware accelerator for neural-network inference covering both **MLP** and **small-CNN** workloads:

- **ISA**: 8 instructions — `LOAD_V`, `LOAD_M`, `STORE`, `GEMV`, `RELU`, `CONV2D_CFG`, `CONV2D_RUN`, `MAXPOOL`
- **Data Path**: 8-bit signed integer (int8) arithmetic; int32 internal accumulator
- **Memory**: Unified 32 KB DRAM (single instance: Gowin_SP BRAM on FPGA, register array in simulation)
- **Computation**: Tiled GEMV + 2D conv MAC array, BSRAM-backed accumulator and x-vector
- **Tiles**: 8 elements per tile on FPGA (TILE_ELEMS=8); 32 elements per tile in simulation (TILE_ELEMS=32)
- **Control**: Hierarchical FSM-based execution with one execution-unit module per opcode
- **MLP performance**: 89 MHz Fmax on Gowin GW2AR-18, 25,470 cycles/img (0.286 ms, ~3,500 img/s) on the 784→12→32→10 MNIST classifier
- **CNN status**: simulation RTL bit-exact against `compiler/golden_model.py` on a trained SmallCNN; FPGA bring-up of conv2d/maxpool is open work

---

## Module Hierarchy

```
fpga_top.sv (FPGA) / tinyml_accelerator_top.sv (Sim)
├── fetch_unit.sv                  — Instruction fetch from unified DRAM
│   └── (fetch_unit_fpga.sv on FPGA — adds FETCH_PRIME for BRAM latency)
│
├── i_decoder.sv                   — Decode 8-instruction ISA + relu_flag
│
├── simple_memory.sv               — Unified 32 KB DRAM
│   ├── (FPGA: Gowin_SP BRAM + UART loader)
│   └── (Sim: register array + $readmemh)
│
└── modular_execution_unit.sv
    ├── buffer_controller.sv       — Vector/matrix buffer management
    │   ├── buffer_file.sv         — Vector buffers (tile-indexed, 16 × 4 KB in sim)
    │   └── buffer_file.sv         — Matrix buffers (tile-indexed, 2 × 16 KB in sim)
    │
    ├── load_execution.sv          — LOAD_V / LOAD_M (length 18-bit per ISA)
    │   ├── load_v.sv              — Vector loading (DRAM → buffer)
    │   └── load_m.sv              — Matrix loading (DRAM → buffer, row-aware)
    │
    ├── store_execution.sv         — STORE: buffer → DRAM (length 18-bit)
    │   └── store.sv               — Tile-based memory write
    │
    ├── gemv_execution.sv          — GEMV tile bridging
    │   └── (sim) top_gemv  /  (FPGA) gemv_unit_core
    │       ├── pe.sv (×TILE_ELEMS)         — 8-bit signed multiply (1-cycle latency)
    │       ├── Gowin_SDPB_32      — x-vector BSRAM (packed 4:1, B1)
    │       ├── Gowin_SDPB_32      — Accumulator BSRAM (32-bit)
    │       ├── scale_calculator.sv
    │       │   └── wallace_32x32.sv
    │       │       └── compressor_3to2.sv
    │       └── quantizer_pipeline.sv
    │
    ├── relu_execution.sv          — Standalone RELU (tile-streamed, FC outputs)
    │   └── relu.sv                — Element-wise max(0, x)
    │
    ├── conv2d_execution.sv        — CONV2D_CFG/RUN: 2D conv + bias + quant + fused ReLU
    │   │                            [simulation RTL only — see "Two RTL trees" in docs/README.md]
    │   ├── pe.sv (×TILE_ELEMS)
    │   ├── accum_ram[0..ACCUM_DEPTH-1] (int32, parameterized depth)
    │   ├── scale_calculator.sv
    │   └── quantizer_pipeline.sv
    │
    └── maxpool_execution.sv       — MAXPOOL: NCHW sliding-window max-pool
                                     [simulation RTL only]
```

---

## Top-Level Architecture

### Block Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│              tinyml_accelerator_top (Sim) / fpga_top (FPGA)      │
│                                                                    │
│  ┌────────────┐     ┌────────────┐     ┌──────────────────────┐ │
│  │  Fetch     │────▶│ Instruction│────▶│  Modular Execution   │ │
│  │  Unit      │     │  Decoder   │     │       Unit           │ │
│  └─────┬──────┘     └────────────┘     └──────────┬───────────┘ │
│        │                                           │             │
│        │         ┌─────────────────────┐           │             │
│        └────────▶│  Unified Memory     │◀──────────┘             │
│                  │  (32 KB DRAM)       │                         │
│                  │  Gowin_SP BRAM      │                         │
│                  └─────────────────────┘                         │
│                                                                    │
│  Controls: clk, rst, start ──▶         ◀── done                 │
│  Output:   y[0:9] (inference result)                             │
└──────────────────────────────────────────────────────────────────┘
```

### Top-Level FSM

```
        ┌─────────┐
        │  IDLE   │◀──────────────────────────────────┐
        └────┬────┘                                   │
             │ start=1                                │
             ▼                                        │
        ┌─────────┐                                   │
        │  FETCH  │                                   │
        └────┬────┘                                   │
             │ fetch_en=1                             │
             ▼                                        │
     ┌──────────────┐                                 │
     │ WAIT_FETCH   │                                 │
     └──────┬───────┘                                 │
            │ fetch_done=1                            │
            ▼                                         │
     ┌──────────────┐                                 │
     │   DECODE     │                                 │
     └──────┬───────┘                                 │
            │ (latch opcode, operands, address)       │
            ▼                                         │
  ┌──────────────────┐                                │
  │ EXECUTE_START    │                                │
  └────────┬─────────┘                                │
           │ exec_start=1                             │
           ▼                                          │
  ┌──────────────────┐                                │
  │ EXECUTE_WAIT     │                                │
  └────────┬─────────┘                                │
           │ exec_done=1                              │
           ▼                                          │
        ┌──────┐                                      │
        │ DONE │──────────────────────────────────────┘
        └──────┘ (done pulse; loops if instr != 0)
```

The top module fetches instructions sequentially from DRAM, decodes them, and dispatches to the execution unit. When the zero instruction is fetched (end of program), the accelerator halts.

### Instruction Format (64-bit)

Opcode in bits **[4:0]**; per-opcode fields packed above. Stored in DRAM **MSB-first across 8 bytes** (the fetch unit re-assembles bit `instr[63:56]` from byte 0, `instr[55:48]` from byte 1, etc.).

The single source of truth for every opcode bit layout is **`compiler/isa_spec.py`**. The assembler, disassembler, golden-model decoder, and `rtl/i_decoder.sv` (auto-generated by `tools/generate_i_decoder.py`) all read from that file — they no longer have to be hand-synced. The cross-consistency tests in `compiler/test_isa_spec.py` (28 cases) assert every consumer agrees on every opcode, including a check that the on-disk RTL decoder matches what the generator would emit. The per-opcode bit tables below are reference documentation — when they drift from `isa_spec.py`, regenerate them or fix the spec.

#### Per-opcode bit layouts

```
LOAD_V (0x01) / STORE (0x03):
  [ 4: 0] opcode       (5)
  [ 9: 5] dest          (5)  buffer ID
  [27:10] length        (18) — bumped from 10b in Patch H to support inputs > 1023
  [63:40] addr          (24) DRAM address

LOAD_M (0x02):
  [ 4: 0] opcode       (5)
  [ 9: 5] dest          (5)
  [19:10] cols          (10)
  [29:20] rows          (10)
  [63:40] addr          (24)

GEMV (0x04):
  [ 4: 0] opcode       (5)
  [ 9: 5] dest          (5)
  [19:10] cols          (10)
  [29:20] rows          (10)
  [34:30] b             (5)  bias buffer ID
  [39:35] x             (5)  x-vector buffer ID
  [44:40] w             (5)  weight buffer ID

RELU (0x05):
  [ 4: 0] opcode       (5)
  [ 9: 5] dest          (5)
  [14:10] x             (5)
  [29:20] length        (10) — RELU only operates on FC-sized outputs

CONV2D_CFG (0x06):
  [ 4: 0] opcode       (5)
  [ 9: 5] dest          (5)
  [15:10] fmap_h        (6)
  [21:16] fmap_w        (6)
  [27:22] in_c          (6)
  [33:28] out_c         (6)
  [37:34] kh            (4)
  [41:38] kw            (4)
  [44:42] stride        (3)
  [47:45] pad           (3)

CONV2D_RUN (0x07):
  [ 4: 0] opcode       (5)
  [ 9: 5] dest          (5)
  [14:10] x             (5)
  [19:15] w             (5)
  [24:20] b             (5)
  [25]    relu_flag     (1) — fused ReLU, added/decoded in Patch F

MAXPOOL (0x08):
  [ 4: 0] opcode       (5)
  [ 9: 5] dest          (5)
  [14:10] x             (5)
  [17:15] pool_size     (3)
  [20:18] stride        (3)
  [26:21] fmap_h        (6)
  [32:27] fmap_w        (6)
  [37:33] channels      (5)  — muxed onto the in_channels port in i_decoder
```

ISA capability ceilings (from these field widths):
- `fmap_h`/`fmap_w`: 6 bits → max 63
- `in_c`/`out_c` (CONV2D_CFG): 6 bits → max 63
- MAXPOOL `channels`: 5 bits → max 31
- `kh`/`kw`: 4 bits → max 15
- `stride`/`pad`/`pool_size`: 3 bits → max 7
- LOAD_V / STORE length: 18 bits → max 262 143
- LOAD_M / GEMV cols: 10 bits → max 1023
- ADDR_WIDTH = 16 → 64 KB DRAM

---

## Execution Unit Architecture

### Block Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                  modular_execution_unit                          │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                 Main FSM Controller                        │ │
│  │  IDLE → DISPATCH → WAIT_* → COMPLETE                       │ │
│  └─────────────────────┬──────────────────────────────────────┘ │
│                        │                                         │
│  ┌─────────────────────┴──────────────────────────────────────┐ │
│  │                                                             │ │
│  │  ┌─────────────────┐    ┌──────────────────┐              │ │
│  │  │ Buffer          │    │ Load Execution   │              │ │
│  │  │ Controller      │◀───│   - load_v       │              │ │
│  │  │  • Vec Buffers  │    │   - load_m       │              │ │
│  │  │  • Mat Buffers  │    └──────────────────┘              │ │
│  │  └───────┬─────────┘                                       │ │
│  │          │                                                  │ │
│  │          │         ┌──────────────────┐                    │ │
│  │          ├────────▶│ GEMV Execution   │                    │ │
│  │          │         │  → gemv_unit_core│                    │ │
│  │          │         │    • 8 PEs       │                    │ │
│  │          │         │    • x_mem BSRAM │                    │ │
│  │          │         │    • res_mem BSRAM│                   │ │
│  │          │         │    • Quantization│                    │ │
│  │          │         └──────────────────┘                    │ │
│  │          │                                                  │ │
│  │          │         ┌──────────────────┐                    │ │
│  │          ├────────▶│ ReLU Execution   │                    │ │
│  │          │         └──────────────────┘                    │ │
│  │          │                                                  │ │
│  │          │         ┌──────────────────┐                    │ │
│  │          └────────▶│ Store Execution  │                    │ │
│  │                    └──────────────────┘                    │ │
│  │                                                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│           ┌───────────────────────┐                              │
│           │ Unified Memory Port   │                              │
│           │ (arbitrated by FSM)   │                              │
│           └───────────────────────┘                              │
└──────────────────────────────────────────────────────────────────┘
```

### Execution Unit FSM

```
    ┌──────┐
    │ IDLE │◀──────────────────────────────────────────┐
    └───┬──┘                                           │
        │ start=1                                      │
        ▼                                              │
  ┌──────────┐                                         │
  │ DISPATCH │──┐                                      │
  └──────────┘  │                                      │
                │ (based on opcode)                    │
     ┌──────────┼─────────────┬───────────────┐       │
     │          │             │               │       │
     ▼          ▼             ▼               ▼       │
 WAIT_LOAD  WAIT_GEMV    WAIT_RELU     WAIT_STORE    │
     │          │             │               │       │
     └──────────┴─────────────┴───────────────┘       │
                        │                              │
                        ▼                              │
                  ┌───────────┐                        │
                  │ COMPLETE  │────────────────────────┘
                  └───────────┘ done=1
```

The execution unit uses a single unified memory port. During LOAD operations, load_v or load_m drive the memory interface. During STORE, the store module drives it. GEMV and RELU operate on buffer data only (no direct DRAM access).

---

## Memory Subsystem

### Unified DRAM (32 KB)

The accelerator uses a **single unified memory** instance:

```
┌──────────────────────────────────────────────────────┐
│              Unified Memory (32 KB)                   │
│              simple_memory.sv                         │
│                                                        │
│  FPGA: Gowin_SP BRAM + UART loader                   │
│  Sim:  Register array + $readmemh(dram.hex)          │
│                                                        │
│  Interface:                                           │
│    mem_addr [ADDR_WIDTH-1:0]  — byte address          │
│    mem_rdata [7:0]            — read data (1 byte)    │
│    mem_wdata [7:0]            — write data (1 byte)   │
│    mem_we                     — write enable           │
│    mem_valid                  — read data valid        │
│                                                        │
│  FPGA: 1-cycle synchronous read (Gowin_SP, OCE=0)    │
│  Sim:  Combinational read (register array)            │
└──────────────────────────────────────────────────────┘
```

### DRAM Memory Map

| Region | Address | Size | Contents |
|--------|---------|------|----------|
| Instructions | 0x000 | ~192 B | Program code (fetched by fetch_unit) |
| Inputs | 0x0C0 | ~784 B | Input vector (loaded per image) |
| Biases | 0x4C0 | ~54 B | Layer biases (fc1, fc2, fc3) |
| Outputs | 0x8C0 | ~10 B | Inference results |
| Weights | 0x940 | ~10 KB | Weight matrices (fc1: 12x784, fc2: 32x12, fc3: 10x32) |

Addresses are configured in `compiler/accelerator_config.py`.

### Buffer System

```
┌──────────────────────────────────────────────────────────────┐
│                    buffer_controller.sv                       │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Vector Buffer File (buffer_file.sv)                  │   │
│  │  - 16 buffers (configurable via VECTOR_BUFFER_COUNT) │   │
│  │  - Tile-based access (8 elements = 64 bits per tile) │   │
│  │  - Separate read/write tile pointers per buffer       │   │
│  │  - Stored as flat register array, indexed by tile     │   │
│  │  - Used for: x-vectors, biases, y-results            │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Matrix Buffer File (buffer_file.sv)                  │   │
│  │  - 2 buffers (weight storage)                         │   │
│  │  - Tile-based access (64 bits per tile)               │   │
│  │  - Row-major weight storage                           │   │
│  │  - Cleared between GEMV invocations (clr_cache)      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                                │
│  Arbitration: buffer_controller routes read/write requests   │
│  from load, gemv, relu, and store execution modules          │
└──────────────────────────────────────────────────────────────┘
```

---

## GEMV Core Pipeline

The GEMV core (`gemv_unit_core.sv`) is the most complex module, implementing a multi-stage pipelined FSM with BSRAM-backed storage.

### Internal Storage

| Resource | Implementation | Size | Purpose |
|----------|---------------|------|---------|
| **x_mem** | Gowin_SDPB_32 (BSRAM) | 1024×32 | x-vector storage, packed 4:1 (B1) |
| **res_mem** | Gowin_SDPB_32 (BSRAM) | 1024×32 | 32-bit accumulator per output row |

Both use synchronous (registered) reads with 1-cycle latency. READ_xxx states in the FSM prime the address one cycle before the consuming state.

### Phase 1: Data Loading

```
IDLE ──▶ LOAD_X ──▶ STORE_X ──▶ ... ──▶ LOAD_BIAS ──▶ STORE_BIAS ──▶ ...
```

1. **LOAD_X / STORE_X** — Receive x-vector tiles from gemv_execution, pack into x_mem BSRAM.
   - B1: 4 int8 elements per 32-bit word → 2 writes per tile (was 8)
   - x_mem depth: 784/4 = 196 words (fits in 1024×32 BSRAM)
2. **LOAD_BIAS / STORE_BIAS** — Receive bias tiles, sign-extend int8→int32, write to res_mem as initial accumulator values.

### Phase 2: Weight Processing (per weight tile)

```
     ┌──────────────────────────────────────────────────────────────┐
     │                                                              │
     ▼                                                              │
 READ_X_TILE ──▶ LOAD_X_TILE ──┐  (cold start only, first tile)   │
                                │                                   │
 WAIT_TILE ◀────────────────────┘                                   │
     │                                                              │
     │ w_valid=1                                                    │
     ▼                                                              │
 WAIT_PE ──▶ SUM_PARTIAL ──▶ READ_ACCUM ──▶ PREP_ACCUM ──▶ ACCUMULATE
                                                                │
                                         ┌──────────────────────┤
                                         │ (overflow)           │ (no overflow)
                                         ▼                      ▼
                                    READ_ACCUM_2 ──▶ ACCUMULATE_2 ──▶ WAIT_NEXT
                                                                          │
                                                     ┌────────────────────┤
                                                     │ last_in_row        │ !last_in_row
                                                     ▼                    ▼
                                              (next row or         WAIT_TILE ◀─┘
                                               READ_MAX)           (B2: tile prefetched)
```

**Per-tile cycle budget (steady state with B1+B2):**

| State | Cycles | Purpose |
|-------|--------|---------|
| WAIT_TILE | 1 | Handshake weight tile from gemv_execution |
| WAIT_PE | 1 | PE multiply latency |
| SUM_PARTIAL | 1 | A2: pairwise PE output sums (4 pairs) |
| READ_ACCUM | 1 | A2 stage-2 sum + prime res_mem BSRAM read |
| PREP_ACCUM | 1 | A3: register res_dout + sum; B2: prime x_mem word 0 |
| ACCUMULATE | 1 | Write accumulator; B2: capture x_mem word 0, prime word 1 |
| WAIT_NEXT | 1 | Update counters; B2: capture x_mem word 1 |
| **Total** | **7** | **(+3 for overflow tiles)** |

**B2 Prefetch:** During PREP_ACCUM/ACCUMULATE/WAIT_NEXT, x_mem is idle from PE computation. B2 uses these cycles to read the next tile's two packed words from x_mem, eliminating READ_X_TILE + LOAD_X_TILE from the steady-state loop (saves 3 cycles/tile).

### Phase 3: Post-Processing

```
READ_MAX ──▶ PREP_MAX ──▶ FIND_MAX ──▶ ... ──▶ COMPUTE_SCALE ──▶ READ_QUANTIZE ──▶ QUANTIZE ──▶ ... ──▶ READ_OUTPUT_Y ──▶ OUTPUT_Y ──▶ DONE
```

1. **FIND_MAX** — Scan res_mem for max absolute value across all rows. Uses A4: registered abs isolates BSRAM→abs→compare chain.
2. **COMPUTE_SCALE** — Calculate `reciprocal = 2^23 / max_abs` (iterative division via `scale_calculator`).
3. **QUANTIZE** — Apply scale via 32×32 Wallace tree multiply, round, saturate to int8. Results written back to res_mem.
4. **OUTPUT_Y** — Stream quantized results as 8-element tiles back to gemv_execution.

### Processing Element (PE)

```
┌──────────────────────┐
│         PE           │
│                      │
│  w (int8)  ──┐       │
│              ├──▶ y = w × x (int16)
│  x (int8)  ──┘       │
│                      │
│  1-cycle latency     │
└──────────────────────┘

8 PEs operate in parallel per tile.
Throughput: 8 MACs/cycle during accumulation.
```

### Tile Count Budget(on the tested model)

| Layer | Rows × Tiles/row | Total tiles |
|-------|-------------------|-------------|
| fc1 | 12 × 98 | 1,176 |
| fc2 | 32 × 2 | 64 |
| fc3 | 10 × 4 | 40 |
| **Total** | | **1,280** |

---

## Quantization Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                Quantization (within gemv_unit_core)           │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  1. Calibration: FIND_MAX                             │   │
│  │     max_abs = max(|res_mem[i]|) for i in 0..rows-1   │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  2. Scale: COMPUTE_SCALE (scale_calculator.sv)        │   │
│  │     reciprocal_scale = (127 << 16) / max_abs          │   │
│  │     Uses iterative division (shift-subtract)          │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  3. Quantize: QUANTIZE (quantizer_pipeline.sv)        │   │
│  │     For each res_mem[i]:                               │   │
│  │       product = int32_value × reciprocal_scale         │   │
│  │                 (wallace_32x32 multiplier)             │   │
│  │       shifted = product >> 23                          │   │
│  │       result  = saturate(shifted, -128, 127)          │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  4. Output: OUTPUT_Y                                   │   │
│  │     Stream quantized int8 values as 8-element tiles   │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

The Wallace tree multiplier (`wallace_32x32.sv`) uses `compressor_3to2` full adders to reduce partial products, producing a 64-bit result from two 32-bit operands.

---

## Module Descriptions

### Core Modules

| Module | File | Purpose |
|--------|------|---------|
| **tinyml_accelerator_top** | `rtl/tinyml_accelerator_top.sv` | Top-level FSM, instruction sequencing |
| **fpga_top** | `src/fpga_top.sv` | FPGA top with UART + button I/O |
| **fetch_unit** | `rtl/fetch_unit.sv` | Sequential instruction fetch from DRAM |
| **fetch_unit_fpga** | `src/fetch_unit_fpga.sv` | FPGA variant with FETCH_PRIME for BRAM latency |
| **i_decoder** | `rtl/i_decoder.sv` | Combinational instruction decode |
| **simple_memory** | `rtl/simple_memory.sv` | Unified 32 KB DRAM (sim: reg array) |

### Execution Modules

The simulation RTL (`rtl/execution_unit/`) and the FPGA-targeted simulation mocks (`rtl/fpga_modules/`) parallel each other. Today they share the load/gemv/relu/store path; the conv2d/maxpool path lives only in `rtl/execution_unit/`.

| Module | Sim path | FPGA-mock path | Purpose |
|---|---|---|---|
| **modular_execution_unit** | `rtl/execution_unit/modular_execution_unit.sv` | `rtl/fpga_modules/modular_execution.sv` | Execution coordinator, memory arbitration, opcode dispatch |
| **buffer_controller** | `rtl/execution_unit/buffer_controller.sv` | `rtl/fpga_modules/buffer_controller.sv` | Dual buffer-file management, random-read port |
| **buffer_file** | `rtl/buffer_file.sv` | `rtl/fpga_modules/buffer_file.sv` | Tile-indexed BSRAM storage |
| **load_execution** | `rtl/execution_unit/load_execution.sv` | `rtl/fpga_modules/load_execution.sv` | LOAD_V / LOAD_M coordination (18-bit length per Patch H) |
| **load_v** | `rtl/load_v.sv` | (shared) | Vector load FSM (DRAM → buffer) |
| **load_m** | `rtl/load_m.sv` | (shared) | Matrix load FSM (DRAM → buffer, row-aware) |
| **gemv_execution** | `rtl/execution_unit/gemv_execution.sv` | `rtl/fpga_modules/gemv_execution.sv` | GEMV tile bridging (buffer ↔ core) |
| **gemv core** | `rtl/top_gemv.sv` | `rtl/fpga_modules/gemv_unit_core.sv` | Core GEMV FSM (PE array + quantization) |
| **relu_execution** | `rtl/execution_unit/relu_execution.sv` | `rtl/fpga_modules/relu_execution.sv` | Standalone RELU (10-bit length, FC outputs only) |
| **relu** | `rtl/relu.sv` | (shared) | Element-wise `max(0, x)` |
| **store_execution** | `rtl/execution_unit/store_execution.sv` | `rtl/fpga_modules/store_execution.sv` | Buffer → DRAM write-back (18-bit length) |
| **store** | `rtl/store.sv` | (shared) | Tile-based memory write FSM |
| **conv2d_execution** | `rtl/execution_unit/conv2d_execution.sv` | _(not yet ported)_ | CONV2D_CFG/RUN: 2D conv + bias + quant + fused ReLU |
| **maxpool_execution** | `rtl/execution_unit/maxpool_execution.sv` | _(not yet ported)_ | MAXPOOL: NCHW sliding-window max-pool |

### Computational Modules

| Module | File | Purpose |
|--------|------|---------|
| **pe** | `rtl/fpga_modules/pe.sv` | 8×8→16 bit signed multiply (1-cycle) |
| **scale_calculator** | `rtl/fpga_modules/scale_calculator.sv` | Iterative division for reciprocal scale |
| **quantizer_pipeline** | `rtl/fpga_modules/quantizer_pipeline.sv` | Pipelined int32→int8 quantization |
| **wallace_32x32** | `rtl/fpga_modules/wallace_32x32.sv` | 32-bit Wallace tree multiplier |
| **compressor_3to2** | `rtl/fpga_modules/compressor_3to2.sv` | Full adder (3:2 compression) |

### IP Blocks / Simulation Mocks

| Module | File | Purpose |
|--------|------|---------|
| **Gowin_SDPB_32** | `rtl/fpga_modules/Gowin_SDPB_32.sv` | BSRAM mock (1-cycle synchronous read) |
| **Gowin_RAM16SDP_Mock** | `rtl/fpga_modules/Gowin_RAM16SDP_Mock.sv` | LUTRAM mock (async read) |
| **Gowin_SP** | (Gowin IP, synthesis only) | Single-port BRAM for unified DRAM |

---

## Optimizations

### Applied Optimizations (chronological)

| ID | Name | Effect | Description |
|----|------|--------|-------------|
| — | FIND_MAX loop fix | −2,300 cy | Scan only `rows` entries, not `MAX_ROWS` |
| — | CLEAR_REMAINING removal | −6,700 cy | Stop zeroing unused accumulator rows |
| A1 | PE valid removal | −1,300 cy | Zero x_current_tile for invalid elements instead of gating adder tree |
| A2 | Pipelined adder tree | +1,280 cy, +Fmax | SUM_PARTIAL stage: pairwise sums halve logic depth |
| A3 | Registered write-back | +1,280 cy, +Fmax | PREP_ACCUM: register res_dout+sum before BSRAM write |
| A4 | Registered abs | +54 cy, +Fmax | PREP_MAX: register abs(res_dout) before FIND_MAX comparison |
| A6 | x_mem to BSRAM | +1,280 cy, +Fmax | Replace 128× LUTRAM with Gowin_SDPB_32 (eliminates 6-level MUX) |
| B1 | Pack x_mem 4:1 | −8,200 cy | 4 int8 per 32-bit word → 2 reads per tile (was 8) |
| B2 | Prefetch x tile | −3,831 cy | Load next tile during PREP_ACCUM/ACCUMULATE/WAIT_NEXT |

### Performance History

| State | Cycles | Fmax (MHz) | Latency (ms) |
|-------|--------|------------|--------------|
| Baseline (BSRAM accum) | 43,996 | 67 | — |
| +FIND_MAX/CLEAR fix | 34,942 | — | — |
| +A1–A4 | 36,221 | 84 | — |
| +A6 (x_mem BSRAM) | 37,501 | 91 | 0.411 |
| +B1 (pack x_mem 4:1) | 29,301 | 86 | 0.343 |
| **+B2 (prefetch x tile)** | **25,470** | **89** | **0.286** |

### Key Design Decisions

1. **BSRAM over LUTRAM** — x_mem and res_mem use Gowin_SDPB_32 (block RAM) instead of Gowin_RAM16SDP (LUTRAM). LUTRAM created deep MUX cascades that limited Fmax. BSRAM has 1-cycle read latency, requiring READ_xxx pipeline states, but eliminates the MUX critical path.

2. **Packed x_mem (B1)** — 4 int8 values per 32-bit BSRAM word. Reduces LOAD_X_TILE from 8 reads to 2, saving 6 cycles per weight tile across 1,280 tiles.

3. **Prefetch during accumulate (B2)** — x_mem is idle during WAIT_PE through WAIT_NEXT (the accumulate pipeline). B2 uses PREP_ACCUM and ACCUMULATE to prime and capture x_mem reads for the NEXT tile, eliminating READ_X_TILE + LOAD_X_TILE from steady-state.

4. **Unconditional adder tree (A1)** — Instead of gating PE outputs with validity masks (expensive MUX logic), invalid x elements are zeroed in LOAD_X_TILE. pe_out = w × 0 = 0 naturally, so the adder tree sum is correct without gating.

### Critical Path

The critical path is in `buffer_controller → vector_buffer_inst` (opcode decode → load execution → buffer controller → tile index), not in the GEMV core. Further Fmax improvements require pipelining the buffer controller's opcode decode path.

---

## Conv2D Pipeline (`conv2d_execution.sv`)

Simulation RTL only (`rtl/execution_unit/conv2d_execution.sv`). Produces full-tensor INT8 quantized output through a BSRAM-backed int32 accumulator.

### FSM phases

```
Phase 1 — setup (per CONV2D_RUN):
  IDLE → INIT_CONV (compute out_h, out_w, total_patch_size)

Phase 2 — accumulate (per output channel × per spatial position × per weight tile):
  LOOP_OC_INIT → TILE_LOOP_INIT → FETCH_W_TILE → WAIT_W_TILE
   → LOOP_OH_OW_INIT → FETCH_X_PIXEL (×patch elems) → WAIT_PE
   → ACCUMULATE → STORE_ACCUM → LOOP_OH_OW_NEXT → …
   → TILE_LOOP_NEXT → … → LOOP_OC_NEXT → … → INIT_QUANT

Phase 3 — quantize (per output element):
  MAX_PASS (find max_abs)
   → START_SCALE → WAIT_SCALE
   → STREAM_QUANT (with optional fused ReLU clamp)
   → DONE
```

### Invariants enforced after Patches F+G+H+I+J

- **NCHW layout end-to-end**: `linear_idx = oc·out_h·out_w + oh·out_w + ow`. Conv writes NCHW; MaxPool reads NCHW with `addr = c·fmap_h·fmap_w + ih·fmap_w + iw` and emits `(c, oh, ow)` order.
- **Fused ReLU on int8** (Patch F): `(relu_flag && q < 0) ? 0 : q` in STREAM_QUANT, matching `golden_model.conv2d(apply_relu=True)`.
- **Bias**: read as int8 from buffer, sign-extended to int32, added to accumulator. Address advanced via `quant_oc` (one bias per `out_h × out_w` output elements).
- **Per-tensor max-abs quant**: `max_abs_reg = max(|biased_val|)` in MAX_PASS; same `scale_calculator + quantizer_pipeline` as GEMV.
- **`accum_ram` parameterized** (Patch I): `[0:ACCUM_DEPTH-1]` (default 4096). `$error` fires in INIT_CONV if `out_h × out_w × out_channels > ACCUM_DEPTH`.
- **Vector buffer width** (Patch J): bumped to 32768 bits (4 KB) to fit SmallCNN's 4×26×26 = 2704-byte conv1 output without silent wrap in `buffer_file.sv`.

### MAC array

The conv2d unit instantiates the same `pe[0..TILE_ELEMS-1]` array as GEMV. `mac_sum` is computed combinationally as the sign-extended sum of the PE outputs, gated by `patch_tile_idx` so trailing PE slots (when `total_patch_size % TILE_ELEMS != 0`) don't contribute.

---

## MaxPool Pipeline (`maxpool_execution.sv`)

Simulation RTL only (`rtl/execution_unit/maxpool_execution.sv`). Sliding-window NCHW max-pool, operates directly on int8 (no rescaling).

### FSM

```
IDLE → INIT (compute out_h, out_w)
   → LOOP_INIT (next pixel) → WINDOW_FETCH → WINDOW_EVAL (×pool_size²)
   → EMIT_PIXEL (write result, advance ow → oh → c)
   → … → DONE
```

### NCHW order (Patch G)

- Read addr: `c × fmap_h × fmap_w + ih × fmap_w + iw`
- Loop nest: `c` outermost (incremented after every full `oh × ow` plane); `oh` middle; `ow` innermost
- Output written as `[c, oh, ow]` — matches `compiler/golden_model.py::maxpool` layout

This was a critical correctness fix: pre-Patch-G, MaxPool used NHWC addressing on conv2d's NCHW output, reducing over completely wrong neighborhoods.

---

## CNN Compiler-Fix Series (Patches A–J)

The CNN-capable path was brought up over a sequence of focused patches. Each shipped with its own cocotb regression test (where applicable):

| Patch | Files | Bug closed | Verification |
|---|---|---|---|
| **A** | `compiler/golden_model.py`, `compiler/disassembler.py` | Debug residue, missing CNN op decoders | Lint clean, opcode round-trip |
| **B** | `compiler/compile.py`, `compiler/dram.py`, `compiler/test_cnn_golden.py` | Duplicate Conv bias `LOAD_V`; DRAM bias-region overlap; TILE_ELEMS hardcoded as 32 | Strengthened end-to-end pytest goes red→green |
| **C** | `compiler/golden_model.py`, `compiler/compile.py`, `compiler/assembler.py` | Documentation gap on `LOAD_M` cols asymmetry | Docstrings render |
| **D** | `compiler/compile.py` | Conv→Relu lookahead breaks on Constant nodes | Both `CONV2D_RUN` instructions emit `relu_flag=1` |
| **F** | `rtl/i_decoder.sv`, `rtl/tinyml_accelerator_top.sv`, `rtl/execution_unit/modular_execution_unit.sv`, `rtl/execution_unit/conv2d_execution.sv` | `relu_flag` bit dropped at every RTL layer | `test_conv2d_execution.py` 2/2 PASS |
| **G** | `rtl/execution_unit/maxpool_execution.sv` | NHWC addressing on NCHW input | `test_maxpool_execution.py` 2/2 PASS |
| **H** | `rtl/i_decoder.sv`, `rtl/tinyml_accelerator_top.sv`, `rtl/execution_unit/{modular_execution_unit,load_execution,store_execution}.sv`, `rtl/{load_v,store}.sv` | LOAD_V/STORE length truncated at 10 bits | Lint clean; F+G regressions hold |
| **I** | `rtl/execution_unit/conv2d_execution.sv` | `accum_ram [0:4095]` hardcoded; silent overflow on larger geometries | Lint clean; F regression holds; new `$error` overflow check |
| **J** | `generate_config.py`, `rtl/execution_unit/{modular_execution_unit,conv2d_execution,maxpool_execution}.sv` | `VECTOR_BUFFER_WIDTH=8192` (1 KB) too small for SmallCNN's 2704-byte conv1 output; address-port width hardcoded | **Heavy_test integration: bit-exact RTL ↔ Golden** on every MNIST image |

### Golden-model contract

The Python golden model in `compiler/golden_model.py` is the canonical specification of what the RTL must produce. Every bit-level decision (rounding, saturation, layout, quantization scale) lives in both places and must stay in sync. Per-unit cocotb tests preload buffers via the testbench wrapper and assert byte-equal output against the golden model — there is no tolerance for divergence.

---

## Two RTL Trees

| Tree | Purpose | Top Module | Memory | IP Blocks |
|------|---------|------------|--------|-----------|
| `src/` | FPGA synthesis (Gowin EDA) | `fpga_top.sv` | Gowin_SP BRAM + UART | Native Gowin IP |
| `rtl/` + `rtl/fpga_modules/` | Simulation (Verilator/cocotb) | `tinyml_accelerator_top.sv` | Register array + `$readmemh` | Mock modules |

`rtl/fpga_modules/` mirrors `src/` with simulation-compatible mocks:
- `Gowin_SDPB_32.sv` — Mock for Gowin BSRAM (registered read, 1-cycle latency)
- `Gowin_RAM16SDP_Mock.sv` — Mock for Gowin LUTRAM (async read)

After modifying `rtl/fpga_modules/gemv_unit_core.sv`, sync to `src/top_gemv.sv`.

### FPGA-Only Differences
- `src/fpga_top.sv` — Adds UART RX/TX, button debounce, LED output
- `src/fetch_unit_fpga.sv` — Adds FETCH_PRIME state for Gowin_SP 1-cycle read latency
- `src/simple_memory.sv` — Gowin_SP BRAM with UART write port (OCE must be 0)

---

## FPGA Synthesis Results (Gowin GW2AR-18, MLP-only path)

The numbers below are for the **MLP path** (`src/`), which is the only synthesizable target today. CNN ops live only in `rtl/execution_unit/` and are open work for the FPGA path.

- **Fmax**: 89.201 MHz (11 logic levels)
- **Cycles/image**: 25,470
- **Latency**: 0.286 ms/image (~3,500 images/sec)
- **Logic**: 42% (8,640 / 20,736 LUT4)
- **BSRAM**: 94% (43 / 46 × 18-Kb blocks)
- **DSP**: 5 blocks (1× MULT36X36 + 4× MULTADDALU18X18)
- **MNIST accuracy**: 95% (10,000 images), 100% exact match vs golden model

### Tang Nano 20K resource ceiling

| Resource | Available | MLP path uses | CNN path (`rtl/execution_unit/`) would need |
|---|---:|---:|---|
| LUT4 | 20,736 | 42 % | TBD; expected higher (more execution units) |
| BSRAM | 46 × 18 Kb | 94 % | Doesn't fit on-chip without external PSRAM |
| DSP | 48 | 5 (MLP) | 32 if TILE_ELEMS=32 (sim variant), 8 if TILE_ELEMS=8 |

The simulation RTL with `TILE_ELEMS=32` and 64 KB of vector + 32 KB of matrix + 16 KB accum_ram **does not fit** as on-chip BSRAM (~110 KB requested vs 103 KB available). The path forward for CNN-on-FPGA is to port the conv2d/maxpool units to `rtl/fpga_modules/` at `TILE_ELEMS=8` and either move `simple_memory` to the 8 MB external PSRAM or shrink internal buffers. With TILE_ELEMS=8 SmallCNN's conv1 output (2704 bytes) would require `VECTOR_BUFFER_WIDTH ≥ 21632` bits in the FPGA path too, mirroring Patch J.

---

| Version | Date | Changes |
|---|---|---|
| 3.0 | May 2026 | CNN bring-up: 8-opcode ISA (CONV2D_CFG/RUN, MAXPOOL); `conv2d_execution.sv` and `maxpool_execution.sv` execution units; `relu_flag` plumbing; bit-exact RTL ↔ Golden on SmallCNN MNIST. Per-opcode bit layouts. Patches A–J. |
| 2.0 | Mar 2026 | Complete rewrite for FPGA architecture: unified memory, 8 PEs, BSRAM, B1/B2 GEMV optimizations |
| 1.0 | Dec 2025 | Initial documentation |
