# Modular Execution Unit

Modular execution-unit path supporting both the MLP and the SmallCNN ISA paths. The simulation RTL here is the canonical CNN-capable execution path; the FPGA-targeted variant in `rtl/fpga_modules/` mirrors a subset of these modules (no `conv2d_execution` or `maxpool_execution` yet).

## Architecture Overview

```
modular_execution_unit.sv (Top Coordinator)
├── buffer_controller.sv      Vector + matrix buffer management
├── load_execution.sv         LOAD_V (0x01), LOAD_M (0x02)
├── store_execution.sv        STORE (0x03)
├── gemv_execution.sv         GEMV (0x04) — FC / Gemm path
├── relu_execution.sv         RELU (0x05) — standalone (FC outputs only)
├── conv2d_execution.sv       CONV2D_CFG (0x06) + CONV2D_RUN (0x07) — 2D conv + bias + quant + fused ReLU
└── maxpool_execution.sv      MAXPOOL (0x08) — NCHW sliding-window max pool
```

## Module Descriptions

### 1. `buffer_controller.sv`
**Purpose:** Unified interface for all buffer file operations

**Features:**
- Manages separate vector and matrix buffer files
- Provides clean read/write interfaces
- Handles tile-based I/O automatically
- Generates read valid signals with proper timing

**Interface:**
- Vector buffer: read/write with 32-element tiles
- Matrix buffer: read/write with 256-bit tiles
- Status signals for operation completion

### 2. `load_execution.sv`
**Purpose:** Handles LOAD_V and LOAD_M operations

**Features:**
- Instantiates `load_v` and `load_m` modules
- Coordinates memory reads with buffer writes
- Tracks tile progress
- Supports both vector and matrix loads

**Operations:**
- `LOAD_V (0x01)`: Load vector from memory to buffer
- `LOAD_M (0x02)`: Load matrix from memory to buffer

### 3. `gemv_execution.sv`
**Purpose:** Orchestrates General Matrix-Vector multiplication

**Features:**
- Reads input vector (x) from buffer
- Reads bias vector (b) from buffer
- Streams weight matrix tiles from buffer
- Instantiates `top_gemv` for computation
- **CRITICAL FIX:** Writes results back to destination buffer

**Operation Flow:**
1. Read x vector tiles → assemble in local storage
2. Read bias vector tiles → assemble in local storage
3. Stream weight matrix tiles to GEMV unit
4. Compute y = Wx + b
5. Write result tiles to destination buffer

### 4. `relu_execution.sv`
**Purpose:** Applies ReLU activation function

**Features:**
- **CRITICAL FIX:** Reads from source buffer (`x_id`), not destination
- Applies ReLU element-wise: `max(0, x)`
- Writes activated results to destination buffer
- Processes data tile-by-tile for memory efficiency

**Operation:**
- `RELU (0x05)`: `dest = ReLU(source)`

### 5. `store_execution.sv`
**Purpose:** STORE operation — write a vector buffer back to DRAM.

**Features:**
- Tile-by-tile read from source buffer, byte-by-byte write to DRAM
- 18-bit length field per ISA (Patch H) — handles `LOAD_V` / `STORE` lengths up to 262 143

**Operation:**
- `STORE (0x03)`: `dram[addr +: length] = buffer[src_id][:length]`

### 6. `conv2d_execution.sv`
**Purpose:** Performs 2D convolution + bias + per-tensor quantization, with optional fused ReLU.

**Features:**
- Reads `x_id` (input feature map, NCHW), `w_id` (weights `[out_C, in_C·kH·kW]`), `b_id` (bias) from buffers
- Geometry latched in `tinyml_accelerator_top.sv` from a prior `CONV2D_CFG` (0x06)
- BSRAM-backed `accum_ram` (parameterized depth via `ACCUM_DEPTH`, default 4096); `$error` if out_h × out_w × out_channels > ACCUM_DEPTH
- MAC array: same `pe[0..TILE_ELEMS-1]` as GEMV; per-tile MAC sum with trailing-element gate
- Three-phase FSM: accumulate (per output channel × spatial × weight tile) → MAX_PASS → STREAM_QUANT
- **Fused ReLU** (`relu_flag` bit 25 of `CONV2D_RUN`): clamps int8 outputs to ≥ 0 in STREAM_QUANT, matching `golden_model.conv2d(apply_relu=True)`

**Operation:**
- `CONV2D_CFG (0x06)`: latch `(fmap_h, fmap_w, in_c, out_c, kh, kw, stride, pad)` — does NOT execute
- `CONV2D_RUN (0x07)`: `dest = quant(conv(x, w) + b)`, optionally with ReLU clamp

### 7. `maxpool_execution.sv`
**Purpose:** Sliding-window NCHW max-pool on int8 (no rescaling).

**Features:**
- Read addr: `c × fmap_h × fmap_w + ih × fmap_w + iw` (NCHW)
- Loop nest: `c` outermost, `oh` middle, `ow` innermost — matches conv2d's NCHW output layout (Patch G)
- Operates directly on int8; no quantization step

**Operation:**
- `MAXPOOL (0x08)`: `dest[c, oh, ow] = max(x[c, oh·s : oh·s+k, ow·s : ow·s+k])`

### 8. `modular_execution_unit.sv`
**Purpose:** Top-level coordinator.

**Features:**
- FSM: IDLE → DISPATCH → WAIT_* → COMPLETE
- One WAIT_* state per opcode (`WAIT_LOAD`, `WAIT_GEMV`, `WAIT_RELU`, `WAIT_CONV2D`, `WAIT_MAXPOOL`, `WAIT_STORE`)
- Routes buffer-controller and memory-bus signals based on the active opcode
- Threads `relu_flag` (Patch F) and `length_or_cols[17:0]` (Patch H) from the top through to the per-opcode units

## Key Improvements Over Original

### ✅ Fixed Critical Bugs

1. **ReLU Source Buffer Bug**
   - **Original:** Read from `dest` buffer instead of `x_id`
   - **Fixed:** Correctly reads from source buffer (`x_id`)

2. **GEMV Result Writeback**
   - **Original:** Results stayed in `result` array, never written to buffer
   - **Fixed:** Results written back to destination buffer for subsequent operations

3. **ReLU Length Handling**
   - **Original:** No length information (defaulted to 0)
   - **Fixed:** Length parameter properly passed and used

### ✅ Improved Maintainability

- **Separation of Concerns:** Each module has single responsibility
- **Testability:** Each module can be tested independently
- **Readability:** ~150 lines per module vs 524 lines monolithic
- **Debugging:** Easy to isolate issues to specific operations

### ✅ Better Architecture

- **Clean Interfaces:** Explicit signal routing between modules
- **Reusable Components:** Buffer controller used by all operations
- **Scalability:** Easy to add new operations (e.g., pooling, quantization)

## File Sizes

| Module | Lines | Purpose |
|---|---:|---|
| `modular_execution_unit.sv` | ~700 | Top coordinator & multiplexing (incl. CNN routing) |
| `buffer_controller.sv` | ~200 | Buffer file wrapper + random-read port |
| `load_execution.sv` | ~250 | LOAD_V / LOAD_M |
| `gemv_execution.sv` | ~330 | Matrix-vector multiply |
| `relu_execution.sv` | ~190 | Standalone ReLU |
| `store_execution.sv` | ~130 | STORE |
| `conv2d_execution.sv` | ~440 | 2D conv + bias + quant + fused ReLU |
| `maxpool_execution.sv` | ~170 | Sliding-window NCHW max-pool |
| **Total** | **~2,400** | |

## Testing

### Per-unit cocotb tests (Verilator backend)

These are the canonical regression tests — each preloads buffers via the testbench wrapper and asserts byte-equal output against `compiler/golden_model.py`:

```bash
cd test/cocotb_tests

# Conv2D — bit-exact match vs golden_model.conv2d, both relu_flag=0 and =1
./make_venv.sh TEST_TARGET=conv2d_execution

# MaxPool — bit-exact match vs golden_model.maxpool, NCHW geometry
./make_venv.sh TEST_TARGET=maxpool_execution

# Top-level execution unit (MLP integration)
./make_venv.sh TEST_TARGET=execution_unit

# GEMV unit
./make_venv.sh TEST_TARGET=gemv_execution
```

### Verilator C++ smoke tests (faster build, no Python)

```bash
cd test
verilator --cc --exe --build \
  -I../rtl -I../rtl/execution_unit \
  --top-module conv2d_execution_tb_wrapper \
  ../rtl/accelerator_config_pkg.sv \
  ../rtl/execution_unit/conv2d_execution.sv \
  ../rtl/execution_unit/buffer_controller.sv \
  ../rtl/buffer_file.sv ../rtl/pe.sv \
  ../rtl/scale_calculator.sv ../rtl/quantizer_pipeline.sv \
  conv2d_execution_tb_wrapper.sv conv2d_execution_tb.cpp
./obj_dir/Vconv2d_execution_tb_wrapper
```

### End-to-end integration

```bash
# CNN end-to-end through compiler+golden (pytest)
cd compiler && pytest test_cnn_golden.py -v

# MLP full-MNIST integration on FPGA-tier RTL (cocotb + Verilator)
cd test/heavy_test_fpga && make run_test NUM_IMAGES=20

# Sim RTL full-MNIST (10K images, larger TILE_ELEMS=32 path)
cd test/heavy_test && make run_test
```

## Usage Example

The modular execution unit has the same interface as the original:

```systemverilog
modular_execution_unit #(
    .DATA_WIDTH(8),
    .TILE_WIDTH(256),
    .MAX_ROWS(1024),
    .MAX_COLS(1024)
) exec_unit (
    .clk(clk),
    .rst(rst),
    .start(start),
    .opcode(opcode),
    .dest(dest),
    .length_or_cols(length_or_cols),
    .rows(rows),
    .addr(addr),
    .b_id(b_id),
    .x_id(x_id),
    .w_id(w_id),
    .result(result),
    .done(done)
);
```

## Neural Network Layer Example

```systemverilog
// Layer: 16 inputs → 8 outputs with ReLU

// 1. Load input vector (16 elements to buffer 9)
start_op(LOAD_V, dest=9, length=16, addr=0x1000);

// 2. Load weight matrix (8×16 to buffer 1)
start_op(LOAD_M, dest=1, rows=8, cols=16, addr=0x2000);

// 3. Load bias vector (8 elements to buffer 4)
start_op(LOAD_V, dest=4, length=8, addr=0x3000);

// 4. GEMV: y = Wx + b (result to buffer 5)
start_op(GEMV, dest=5, w_id=1, x_id=9, b_id=4, rows=8, cols=16);

// 5. ReLU: activated = max(0, y) (buffer 5 → buffer 7)
start_op(RELU, dest=7, x_id=5, length=8);
```

## Future Enhancements

### Planned Improvements
1. **Store Module:** Implement `store_v` for complete STORE operation
2. **Quantization:** Add dedicated quantization execution module
3. **Pooling:** Add max/avg pooling operations
4. **Pipeline Optimization:** Overlap operations where possible
5. **Performance Counters:** Add instrumentation for profiling

### Potential Optimizations
- **Prefetching:** Start loading next operation's data early
- **Double Buffering:** Allow computation while loading
- **Parallel Execution:** Execute independent operations concurrently

## Design Decisions

### Why Separate Vector and Matrix Buffers?
- Different size requirements (8KB vs 800KB)
- Optimized access patterns for each type
- Clear separation prevents addressing errors

### Why Write GEMV Results to Buffer?
- Enables operation chaining (GEMV → ReLU)
- Maintains consistency with load/store model
- Allows intermediate results to be reused

### Why Tile-Based Processing?
- Memory efficient (don't need full vector in registers)
- Scalable to large matrices
- Matches hardware memory bandwidth

## Compatibility

The modular execution unit is designed as a **drop-in replacement** for the original `execution_unit.sv`:

- ✅ Same port interface
- ✅ Same operation semantics
- ✅ Same timing characteristics
- ✅ Compatible with existing testbenches

To use: replace `execution_unit` instantiation with `modular_execution_unit`.

## Contributing

When adding new operations:

1. Create new execution module (e.g., `pool_execution.sv`)
2. Add module to `modular_execution_unit.sv`
3. Add case in DISPATCH state for new opcode
4. Add multiplexing logic for buffer access
5. Create unit test in `test/execution_tests/`
6. Update this README

## License

Same as parent tinyML_accelerator project.

## Authors

- Refactored modular design: GitHub Copilot (2025)
- Original execution_unit: tinyML_accelerator contributors
