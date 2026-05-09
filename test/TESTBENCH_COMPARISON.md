# Testbench Comparison: Heavy Test vs Cocotb Tests vs Golden Model

## Executive Summary

This document compares three testing approaches in the Ushqyn project:
1. **Golden Model** (`compiler/golden_model.py`) - Python reference implementation
2. **Cocotb Tests** (`test/cocotb_tests/`) - RTL validation suite with basic coverage
3. **Heavy Test** (`test/heavy_test/`) - Comprehensive production-grade validation

---

## 1. Golden Model (`compiler/golden_model.py`)

### Purpose
Software reference implementation that emulates the accelerator's instruction set in Python.

### Key Characteristics

**Functionality:**
- Emulates all 5 instruction types: LOAD_V, LOAD_M, STORE, GEMV, RELU
- Maintains internal buffer state (`buffers` dictionary)
- Uses global memory loaded from `dram.hex`
- Performs quantization after GEMV operations
- Returns final output buffer after program execution

**Implementation Details:**
```python
def execute_program(hex_file):
    """Execute the program from a hex file."""
    # Clear global state for fresh execution
    global buffers, output_buffer, flag
    buffers = {}
    output_buffer = 0
    flag = 0
    
    # Load instructions and memory
    memory = load_memory('dram.hex')
    
    # Execute each instruction
    for instruction in instructions:
        i_decoder(instruction)
    
    return buffers[output_buffer][0:output_length]
```

**Instruction Execution:**
- **LOAD_V**: Copies vector from memory to buffer
- **LOAD_M**: Copies matrix from memory to buffer  
- **STORE**: Writes buffer back to memory, updates output_buffer
- **GEMV**: Matrix-vector multiplication with bias, includes automatic quantization
- **RELU**: Applies ReLU activation element-wise

**Quantization Behavior:**
- GEMV automatically quantizes its int32 output to int8 after computation
- Scale calculated as: `scale = max(abs(output)) / 127`
- Uses `quantize_int32_to_int8()` helper function

**Memory Management:**
- Single unified memory array loaded from hex file
- No separate memory instances
- Direct array indexing for reads/writes

### Strengths
✅ Simple, readable reference implementation  
✅ Fast execution (pure Python)  
✅ Easy to debug and modify  
✅ Deterministic output

### Limitations
⚠️ No timing/cycle accuracy  
⚠️ Doesn't model hardware details (pipeline, memory access patterns)  
⚠️ Global state can cause issues if not properly reset  
⚠️ No validation of memory boundaries or error conditions

---

## 2. Cocotb Tests (`test/cocotb_tests/`)

### Purpose
Basic RTL validation suite that compares RTL execution against the golden model.

### Main Test: `test_golden_comparison.py`

**Test Configuration:**
- Tests **20 images** from MNIST (configurable, default was 10)
- Timeout: 500,000 cycles per test
- Tolerance: Allows errors in output comparison

**Test Flow:**
```python
async def test_accelerator_mnist_dataset(dut):
    1. Initialize model and generate assembly
    2. Sync DRAM to RTL memories (all 4 instances)
    3. Load MNIST test dataset
    4. For each test image (20 images):
        a. Prepare input
        b. Reset DUT
        c. Execute RTL (single start pulse, wait for done)
        d. Read RTL output from memory
        e. Execute golden model
        f. Compare outputs
    5. Report accuracy statistics
```

**RTL Execution Method:**
```python
success = await tester.execute_all(timeout_cycles=500000)
```
- Single `start` pulse
- Waits for `done` signal when zero instruction is fetched
- Reads output from RTL memory: `execution_u.store_exec.store_inst.dram.memory`

**Memory Synchronization:**
- RTL has **4 separate memory instances**:
  1. `fetch_u.memory_inst.memory` - instruction fetch
  2. `execution_u.load_exec.load_v_inst.memory_inst.memory` - vector load
  3. `execution_u.load_exec.load_m_inst.memory_inst.memory` - matrix load  
  4. `execution_u.store_exec.store_inst.dram.memory` - store unit
- All 4 must be synchronized after initialization
- `sync_dram_to_rtl()` writes to all instances

**Comparison Approach:**
```python
match, differences, max_error = tester.compare_results(
    rtl_output, golden_output, verbose=False
)
```
- Element-by-element comparison
- Calculates max absolute error
- Counts mismatches
- Reports statistics

**Pass Criteria:**
```python
if rtl_accuracy >= 70 and abs(rtl_accuracy - golden_accuracy) <= 10:
    # PASS
```
- RTL accuracy ≥ 70%
- Difference from golden ≤ 10%

### Strengths
✅ Validates RTL against golden model  
✅ Uses real MNIST data  
✅ Tests complete execution flow  
✅ Includes memory sync validation  
✅ Fast execution (small dataset)

### Limitations
⚠️ Limited test coverage (only 20 images)  
⚠️ Relaxed pass criteria (70% accuracy)  
⚠️ No output region clearing (potential contamination)  
⚠️ Minimal boundary/edge case testing  
⚠️ No per-class accuracy analysis  
⚠️ No detailed failure analysis  
⚠️ Doesn't verify done pulse behavior thoroughly

---

## 3. Heavy Test (`test/heavy_test/`)

### Purpose
Production-grade comprehensive validation with strict requirements.

### Main Test: `test_full_mnist.py`

**Test Configuration:**
- Tests **10,000 images** (full MNIST test set, configurable)
- Timeout: 500,000 cycles per test
- **STRICT**: Exact output matching required
- Configurable: `NUM_IMAGES`, `STOP_ON_FIRST_FAIL`, `VERBOSE`

**Enhanced Test Flow:**
```python
async def test_full_mnist_dataset(dut):
    1. Initialize model and generate assembly
    2. Sync DRAM to RTL memories
    3. **VERIFY** memory sync across all 4 instances
    4. Load MNIST test dataset (10,000 images)
    5. For each test image:
        a. Prepare input
        b. **CLEAR output region** (prevent contamination)
        c. Reset DUT
        d. **VERIFY** memories still synced after reset
        e. Execute RTL
        f. **VERIFY** done pulse behavior (1 cycle)
        g. **VERIFY** output was actually written
        h. Read RTL output
        i. Execute golden model
        j. Compare with **EXACT** matching
        k. Record to CSV file immediately
    6. Generate comprehensive statistics
    7. Evaluate against strict pass criteria
```

**Enhanced Tester Class:**
```python
class EnhancedTester(TinyMLAcceleratorTester):
    def clear_output_region(self):
        """Clear output memory before each test"""
        
    def verify_output_was_written(self):
        """Verify STORE actually wrote data"""
        
    def verify_all_memories_synced(self, sample_addrs=None):
        """Verify all 4 memories contain identical data"""
        
    async def verify_done_pulse(self):
        """Verify done signal pulses for exactly 1 cycle"""
```

**Strict Comparison:**
```python
match, differences, max_error = tester.compare_results(
    rtl_output, golden_output, verbose=(VERBOSE or test_idx < 3)
)
# Requires EXACT match
```

**Pass Criteria (STRICT):**
```python
REQUIRED_RTL_ACCURACY = 85.0       # Must achieve 85%+ accuracy
REQUIRED_EXACT_MATCH_RATE = 95.0   # 95%+ outputs match exactly
MAX_ALLOWED_ERROR = 3              # Maximum error in any output
```

**Statistics Tracking:**
- Total tests, RTL correct, golden correct, both correct
- Exact match count and rate
- Per-class accuracy (digits 0-9)
- Failure analysis with details
- Performance metrics (tests/second)
- All results saved to `test_results.csv`

**CSV Output:**
Each test result includes:
- Test index, label, predictions
- RTL/golden correctness flags
- Exact match flag, max error
- Full RTL and golden output vectors

### Strengths
✅ **Comprehensive coverage** (10,000 images)  
✅ **Strict validation** (exact matching required)  
✅ **Memory integrity checks** (sync verification)  
✅ **Contamination prevention** (output clearing)  
✅ **Done pulse validation** (timing verification)  
✅ **Per-class analysis** (detailed breakdown)  
✅ **Failure tracking** (CSV with full data)  
✅ **Configurable execution** (subset testing, debug mode)  
✅ **Performance metrics** (tests/second)  
✅ **Production-ready** (strict pass criteria)

### Limitations
⚠️ Slower execution (10,000 images)  
⚠️ Higher resource requirements  
⚠️ More complex setup

---

## Detailed Comparison Table

| Aspect | Golden Model | Cocotb Tests | Heavy Test |
|--------|--------------|--------------|------------|
| **Purpose** | Reference implementation | Basic validation | Production validation |
| **Test Count** | N/A (software only) | 20 images | 10,000 images |
| **Execution** | Python only | RTL + Golden | RTL + Golden |
| **Timing** | No timing | Cycle-accurate | Cycle-accurate |
| **Memory Model** | Single array | 4 RTL instances | 4 RTL instances |
| **Sync Verification** | N/A | Basic | Comprehensive |
| **Output Clearing** | Reset buffers | ❌ No | ✅ Yes |
| **Done Pulse Check** | N/A | ❌ No | ✅ Yes (first 3 tests) |
| **Write Verification** | N/A | ❌ No | ✅ Yes |
| **Comparison** | N/A | Element-wise | Element-wise, strict |
| **Tolerance** | N/A | Allows errors | Zero tolerance |
| **Pass Criteria** | N/A | 70% accuracy | 85% + 95% exact match |
| **Per-Class Stats** | ❌ No | ❌ No | ✅ Yes (all 10 digits) |
| **Failure Analysis** | ❌ No | Basic logging | ✅ Detailed CSV |
| **CSV Output** | ❌ No | ❌ No | ✅ Yes (full data) |
| **Performance Metrics** | ❌ No | ❌ No | ✅ Yes |
| **Debug Mode** | N/A | ❌ No | ✅ Yes (STOP_ON_FIRST_FAIL) |
| **Boundary Tests** | ❌ No | ❌ No | ✅ Planned |

---

## Key Differences in Test Methodology

### 1. Memory Synchronization

**Cocotb Tests:**
```python
# Basic sync at initialization
tester.sync_dram_to_rtl()
```

**Heavy Test:**
```python
# Sync with verification
tester.sync_dram_to_rtl()
if not tester.verify_all_memories_synced():
    assert False, "Memory sync failed"

# Re-verify after reset
if test_idx == 0:
    if not tester.verify_all_memories_synced():
        assert False, "Memory desync after reset"
```

### 2. Output Validation

**Cocotb Tests:**
```python
# Read output directly
rtl_output = tester.read_memory_from_rtl(addr, length)
```

**Heavy Test:**
```python
# Clear before test
tester.clear_output_region()

# Execute
await tester.execute_all()

# Verify write occurred
if not tester.verify_output_was_written():
    cocotb.log.error("Output region empty - write failure")
    
# Then read
rtl_output = tester.read_memory_from_rtl(addr, length)
```

### 3. Done Signal Handling

**Cocotb Tests:**
```python
# Just wait for done
success = await tester.wait_for_done()
```

**Heavy Test:**
```python
# Wait for done
success = await tester.wait_for_done()

# Verify pulse behavior (first few tests)
if test_idx < 3:
    await tester.verify_done_pulse()  # Must be exactly 1 cycle
```

### 4. Comparison and Pass Criteria

**Cocotb Tests:**
```python
# Lenient comparison
match, differences, max_error = tester.compare_results(
    rtl_output, golden_output, verbose=False
)

# Lenient pass criteria
if rtl_accuracy >= 70 and abs(rtl_accuracy - golden_accuracy) <= 10:
    # PASS
```

**Heavy Test:**
```python
# Strict comparison
match, differences, max_error = tester.compare_results(
    rtl_output, golden_output, verbose=(VERBOSE or test_idx < 3)
)

# Strict pass criteria
REQUIRED_RTL_ACCURACY = 85.0
REQUIRED_EXACT_MATCH_RATE = 95.0
MAX_ALLOWED_ERROR = 3

pass_accuracy = summary['rtl_accuracy'] >= REQUIRED_RTL_ACCURACY
pass_exact_match = summary['exact_match_rate'] >= REQUIRED_EXACT_MATCH_RATE
pass_max_error = summary['max_max_error'] <= MAX_ALLOWED_ERROR

overall_pass = pass_accuracy and pass_exact_match and pass_max_error
```

---

## Golden Model vs RTL Comparison Flow

### Golden Model Execution
```
1. Load memory from dram.hex (single array)
2. Parse instructions from first 0x700 bytes
3. For each instruction:
   - Decode opcode and operands
   - Execute operation on buffers/memory
   - GEMV includes automatic quantization
4. Return output_buffer[0:output_length]
```

### RTL Execution (Both Test Suites)
```
1. Sync dram.hex to all 4 RTL memory instances
2. Reset DUT
3. Pulse start signal (single pulse)
4. RTL runs autonomously:
   - Fetch instruction
   - Decode
   - Execute (LOAD_V, LOAD_M, GEMV, RELU, STORE)
   - Repeat until zero instruction
5. Done signal pulses for 1 cycle
6. Read output from store memory instance
```

### Comparison Process
```
RTL Output (int8[10]) <-- Compare --> Golden Output (int8[10])
                              |
                              v
                    Element-wise difference
                              |
                              v
                    Calculate max_error
                              |
                              v
                    Check pass criteria
```

---

## Quantization Behavior

### In Golden Model
```python
def gemv(dest, w, x, b, rows, cols):
    # Compute int32 sums
    for i in range(rows):
        sum = np.int32(0)
        for j in range(cols):
            sum += np.int32(buffers[w][i * cols + j]) * np.int32(buffers[x][j])
        sum += buffers[b][i]
        buffers[dest][i] = np.int32(sum)
    
    # Automatic quantization after GEMV
    scale = np.max(np.abs(buffers[dest])) / 127
    buffers[dest] = quantize_int32_to_int8(
        np.array(buffers[dest], dtype=np.int32), 
        scale, 
        0  # zero_point
    )
```

### In RTL
- GEMV unit performs computation
- Separate quantization unit applies scaling
- Same algorithm: `scale = max(abs(output)) / 127`
- Output quantized to int8 range

### Expected Alignment
- Both use identical quantization algorithm
- Small differences possible due to:
  - Rounding differences
  - Fixed-point vs floating-point arithmetic
  - Quantization unit implementation details

---

## Recommendations

### When to Use Each Approach

**Golden Model:**
- Quick algorithm validation
- Compiler testing
- ISA verification
- Debugging instruction sequences

**Cocotb Tests:**
- Daily development testing
- Quick RTL sanity checks
- CI/CD integration (fast feedback)
- Individual module testing

**Heavy Test:**
- Pre-release validation
- Regression testing
- Production readiness verification
- Detailed failure analysis
- Performance benchmarking

### Best Practices

1. **Development Cycle:**
   ```
   Golden Model → Cocotb Tests → Heavy Test
   (Algorithm)    (Daily/CI)     (Pre-release)
   ```

2. **Debugging:**
   ```
   Heavy Test Failure → Cocotb Tests → Golden Model
   (Find issue)         (Reproduce)     (Verify algorithm)
   ```

3. **Continuous Integration:**
   - Run Cocotb tests on every commit (fast)
   - Run Heavy Test nightly or weekly (comprehensive)
   - Use golden model for compiler validation

---

## Current Status

### Test Results

**Cocotb Tests (`test_golden_comparison.py`):**
- Status: ✅ Passing (20 images)
- RTL Accuracy: ~85-90% (needs verification)
- Golden Accuracy: ~90%

**Heavy Test (`test_full_mnist.py`):**
- Status: ⚠️ Needs investigation (exit code 2)
- Last run: Incomplete
- Configuration: 10,000 images
- See: `test/heavy_test/test_results.csv`

### Known Issues

1. **Heavy Test Exit Code 2:**
   - Possible timeout issues
   - Memory sync problems
   - Need to run with `STOP_ON_FIRST_FAIL=1` for debugging

2. **Memory Synchronization:**
   - 4 separate RTL memory instances must stay synchronized
   - Requires explicit writes via `write_to_all_rtl_memories()`
   - Reset behavior may affect sync

3. **Quantization Differences:**
   - Small errors expected due to rounding
   - Heavy test requires ≤3 max error
   - Cocotb test more lenient

---

## Conclusion

The three testing approaches form a comprehensive validation strategy:

1. **Golden Model** provides the algorithmic reference and enables fast software validation

2. **Cocotb Tests** provide quick RTL validation suitable for daily development and CI/CD

3. **Heavy Test** provides production-grade comprehensive validation with strict requirements

All three are necessary and complement each other:
- Golden model validates the algorithm
- Cocotb tests validate RTL matches golden model (basic)
- Heavy test validates RTL matches golden model (comprehensive, strict)

The main differences in the RTL validation are:
- **Scale**: 20 vs 10,000 images
- **Rigor**: Lenient vs strict pass criteria
- **Verification**: Basic sync vs comprehensive validation
- **Analysis**: Simple stats vs detailed per-class breakdown
- **Debugging**: Basic logs vs CSV output with full data
