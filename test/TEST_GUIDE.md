# Ushqyn Test Suites

This directory contains two complementary test suites for validating the Ushqyn RTL implementation.

## Test Suites Overview

### 1. Basic Tests (`cocotb_tests/`)

**Purpose**: Fast iteration and component-level testing

**Characteristics**:
- Quick execution (~10-30 seconds)
- Tests individual modules and basic integration
- Uses small sample dataset (10 images)
- Lenient pass criteria (70% accuracy, ±2 error tolerance)
- Good for development and quick validation

**Use Cases**:
- During active development
- Quick sanity checks after changes
- Component-level debugging
- CI/CD for every commit

**Run**:
```bash
cd cocotb_tests
make TEST_TARGET=golden_comparison run_test  # Basic golden model comparison
make TEST_TARGET=load_m run_test              # Individual module tests
```

### 2. Heavy Tests (`heavy_test/`)

**Purpose**: Comprehensive production validation

**Characteristics**:
- Thorough validation (30-60 minutes for full run)
- Tests complete MNIST dataset (10,000 images)
- Strict pass criteria (85% accuracy, exact matches required)
- Comprehensive memory and timing validation
- Detailed statistics and failure analysis

**Use Cases**:
- Final validation before release
- Regression testing
- Accuracy benchmarking
- Finding corner cases
- Nightly CI/CD builds

**Run**:
```bash
cd heavy_test
make quick_test    # 100 images (~1 minute)
make run_test      # Full 10,000 images (~30-60 minutes)
make debug_test    # Debug mode with verbose output
```

## Comparison Matrix

| Feature | Basic Tests | Heavy Tests |
|---------|-------------|-------------|
| **Execution Time** | ~10-30 seconds | ~30-60 minutes (full) |
| **Dataset Size** | 10 images | 10,000 images |
| **Pass Criteria** | 70% accuracy, ±2 tolerance | 85% accuracy, exact match |
| **Memory Validation** | Basic | 4 instances verified |
| **Output Clearing** | No | Yes (prevents contamination) |
| **Statistics** | Basic | Comprehensive + per-class |
| **Failure Analysis** | Limited | Detailed with debugging |
| **Configurable** | Fixed | Highly configurable |
| **Stop on Fail** | No | Yes (optional) |
| **Verbose Mode** | Limited | Full control |

## Recommended Workflow

### During Development
1. Run basic tests frequently for quick feedback
2. Fix any failures immediately
3. Use individual module tests for targeted debugging

### Before Committing
1. Run basic golden model test
2. Run heavy quick_test (100 images)
3. Ensure both pass

### Before Release/Merge
1. Run full heavy test suite (10,000 images)
2. Review comprehensive statistics
3. Investigate any failures
4. Verify accuracy meets requirements

### Debugging Failures
1. Start with heavy test in debug mode:
   ```bash
   cd heavy_test
   make debug_test
   ```
2. Identify failing test index
3. Use basic tests for targeted module debugging
4. Re-run heavy test to verify fix

## Test Configuration

### Basic Tests
Located in `cocotb_tests/test_golden_comparison.py`:
```python
num_tests = 10  # Number of images to test
```

### Heavy Tests
Configured via environment variables:
```bash
NUM_IMAGES=1000           # Test first N images
STOP_ON_FIRST_FAIL=1      # Stop on first failure
VERBOSE=1                 # Enable verbose output
```

## Understanding Test Results

### Basic Tests Pass
- RTL is functionally correct for basic cases
- Safe for continued development
- May not catch edge cases

### Heavy Tests Pass
- RTL is production-ready
- Handles full dataset correctly
- Meets accuracy requirements
- Memory synchronization verified

### Heavy Tests Fail
- Check specific failure patterns:
  - Low accuracy: Neural network issue
  - Output mismatches: Quantization or arithmetic errors
  - Memory sync failures: DRAM access issues
  - Timeouts: FSM hangs or performance issues

## CI/CD Integration

### Fast CI (every commit)
```yaml
- name: Quick Validation
  run: |
    cd test/cocotb_tests
    make TEST_TARGET=golden_comparison run_test
```

### Thorough CI (nightly or pre-merge)
```yaml
- name: Heavy Validation
  run: |
    cd test/heavy_test
    make run_test NUM_IMAGES=1000  # Subset for CI time
```

### Full Validation (pre-release)
```yaml
- name: Full Heavy Test
  run: |
    cd test/heavy_test
    make run_test  # All 10,000 images
```

## Directory Structure

```
test/
├── cocotb_tests/              # Basic/fast tests
│   ├── test_golden_comparison.py
│   ├── test_load_m.py
│   ├── test_load_v.py
│   ├── test_buffer_controller.py
│   └── utils/
│       └── accelerator_tester.py
│
├── heavy_test/                # Comprehensive tests
│   ├── test_full_mnist.py
│   ├── Makefile
│   ├── README.md
│   └── quick_validate.py
│
└── TEST_GUIDE.md (this file)
```

## Troubleshooting

### Basic Tests Pass, Heavy Tests Fail
- Issue likely in edge cases or specific input patterns
- Run heavy test with STOP_ON_FIRST_FAIL=1 to find first failure
- Check if issue is systematic or isolated

### Both Tests Fail
- Fundamental RTL issue
- Start with basic component tests
- Fix at module level before re-running integration

### Intermittent Failures
- Memory synchronization issue
- Check that all 4 RTL memory instances are synced
- Verify output region clearing

### Performance Issues
- Heavy test too slow: Test subset first
- Use `NUM_IMAGES=100` for faster iteration
- Disable VCD tracing if not needed

## Best Practices

1. **Run basic tests frequently** during development
2. **Run heavy quick_test** before committing
3. **Run full heavy test** before major releases
4. **Use debug mode** when investigating failures
5. **Review statistics** to understand failure patterns
6. **Clear artifacts** between test runs:
   ```bash
   make clean_sim
   ```

## Adding New Tests

### To Basic Tests
Add to `cocotb_tests/`:
- Good for: Component tests, specific scenarios
- Fast execution required
- Add as new `@cocotb.test()` function

### To Heavy Tests
Add to `heavy_test/test_full_mnist.py`:
- Good for: Dataset variations, boundary cases
- Can be slower, more thorough
- Integrate with main loop or add as separate test

## Getting Help

If tests fail and you're unsure why:
1. Review test output and statistics
2. Run in debug/verbose mode
3. Check the README in each test directory
4. Review RTL module documentation
5. Use VCD traces for signal-level debugging

## Summary

- **Quick iteration**: Use `cocotb_tests`
- **Thorough validation**: Use `heavy_test`
- **Both should pass** for production-ready code
- **Heavy tests** are the final authority on correctness
