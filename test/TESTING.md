# Ushqyn — Testing Documentation

## Quick Start

```bash
# Quick validation (recommended for development)
cd test
./run_all_tests.sh quick

# Full validation (recommended before release)
cd test
./run_all_tests.sh full
```

## Test Suites

### 1. Basic Tests (`test/cocotb_tests/`)
- **Purpose**: Fast iteration and component testing
- **Execution Time**: ~10-30 seconds
- **Dataset**: 10 MNIST images
- **Pass Criteria**: 70% accuracy, ±2 error tolerance
- **Use For**: Development, quick validation, CI per-commit

```bash
cd test/cocotb_tests
make TEST_TARGET=golden_comparison run_test
```

### 2. Heavy Tests (`test/heavy_test/`) ⭐ NEW
- **Purpose**: Comprehensive production validation
- **Execution Time**: 1-60 minutes (configurable)
- **Dataset**: Up to 10,000 MNIST images
- **Pass Criteria**: 85% accuracy, exact matches required
- **Use For**: Final validation, regression testing, releases

```bash
cd test/heavy_test
make quick_test    # 100 images (~1 min)
make run_test      # 10,000 images (~30-60 min)
make debug_test    # Debug mode
```

## Key Improvements in Heavy Tests

Based on thorough analysis of the original test suite, the heavy tests address:

✅ **Memory Contamination Prevention**
- Clears output region before each test
- Prevents false positives from stale data

✅ **Strict Validation**
- Requires exact output matches (no tolerance)
- Verifies all 4 RTL memory instances stay synced
- Validates done pulse behavior

✅ **Comprehensive Statistics**
- Overall accuracy with per-class breakdown
- Detailed failure analysis
- Performance metrics (tests/second)

✅ **Configurable Execution**
- Test any subset of dataset
- Stop on first failure for debugging
- Verbose mode for detailed output

✅ **Production-Ready**
- Strict pass criteria (85% accuracy, exact matches)
- Exhaustive testing (10,000 images)
- Detailed failure reporting

## Configuration

### Heavy Test Options

Environment variables:
```bash
NUM_IMAGES=1000           # Test first N images (default: 10000)
STOP_ON_FIRST_FAIL=1      # Stop on first failure (default: 0)
VERBOSE=1                 # Verbose output (default: 0)
```

Examples:
```bash
# Test 500 images
make run_test NUM_IMAGES=500

# Debug mode - stop on first failure with verbose output
make run_test NUM_IMAGES=100 STOP_ON_FIRST_FAIL=1 VERBOSE=1

# Quick validation during development
make quick_test
```

## Test Results Interpretation

### If Basic Tests Pass ✓
- RTL functionally correct for basic cases
- Safe for continued development
- Ready for more thorough testing

### If Heavy Tests Pass ✓
- **RTL is production-ready**
- Handles full dataset correctly
- Meets accuracy requirements
- Memory synchronization verified
- Ready for deployment

### If Heavy Tests Fail ✗
Check the failure pattern:
- **Low accuracy**: Neural network computation issue
- **Output mismatches**: Quantization or arithmetic errors
- **Memory sync failures**: DRAM access or synchronization issues
- **Timeouts**: FSM hangs or performance problems

## Recommended Workflow

### During Development
```bash
# Run basic tests frequently
cd test/cocotb_tests
make TEST_TARGET=golden_comparison run_test
```

### Before Committing
```bash
# Run both basic and quick heavy test
cd test
./run_all_tests.sh quick
```

### Before Release
```bash
# Run full validation
cd test
./run_all_tests.sh full
```

### When Debugging Failures
```bash
# Use heavy test debug mode
cd test/heavy_test
make debug_test

# Or stop on specific failure
make run_test NUM_IMAGES=<failure_index+1> STOP_ON_FIRST_FAIL=1 VERBOSE=1
```

## CI/CD Integration

### Fast CI (every commit)
```yaml
- name: Quick Tests
  run: |
    cd test/cocotb_tests
    make TEST_TARGET=golden_comparison run_test
```

### Nightly/Pre-merge CI
```yaml
- name: Comprehensive Tests
  run: |
    cd test
    ./run_all_tests.sh quick
```

### Release Validation
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
│   └── utils/
│       └── accelerator_tester.py
│
├── heavy_test/                # NEW: Comprehensive tests
│   ├── test_full_mnist.py    # Main test suite
│   ├── Makefile              # Build and run configuration
│   ├── README.md             # Detailed documentation
│   └── quick_validate.py     # Quick validation script
│
├── run_all_tests.sh          # NEW: Run both suites
└── TEST_GUIDE.md             # NEW: Detailed guide
```

## Documentation

- **Test Guide**: `test/TEST_GUIDE.md` - Comprehensive guide comparing both test suites
- **Heavy Test README**: `test/heavy_test/README.md` - Detailed heavy test documentation
- **This File**: Quick reference and getting started

## Performance Expectations

| Test Suite | Quick Mode | Full Mode |
|------------|-----------|-----------|
| Basic | 10s | 30s |
| Heavy | 60s (100 img) | 30-60min (10k img) |

## Troubleshooting

### Tests Timeout
- Increase timeout in test configuration
- Check for RTL hangs with VCD traces
- Verify clock is running

### Memory Sync Failures
- Check that DRAM sync is called after changes
- Verify all 4 memory instances are accessible
- Review memory hierarchy paths

### Output Mismatches
- Enable verbose mode to see detailed comparison
- Check if output region was cleared
- Verify golden model produces expected results
- Review quantization settings

### Performance Issues
- Test smaller subset first: `NUM_IMAGES=100`
- Disable VCD tracing if not needed
- Run on faster machine for full tests

## Getting Help

1. Review test output and statistics
2. Run in debug/verbose mode
3. Check README files in test directories
4. Use VCD traces for signal-level debugging
5. Review `test/TEST_GUIDE.md` for detailed comparison

## Summary

- **Development**: Use basic tests for quick feedback
- **Pre-commit**: Run `./run_all_tests.sh quick`
- **Pre-release**: Run `./run_all_tests.sh full`
- **Both must pass** for production-ready code
- **Heavy tests** are the final authority on correctness
