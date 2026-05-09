# Heavy MNIST Test Suite

This directory contains comprehensive validation tests for the Ushqyn RTL implementation.

## Overview

The heavy test suite performs exhaustive validation against the golden model using the complete MNIST test dataset (10,000 images). This is separate from the basic `cocotb_tests` and provides production-grade validation.

## Key Features

### Improvements Over Basic Tests

1. **Full Dataset Coverage**: Tests all 10,000 MNIST test images (configurable)
2. **Strict Validation**: Requires exact output matches - no tolerance for errors
3. **Memory Contamination Prevention**: Clears output region before each test
4. **Comprehensive Verification**:
   - Verifies DRAM contents (single unified `main_memory`; the legacy 4-memory check in `accelerator_tester.verify_all_memories_synced()` gracefully handles `AttributeError` for the old paths)
   - Validates done pulse behavior
   - Checks that STORE actually wrote data
5. **Detailed Statistics**:
   - Per-class accuracy breakdown
   - Failure analysis with debugging info
   - Performance metrics (tests/second)
6. **Configurable Execution**:
   - Test subset or full dataset
   - Stop on first failure for debugging
   - Verbose mode for detailed output

### Test Coverage

- **Functional Correctness**: RTL outputs match golden model exactly (bit-exact, no tolerance)
- **Accuracy Validation**: Neural network achieves expected classification accuracy on MNIST
- **Memory Integrity**: Unified `main_memory` matches the in-Python DRAM image after every load
- **Timing Behavior**: Done signal pulses correctly only on program-end (zero instruction)
- **Output Verification**: STORE writes complete results

## Usage

### Quick Start

```bash
# Run quick test (first 100 images)
make quick_test

# Run full test suite (all 10,000 images)
make run_test

# Debug mode (stop on first failure, verbose output)
make debug_test
```

### Advanced Usage

```bash
# Test specific number of images
make run_test NUM_IMAGES=1000

# Stop on first failure
make run_test STOP_ON_FIRST_FAIL=1

# Verbose output for all tests
make run_test VERBOSE=1

# Combination
make run_test NUM_IMAGES=500 STOP_ON_FIRST_FAIL=1 VERBOSE=1
```

### Configuration Options

Environment variables:

- `NUM_IMAGES`: Number of images to test (default: 10000)
- `STOP_ON_FIRST_FAIL`: Stop on first mismatch (default: 0)
- `VERBOSE`: Enable detailed output (default: 0)

## Test Structure

### Main Test: `test_full_mnist_dataset`

Comprehensive validation with:
1. **Initialization**: Load model, generate assembly, sync memories
2. **Dataset Loading**: Load MNIST test set
3. **Main Test Loop**: For each image:
   - Prepare input and clear output region
   - Execute RTL
   - Verify done pulse and output written
   - Execute golden model
   - Compare outputs with strict exact matching
   - Record statistics
4. **Results Analysis**: 
   - Overall accuracy
   - Per-class breakdown
   - Failure analysis
   - Performance metrics

### Boundary Test: `test_boundary_cases`

Tests edge conditions:
- All-zeros input
- Maximum values
- Minimum values
- (Extensible for additional cases)

## Pass Criteria

The test suite applies strict pass criteria:

1. **RTL Accuracy**: Must achieve ≥85% classification accuracy
2. **Exact Match Rate**: ≥95% of outputs must match golden model exactly
3. **Maximum Error**: No output error exceeds 3

All three criteria must pass for overall test pass.

## Output and Debugging

### Console Output

The test provides:
- Progress indicators every 100 tests
- Per-test results (for verbose mode or first few tests)
- Comprehensive statistics at end
- Failure analysis with first 10 failures detailed

### Artifacts

- `results.xml`: JUnit-style test results
- `test_results.csv`: Detailed per-image results (see below)
- VCD traces (if enabled): Signal waveforms for debugging
- Console logs: Detailed execution trace

### CSV Results File

After each test run, detailed results are saved to `test_results.csv` with the following columns:

- `test_idx`: Test index (0-based)
- `label`: True label (0-9)
- `rtl_prediction`: RTL predicted digit
- `golden_prediction`: Golden model predicted digit
- `rtl_correct`: Whether RTL prediction matches label
- `golden_correct`: Whether golden prediction matches label
- `exact_match`: Whether RTL and golden outputs match exactly
- `max_error`: Maximum error between RTL and golden outputs
- `rtl_output`: Full RTL output vector (10 values)
- `golden_output`: Full golden model output vector (10 values)

**Example analysis:**
```bash
# Use the built-in analysis script
python analyze_results.py test_results.csv

# Or manual analysis:

# View all failures
grep "False" test_results.csv

# Count correct predictions
awk -F, '$5=="True" {count++} END {print count}' test_results.csv

# Find tests with high error
awk -F, '$8>2 {print}' test_results.csv

# View in spreadsheet
open test_results.csv  # macOS
# or import into Excel/LibreOffice
```

The `analyze_results.py` script provides:
- Overall accuracy statistics
- Per-class breakdown
- Error analysis
- First 20 failures with details
- Pass/fail evaluation

### Debugging Failed Tests

If tests fail:

1. Run debug mode to stop on first failure:
   ```bash
   make debug_test
   ```

2. Check the failure details in console output:
   - RTL vs Golden output comparison
   - Predicted vs actual labels
   - Maximum error value

3. For specific test index, set:
   ```bash
   make run_test NUM_IMAGES=<failed_index+1> STOP_ON_FIRST_FAIL=1 VERBOSE=1
   ```

4. Review memory sync verification messages

5. Check VCD traces for signal-level debugging

## Performance

Typical performance:
- ~2-5 tests/second (depending on host performance)
- Full 10,000 image run: ~30-60 minutes
- Quick test (100 images): ~30-60 seconds

## Integration with CI/CD

The test suite is designed for CI/CD integration:

```yaml
# Example GitHub Actions
- name: Run Heavy Tests
  run: |
    cd test/heavy_test
    make run_test NUM_IMAGES=1000  # Subset for CI
```

For full validation, run locally or in nightly CI jobs.

## Comparison with Basic Tests

| Feature | Basic (`cocotb_tests`) | Heavy (`heavy_test`) |
|---------|----------------------|---------------------|
| Dataset size | 10 images | 10,000 images (configurable) |
| Pass criteria | 70% accuracy, ±10% tolerance | 85% accuracy, exact match required |
| Memory validation | Basic | Comprehensive (4 instances) |
| Output clearing | No | Yes |
| Statistics | Basic | Comprehensive with per-class |
| Debugging | Limited | Extensive |
| Execution time | Fast (~10s) | Thorough (~30-60min full) |

Use basic tests for quick iteration, heavy tests for validation.

## Troubleshooting

### Common Issues

1. **Memory sync failures**:
   - Check that all RTL memory instances are being written
   - Verify `sync_dram_to_rtl()` is called after DRAM changes

2. **Timeout errors**:
   - Increase timeout in `execute_all(timeout_cycles=...)`
   - Check for RTL hangs with VCD traces

3. **Output mismatches**:
   - Verify output region is cleared before test
   - Check golden model execution
   - Review quantization settings

4. **Performance issues**:
   - Test subset first: `NUM_IMAGES=100`
   - Disable VCD tracing if not needed
   - Run on faster machine

## Future Enhancements

Potential additions:
- Parallel test execution
- Additional boundary cases
- Custom input validation
- Layer-by-layer output verification
- Adversarial test cases
- Regression tracking across commits

## Contact

For issues or questions about the heavy test suite, refer to the main project documentation or open an issue in the repository.
