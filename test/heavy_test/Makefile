# Makefile for Heavy MNIST Test Suite
# This runs comprehensive validation on the full MNIST dataset

# Paths
RTL_DIR = ../../rtl
TEST_DIR = .
COCOTB_DIR = ../cocotb_tests

# Verilator settings
TOPLEVEL_LANG = verilog
SIM = verilator

# DUT
TOPLEVEL = tinyml_accelerator_top
MODULE = test_full_mnist

# Verilator flags
EXTRA_ARGS += --trace --trace-structs
EXTRA_ARGS += -Wno-WIDTH -Wno-CASEINCOMPLETE -Wno-CASEX -Wno-TIMESCALEMOD
EXTRA_ARGS += -Wno-UNOPTFLAT -Wno-PINMISSING -Wno-UNUSED -Wno-BLKANDNBLK -Wno-UNSIGNED
EXTRA_ARGS += --timing  # Enable timing support
EXTRA_ARGS += -CFLAGS "-std=c++14"

# Source files - include all RTL modules
VERILOG_SOURCES = $(RTL_DIR)/accelerator_config_pkg.sv
VERILOG_SOURCES += $(RTL_DIR)/tinyml_accelerator_top.sv
VERILOG_SOURCES += $(RTL_DIR)/fetch_unit.sv
VERILOG_SOURCES += $(RTL_DIR)/i_decoder.sv
VERILOG_SOURCES += $(RTL_DIR)/simple_memory.sv
VERILOG_SOURCES += $(RTL_DIR)/load_v.sv
VERILOG_SOURCES += $(RTL_DIR)/load_m.sv
VERILOG_SOURCES += $(RTL_DIR)/store.sv
VERILOG_SOURCES += $(RTL_DIR)/buffer_file.sv
VERILOG_SOURCES += $(RTL_DIR)/pe.sv
VERILOG_SOURCES += $(RTL_DIR)/top_gemv.sv
VERILOG_SOURCES += $(RTL_DIR)/quantization.sv
VERILOG_SOURCES += $(RTL_DIR)/quantizer_pipeline.sv
VERILOG_SOURCES += $(RTL_DIR)/scale_calculator.sv
VERILOG_SOURCES += $(RTL_DIR)/wallace_32x32.sv
VERILOG_SOURCES += $(RTL_DIR)/compressor_3to2.sv
VERILOG_SOURCES += $(RTL_DIR)/relu.sv

# Execution unit modules
VERILOG_SOURCES += $(RTL_DIR)/execution_unit/modular_execution_unit.sv
VERILOG_SOURCES += $(RTL_DIR)/execution_unit/buffer_controller.sv
VERILOG_SOURCES += $(RTL_DIR)/execution_unit/load_execution.sv
VERILOG_SOURCES += $(RTL_DIR)/execution_unit/gemv_execution.sv
VERILOG_SOURCES += $(RTL_DIR)/execution_unit/conv2d_execution.sv
VERILOG_SOURCES += $(RTL_DIR)/execution_unit/maxpool_execution.sv
VERILOG_SOURCES += $(RTL_DIR)/execution_unit/relu_execution.sv
VERILOG_SOURCES += $(RTL_DIR)/execution_unit/store_execution.sv

# Python path setup
# Python path setup
export PYTHONPATH := $(TEST_DIR):$(COCOTB_DIR)/utils:$(PYTHONPATH)

# Test configuration via environment variables
# Usage examples:
#   make run_test                              # Run all 10,000 images
#   make run_test NUM_IMAGES=100               # Test first 100 images
#   make run_test START_IMAGE=1000             # Start from image 1000
#   make run_test START_IMAGE=1000 NUM_IMAGES=500  # Test images 1000-1499
#   make run_test STOP_ON_FIRST_FAIL=1         # Stop on first failure for debugging
#   make run_test VERBOSE=1                    # Enable verbose output for all tests

# Default: test all images
NUM_IMAGES ?= 10000
export NUM_IMAGES

# Start from specific image (useful for resuming failed tests)
START_IMAGE ?= 0
export START_IMAGE

# Stop on first failure (useful for debugging)
STOP_ON_FIRST_FAIL ?= 0
export STOP_ON_FIRST_FAIL

# Verbose mode
VERBOSE ?= 0
export VERBOSE

# Include cocotb makefile
include $(shell cocotb-config --makefiles)/Makefile.sim

# Custom targets
.PHONY: run_test clean_sim quick_test debug_test

# Run full test suite
# Workaround: Verilator + Cocotb $finish exits before results.xml is written
run_test:
	@$(MAKE) sim 2>&1 | tee test_output.log; \
	if grep -q "TEST SUITE PASSED" test_output.log || (grep -q "FAIL=0" test_output.log && grep -q "PASS=" test_output.log); then \
		echo "\n✓ All tests PASSED (results.xml issue ignored)"; \
		exit 0; \
	else \
		echo "\n✗ Tests FAILED - check output above"; \
		exit 1; \
	fi


# Quick test - first 100 images
quick_test:
	$(MAKE) run_test NUM_IMAGES=100

# Debug test - stop on first failure with verbose output
debug_test:
	$(MAKE) run_test NUM_IMAGES=100 STOP_ON_FIRST_FAIL=1 VERBOSE=1

# Clean simulation artifacts
clean_sim:
	rm -rf sim_build
	rm -rf __pycache__
	rm -rf *.vcd
	rm -rf results.xml
	rm -rf .pytest_cache

# Help
help:
	@echo "Heavy MNIST Test Suite"
	@echo "======================="
	@echo ""
	@echo "Targets:"
	@echo "  run_test       - Run full test suite (default: all 10,000 images)"
	@echo "  quick_test     - Run quick test (first 100 images)"
	@echo "  debug_test     - Run debug test (100 images, stop on first fail, verbose)"
	@echo "  clean_sim      - Clean simulation artifacts"
	@echo "  help           - Show this help"
	@echo ""
	@echo "Configuration (environment variables):"
	@echo "  NUM_IMAGES=N           - Test N images (default: 10000)"
	@echo "  START_IMAGE=N          - Start from image N (default: 0)"
	@echo "  STOP_ON_FIRST_FAIL=1   - Stop on first failure (default: 0)"
	@echo "  VERBOSE=1              - Enable verbose output (default: 0)"
	@echo ""
	@echo "Examples:"
	@echo "  make run_test NUM_IMAGES=1000"
	@echo "  make run_test START_IMAGE=5000 NUM_IMAGES=100     # Test images 5000-5099"
	@echo "  make debug_test"
	@echo "  make run_test STOP_ON_FIRST_FAIL=1 VERBOSE=1"
