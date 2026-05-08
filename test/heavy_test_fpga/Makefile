# Makefile for Heavy MNIST Test Suite - FPGA Module Variant
# This runs comprehensive validation on the full MNIST dataset using the
# FPGA-optimized execution modules from rtl/fpga_modules/.
#
# Differences from heavy_test/Makefile:
#   - Uses rtl/fpga_modules/ execution modules (BRAM-based GEMV, smaller accumulators)
#   - Uses a local tinyml_accelerator_top.sv adapted for the FPGA module interfaces
#   - The FPGA modules use the same TILE_WIDTH=256/TILE_ELEMS=32 for simulation
#     but internally implement GEMV with a 6-element tile-bridging core (gemv_unit_core)
#   - Buffer sizes are overridden to simulation values so the same MNIST model fits

# Paths
RTL_DIR  = ../../rtl
TEST_DIR = .
COCOTB_DIR = ../cocotb_tests

# Verilator settings
TOPLEVEL_LANG = verilog
SIM = verilator

# DUT
TOPLEVEL = tinyml_accelerator_top
MODULE   = test_full_mnist

# Verilator flags
EXTRA_ARGS += --trace --trace-structs
EXTRA_ARGS += -Wno-WIDTH -Wno-CASEINCOMPLETE -Wno-CASEX -Wno-TIMESCALEMOD
EXTRA_ARGS += -Wno-UNOPTFLAT -Wno-PINMISSING -Wno-UNUSED -Wno-BLKANDNBLK
EXTRA_ARGS += -Wno-PINCONNECTEMPTY  # FPGA buffer_file exposes debug_w_tile_index
EXTRA_ARGS += --timing  # Enable timing support
EXTRA_ARGS += -CFLAGS "-std=c++14"

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------

# Local FPGA-adapted top-level (uses exec_result[0:31] and FPGA buffer params)
VERILOG_SOURCES  = $(TEST_DIR)/tinyml_accelerator_top.sv

# Shared RTL modules (same between simulation and FPGA variants)
VERILOG_SOURCES += $(RTL_DIR)/fetch_unit.sv
VERILOG_SOURCES += $(RTL_DIR)/i_decoder.sv
VERILOG_SOURCES += $(RTL_DIR)/simple_memory.sv
VERILOG_SOURCES += $(RTL_DIR)/load_v.sv
VERILOG_SOURCES += $(RTL_DIR)/load_m.sv
VERILOG_SOURCES += $(RTL_DIR)/store.sv
VERILOG_SOURCES += $(RTL_DIR)/pe.sv
VERILOG_SOURCES += $(RTL_DIR)/quantization.sv
VERILOG_SOURCES += $(RTL_DIR)/quantizer_pipeline.sv
VERILOG_SOURCES += $(RTL_DIR)/scale_calculator.sv
VERILOG_SOURCES += $(RTL_DIR)/wallace_32x32.sv
VERILOG_SOURCES += $(RTL_DIR)/compressor_3to2.sv
VERILOG_SOURCES += $(RTL_DIR)/relu.sv

# FPGA-specific execution modules (replace rtl/execution_unit/ equivalents)
# These use BRAM-based storage and a tile-bridging GEMV core to save resources.
VERILOG_SOURCES += $(RTL_DIR)/fpga_modules/buffer_file.sv
VERILOG_SOURCES += $(RTL_DIR)/fpga_modules/buffer_controller.sv
VERILOG_SOURCES += $(RTL_DIR)/fpga_modules/modular_execution.sv
VERILOG_SOURCES += $(RTL_DIR)/fpga_modules/load_execution.sv
VERILOG_SOURCES += $(RTL_DIR)/fpga_modules/gemv_execution.sv
VERILOG_SOURCES += $(RTL_DIR)/fpga_modules/relu_execution.sv
VERILOG_SOURCES += $(RTL_DIR)/fpga_modules/store_execution.sv
VERILOG_SOURCES += $(RTL_DIR)/fpga_modules/gemv_unit_core.sv
VERILOG_SOURCES += $(RTL_DIR)/fpga_modules/Gowin_RAM16SDP_Mock.sv
VERILOG_SOURCES += $(RTL_DIR)/fpga_modules/Gowin_SDPB_32.sv

# Python path setup. After P3 we no longer ship a local `accelerator_config.py`
# overriding compiler/accelerator_config.py; the FPGA-tier config is selected
# via the TINYML_PROFILE env var below.
export PYTHONPATH := $(TEST_DIR):$(COCOTB_DIR)/utils:$(PYTHONPATH)

# Profile selection (P3): compiler/accelerator_config.py reads this at import
# time and exposes the matching profile under the historical AcceleratorConfig
# name. Must be exported so the cocotb subprocess inherits it.
export TINYML_PROFILE := fpga

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

# cocotb's Makefile.inc exports PYTHONHOME pointing to its Python (e.g. 3.10 framework).
# Verilator's verilated.mk invokes 'python3' which may be a different version (e.g. 3.13
# from Homebrew), causing "Failed to import encodings" because PYTHONHOME doesn't match.
# Fix: prepend the directory of cocotb's Python so 'python3' resolves to the same version.
export PATH := $(dir $(shell cocotb-config --python-bin)):$(PATH)

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
	@echo "Heavy MNIST Test Suite - FPGA Module Variant"
	@echo "============================================="
	@echo ""
	@echo "This test uses rtl/fpga_modules/ (BRAM-based GEMV, tile-bridging"
	@echo "architecture) instead of rtl/execution_unit/ (register-based)."
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
