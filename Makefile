# Top-level Makefile for the tinyML accelerator project.
# Run `make help` for the target list.
#
# Layered targets:
#   - `ci`            — fast, simulator-free; runs on every push (GitHub Actions)
#   - `heavy-test`    — full RTL bit-exactness through Verilator + cocotb (local)
#   - `clean`         — remove generated artifacts and Python/Verilator caches
#
# The `ci` target is the lock-in for the P1/P2/P3 structural fixes:
# it runs the cross-consistency tests for the ISA spec, the AcceleratorConfig
# profiles, the unified ONNX walker, and the CNN end-to-end golden, plus a
# `--check` of the on-disk RTL decoder against the spec. Any drift breaks CI.

PYTHON ?= python3

.PHONY: help ci test-compiler check-isa config heavy-test clean

help:
	@echo "Targets:"
	@echo "  ci             Fast tier (no simulator). Runs all compiler-side"
	@echo "                 pytest plus tools/generate_i_decoder.py --check."
	@echo "                 This is what GitHub Actions runs."
	@echo "  test-compiler  Compiler-side pytest (44 cases, ~3-5 sec)"
	@echo "  check-isa      Verify rtl/i_decoder.sv matches compiler/isa_spec.py"
	@echo "  config         Regenerate rtl/accelerator_config_pkg.sv from SimProfile"
	@echo "  heavy-test     Full MLP MNIST cocotb test (requires Verilator)"
	@echo "                 Pass NUM_IMAGES=N to test N images (default: 2 here)."
	@echo "  clean          Remove generated artifacts and caches"

# ── CI tier (no simulator) ───────────────────────────────────────────────────

ci: test-compiler check-isa
	@echo ""
	@echo "✓ CI tier passed (compiler pytest + ISA --check)"

test-compiler:
	cd compiler && $(PYTHON) -m pytest \
	    test_isa_spec.py \
	    test_cnn_golden.py \
	    test_accelerator_config.py \
	    test_unified_walker.py \
	    test_buffer_allocator.py \
	    -q --tb=short

check-isa:
	$(PYTHON) tools/generate_i_decoder.py --check

config:
	$(PYTHON) generate_config.py

# ── Heavy tier (Verilator required) ──────────────────────────────────────────

# Override NUM_IMAGES on the command line: `make heavy-test NUM_IMAGES=20`
NUM_IMAGES ?= 2

heavy-test:
	$(MAKE) -C test/heavy_test run_test NUM_IMAGES=$(NUM_IMAGES)

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean:
	@find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .pytest_cache -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name sim_build -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name obj_dir -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.vcd" -not -path "./.git/*" -delete 2>/dev/null || true
	@find . -type f -name "results.xml" -not -path "./.git/*" -delete 2>/dev/null || true
	@find . -type f -name "test_output.log" -not -path "./.git/*" -delete 2>/dev/null || true
	@rm -f compiler/dram.hex compiler/disassembled.asm
	@echo "✓ Cleaned"
