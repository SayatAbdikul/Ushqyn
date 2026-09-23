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
	@echo "  test-compiler  Historical compiler and static INT8 v2 regressions"
	@echo "  check-isa      Verify rtl/i_decoder.sv matches compiler/isa_spec.py"
	@echo "  config         Regenerate historical v1 simulation package"
	@echo "  v2-config      Regenerate active Tang Nano target package"
	@echo "  v2-lint        Strict Verilator lint of active board hierarchy"
	@echo "  v2-fixture     Build trained MLP image and 1,000 exact oracle cases"
	@echo "  v2-test        Full phase-2 RTL simulation suites (includes 1,000 jobs)"
	@echo "  p3-fixture     Build trained SmallCNN and 1,000 independent cases"
	@echo "  p3-native      Run 1,000 SmallCNN jobs through board-system RTL"
	@echo "  p3-quality     Evaluate all 10,000 MNIST images in float and INT8"
	@echo "  p3-switch      Test MLP↔SmallCNN model switching in one RTL system"
	@echo "  heavy-test     Full MLP MNIST cocotb test (requires Verilator)"
	@echo "                 Pass NUM_IMAGES=N to test N images (default: 2 here)."
	@echo "  clean          Remove generated artifacts and caches"

# ── CI tier (no simulator) ───────────────────────────────────────────────────

ci: test-compiler check-isa check-v2-target
	@echo ""
	@echo "✓ CI tier passed (compiler pytest + ISA/target --check)"

test-compiler:
	cd compiler && $(PYTHON) -m pytest \
	    test_isa_spec.py \
	    test_cnn_golden.py \
	    test_accelerator_config.py \
	    test_unified_walker.py \
	    test_buffer_allocator.py \
	    test_static_pipeline.py \
	    test_hardware_v2.py \
	    test_memory_planner.py \
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

# Active phase-2 target. Legacy targets above preserve historical regression paths.
.PHONY: check-v2-target v2-config v2-lint v2-fixture v2-test
check-v2-target:
	$(PYTHON) tools/phase2/generate_target.py --check

v2-config:
	$(PYTHON) generate_config.py --target v2

v2-lint:
	$(PYTHON) tools/phase2/lint.py

v2-fixture:
	$(PYTHON) tools/phase2/prepare_mlp.py

v2-test: check-v2-target v2-lint v2-fixture
	$(PYTHON) test/phase2/run.py requantizer
	$(PYTHON) test/phase2/run.py engine
	$(PYTHON) test/phase2/run.py board
	$(PYTHON) test/phase2/run.py system

.PHONY: p3-fixture p3-native p3-quality p3-switch
p3-fixture:
	$(PYTHON) tools/phase3/prepare_smallcnn.py --jobs 1000

p3-native: check-v2-target v2-lint p3-fixture
	$(PYTHON) tools/phase3/run_native.py

p3-quality:
	$(PYTHON) tools/phase3/prepare_smallcnn.py --jobs 10000 --quality-only --output work/phase3/full-quality

p3-switch: check-v2-target v2-lint p3-fixture v2-fixture
	$(PYTHON) test/phase3/run.py
