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
	@echo "  p4-kernels     Exact KWS/VWW-shaped kernel RTL tests"
	@echo "  p4-dma         Standalone abstract-port tile-transfer RTL tests"
	@echo "  p4-core        Concurrent engine/DMA shared-SRAM RTL test"
	@echo "  p4-tiled-program  Packed multi-layer and multi-tile RTL/oracle test"
	@echo "  p4-tiled-host  UART-framed SDRAM upload/DMA/engine RTL test"
	@echo "  p4-refresh     Standalone SDRAM refresh scheduler RTL test"
	@echo "  p4-audit       Audit frozen KWS/VWW operator and storage inventories"
	@echo "  p4-test        Run the reproducible Phase 4 simulation tier"
	@echo "  p5-plan        Pin the balanced 10,000-job audio/vision switch plan"
	@echo "  p5-audit       Inventory Phase 5 evidence and run its regressions"
	@echo "  p6-check       Boardless scheduler, semantics and command checks"
	@echo "  p6-boardless   Pinned KWS/VWW candidate and spatial experiments"
	@echo "  p6-rtl         Isolated candidate sequencer/engine/DMA simulation"
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
	    test_phase4_kernels.py \
	    test_phase4_tiling.py \
	    test_phase4_sequence.py \
	    test_phase4_compile.py \
	    test_phase4_rebase.py \
	    test_phase4_sdram_geometry.py \
	    test_memory_planner.py \
	    -q --tb=short

check-isa:
	$(PYTHON) tools/generate_i_decoder.py --check

# Isolated research preparation; no board access or Phase 5 payload mutation.
.PHONY: p6-check p6-boardless p6-rtl
p6-check:
	$(PYTHON) -m pytest compiler/test_scheduler_*.py -q
	$(PYTHON) tools/phase6/check_contract.py

p6-boardless:
	$(PYTHON) tools/phase6/run_boardless.py

p6-rtl:
	$(PYTHON) tools/phase6/run_rtl.py

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

.PHONY: p4-kernels p4-dma p4-core p4-tiled-program p4-tiled-host p4-real-host p4-refresh p4-audit p4-plan p4-pingpong-audit p4-test
p4-kernels: check-v2-target v2-lint
	$(PYTHON) test/phase4/run.py

p4-dma:
	$(PYTHON) test/phase4/run_dma.py

p4-core:
	$(PYTHON) test/phase4/run_tiled_core.py

p4-tiled-program:
	$(PYTHON) test/phase4/run_tiled_program.py

p4-tiled-host:
	$(PYTHON) -m pytest tools/phase4/test_tiled_host.py -q --tb=short
	$(PYTHON) test/phase4/run_tiled_host.py

p4-real-host:
	$(PYTHON) test/phase4/run_tiled_host.py --fixture work/phase4/rtl-kws
	$(PYTHON) test/phase4/run_tiled_host.py --fixture work/phase4/rtl-vww

p4-refresh:
	$(PYTHON) test/phase4/run_sdram_refresh.py

p4-burst-port:
	$(PYTHON) test/phase4/run_burst_port.py

p4-audit:
	$(PYTHON) tools/phase4/audit_inventories.py

p4-plan:
	$(PYTHON) tools/phase4/plan_tiling.py --check

p4-pingpong-audit:
	$(PYTHON) tools/phase4/audit_pingpong_feasibility.py

p4-test: ci p4-kernels p4-dma p4-core p4-tiled-program p4-tiled-host p4-refresh p4-burst-port p4-audit p4-plan p4-pingpong-audit

.PHONY: p5-plan p5-audit
p5-plan:
	$(PYTHON) tools/phase5/schedule.py --summary docs/research/evidence/phase5/switch-plan.json

p5-audit: p5-plan
	$(PYTHON) -m unittest discover -s tools/phase5 -p 'test_*.py' -v
	$(PYTHON) tools/phase5/audit.py --summary docs/research/evidence/phase5/readiness.json
