# Phase 6 integration and selected release — 1 October 2026

The integration combines the existing phase history, outstanding parallel compiler work and completed research evidence with a durable selection of the fastest board-tested KWS/VWW accelerator. All local phase branches and detached research HEAD `8f3b752` were ancestors of `codex/phase-5` at `53428af`. The integration branch is `codex/phase6-current-release`; advancing `main` through it includes every phase branch without replaying obsolete snapshots.

The selected engine is the cached-weight co-issue implementation, SHA256 `9f934dcc27ae891e4c7b8b84bcc4b65e82d903492c1867572784fbe1cd9484ee`. The original tested image is unchanged, SHA256 `eaf74e99519d3dfac876bf21e9087dd005071bb8b8682d3f8ae6fb23bb76644d`, with a 27 MHz clock, 750,000-baud UART and existing 256-byte bridge. Its matched 14-execution physical screen records median device time 41.450 ms KWS and 79.898 ms VWW. Routing gives 28.826 MHz Fmax and zero setup/hold violations. These remain short-screen results; no new board campaign or energy measurement occurs during integration.

The release is under `hardware/releases/phase6/selected/`, with exact image and four KWS/VWW pinned/stress command/payload/input/oracle fixtures. The manifest binds 18 promoted physical source identities to the archived route inventory and checks signed board records and reload/warmup ordering. A relocation-safe build uses `rtl/phase6/selected_engine.sv`, `hardware/phase6/selected_host27.sv` and `selected_pll27.v`. The generated encrypted SDRAM IP remains a hash-pinned local build artifact under ignored `work/`, recoverable from the preserved archive. The build recipe changes path resolution only; a fresh bitstream still needs its own identity and physical validation.

Validation of the promoted release passes eight fresh native jobs: KWS/VWW × pinned/stress × fixed/stalled RAM seed 6063. All outputs and the archived elapsed/engine/DMA/overlap counters match. Native evidence uses the promoted source and pins the final driver, harness and manifest. Relocation materialization/verification succeeds in a separate checkout path; a mock Tcl evaluation verifies the 18 actual add_file paths without claiming a new route. Reports are durable under `hardware/releases/phase6/selected/validation/`.

The calibration CLI now loads each NPZ input array once and yields views, avoiding a decompressed array copy for every sample. Its new memory-sharing/count regressions are included in CI. Final `make ci` passes 142 compiler tests, ISA/target generation checks and selected-release source/image/fixture/physical-record verification. No selected accelerator arithmetic or fixture bytes change.

The detached research integration preserves 4,664 evidence files byte for byte and adds 43 source/test/document paths. Its broader matched driver is named `matched_b1b2_generic.py`, preserving the existing `matched_b1b2.py` API and completed board workflows. The generic protocol is `PHASE_6_GENERIC_MATCHED_BASELINES_V1.md`; the newer physical campaign record is retained. Imported clients and live source-pin lists use the new generic module name. Archived original source bytes and hashes are unchanged.

`chain_resident.py` supports explicit prepared tile plans and conventional materialization; `constant_filter.py` forwards those options. The default KWS/VWW command and payload bytes remain identical to historical fixtures. Caller-supplied plans use an input-extent guard before retaining a producer tensor. Original and updated pinned/stress programs pass eight independent exact replays; six resident-weight backend cases, partial-consumer retention and prefetch-dependency checks also pass.

The focused integration pytest command yields 69 passes and two historical preservation-guard failures:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 -m pytest \
  compiler/test_matched_defines_regions.py tools/phase6/test_matched_defines.py \
  tools/phase6/test_archive_matched_baselines.py tools/phase6/test_matched_campaign.py \
  compiler/test_scheduler_resident.py test/phase6/test_chain_sibling.py \
  test/phase6/test_constant_sibling.py test/phase6/test_constant_filter.py -q --tb=short
```

Both failures occur before their functional checks because `check_frozen()` compares `compiler/static_cli.py` with the old Phase 5 source snapshot. The CLI fix intentionally changes that source. A diagnostic in-memory rerun validates the other 108 frozen files and both functional tests pass when only that known mismatch is admitted. No historical manifest, archived fixture or committed hash-check bypass is changed. The current release has its own new source identities; historical preservation audits should use their corresponding archived snapshots rather than treating an evolving checkout as frozen.

The project-analysis skill edit that predates the research round is captured unchanged. Generated datasets, native objects/executables, local bitstreams and work products remain ignored. Factorization, deadline-policy and failed resident-controller prototypes are committed as research artifacts and do not replace the selected accelerator. Root and hardware documentation and Make targets now point to the selected release.

Use `make p6-selected-check PYTHON="$PWD/.venv/bin/python3"` for source/image/fixture verification, `make p6-selected-native` with the same Python setting for the eight native checks, and `make p6-selected-extract` to materialize the tested image/local IP. No board is accessed by these targets.
