# Phase 0–1 implementation record — updated 2026-09-26

**[Phase 0 is complete for the declared research contract](PHASE_0_CLOSURE.md).**
The September 26 closure adds AD data/calibration provenance, a refreshed claim
and prior-work decision, and the lab measurement plan. Minimal physical UART
readback already passed all 1,280 bytes. The FPGA part and revision C are known;
the PCB silkscreen revision remains unknown. No meter or second FPGA is available.
Those constraints are recorded explicitly, with physical energy and second-target
validation remaining later-phase requirements.

G1's software numerical and primary-quality gate passes on its declared research
splits. Official MLPerf eligibility and SOTA remain unestablished. Numerical and
build tables below preserve the original September 9 evidence; later Phases 2–4
resolved the old synthesis/RTL restrictions and add their own physical records.

## Acceptance checklist

| Task | Status | Durable evidence / remaining work |
|---|---|---|
| R01 claim/prior work | Complete, decision gate | `novelty_matrix.md` and `PRIOR_WORK_AUDIT.md` freeze the hypothesis, eligibility, DeFiNES B3, code inspection and overlap/pivot decision. Baseline reproduction remains B03; final prior-work refresh remains E04. |
| R02 benchmark freeze | Complete, research splits | All source conversions/inventories pass. KWS/VWW and AD data/calibration content hashes are frozen; AD adds 248 evaluation and 112 calibration recordings with pinned preprocessing. Official MLPerf eligibility remains separate. |
| R03 baseline provenance | Complete | Original `3aa5fe8`, environment, dirty-path manifest, 58-test baseline, five reproduced defects and Gowin evidence in `evidence/`. |
| R04 lab/toolchain | Complete, capability/plan gate | Minimal physical UART passed; specimen/toolchain identified. `LAB_MEASUREMENT_PLAN.md` records the acquisition route, supply/clock/marker setup and uncertainty protocol. Instrument/second-target access is unconfirmed; PCB revision stays null. |
| Q01 arithmetic/oracle | Complete, software | Explicit version-2 INT8 contract, independent centered-input oracle, rounding/bias/overflow regressions. |
| Q02 graph semantics | Complete for declared subset | BN/affine folding, explicit biases/layouts/outputs, Conv/Gemm/pool attribute validation; unsupported cases reject. |
| Q03 quantization IR | Complete, software | Saved disjoint calibration, per-channel weights, product-unit INT32 bias, corrected bias and fixed-point requantization survive image serialization. |
| Q04 image safety | Complete, software | Versioned target, aligned nonoverlapping segments, bounds/field/parameter checks and atomic replacement; FPGA v2 target rejects. |
| Q05 implementation/quality | Complete at software gate | 115 compiler tests and ISA check; independent all-layer checks on three real inputs per primary model; complete declared KWS/VWW quality above targets; legacy heavy RTL comparisons restored to 100% exact / zero maximum error. Corrected multi-tile RTL execution remains H03/C01/H05, not claimed here. |

**G0 passes its evidence-contract gate. G1 passes its software gate.** Neither
certifies official MLPerf eligibility, measured energy, or later research gates.

## Numerical results

| Workload / full declared split | Original float | Static INT8 software v2 | Threshold |
|---|---:|---:|---:|
| KWS, 4,890 canonical Speech Commands v2 test clips | 92.17% (4,507) | **92.31% (4,514)** | 90% |
| VWW, 10,961 training-recipe validation images | 84.49% (9,261) | **84.29% (9,239)** | 80% |

Calibration uses 96 KWS training clips/windows and the 11 pinned upstream VWW
images, disjoint by both IDs and feature hashes. VWW uses the complete first 10%
per-class filename partition of the Silicon Labs 96×96 archive with no random
augmentation. It is **not certified as the official MLPerf accuracy split**.
Logits are produced by the software engine, with host argmax; probability outputs
are outside this boundary. Full accuracy uses the production integer evaluator.
The separate independent oracle checks all 22 KWS / 58 VWW layers on first,
middle and last real samples: 144,630 / 491,266 integer values per sample,
respectively, with zero mismatches. It does not reuse production quantization or
execution helpers. No corrected RTL/board inference was run.

All source-framework conversions passed 16 deterministic probes at `atol=1e-5`,
`rtol=1e-4`. Rejected KWS hybrid-TFLite and AD TFLite-to-source routes are preserved;
valid original-source conversions replace them. AD's exported dense BN Mul/Add
pattern folds with float parity. Its ROC-AUC was unmeasured on September 9;
the September 26 closure adds source-float research-split ROC-AUC, while INT8/FPGA
AD quality remains unmeasured. Published
INT8 TFLite and newly calibrated v2 are distinct numerical implementations.

## Gowin and physical evidence — historical September 9 snapshot

Fresh target: GW2AR-LV18QN88C8/I7 revision C, Gowin Education V1.9.11.03.
The minimal UART design uses 75 LUT and 55 registers and completes routing and
bitstream generation. Its internal-path Fmax is 304.389 MHz under a 27-MHz
constraint, with a generic-clock-routing warning and incomplete asynchronous I/O
constraints. It has not been programmed/read back on a board.

The current accelerator fails synthesis: **273,847 inferred DFF versus 15,750
available**. Current post-route timing and bitstream therefore do not exist.
Recovered installed historical reports show 89.201 MHz synthesis Fmax and
**37.502 MHz routed Fmax**, with 43 BSRAM primitives; source hashes differ from
the current checkout. Historical reports cannot prove current implementation fit.

Minimal-design Gowin power is an **estimate of 125.162 mW** under default activity,
including 122.800 mW quiescent power. It is not board power or inference energy.
See [hardware procedure](../../hardware/README.md) for exact build/readback commands
and outstanding physical acquisition requirements.

## Software reproduction — September 9 snapshot

Run `make ci PYTHON=/absolute/path/to/compiler/python`. The final run passed
115 tests plus generated-ISA consistency; warnings originate in the existing
legacy NumPy/PyTorch path and are preserved in the log. The original baseline's
58 tests required the local ignored trained MLP weight fixture, whose hash is
recorded; that fixture is not redistributed. The new static test module adds
57 tests. No tests were skipped in these two recorded runs.

[Benchmark instructions](../../benchmarks/README.md) provide source fetching,
conversion, preprocessing, compilation and full evaluation commands. Manifests
contain every sample hash plus image/code/environment hashes. Both primary
images rebuilt byte for byte from saved calibration after the last arithmetic
import change. Large model/data/image payloads are outside Git and must be
regenerated or cached from those manifests.

The legacy generator is now explicit `generate_legacy_assembly`; the ordinary
entry point directs callers to the v2 compiler rather than silently producing a
known-invalid new executable. Existing hardware deployments need H01–H03 migration.
The heavy RTL tests now fail on any integer mismatch; restoring strict checks is
not equivalent to passing those tests with the future arithmetic.

The original next steps were AD provenance, prior-work review, UART/instrument
access and H01–H03 target consolidation. The September 26 G0 closure and later
hardware records now supersede that queue; physical instrumentation remains
unavailable. Research novelty still must survive tuned B3/COSMA comparisons
before any SOTA wording is justified. Follow the [current backlog](../RESEARCH_BACKLOG.md).
