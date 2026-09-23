# Phase 0–1 implementation record — 2026-09-09

**2026-09-23 physical addendum:** the reset-corrected minimal UART passes all
1,280 bytes on the connected Tang Nano 20K, and the active accelerator passes
MLP/SmallCNN model switching and full scratchpad readback. The supplied chip
marking matches `GW2AR-LV18QN88C8/I7`; JTAG identifies revision C. The user has
no power instrument, and PCB revision/second-target access remain unknown.
R04 therefore stays partial. See the [physical record](PHYSICAL_BOARD_STATUS.md).
The report below preserves the original September 9 software/build evidence;
its old accelerator synthesis failure has since been resolved by Phases 2–4.

The phase-1 **software numerical and primary-quality gate passes on the declared
research splits**. Phase 0 remains partial: physical board readback/instrument
access, AD data/calibration provenance, and the remaining prior-work audit are
open. No SOTA, full MLPerf eligibility, corrected RTL, or measured energy claim
is made. This record accompanies two scoped commits on `codex/phase-0-1`.

## Acceptance checklist

| Task | Status | Durable evidence / remaining work |
|---|---|---|
| R01 claim/prior work | Partial | `novelty_matrix.md` freezes hypothesis, comparison eligibility, DeFiNES B3 and overlap/pivot decision. Selected full texts and DeFiNES code inspected; exhaustive full-text/code audit and baseline reproduction remain open. |
| R02 benchmark freeze | Partial | All three original models convert within stated tolerances; TFLite and canonical ONNX inventories agree on MAC counts. Complete declared KWS/VWW data/calibration hashes and source recipes are frozen. AD dataset/calibration content remains unverified; official VWW MLPerf accuracy-set eligibility remains separate. |
| R03 baseline provenance | Complete | Original `3aa5fe8`, environment, dirty-path manifest, 58-test baseline, five reproduced defects and Gowin evidence in `evidence/`. |
| R04 lab/toolchain | Partial | Minimal build and 1,280-byte physical UART readback now pass; programmer/UART path and FPGA part identified. PCB revision, instrument acquisition/borrow plan and second-target access remain open. |
| Q01 arithmetic/oracle | Complete, software | Explicit version-2 INT8 contract, independent centered-input oracle, rounding/bias/overflow regressions. |
| Q02 graph semantics | Complete for declared subset | BN/affine folding, explicit biases/layouts/outputs, Conv/Gemm/pool attribute validation; unsupported cases reject. |
| Q03 quantization IR | Complete, software | Saved disjoint calibration, per-channel weights, product-unit INT32 bias, corrected bias and fixed-point requantization survive image serialization. |
| Q04 image safety | Complete, software | Versioned target, aligned nonoverlapping segments, bounds/field/parameter checks and atomic replacement; FPGA v2 target rejects. |
| Q05 implementation/quality | Complete at software gate | 115 compiler tests and ISA check; independent all-layer checks on three real inputs per primary model; complete declared KWS/VWW quality above targets; legacy heavy RTL comparisons restored to 100% exact / zero maximum error. Corrected multi-tile RTL execution remains H03/C01/H05, not claimed here. |

**G0 is open. G1's software criteria pass with the explicit split scope above; this
does not waive G0 or certify official MLPerf eligibility.** Finish the open data
and lab work alongside H01/H02 before interpreting future hardware results as a
paper-ready benchmark.

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
pattern folds with float parity; its full ROC-AUC remains unmeasured. Published
INT8 TFLite and newly calibrated v2 are distinct numerical implementations.

## Gowin and physical evidence

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

## Reproduction and next actions

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

Next: finish AD dataset/calibration and the remaining prior-work audit; obtain
physical UART and instrument access; then execute H01–H03 target consolidation,
descriptors and corrected RTL. The register-fit failure makes the planned BSRAM
memory redesign essential. Research novelty must survive R05 and tuned B3/COSMA
comparisons before any SOTA wording is justified.
