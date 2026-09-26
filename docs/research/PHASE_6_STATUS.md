# Phase 6 — boardless milestone, 2026-09-26

**Later milestone:** the [cache, spatial SIMD and retention implementation](PHASE_6_OPTIMIZATION.md)
adds routed cache/parameter-reuse and spatial pointwise hardware, executable exact activation retention,
full-model RTL checks, overlap-aware bounded search and an upstream DeFiNES
geometry cross-check. It produces a substantial simulated hardware gain; the
new search nearly ties a simple retention baseline. Full B3 adaptation and G6
remain open. The rest of this file preserves the earlier foundation milestone.

**The boardless search/semantics/integration milestone passes; G6 remains open.**
This is implementation and verification, not evidence of a new state-of-the-art
accelerator. The restricted real-model search found **no gain over its
largest-tile serialized baseline**. Spatial halo caching has useful software
tradeoffs, but has not been lowered to a physically feasible fused schedule or
measured on the FPGA.

New work is isolated in `compiler/scheduler/`, `tools/phase6/`, `test/phase6/`
and `work/phase6/`. SHA256 checks confirm preservation of 109 pre-existing
Phase 5 compiler, runner, RTL and release files. No device was opened, FPGA
image changed, or Phase 5 job interrupted.

## Implementation and acceptance scope

| Item | Implemented and checked without a board | Still required for the full backlog item |
|---|---|---|
| S01 | Independent timing/placement/resource certificates; halo and INT32 reduction contracts; quantization-preserving segment executor; real-model candidates and independent command replay | Bind fused live buffers, port traces and transfers to actual lowering; agree fair B3 candidates; prove complete physical SRAM fit |
| S02 | Exact integer-time/address enumeration and catalogue DP; independent exhaustive timing/address and path oracles | Extend the declared catalogue to the final fused/B3-compatible implementation |
| S03 | Bounded beam search, valid catalogue bounds, feasible fallback, measured DMA cost integration; KWS/VWW runtime and exact-gap checks | Optimize the actual fusion/recompute/placement space; calibrate new costs and evaluate held-out prediction errors |
| S04 | DMA-cost ablation, paired serialized/prefetch artifacts, cache/recompute semantic sweeps, pending physical matrix, negative result recorded | Matched B1/B2/B3/B4 measurements and causal fusion/placement/bank/overlap ablations; an explained physical benefit |
| S05 | Optional dynamic-quantization experiment deferred | Consider only after the main mechanism demonstrates a benefit |

B03's dependency has not been waived. `retain-last-region` is a restricted
software cache policy, **not a DeFiNES reproduction**. Neither the baseline
tuner nor the spatial executor is presented as completed B4.

## Search and legality

The immutable contract binds versioned buffers and arithmetic tasks to a SHA256.
Its independent verifier checks complete production/consumption, INT32 lifetime
through requantization, aligned addresses, memory overlap, resource capacities
and half-open request intervals.

`search.py` exhaustively considers integer starts and aligned addresses within
an explicit horizon. `catalogue.py` performs exact DP or bounded beam search
over positive-length segments. Dominance is allowed only at the same position
and **complete frontier state**. The caller must encode every future-relevant
distinction in that state. The adapter's fully materialized layer boundaries
make its state sufficient; arbitrary fused frontiers do not inherit that proof.

Candidate order and expansion limits are deterministic. Wall-time cutoffs are
explicitly marked as non-reproducible truncation. Interruption never proves
infeasibility; a feasible fallback is retained when available. Optimality is
limited to the declared integer-time problem/catalogue. The relaxed catalogue
bound ignores boundary compatibility and is admissible for its additive cost,
not a lower bound on physical latency.

The **112 regression tests** include 12 independently enumerated placement/timing
problems, 30 independently enumerated random catalogues, an adversarial frontier
case, budget/timeout failures, numerical and command corruptions, and the
earlier 20 fixed-contract checks.

## Numerical and command integration

`spatial.py` executes batch-one NCHW Conv/Relu/Clip segments. Backward halos
include stride, dilation and asymmetric padding. Group/depthwise convolutions
are covered. Reduction chunks partition K exactly once, add corrected bias once,
and require an all-INT8-input INT32 bound. Requantization remains at **every
original layer boundary**, with original multiplier, shift, zero point and
rounding semantics.

Cache/no-cache execution, odd tails, extreme zero points and negative ties are
compared to `integer_reference.py`, which uses centered inputs, uncorrected
bias and independent rounding. The real-model sweep covers 18 KWS and 54 VWW
spatial layers in segments of at most four layers: **152 exact segment checks**
across two inputs, two tile sizes and two cache modes. Reshape/Transpose/pooling/
Gemm remain explicit fusion barriers. This is not an accuracy-set evaluation or
a fused full-model hardware run.

`abi_verify.py` replays raw commands without importing the tiler or trusting its
manifest. It checks graph-derived descriptor geometry, input versions and
coordinates, parameters, DMA bounds, waits, live ownership, output coverage
and SDRAM results. The oracle supplies engine outputs; this checker does not
simulate arithmetic or port timing. Six policies per model pass on a pinned
fixture and a seeded INT8 stress input: **24 complete command replays**. Pinned
fixture tensors also match the previously archived per-layer hashes.

The new integration simulation runs generated bytecode on the actual sequencer,
engine, DMA and SRAM RTL with random external-port stalls. **Four distinct
synthetic-chain artifacts pass**; two other policies are identical aliases.
Both prefetch variants exercise overlap. The fixture crosses the 16 KiB tiling
threshold. External memory is an abstract stalled RAM, not the physical SDRAM
controller. Its counters are diagnostic, not a matched physical speedup
comparison. No new synthesis or board claim follows.

## Real-model results and negative result

The executable catalogue chooses full-32KiB or preferred-16KiB tiles per layer,
with materialized SDRAM boundaries. All candidate combinations fit command and
payload capacities before frontier-state merging. Ranking uses the pinned
Phase 4 uncontended engine model plus fitted DMA costs. It excludes dispatch/
host uncertainty and overlap contention. The inherited DMA holdout errors and
calibration bitstream remain in the [cost evidence](evidence/phase4/physical-sequence-costs.json);
coefficients were not refitted. Costs are estimates, not upper bounds.

| Model | Local choices | Construction + costing + search | Beam gap to restricted exact optimum | Selected serialized component cycles |
|---|---:|---:|---:|---:|
| KWS | 30 across 22 layers | 1.40 s | 0% | 5,725,029 |
| VWW | 84 across 58 layers | 6.70 s | 0% | 18,654,475 |

Host: macOS 26.4 arm64, Python 3.13.7, with Phase 5 and the isolated RTL test
also running. CPU model is not recorded; these are observations on this host,
not portable runtime guarantees. The ≤60 s and ≤10% targets pass **only for this
restricted additive catalogue**. Each layer has the same materialized boundary
state, so its exact optimum is easy; the general placement/fusion problem has
not been solved.

Full32, mixed and DMA-blind select identical serialized artifacts. Preferred-
16KiB serialized estimates are 5,760,565 KWS and 18,794,264 VWW component cycles.
Prefetch artifacts receive no additive latency prediction. **Tuning only this
two-geometry serialized catalogue does not establish a research contribution.**

For fixed ≤4-layer spatial segments, the software experiment shows:

| Model / output tile | MACs without cache | MACs with last-region cache | Useful MACs | Peak retained cache bytes |
|---|---:|---:|---:|---:|
| KWS / 4×4 | 4,645,632 | 3,190,528 | 2,656,000 | 4,224 |
| KWS / 8×8 | 3,101,440 | 2,656,000 | 2,656,000 | 8,320 |
| VWW / 4×4 | 10,910,048 | 8,653,184 | 7,489,152 | 11,392 |
| VWW / 8×8 | 8,006,368 | 7,718,784 | 7,489,152 | 20,736 |

These count semantic Conv MACs within the supported spatial scope. Cache bytes
exclude transient inputs/outputs, weights, descriptors, banks and INT32 working
storage. They **do not prove a 32 KiB SRAM fit**, bandwidth reduction or faster
inference. Larger tiles reduce repeated work while increasing retained storage;
this is a direction to implement and test, not a measured accelerator result.

## Reproduction and evidence

Run from the repository root with pinned Phase 4 models/calibrations and fixtures:

```sh
make p6-check PYTHON="$PWD/.venv/bin/python3"
.venv/bin/python3 -m pytest compiler/test_scheduler_*.py -q --junitxml=work/phase6/tests.xml
make p6-boardless PYTHON="$PWD/.venv/bin/python3"
make p6-rtl PYTHON="$PWD/venv/bin/python3"
.venv/bin/python3 tools/phase6/summarize.py
```

The summary expects the preservation manifest at
`work/phase6/phase5-frozen-files.json`; restore it from the archived `.json.gz`
below when reproducing this snapshot. Changed pinned sources/artifacts fail
verification. The boardless runner recreates command/payload binaries under
`work/phase6/boardless/`; large binaries are not added to Git.

- [Compact results](evidence/phase6/summary.json)
- [Source-pinned software report](evidence/phase6/boardless-report.json.gz)
- [Source-pinned RTL results](evidence/phase6/rtl-candidates.json.gz)
- [112-case JUnit record](evidence/phase6/tests.xml.gz)
- [Frozen Phase 5 manifest](evidence/phase6/phase5-frozen-files.json.gz)
- [Pending physical experiment matrix](evidence/phase6/physical-matrix.json)

## Next work and gate boundary

Further boardless research work remains: implement a fair B3 policy, choose the
comparable fused catalogue, lower halo/carry/INT32 storage and strided transfers
to actual hardware, and validate the lowering in RTL. These are substantial
design tasks; the software cache is not automatically executable by the
current ABI. Certificates, oracles and search infrastructure are now available
to check those changes.

After Phase 5 releases the board, run the prepared current-ABI comparisons and
calibrate new hardware modes. Full B4 evaluation needs matched B1/B2/B3,
held-out cost errors, causal ablations, correctness/quality constraints and an
explained measured gain. **A board connection alone cannot close G6, and this
milestone is not a SOTA claim.**
