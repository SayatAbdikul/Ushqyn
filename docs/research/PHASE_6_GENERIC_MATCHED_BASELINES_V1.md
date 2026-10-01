# Matched baseline campaign

Date: 2026-09-28. This campaign implements and tunes the baselines before using
their performance to assess a research contribution. A restricted or unfinished
adaptation is not evidence of superiority over the complete published method.

## Fixed comparison contract

Every policy uses the frozen KWS and VWW ONNX models and calibration files, the
same original integer oracle, and identical pinned and seeded stress inputs.
Stress input seed is 6157. Exact channel permutation/compaction, symbolic final
constant folding where applicable, constant-filter handling, activation LUT
epilogues, and legal prefetch are common compiler opportunities. They preserve
the original arithmetic and output bytes; none is counted as a new scheduler's
exclusive advantage.

The common hardware is the already routed and physically screened pooled engine:
eight INT8 lanes, INT32 accumulation, 32 KiB scratchpad, 8 MiB board SDRAM,
32 KiB command memory, 27 MHz, and the 256-byte UART bridge at 750 kbaud.
Bitstream SHA256:
`3e943639382e4efea62d4d2072c016fa40eacc7a97ef6bad7f5bfdc69d8dafce`.
All policies therefore use the same 10,095 CLS, 38 BSRAM and 9.5 DSP resources.
The routed Fmax is 27.613 MHz with zero reported setup/hold violations.

Measured device latency is the hardware sequencer's start-to-HALT counter. It
includes execution, scheduled DMA, packing, command processing and stalls. It
excludes model/input UART upload and output readback; those host times are
recorded separately. Native RTL RAM-model cycles are a tuning screen, not a
substitute for board SDRAM timing or an energy measurement.

## Policies

| Policy | Decisions under test | Common features retained |
| --- | --- | --- |
| B1 | Conventional channel tiling, fixed alternating scratchpad halves where tiles fit, full-memory fallback and materialized inter-macro boundaries | Exact graph optimizations, producer activation epilogue, legal double-buffer/input/parameter prefetch |
| B2 | Choose tiles first, then first-fit allocation using tensor liveness and legal sibling-input retention; allow B1 fallback | The same graph, arithmetic, engine and prefetch opportunities |
| B3 adaptation | DeFiNES depth-first stack/geometry choices with recomputation, horizontal caching, or horizontal plus vertical caching, lowered to this FPGA's actual DMA/COPY/descriptor ABI | The same model transformations, constant handling, epilogue, retained immutable data and legal prefetch |
| Current | Existing three-pair VWW strip schedule plus exact final-constant folding; existing compacted KWS schedule | Same optimized graph and hardware |

The updated B1 is stronger than the historical serialized preliminary B1. Do not
mix its new results with historical labels without describing this change.
DeFiNES source is pinned at revision
`7097d6090dc22321e44ce91434e7cc23b065864f`; this is a target-specific adaptation,
not a reproduction of another paper's hardware, power or published throughput.

The existing three-pair schedule is a feasible full-width depth-first policy:
each VWW DW/activation/PW/activation stack uses two strips, while source prefetch
protects an aliased external slot. These are code generation and placement
choices available to the adaptation. Include this explicitly manual incumbent
in B3's candidate set; do not claim that upstream's unmodified search selected
it. A speedup over a slower generic lowering would not establish novelty.

## Tuning and verification

B1/B2 initially enumerate ten preferred tile capacities (including 256-byte LUT
reserve variants) and both prefetch choices. The physical capacity is always
32 KiB; the preferred tile size is not an artificial reduction of resources.
Report all rejected candidates, equivalent execution programs and construction
time. Both native stall seeds 0 and 6063 are scored; do not select from seed 0
alone. The declared board shortlist has at most three distinct schedules per
model and policy. The search is bounded, not a claim of global optimality.

B3 must report its complete implemented search axes, actual candidate counts,
selection cost, feasibility failures and remaining coverage limitations. Geometry
or traffic estimates without executable commands do not qualify as measured
baseline results. Packing, overlap caches, descriptor memory and buffer guards
must fit the actual machine. Cache-induced recomputation savings alone are not
latency savings: COPY and command overhead are also executed and counted.

The first B2 implementation retains complete inputs and repeated sibling inputs.
It conservatively materializes a producer for partial consuming tiles. The
remaining opportunity for range-aware retention is being audited; the bounded
catalogue must remain explicit until that audit or extension is complete.

Before a physical run, `tools/phase6/matched_campaign.py` checks the 109-file
milestone freeze, the sealed physical reference, the exact programmed bitstream,
the 18-source Gowin project, raw timing/route evidence, the native executable,
compiler/model hashes, and every fixture. It explicitly maps historical absolute
build paths to this worktree without rewriting historical evidence. New native
results are bound to their executable and complete fixture file hashes.

For each distinct model/schedule, the short board screen executes one stress
inference and three timed inferences, each immediately preceded by a warmup after
reloading that schedule. Policy order alternates across timing rounds and rotates
across models. Byte-identical execution/check artifacts share one measured group;
the report identifies this reuse rather than inflating the sample count. Report
each median and range, then choose the fastest measured shortlist candidate for
each baseline/model. These are exploratory tuning results; a later held-out
confirmation and full accuracy/stability campaign remain separate.

## Parallel hypothesis track

Task 2 can construct counterexamples while task 1 builds these baselines. Its
first bounded screen rejected an INT32 scratchpad-lifetime novelty claim: the
current engine holds accumulators in registers, and an ordinary joint formulation
explains the abstract example once resources are represented correctly. See
[the hypothesis decision](PHASE_6_HYPOTHESIS_LIVE_STATE_V1.md).

A positive novelty or speedup claim must wait for the strongest applicable
matched baseline and causal ablations. Completing compiler machinery, passing
small RTL tests, or beating an unfinished B3 does not close Phase 6.

## Evidence locations

- B1/B2 final search and validation:
  `work/phase6/matched-baselines-v1/b1b2-final-v1/`.
- Generic rectangular/caching native checks:
  `work/phase6/matched-defines-native-smoke-v1/report.json`.
- Current schedule revalidation: `work/phase6/matched-baselines-v1/current-v4/`.
- Physical plan/results: `work/phase6/matched-baselines-v1/`.

Result status and final output paths must be read from the generated reports.
This protocol does not assert that the physical campaign or complete B3 tuning
has finished.
