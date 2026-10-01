# Live-state hypothesis: bounded negative screen

Date: 2026-09-28. This independent task-2 screen ran while the matched B1/B2/B3
baseline work proceeded. It does not use the board, change a model, or implement
the excluded continuous audio/video experiment.

**Decision: this example does not establish a novel scheduling mechanism.** It
demonstrates why an incorrect temporary-buffer size can produce a false result,
then shows that an ordinary joint task/address formulation resolves the example.
It also exposes a mismatch with the current accelerator: spatial accumulators
remain in engine registers through requantization, so treating their INT32 bytes
as scratchpad residents invents a current storage constraint.

## Reproducible test

Run `python3 tools/phase6/hypothesis_live_state.py`. Python's standard library is
sufficient. The report is `work/phase6/hypothesis-v1/report.json`; it contains the
complete fixed-task graphs, exact schedules, placements, independent exhaustive
results, legality checks, arithmetic checks, frozen-RTL evidence and source hashes.

The four abstract tasks are an eight-element MAC, its explicit requantization,
a consuming operation, and a next-input DMA. The MAC and quantization share one
engine. Memory placement retains the accumulator until quantization completes,
and the next input remains live at the exit. The arithmetic uses asymmetric
input/output zero points and preserves both rounding boundaries.

**The unit is a synthetic tick, not a board cycle.** Durations and reservation
windows are stipulated for a small scheduling problem. They are not memory-word
traces or calibrated Tang Nano latencies. In particular, varying accumulator
storage below is a sensitivity test within this declared abstraction, not an
implemented spill/reload mechanism or a physical architecture comparison.

| Model of the same abstract tasks | Exact optimum |
| --- | ---: |
| Incorrectly count eight INT32 accumulators as eight INT8 bytes; 40-byte scratchpad, one port | 7 ticks |
| Correct 32-byte accumulator in that scratchpad | 9 ticks |
| Correct accumulator, 48-byte scratchpad | 7 ticks |
| Correct accumulator, 40-byte scratchpad and two ports | 8 ticks |
| Accumulator in a separate register resource, 40-byte scratchpad and one port | 7 ticks |

The search and a separate complete enumeration agree for all five problems.
The verifier rejects the first schedule when replayed with the correct INT32
buffer size. All 256 scalar INT8 values pass the raw/corrected-bias versus
centered-arithmetic equality check. Prematurely narrowing the accumulator changes
requantized outputs for many values; it cannot repair the capacity problem.

The sealed selected-engine source is inspected directly from the existing
archive. Its `pw_acc[0:7]` declarations and `Q_STREAM` transition retain internal
accumulator values until the requantizer accepts them. This source inspection
does not turn the abstract timings above into hardware measurements.

## What the test establishes about prior work

No DeFiNES or COSMA implementation is executed here. Calling this result a
speedup over either method would be incorrect. The exact solver is a
representability check for explicit tasks, live buffers, addresses and resource
reservations.

The pinned DeFiNES inspection already establishes memory ports, operand
precision, overlap caching and memory-level choices. Its hardware description
must be adapted faithfully before concluding that it cannot handle a constraint.
The [source audit](evidence/phase0/prior-code.json) pins revision
`7097d6090dc22321e44ce91434e7cc23b065864f`.

[COSMA §III](https://arxiv.org/html/2311.18246v1) operates on atomic operators and
jointly handles tensor lifetimes, contiguous placement and replacement. Making a
MAC and requantization separate graph nodes with an INT32 edge is an available
conceptual expansion. Sub-operation port timing would require an adaptation;
neither its implementation nor its resulting performance is established here.
The simple example therefore does not exclude a suitably expanded joint method.

## Focused next hypothesis

Test whether a compact, executable description of the live state can make
search substantially cheaper while preserving good legal schedules. That is a
search-efficiency and verification hypothesis, not an asserted new hardware
capability. First give every method the same executable catalogue, exact integer
semantics, memory costs and common optimizations. Compare exact small instances,
held-out search quality and search runtime; then measure only the surviving
full-model schedules. If the tuned baseline selects the same schedules, retain
the compiler/backend as engineering work and reject a scheduler-speedup claim.

Task 2 can construct and falsify hypotheses in parallel with task 1. A positive
claim against the strongest applicable baseline must await task 1's matched,
tuned implementation and evidence.
