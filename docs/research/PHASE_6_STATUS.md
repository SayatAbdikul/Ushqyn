# Phase 6 preparation — 2026-09-26

**Boardless preparation has started; G6 remains open.** This work is isolated in
`compiler/scheduler/` and `tools/phase6/`. Active Phase 5 runners do not import
these modules. Their RTL, bitstream, compiler lowering, command images and
physical test fixtures are unchanged. No UART or JTAG access is needed here.

## What can proceed alongside Phase 5

| Item | Useful work without the FPGA | What still depends on later evidence |
|---|---|---|
| S01 candidates and certificates | Define the fixed arithmetic/task contract; check dependencies, live INT32/INT8 storage, addresses, banks/ports and transfer ownership. Add halo/partial-sum semantics and a real-model adapter next. | Actual command/RTL expressiveness and the fair B3 candidate space must agree before S01 closes. |
| S02 exact small search | Implement bounded exhaustive/DP search and compare with independently enumerated small cases. | Optimality applies only to the declared candidate catalog, not arbitrary schedules. |
| S03 practical search | Deterministic search budgets, feasible fallback, cost-model integration and runtime/quality evaluation in software. | New fusion/caching costs need simulation and eventually physical validation; old kernel costs alone do not establish those costs. |
| S04 ablations | Generate matched configurations, analysis scripts and experimental manifests. | Board latency/traffic, fair tuned B1/B2/B3 and measured benefits are required to close the research gate. |

The backlog dependency on B03 remains intact for gate acceptance. Independent
specification, checker and small-instance work can be prepared before B03 is
finished. The current Phase 5 command format lacks DeFiNES's general spatial
cross-layer caching/recomputation modes. Neither renaming B2 nor constraining B3
to fit a convenient optimizer establishes a fair comparison.

## Implemented starting point

`contract.py` defines immutable memories, resource capacities, versioned buffers
and fixed tasks. A certificate supplies only task start times and buffer
addresses. Its SHA256 binds it to the complete problem, including each task's
arithmetic identity and explicit requantization operation.

`verify.py` is independent of a candidate generator. It checks:

- Every fixed task appears once and every buffer has a placement and producer.
- Consumers start only after the complete producer finishes; cycles reject.
- Sizes, alignment, bounds and simultaneous live ranges permit each placement.
- INT32 accumulator storage stays live through its quantization consumer.
- Half-open resource reservations fit their capacities, including a shared
  SRAM request port. Different addresses alone do not eliminate port conflicts.

Two hand-derived examples cover an INT32→INT8 chain with legal lifetime reuse
and a compute/DMA overlap with nonintersecting SRAM request intervals. The
20 regression cases include a 25-timeline exhaustive check against independent
interval conditions, corruption/reordering cases, quantization identity drift,
invalid geometry, cycles and retained-buffer overwrites. Reproduce with:

```sh
make p6-check PYTHON=/absolute/path/to/tinyML_accelerator/.venv/bin/python3
```

The [starting evidence](evidence/phase6/contract.json) pins source hashes, complete
synthetic problems, proposed certificates and verification reports.

## Explicit limits and next steps

This checker verifies a **fixed-task abstraction**. Arithmetic identities are
opaque labels bound by the problem hash; numerical equivalence needs the
independent integer oracle. Durations and port traces in these examples are
synthetic ticks. Their makespans are neither predictions nor measured FPGA
latencies. A caller-supplied port trace cannot certify hardware behavior unless
that trace has itself been validated against the relevant implementation.

Whole-buffer availability, one producer per version and conservative lifetimes
are intentional initial restrictions. No in-place operations, partial-output
streaming, halo/recomputation graph transformation, physical BSRAM-mode proof,
real-model certificate adapter, search optimizer or RTL lowering is implemented
by this change. S01 is partial; S02–S04 are not complete. No Phase 6 gate or
performance claim follows from passing these examples.

Next implement a versioned tile/halo/partial-sum graph contract and its semantic
oracle; then add real-model candidate adapters and bounded exact search. Keep
the executing Phase 5 release frozen. New RTL/physical experiments should wait
for its campaigns to finish or use a separately available board.
