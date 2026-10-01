# Task 2 prior-art screen: control storage and reusable executable plans

Date: 2026-09-28. Scope: independent, boardless review of two proposed research
directions after the matched-baseline campaign. This review changes no baseline,
model, hardware, or experimental evidence. It is a bounded primary-source audit,
not a claim that all relevant literature has been exhausted.

**Decision: neither direction, as currently phrased, establishes a novel
mechanism.** Finite instruction storage, explicit transfer costs, state-aware
compiler rewrites, memoized scheduling, and reusable verified transformations
all have close precedents. One narrow search-efficiency hypothesis remains
testable, below. It has no positive result yet and is not a new architecture.

## Disqualification matrix

“Not established” below means the inspected source does not establish that exact
feature. It does not mean the prior system cannot be extended to support it.

| Primary source and inspected location | Already established | Claim this disqualifies; remaining distinction |
| --- | --- | --- |
| [DeFiNES, HPCA 2023](https://arxiv.org/html/2212.05344v1), §§II–IV | Searches spatial tiles, fusion and overlap caching; models memory levels and COPY costs with word lengths and port conflicts. Identical tile types are evaluated once. It explicitly connects tile diversity to code/control complexity and identifies unmodeled controller stalls in its validation. | Reject “first transfer-aware depth-first search,” “first tile memoization,” and “first to notice control overhead.” A hard bound on actual target command bytes and byte-addressed lowering are more specific than its published analytical model, but specificity alone is not algorithmic novelty. |
| [COSMA](https://arxiv.org/html/2311.18246v1), §§III–IV | Joint operator ordering, contiguous tensor placement and replacement; exact ILP plus a divide-and-conquer heuristic. Operators are atomic in the published model. | Reject “first joint schedule/address/liveness optimizer.” Compare an explicit tile/task expansion with aligned transfers and a command-budget constraint before attributing a gap to COSMA. Such an extension is a comparison proposal, not a reproduced published result. |
| [VTA](https://arxiv.org/pdf/1807.04188), §§3.1–3.2, Figs. 4–5 | Separate task/micro-op ISAs, DMA and dependency instructions, bounded micro-op storage, affine nested loops to compress microcode, and JIT loading of kernels that do not all fit at once. | Reject “finite control store,” “compact repetitive commands,” and “explicit DMA instructions” as standalone novelty. VTA fetches tasks from DRAM; its storage problem is not identical to this engine's resident whole-program limit. Comparing with it requires accounting for fetch/reload resources and latency, not silently giving one machine unlimited code storage. |
| [msf-CNN](https://arxiv.org/html/2505.11483v3), §§5–7 | Represents fusion blocks as graph edges with memory/compute costs, searches constrained paths, and emits code through microTVM. | Reject “fusion-block catalogue plus resource-constrained path search.” Adding emitted command bytes as another path resource is an ordinary extension unless a nontrivial coupling and a better solution method are demonstrated. |
| [SERENITY, ICML 2020](https://arxiv.org/pdf/2003.02369), §3.1 and Appendix C | Derives a state signature to merge repeated scheduling subproblems, applies dynamic programming, and proves the corresponding dominance rule for its peak-memory objective. | Reject “memoize equivalent scheduling states” and “prove a compact state is sufficient” at this level of generality. A new signature must preserve additional physical constraints and demonstrably improve an already memoized comparison. |
| [TASO, SOSP 2019](https://theory.stanford.edu/~aiken/publications/papers/sosp19.pdf), §§4–6 and Algorithm 2 | Formally verifies reusable graph substitutions, searches using them, and measures each operator configuration/layout once for its cost model. | Reject “verified building blocks plus measured-cost search.” Graph substitution is not the same as a physical byte-addressed fragment, but cached measurement and reusable verification are established mechanisms. |
| [Exo, PLDI 2022](https://cap.csail.mit.edu/sites/default/files/research-pdfs/Exocompilation%20for%20Productive%20Programming%20of%20Hardware%20Accelerators.pdf), §§2.4, 3.3–3.4, 5–6 | Models accelerator configuration state and memories; uses composable checked rewrites and equivalent subprocedures modulo configuration state. Provenance permits simpler equivalent procedures in SMT queries to reduce verification cost. | Reject “stateful reusable verified subplans” and “reuse equivalence to reduce solver work” as generic novelty. Its hardware instruction annotations remain a trusted interface; it is not automatically a proof of arbitrary RTL or our exact INT8 rounding semantics. |
| [Automated Translation Validation of a Compiler for Statically Scheduled Accelerators, FMCAD 2025](https://repositum.tuwien.at/handle/20.500.12708/219556), §§I–V | Represents compiler stages and hardware as symbolic transition systems, validates equivalence with schedule information, and accelerates checking with symbolic starting states. | Reject “validate executable compiler output against hardware” as a new contribution by itself. Two exact simulator samples are substantially weaker than this kind of formal validation. |
| [MATCHA, DAC 2026](https://arxiv.org/html/2604.09124v1), §§3.1–3.3 | Tile/pattern mapping, invocation overhead, physical address/lifetime planning, transfer costs and generated execution plans. Its reported global transfer model is serialized. | Reject “executable plan with physical placement and transfer cost.” A target-specific asynchronous timing summary would require its own evidence; there is no basis for claiming every existing planner ignores these costs. |
| [Depth-First Fusion and Tiling for CNN Memory Footprint Reduction in TVM, AccML 2026](https://accml.dcs.gla.ac.uk/papers/2026/8th_AccML_paper_9.pdf), §III | Compiles fused producer/consumer tiles, bounds circular-buffer overwrite, and integrates buffer liveness with standard memory planning. | Reject “a compiler for depth-first inference” and “checking tile-buffer overwrite” alone. This work is a preliminary memory-footprint study, not a reproduction on our engine. |

The pinned DeFiNES implementation corroborates its memoization claim:
[`DepthFirstStage.py`, revision `7097d609`](https://github.com/ZigZag-Project/DeFiNES/blob/7097d6090dc22321e44ce91434e7cc23b065864f/classes/stages/DepthFirstStage.py#L456)
keys reused tile evaluations by geometry, weight source, cache memory levels, and
cache read/write extents. Reusing results only for identical tile shapes is not
the strongest applicable comparison.

## Consequences for this project

The existing [B3 search](../../tools/phase6/matched_defines_baseline.py) already
uses actual emitted command bytes in `k_paths(max_bytes=32768)` and rejects an
oversized composed program. Its edges remove their standalone HALT and add one
final HALT. The [generic backend](../../compiler/scheduler/matched_defines.py)
also checks the emitted command capacity; the
[rectangle lowering](../../compiler/scheduler/matched_defines_regions.py)
already includes aligned transfers, guarded scratch regions, prefix-preserving
COPY and external read/modify/write. These are existing Task 1 capabilities,
not Task 2 discoveries. A better beam or cost function is useful engineering,
but needs evidence beyond beating the old proxy to support a new method.

The two proposed directions have different risks:

* **Command-store-aware depth-first compilation:** a useful systems question is
  whether physical command expansion changes the useful tiling/fusion frontier.
  Showing that alone establishes a target limitation, not a new search
  algorithm. First compare exact emitted costs with ordinary constrained path
  search, including a Pareto frontier over latency and command bytes. A
  post-lowering rejection-only strawman is insufficient. Looped command formats
  or instruction streaming are legitimate alternative solutions, with their
  actual area, timing and transfer costs reported separately.
* **Executable-state search with reusable checked fragments:** the potentially
  useful part is a small *sufficient* physical boundary description that allows
  safe reuse across different placements and surrounding schedules. The
  current B3 composer materializes activation boundaries in external memory;
  this simplifies composition and does not establish that arbitrary live SRAM
  states can be joined. A new comparison must expose the same retained-state
  choices to every search method.

For either direction, do not confuse exact command/data bytes with exact timing.
A fragment's isolated cycle count is not automatically additive when the
previous fragment leaves DMA requests, queues, descriptor state or prefetches
active. Either require and account for quiescent boundaries, or include those
states and their effects. A variable memory-response trace also prevents a
single measured scalar from being a universally exact latency certificate.

## One defensible hypothesis to test

**H2, proposed and unverified:** On a fixed catalogue of legal fragments for the
frozen INT8 engine, a canonical physical entry/exit-state summary can reduce
complete compile/search/check time by at least **5×** against the fastest
applicable ordinary memoized search, while returning schedules within **1%** of
the exhaustive executable optimum on bounded test instances, with zero
legality or arithmetic failures. Both methods enforce the same 32 KiB command
store and 32 KiB scratchpad and can choose the same schedules. Compare search
time at that same quality target; do not compare an approximate H2 search only
against a competitor required to prove exact optimality.

The numerical thresholds are proposed go/no-go criteria, not measured results
or requirements inherited from a paper. This is a search-efficiency claim; it
does not promise faster inference or top-tier acceptance. If no compact
sufficient summary is found, reject H2 rather than renaming ordinary caching.

A candidate summary must distinguish live tensor identities and byte regions,
alignment and guards, retained parameter/descriptor contents, free regions,
input/output layout, remaining command capacity, and any relevant control or
asynchronous state. Only quotient away an address or state component after
showing that doing so preserves all subsequent legal continuations and costs.
Model weights, quantizers, code-generator version and engine semantics belong
in the reuse identity. Do not merge fragments merely because their shapes or
measured output samples match.

The bounded first test should use small real KWS/VWW subgraphs plus held-out
shapes with misalignment, halo caches, partial tiles, and competing retained
weights. Exhaustively enumerate their shared fragment catalogue and address
choices. Compare H2 with exact-state memoization, identical-tile reuse, and
ordinary constrained path/DP search with the same lowerer, cost oracle and
legality checker. Where the graph has nontrivial ordering/replacement, include
an explicit task-expanded joint placement formulation; label any implementation
an adaptation. Reusable results and common optimizations must be available to
the comparisons too.

Count cold end-to-end compilation time, lowering calls, validation time,
search time, peak host RAM, states retained, optimality gap, and invalid
compositions. Report warm-cache results separately and charge certificate
construction to the cold run. An algorithm must not appear faster because its
expensive fragment catalogue was built outside the timed interval. Native RTL
checks establish sampled executable correctness; claiming a certificate for
all inputs additionally requires a precise specification and a proof or sound
symbolic validation, including integer overflow and requantization boundaries.

Stop this hypothesis if ordinary memoization already achieves the same result,
if the summary merges inequivalent states, if retaining enough state eliminates
the search saving, or if the result disappears against a full latency/command
Pareto frontier. A positive search result would justify a compiler-methodology
investigation. An accelerator speedup claim would still require separate
matched full-model measurements; the project's existing latency acceptance
target is not replaced by H2.

## Audit limits

Searches covered DeFiNES, COSMA, VTA, instruction-memory-aware compilation,
accelerator translation validation, verified graph substitutions, and stateful
compiler scheduling, then followed primary papers and author sources. Paper
sections above were inspected directly; the DeFiNES source was also inspected
locally at the pinned revision. This audit executed no external framework and
does not compare published numerical speedups with this project. Lack of a
published Tang Nano backend is not evidence of a missing mechanism. A positive
H2 result would still need a broader claim-specific literature check before
using “novel,” “first,” or “state of the art.”
