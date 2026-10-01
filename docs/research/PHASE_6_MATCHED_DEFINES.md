# Executable DeFiNES policy adaptation on Tang Nano 20K

This adaptation executes rectangular depth-first schedules on the same frozen
27 MHz eight-lane engine used by B1/B2. It implements the three overlap modes
from the pinned author implementation: recomputation, horizontal caching with
vertical recomputation, and horizontal plus vertical caching. It is an
adaptation to the fixed descriptor engine, not a reproduction of the paper's
ideal accelerator-array results or energy model.

## Published policy and hardware adaptation

The source is pinned at
[`7097d6090dc22321e44ce91434e7cc23b065864f`](https://github.com/KULeuven-MICAS/DeFiNES/tree/7097d6090dc22321e44ce91434e7cc23b065864f).
The runner invokes the unmodified `DfStackCutIfWeightsOverflowStage` and
`backpropagate_tilesize` on both common transformed graphs. The former records
the upstream weight-capacity stack cuts; the latter checks all three cache-mode
geometries. The final evidence manifest binds the complete upstream Python
source tree and the downloaded source archive where available.

| Published choice | Executable adaptation |
|---|---|
| Output tile height and width | Explicit 2D grid, including full shape, half/quarter limits and 1/4/8/16 limits |
| Fusion depth and stack cuts | Every contiguous Conv/activation stack; additional cuts can satisfy the fixed SRAM and command limits |
| Recompute overlap | Recursive demand for exact clipped input halos |
| Horizontal overlap cache | Previous/next horizontal cache rectangles in physical SRAM |
| Horizontal and vertical caches | Horizontal caches plus previous/next tile-row cache rectangles; only initialized regions are reusable |
| Memory hierarchy and latency model | Actual 32 KiB SRAM/8 MiB SDRAM layout, DMA and COPY commands; complete-program native RTL timing determines final ranking |
| Spatial/temporal MAC mapping | Fixed to the same eight-lane descriptor dataflow as all comparison policies |
| Energy minimization | Not claimed; no synthetic energy constants are substituted for measurements |

The current manual three-pair schedule is a legal incumbent for this policy
family. Its VWW DW/activation/PW/activation regions use full-width depth-first
strips; the KWS retained chain is a full-tile case. Aliased external-slot
protection, safe prefetch, exact constant evaluation and activation epilogues
are backend improvements available to every policy. Excluding this incumbent
would make the adapted baseline artificially weaker than the implementation
already available in the project.

## Exact physical lowering

Every tensor remains signed INT8 with the original bias correction,
requantization and activation boundaries. A Conv followed by Relu/Clip uses
the existing, exhaustively checked 256-byte lookup epilogue. A zero-filter
tail is filled only with its exact bias-derived, requantized and activated
codes. The common model preparation applies the same exact channel grouping,
dead-channel compaction and final constant folding as B1/B2.

The SRAM arena reserves descriptor bytes, an eight-byte guard for every tensor,
all live recursive operands, cache rectangles, weights, parameters, activation
tables and packing scratch. Cache validity is tracked as rectangles; absent
bytes are recomputed rather than read speculatively. An immutable load can be
eliminated only while every destination byte still holds the identical known
value. Safe DMA prefetch uses the shared dependency checker.

Rectangles with unaligned row/channel starts are implemented by aligned DMA
and the existing byte COPY opcode. Copies preserve bytes preceding an
unaligned destination; external stores preserve neighboring bytes by
read/modify/write. Whole tensors and aligned full-width strips use direct
DMA/COPY paths. Packing instructions, cache maintenance, halo reads,
recomputation and all associated descriptors count toward both SRAM/command
capacity and measured runtime.

The separate full-width lowering provides a more efficient alternating
placement for compatible stacks. It is included as another exact mode-1
implementation, rather than restricting the baseline to the more general
recursive allocator. This specialized lowering serializes transfers and reloads
immutable data between strips; the generic alternatives and manual incumbent
provide the common retention/prefetch opportunities. The comparison does not
claim every alternative uses every optional optimization simultaneously.

## Search and validation

`tools/phase6/matched_defines_baseline.py` records every attempted geometry and
its status. A documented structural proxy retains a small set of choices per
stack and proposes complete paths. The proxy is not a calibrated latency claim.
The all-stack catalogue covers 5,040 KWS configurations (180 executable) and
28,504 VWW configurations (1,953 executable). Capacity/alignment rejections
remain in the evidence. Complete candidate programs are then measured with the frozen native RTL
executable, and the final frontier is ranked using both fixed and stalled
memory seeds. `matched_defines_finalize.py` validates each exported frontier
entry on the pinned input and independently generated INT8 stress input
(seed 6157), with native stall seeds 0 and 6063.

Each segment is independently replayed against the integer oracle. Composition
relocates its activation and immutable addresses into two disjoint full-model
activation slots without changing its instructions. The complete composed
program must additionally pass native RTL against the original model logits.
Native rows bind the executable and every fixture file before and after
execution. Physical measurements are the responsibility of the common matched
campaign; native cycles must not be presented as board measurements.

The generic cache-mode smoke campaign covers standard and depthwise
convolution, stride two, asymmetric padding, negative quantization zero points,
both activation forms, all three overlap modes, and both native stall seeds.
The rectangle helper tests separately check byte alignment, guards and preserved
neighbors. See the final report's `cache_mode_native_proof` and source hashes
for the current reproducible evidence paths.

## Limits that remain visible

- Search is bounded by an explicit grid, four proxy survivors per stack, and
  eight complete native candidates per model. It is not exhaustive over all
  possible integer tile dimensions or SRAM placements. The generated paths
  cover the spatial prefix with depth-first segments, followed by a conventional
  nonspatial tail; arbitrary mixtures with conventional spatial segments are
  not enumerated. The common manually tuned incumbent remains eligible.
- The current cache implementation uses separate old/new buffers. A more
  complex in-place rotation may make some rejected configurations feasible.
- The primary catalogue uses opportunistic immutable residency. A separate
  bounded paired screen preallocates whole-stack weights, parameters and LUTs,
  charges their persistent SRAM storage, and compares against the same geometry
  without that reservation. It does not repeat the entire catalogue.
- The fixed engine does not expose alternative MAC mappings or partial INT32
  reduction spilling. Their absence is shared hardware capability, not a
  limitation attributed to DeFiNES itself.
- A rejected case is **infeasible for this concrete backend**, not a proof that
  no implementation on the FPGA could execute that schedule.
- Final performance and baseline selection are authoritative only in the sealed
  matched campaign report. Implementing these modes does not establish novelty
  for the project's proposed optimizer.
