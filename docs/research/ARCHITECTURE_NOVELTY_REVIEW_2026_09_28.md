**Architecture research review — 28 September 2026**

**Follow-up:** The [parallel experiment screens](PARALLEL_ARCHITECTURE_EXPERIMENTS_2026_09_28.md) tested the leading resumable-fusion and joint-memory hypotheses against stronger controls. Their current modeled results are negative or conditional; the investment recommendation below is superseded by that follow-up decision.

Reviewed checkout: `53428af` plus the existing working tree. This is a project-wide research assessment of the active architecture, compiler, verification infrastructure, physical evidence, and selected primary literature. It is not an exhaustive line-by-line audit or a certification of novelty. No new simulation, synthesis, board run, or energy measurement was performed. Existing RTL and experiments were not modified.

**Recommendation.** Investigate a time-multiplexed microtile engine that can suspend and resume reductions, preserve every original quantization boundary, and transfer ownership of small physical memory regions between producer and consumer operations. Couple it to compact control programs and a measured allocation of control storage, activations, halos, and partial sums. Treat this as a falsifiable architectural hypothesis. More search over the current command interface is not supported as the central paper contribution by the present results.

**What the project actually contains**

The historical instruction-driven design in `src/`, `rtl/execution_unit/`, and much of `docs/RTL_ARCHITECTURE.md` is not the current measured system. The current board path is the descriptor-driven v2 hierarchy with a generated Phase 6 engine. The selected build uses `work/phase6/pool-timing-v1/engine.sv`; its hash and route are archived in the [route record](evidence/phase6/matched-baselines-v1/route27-report.json). See [runner selection](../../tools/phase6/matched_baselines_board.py:35).

| Subsystem | Established capability | Research implication |
| --- | --- | --- |
| Compute | Eight INT8 multipliers, spatial pointwise/depthwise modes, exact requantization and activation epilogues, pooling | A working reusable engine is a substantial foundation. Another operator fast path alone is unlikely to establish a general architectural result. |
| Data memory | 32 KiB data scratchpad, eight byte banks sharing one 64-bit request port; compute/DMA arbitration | Physical banking does not currently provide independent concurrent bank access. Inspect actual service traces before assigning speedup to more bandwidth. |
| Control memory | Separate 32 KiB program SRAM, storing 2,048 16-byte commands | Total on-chip storage is larger than the advertised data scratchpad. Command expansion competes indirectly with the physical memory budget available for tensors. |
| External memory | 8 MiB SDRAM, autonomous transfers, physical burst adapter, single-outstanding upstream word DMA | Contiguous DMA is established. Small off-axis tiles incur packing, copying, and command costs. |
| Compiler | Calibrated affine INT8, per-channel weights, corrected bias, explicit rounding; tiled hardware path restricted to linear graphs | Do not infer residual/branch support from broader software support. |
| Scheduling | Exact/beam catalogue search, software spatial semantics, executable retention and selected strip fusion | Software expressiveness exceeds the hardware interface. Existing search results do not establish a new scheduling algorithm. |
| Verification | Independent centered-integer oracle, command replay, native RTL, malformed-input tests, sealed physical campaigns | Excellent infrastructure for testing a new mechanism. A certificate must be bound to actual RTL accesses before it proves physical port feasibility. |

Sources: [scratchpad](../../rtl/v2/scratchpad.sv:1), [arbiter](../../rtl/v2/tiled_core.sv:73), [DMA](../../rtl/v2/tile_dma.sv:82), [sequencer](../../rtl/v2/tile_sequencer.sv:1), [linear-graph restriction](../../compiler/phase4_compile.py:19), [software spatial semantics](../../compiler/scheduler/spatial.py:1), [certificate limitation](../../compiler/scheduler/contract.py:1), [command replay scope](../../compiler/scheduler/abi_verify.py:1).

**What the latest measurements say**

The [matched campaign](PHASE_6_MATCHED_BASELINES_V1.md:132) compares schedules on the same eight-lane, 27 MHz image, with common arithmetic and engine optimizations:

| Policy | KWS device latency | VWW device latency |
| --- | ---: | ---: |
| B1: tuned materializing schedule | 49.917 ms | 100.414 ms |
| B2: tuned physical liveness/retention | 44.513 ms | 94.942 ms |
| B4: selected retention and strip fusion | 44.513 ms | 90.429 ms |

B4 improves two-model geometric-mean throughput by 1.116× over B1, but only 1.025× over B2. KWS B2/B4 command programs are identical. A restricted DeFiNES-derived eight-cut catalogue selects the same B4 program; that comparison is a tie and is not a full DeFiNES reproduction. The subsequent 45-run finalist screen confirms VWW B4 at 90.424 ms. Three timed repetitions per policy on a pinned input establish a short repeatability screen, not a dataset-wide latency distribution.

The selected route consumes 10,095/10,368 CLS (97.37%), 38/46 BSRAM blocks, and 9.5/24 DSP equivalents. Routed Fmax is 27.613 MHz. Only 273 CLS sites remain; spare DSPs do not imply a second engine or wide crossbar will fit.

The physical [primary records](evidence/phase6/matched-baselines-v1/board/records.jsonl) give these medians:

| Counter | KWS | VWW |
| --- | ---: | ---: |
| Total cycles | 1,201,859 | 2,441,571 |
| Engine-busy cycles | 1,170,262 | 2,133,411 |
| DMA-busy cycles | 39,115 | 311,445 |
| Engine/DMA overlap | 7,866 | 6,968 |

Engine busy occupies 97.37% and 87.38% of total time. It includes memory stalls and control, so it is not MAC utilization. Holding engine cycles fixed and removing *all* time outside the engine would yield only approximately 1.027× KWS, 1.144× VWW, and 1.084× geometric-mean speedup. This is a conditional sensitivity calculation, not a universal bound: improved operand delivery can also reduce engine time. It explains why external traffic reduction alone is an insufficient thesis on these two models.

The most useful negative result is the [2D schedule](PHASE_6_MATCHED_BASELINES_V1.md:74). Direct output scatter needs 1,536 DMA commands plus 1,536 waits, already exceeding the 2,048-command store. A valid alternative uses overfetch and SRAM COPY, fits 1,471 commands, and passes exact checks, but takes 110.141 ms on the board versus 90.424 ms for B4: 21.8% more time. Fewer logical tensor bytes do not automatically mean faster physical execution.

Qualification is image-specific. Full KWS board accuracy passed on an earlier 20.25 MHz spatial image: 4,514/4,890 correct with zero integer-output mismatches. VWW stopped after 212 samples. These results do not qualify the newer 27 MHz image. Older Phase 5 evidence also contains a completed 10,000-job alternating-model run; it must not be attributed to the selected image. The [benchmark record](../../benchmarks/README.md:23) identifies the VWW research split and does not certify it as the official MLPerf accuracy set. Board energy remains unmeasured.

**Idea 1 — Resumable reductions and quantized microtile handoff. Highest priority.**

Today, each compute descriptor initializes its accumulators from bias, finishes its reduction, and enters requantization. There is no descriptor-visible continuation of INT32 state across operations. The selected engine's `P_CHECK` and `PW_MAC` paths establish this restriction ([initialization](../../work/phase6/pool-timing-v1/engine.sv:649), [completion](../../work/phase6/pool-timing-v1/engine.sv:735)).

Add a small context store and explicit `INIT`, `ACCUMULATE`, and `FINALIZE` operations. A context identifies the output region, completed reduction range, bias status, arithmetic identity, and accumulator storage. Execute a producer microtile, apply its complete original INT8 requantization and activation, then time-share the same MAC lanes to update the consumer's retained partial sums. Resume either operation later. Publication and last-use tokens govern when memory can be reused.

For example, a depthwise operator produces eight spatial values for a channel. After their original numerical boundary, a pointwise consumer uses them to update persistent outputs. The next channel's producer work can follow without materializing the full intermediate feature map. Retaining eight pixels across 64 consumer output channels requires 8 × 64 × 4 = 2,048 bytes of INT32 state, before metadata and bank rounding. Smaller output-channel tiles reduce state but can increase producer replay or retained activation storage. The compiler must charge that tradeoff.

This can lose badly on pointwise expansion: a producer tile costs approximately `P*C` INT8 bytes, whereas retaining all consumer partial sums costs `4*P*M` bytes, for P pixels, C producer channels, and M consumer outputs. Reading and rewriting those partial sums at every channel chunk can also exceed the activation traffic saved. Compare against hand-tuned output-stationary blocked fusion that keeps a small accumulator tile and reuses a compact INT8 producer tile. That simple control is potentially fatal to the proposed novelty claim and should be tested first.

Do not bypass an intermediate requantization or activation by algebraically combining convolutions. Bias must be applied once, reduction coordinates covered exactly once, and overflow/error behavior preserved. A single time-shared array does not execute both layers simultaneously; any claimed gain must exceed context-switch, weight-reload, and partial-sum service costs.

The possible contribution is a compact execution mechanism that makes useful fine-grained fused schedules physically realizable under strict port and state budgets. It is not the first fusion engine or persistent accumulator. [DeFiNES](https://arxiv.org/abs/2212.05344) and [LoopTree](https://arxiv.org/abs/2409.13625) already explore fusion, retention, and recomputation. Give their adapted policies the same improved backend. Compare against ordinary split-reduction support plus conventional fusion before attributing a benefit to a more elaborate ownership mechanism.

First experiment: exact DW→PW and PW→DW blocks, a pointwise expansion case, a non-fused control, and one residual fork/join stress case. Include an early VWW block that spills, a KWS block that already fits, a smaller SRAM budget, and a held-out geometry. Build an executable trace model that charges each bank access, original quantization boundary, context switch, repeated parameter load, and partial-sum spill. If even an optimistic but resource-accounted version cannot beat ordinary output-stationary blocked fusion, stop before a large RTL redesign.

**Idea 2 — Joint control and tensor memory allocation. Strong supporting mechanism.**

The [sequencer](../../rtl/v2/tile_sequencer.sv:46) reserves a complete 32 KiB scratchpad for commands. Introduce compact loop/region commands and a small instruction working set. At compile/load time, allocate physical memory groups among program storage, activation tiles, halos, and partial sums. Fine-grained last-use release can subsequently reuse data regions during execution. Start with static physical bank partitioning; runtime bank reassignment is a separate complexity to justify.

The research question is whether code expansion is an important, general resource constraint in tiny accelerators, and whether jointly selecting program representation and live tensor state changes the capacity/performance frontier. A modest controller can be worthwhile if it releases enough physical BSRAM for the contexts in idea 1. Logical byte savings alone do not establish recovered BSRAM blocks, and instruction fetch must not simply move contention to the data path.

Nested-loop micro-operations already exist in [VTA](https://arxiv.org/abs/1807.04188) and its [open implementation](https://github.com/apache/tvm-vta/blob/main/hardware/intelfocl/src/vta.cl). Multidimensional data movement and layout transformations are covered by [XDMA](https://arxiv.org/abs/2508.08396); bank-conscious decoupled operand streaming is covered by [DataMaestro](https://arxiv.org/abs/2504.14091). A loop instruction or 2D DMA by itself is an engineering improvement. The proposed distinction needs to be a demonstrated coupling of control-memory footprint, exact live computation state, and physical memory allocation.

First experiment: lower existing schedules to compact loops without changing arithmetic or transfer order; count control bytes, physical blocks, fetch traffic, and execution overhead. Then compare (a) a fixed optimized code/data partition, (b) several independently tuned fixed partitions, and (c) joint allocation at the same total physical SRAM. If a simple fixed partition explains the gain, use it and drop the adaptive-allocation claim.

**Idea 3 — Lossless partial-sum storage selected from proven bounds. Conditional extension.**

Resumable fusion may make partial-sum storage the new bottleneck. Keep the arithmetic exact, but store each channel or reduction phase using a proven sufficient signed width. Decode to the normal arithmetic width when resuming. Store wide values wherever the proof is insufficient. Initially choose widths statically; data-dependent compression adds metadata and variable-latency costs.

For the present raw-byte/corrected-bias arithmetic, a conservative bound after a consumed prefix is `abs(corrected_bias) + 128 * sum(abs(consumed_weights))`. The proof must cover all legal input codes and every required intermediate update, not observed dataset ranges. Packing must respect actual BSRAM widths, alignment, and read/write ports. A 21-bit value does not automatically occupy less physical memory than a 32-bit value.

This differs in purpose from simply narrowing the global MAC datapath: it aims to keep more suspended reductions resident and avoid spills. Nevertheless, accumulator-width optimization is established by [A2Q](https://arxiv.org/abs/2308.13504), and additional partial-sum compression prior art must be checked before any novelty claim. The project already tried a narrower accumulator with no useful physical gain; this proposal is justified only if the new continuation traces reveal a storage-capacity bottleneck.

First experiment: compare always-32-bit state, per-layer fixed safe widths, and per-context/per-phase safe widths, including packing ports and metadata. Proceed only when the latter enables a materially better fused working set or reduces measured spill costs beyond the simpler safe-width baseline. Otherwise omit it from the paper.

**Experiments that would make the paper convincing**

Use one coherent thesis. Idea 1 is the central execution mechanism, idea 2 may make it fit, and idea 3 is optional. Do not present a collection of unrelated small optimizations as three independent novelty claims.

1. Strengthen the ordinary baseline first: pipelined operand delivery, sensible bank partitioning, compact loops, and strided DMA. Measure byte-at-a-time writes and request/MAC bubbles in the selected engine. Preserve these improvements in every comparison.
2. Compare the same eight MAC lanes and total physical memory budget. Report equal-frequency results and independently routed achievable-frequency results. Charge context RAM, control RAM, caches, FIFOs, LUT/CLS overhead, and physical block fragmentation to every design.
3. Separate ablations for reduction continuation, publication/last-use ownership, control/data allocation, and optional compressed state. Give DeFiNES/LoopTree adapted policies the same hardware. Include a tile-expanded [COSMA](https://arxiv.org/abs/2311.18246) placement/replacement comparison where applicable. Label adaptations and excluded choices precisely.
4. Expand beyond two linear models: retain KWS/VWW, add a dense model without the present large constant-channel simplification, a residual network with joins, and a larger activation working set. Hold out shapes/models from cost-model tuning. Additional operator coverage is evaluation infrastructure, not novelty itself.
5. Sweep physical storage budgets and port organizations. Explain where the mechanism wins, where a simple schedule matches it, and where context/weight overhead makes it lose. A stronger capacity frontier can be a useful result even when a model already fits the baseline.
6. Bind the legality checker to emitted bank-access and publication traces. Include backpressure, malformed contexts, stale tensor versions, partial reduction overlap, bias duplication, overflow, abort/reset, and joins. Correct byte replay alone does not verify port timing or deadlock freedom.
7. Qualify the final image on complete declared accuracy sets, multiple programming sessions, and measured energy per inference. Keep device execution, input transport, preprocessing, and diagnostic readback separate. Demonstrate a second physical memory geometry or platform when feasible; do not frequency-normalize unrelated boards into a controlled comparison.

For an internal investment gate, require a clear advantage over the strengthened baseline across several held-out cases—such as a substantial end-to-end latency/energy gain or a new useful capacity frontier—before expanding the hardware campaign. This is a research management criterion, not a publication threshold or predicted result. The repository's existing prospective targets remain unproven.

**Directions to deprioritize**

The current [early-exit gate](PHASE_6_EARLY_EXIT_GATE.md:13) skips less than 0.04% of MAC slots. The [nonstream screens](PHASE_6_NONSTREAM_NOVELTY_V1.md:19) find a 1.02147× ideal one-cycle zero-block opportunity before metadata costs, no cycle gain from compiler hints, and only 0.22–0.38% VWW native improvement from the final constant slice. [Padding and alternative lane mapping](PHASE_6_FOLLOWON_OPTIMIZATIONS.md:81) also have small measured or optimistic gains on the present models. These are useful negative results. Repeating them without changing the underlying bottleneck is a poor research investment.

An appropriate working thesis is: **“Exact quantized fusion under a joint control, tensor-state, and memory-port budget.”** Whether the eventual result belongs in a top-tier architecture paper depends on a distinct mechanism surviving the strong baselines above and demonstrating a general, reproducible benefit. The current evidence establishes a credible experimental platform; it does not yet establish that contribution.
