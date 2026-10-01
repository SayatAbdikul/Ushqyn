# Three new research hypotheses — 1 October 2026

Status: all three bounded experiments completed, with native follow-up on the two surviving directions and a passed raw-evidence audit. VWW compression and drained-boundary interruption have useful evidence. Neither currently establishes architectural novelty or a new board-performance result.

## Completed results and decisions

| Direction | Concrete result | Research decision |
| --- | --- | --- |
| Model/representation co-design | Frozen VWW INT8 factors reduce useful products 40.78%, save 155,008 logical weight bytes, and score 84.0293% versus dense 84.2263% on all 10,657 previously untouched examples. AD factors reduce products 16.96% and score AUC 0.888855 versus 0.877668 on 200 held-out recordings. | A useful compression result. The tested wide-latent mechanism and fusion have no unique advantage over ordinary factor execution/blocked fusion. No novel architecture is established. |
| Deadline-aware sharing | An isolated pause/controller extension passes real VWW→AD→VWW native execution, including destructive scratchpad clobber and live-state restoration. Under fixed RAM, splitting reduces AD response 375,495→316,339 cycles and meets a 353,303-cycle deadline. | A conditional controller/compiler opportunity. Stalled RAM misses the same deadline. Virtual UART and one selected arrival prohibit a physical schedulability claim. |
| Compiler optimality certificates | 18 finite families, 1,206 independently checked choices, 47,212 exhaustively enumerated paths, 46 exact INT8 region replays, and 90 rejected corrupt certificates. Maximum strong-control cost gap 1.9833%; matched serial executable controls have 0% gap. | Stop this restricted mapping family as a central novelty claim. The certificate is useful as a bounded stopping result; it does not cover arbitrary architectures or all compiler schedules. |

### Representation: stronger quality and native checks

The first 256-example classification screens were approximately label-balanced. All ranks and factor methods were frozen before those results. A separate follow-up evaluates every remaining classification example, excluding both the 48 development examples and those 256 previously observed examples. Candidate selection was not changed.

| Untouched population | Dense | INT8 factors | Original column mask | Dev-only pruning refit |
| --- | ---: | ---: | ---: | ---: |
| KWS, 4,586 examples | 92.3463% | 92.4989% | 92.3027% | 92.6734% |
| VWW, 10,657 examples | 84.2263% | 84.0293% | 82.8845% | 82.7156% |

VWW factor-minus-dense paired bootstrap 95% interval is [-0.5067, +0.0938] percentage points. Factor-minus-mask is [+0.6662, +1.6046] points. These are research-protocol quality estimates at the untouched population's original label proportions. The mask is reconstructed from development data and exactly reproduces all 256 earlier mask predictions. Its arithmetic saving assumes a gather or propagated channel-compaction implementation; the zero-column executable itself still executes dense operations on current RTL. Neither tested pruning control is a globally retrained pruning/QAT baseline.

AD uses the original accuracy archive re-partitioned into 48 labeled development recordings and 200 held-out recordings (39,200 windows). Its AUC improvement has an exploratory interval [-0.01540, +0.03654], so formal noninferiority/superiority is not established. No end-to-end training or QAT was performed; activation-aware fitting is development-only teacher regression. These protocols are not official unchanged benchmark scores.

An accelerated CPU-double evaluator proves each integer dot-product's absolute partial-sum budget is below 2^53, checks integral results, and performs requantization in INT64. It matches every layer of the independent integer oracle on real, random and both INT8 endpoint inputs for dense, factor and pruning executables. NPZ members are checked against their mapped NPY bytes. No accuracy archive or frozen export was changed.

Thirty-two selected-engine native runs then check VWW/AD dense and factor models on pinned real and INT8-stress inputs, fixed/stalled external RAM, and diagnostic/timed programs. Diagnostic programs check every materialized layer; timing programs omit snapshot DMA.

| Matched generic native compilation | Fixed-RAM dense cycles | Factor cycles | Reduction | Stalled-RAM reduction |
| --- | ---: | ---: | ---: | ---: |
| VWW | 4,347,606 | 3,029,706 | 30.3132% | 29.8821% |
| AD | 294,419 | 252,084 | 14.3792% | 14.3209% |

**The VWW comparison is not against the best tuned schedule.** The existing selected B3 native control takes 2,014,205 cycles, already below the generic factor program's 3,029,706. Therefore this experiment demonstrates a decomposition effect under matched generic compilation, not an improvement over our strongest available accelerator. Applying equivalent constant compaction/strip tuning to the changed graph is an unresolved requirement. No generic native reduction is added to a previous board improvement. KWS's 4.22% arithmetic opportunity fails the investment screen and was not advanced to native timing.

Reproduction and details: [representation screen](../../work/phase6/representation-screen-v1/README.md), [expanded quality](../../work/phase6/representation-quality-v1/README.md), [native comparison](../../work/phase6/representation-native-v1/README.md).

### Deadline mechanism: actual pause and state restoration

The isolated prototype copies the sequencer, host bridge and packet decoder. A CRC-valid one-byte pause request is the sole extra mutation allowed while busy. The sequencer stops after an engine/DMA-drained WAIT, exposes the next command index, and resumes by refetching the next command. Three unsafe busy writes are rejected in every interleaving test. The arithmetic engine remains source-identical to the selected co-issue engine.

Both jobs have private external-memory namespaces. Real DMA saves/restores the conservative live SRAM granules. Two real jobs plus temporary checkpoint commands fit the existing program store: 23,808 B for ordinary descriptors or 29,808 B for the split variant. This is a two-job construction; it does not solve the three-job/multiple-instance addressing problem.

Twelve native runs cover both schedules, fixed/stalled RAM, no-interruption controls and interruption with/without a full scratchpad clobber. Every expected output is exact; live restoration has zero byte mismatches. Skipping restoration deliberately fails the VWW output check. No-interruption fixed-RAM cycle totals remain exactly equal to their source traces.

| Native two-job witness, no artificial clobber | Ordinary descriptors | Split VWW |
| --- | ---: | ---: |
| Fixed-RAM AD response | 375,495 cycles, misses | 316,339 cycles, meets |
| Fixed-RAM blocking after accepted request | 77,059 cycles | 17,216 cycles |
| Saved/restored live SRAM | 8,640 B | 10,176 B |
| Stalled-RAM AD response | 493,825 cycles, misses | 391,670 cycles, misses |

The deadline is fixed at 353,303 cycles. Smaller descriptors increase VWW isolated service by 5.40% and context bytes; they trade throughput/other-job latency for interruption latency. The native host moves UART bytes in one cycle and installs temporary checkpoint programs dynamically. These timings cannot establish that the selected 750-kbaud physical transport meets the deadline. A resident autonomous context controller, routed fit/timing and a broader worst-case response analysis remain unimplemented.

The first screen contains 20,640 schedule records and 4,128 paired sensitivity cases. Uniform finer splitting helps some scenarios and worsens others. Some labels encode identical job sequences; these counts are coverage, not independent trials. Software UART full-program reload has zero zero-miss schedules in the grid; software HALT at every fine boundary exceeds the combined 32 KiB program capacity.

Details: [deadline screen](../../work/phase6/deadline-screen-v1/README.md), [native pause prototype](../../work/phase6/deadline-pause-v1/README.md).

### Audit, resource use and remaining scope

The [raw-evidence audit](../../work/phase6/three-hypothesis-campaign-v1/audit.json) independently recomputes saved prediction tallies, split coverage, native reductions, fixture/report hashes, schedule conservation, checkpoint extents and final memory guards. It confirms all 165 production/compiler/RTL/target/benchmark-manifest files in the start snapshot are unchanged. Five deadline invariant tests, saved-certificate checking, Python compilation and whitespace checks pass.

The largest final Python process observed 790.7 MiB RSS during expanded VWW quality evaluation. The column-control run now closes/reopens the feature mmap between small blocks and peaks at 263.0 MiB; its first broad mmap scan exceeded the 1 GiB target because resident file-backed pages accumulated, and was replaced and rerun with the bounded mapping. No 40 GB retained-array behavior occurred. Builds used two compiler jobs; all numerical workers used one thread.

All three proposed directions have executable first-stage evidence, and the two survivors have native follow-up. They have not received new synthesis, route, board/energy qualification, end-to-end retraining or full published-method comparisons. The results support further compression/QoS research, but a Q1/top-tier architectural novelty claim remains unsupported.

The [portable evidence manifest](evidence/phase6/three-hypotheses-v1/manifest.json), compressed source/model/result archive and streaming verification are saved in `docs/research/evidence/phase6/three-hypotheses-v1/`. Feature maps, original benchmark archives, build objects and executables are excluded. The unchanged selected engine source is included so its external worktree is not needed merely to recover the arithmetic RTL. Existing benchmark data are still required for reproduction.

## Scope and advancement rules

The user authorized trying all three hypotheses discussed in the preceding research review. The first stage consists of executable, bounded screens of model/representation co-design, discrete inference-job sharing under deadlines, and independently checked optimality gaps. Production engine and compiler sources remain outside the experiment edits.

The existing Tang Nano 20K reference has eight INT8 MAC lanes, one shared 64-bit data scratchpad port, 32 KiB of data scratchpad, a separate program memory, and full-model weights in external SDRAM. An active weight tile resides in the scratchpad. A wider latent operand cannot inherit the throughput of the existing INT8 multipliers without a hardware change or multiple limb operations.

The existing project's matched-latency investment gate is 15%. This is an internal decision threshold, not a publication criterion. A software traffic/MAC proxy does not meet it. For representation exploration, the provisional quality gate is at most one percentage point of classification accuracy loss or 0.01 recording ROC-AUC loss relative to the same evaluation baseline. Candidate choice must use development data; evaluation results must identify whether they are full frozen sets or bounded disjoint subsets. These quality gates are exploratory and do not authorize an accuracy claim based on fixtures alone.

Runs load compressed NPZ array members once or extract and memory-map them, retain bounded batches, and use a single numeric worker thread. Each report records observed peak RSS. A target of less than 1 GiB per process prevents repeating the earlier retained-NPZ-backing-array memory incident.

## 1. Model/intermediate-representation/datapath co-design

Test actual pointwise/GEMM matrices, not only synthetic square examples. Compare the dense model, ordinary unfused factorization, fused factorization with explicitly charged latent precision, and a structured pruning control. Singular-spectrum energy is a screening statistic, not accuracy evidence. Preserve the original output quantization boundary where possible; changed arithmetic is checked against a reference for that changed model rather than represented as bit-exact equivalence to the original model.

Advance only if a frozen candidate preserves useful held-out quality and an honest eight-lane/64-bit-port model shows a worthwhile gain after factor weights, intermediates, lane tails and wider-operand execution. Native lowering and physical feasibility remain separate obligations.

Generic decomposition and joint precision/mapping are established research directions: [TinyML approximate matrix decomposition](https://arxiv.org/abs/2604.16113), [quantization and mapping synergy](https://arxiv.org/abs/2404.05368). Any future novelty claim must identify a specific new representation/storage/execution mechanism and compare against these approaches.

## 2. Predictable sharing of discrete inference jobs

Use source-pinned native engine traces where available. Compare nonpreemptive earliest-deadline-first (EDF), EDF at existing descriptor boundaries, static service slots, and any compiler-selected smaller safe boundaries. Charge saving/restoring live scratchpad state, program state, invalidated caches and DMA reloads. Sweep unmeasured context/DMA costs rather than assigning a single favorable value. Existing descriptor yielding is a proposed software control, not a currently board-qualified facility.

Advance only if the proposed boundary/state mechanism provides a useful deadline capability beyond strong software controls over multiple arrival regimes and overhead assumptions. Scheduling simulation is not a hardware worst-case response-time proof. This experiment concerns separate requests, not continuous-input reuse.

Preemptible neural accelerators and scheduling already exist; [PREMA](https://arxiv.org/abs/1909.04548) is a required comparison before treating preemption as novelty.

## 3. Independently checked compiler optimality gaps

Declare the finite schedule family, its cost model, exact quantization barriers, memory geometry and reservations. Exhaustively solve small instances and verify the solutions and lower bounds using a separate checker. Compare current heuristics, a strengthened ordinary control and the exact optimum. Sweep 8/16/32 KiB budgets where meaningful. Report larger real-workload certificates only within the declared family; a catalogue optimum is not the optimum over all hardware schedules.

Advance toward a compiler contribution only if the bounds are useful beyond the existing infeasibility classifier and explain significant heuristic gaps or reliably certify small remaining gaps. Bounds in a modeled MAC/port-work objective are not device-cycle bounds.

[The Turbo-Charged Mapper](https://arxiv.org/abs/2602.15172) already targets optimal mappings with aggressive pruning. Generic optimal search or pruning alone is insufficient as a new contribution.

## Reproducible outputs

- `tools/phase6/representation_screen.py` → `work/phase6/representation-screen-v1/`
- `tools/phase6/deadline_screen.py` → `work/phase6/deadline-screen-v1/`
- `tools/phase6/optimality_screen.py` → `work/phase6/optimality-screen-v1/`

## Stopping interpretation

Each direction ends this stage with a benefit decision, an evidence level and a list of unresolved requirements. Failure of an RTL implementation rejects that implementation on the chosen device. Exhaustive evaluation rejects or optimizes only the declared finite family. None of the three screens can exhaust the whole architectural research space.
