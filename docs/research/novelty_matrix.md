# Candidate claim and baseline decision — updated 2026-09-26

This is a research decision record, not a novelty or SOTA certification. The
search was expanded beyond the initial roadmap because close prior work already
co-optimizes memory and scheduling. A cheap FPGA port alone is not a contribution.

| Work and primary source | Established overlap | Unresolved distinction for this project |
|---|---|---|
| [FINN-R](https://arxiv.org/html/1809.04570), resource models | Quantized FPGA design exploration; physical BRAM fragmentation is explicitly modeled. | Whether a fixed eight-lane temporal engine plus executable quantization boundaries adds a useful constraint absent from applicable backends is unknown. Do not claim first physical memory awareness. |
| [SAMO](https://arxiv.org/html/2112.00170v2), §§III–IV | Joint partition/channel/kernel folding with resource and bandwidth constraints; integrates FINN, fpgaConvNet and hls4ml. | Streaming/reconfiguration formulation differs from the proposed fixed engine. A restricted adaptation must be labeled and must retain its resource/bandwidth checks. |
| [DNNExplorer](https://arxiv.org/html/2008.12745), architecture and DSE sections | Hybrid pipeline/layer-reuse architecture and memory/performance exploration. | A shared engine is not novel by itself. Tiny-device deployment and verified bank allocation need separate evidence. |
| [Fused-Layer CNN Accelerators](https://compas.cs.stonybrook.edu/%7Emferdman/downloads.php/MICRO16_Fused_Layer_CNN_Accelerators.pdf), §III | Backward dependency pyramids, inter-layer fusion, and explicit overlap caching versus recomputation tradeoffs reduce off-chip traffic. | The full author manuscript was retrieved on September 9 after an initial URL failed. Its storage/arithmetic exploration further rules out novelty based on fusion or recomputation alone; no implementation reproduction is claimed. |
| [MCUNetV2](https://arxiv.org/html/2110.15352) | Patch inference reduces peak activation storage; model/inference co-design. | Comparison must freeze the model or separately label changes to its receptive field and accuracy. |
| [msf-CNN](https://arxiv.org/html/2505.11483v3), §§4–6 | Searches multiple fusion blocks using memory/MAC edge costs, horizontal caching and constrained graph search. | Our proposed graph-search mechanism substantially overlaps. Its published MCU formulation is not itself a Gowin bank allocation certificate. Backend extensions remain possible. |
| [DeFiNES](https://arxiv.org/html/2212.05344v1), §§II–III and artifact | Tile size, fusion depth, overlap caching/recomputation, operand placement across memory levels, data copies, energy and latency. Code explicitly represents physical memory ports. | The useful question is whether executable BSRAM placement/quantization constraints change its best realizable result. “Joint memory and scheduling” and “port aware” are already insufficient claims. |
| [COSMA](https://arxiv.org/html/2311.18246v1), §III | Joint operator schedule, memory address allocation and tensor replacement; ILP and scalable decomposition. Operators are atomic in the formulation. | Tile-level quantized live state/port timing could distinguish scope, but applying COSMA to a tile graph is a necessary counterargument. |
| [Depth-First Fusion and Tiling, AccML 2026](https://accml.dcs.gla.ac.uk/papers/2026/8th_AccML_paper_9.pdf) | Recent work again targets CNN memory through fusion/tiling. | September 26 method review confirms buffer-liveness/circular-window overlap; implementation reproduction is not claimed. |

## B3 selection and fair adaptation

Select **DeFiNES** as B3 for the memory/schedule mechanism. Keep msf-CNN as a
secondary fusion-policy baseline and COSMA as a mandatory placement/replacement
comparison or explicit limitation. This is stronger than choosing only an MCU
fusion method that lacks an FPGA backend.

Code inspected at DeFiNES commit `7097d6090dc22321e44ce91434e7cc23b065864f`:

* [DepthFirstStage.py](https://github.com/KULeuven-MICAS/DeFiNES/blob/7097d6090dc22321e44ce91434e7cc23b065864f/classes/stages/DepthFirstStage.py): horizontal/vertical caches, tile backpropagation and weight storage.
* [memory_level.py](https://github.com/KULeuven-MICAS/DeFiNES/blob/7097d6090dc22321e44ce91434e7cc23b065864f/classes/hardware/architecture/memory_level.py): port attributes, bandwidth and operand-level-direction allocation.

Adapt the same eight-lane MAC throughput, precision, per-layer quantization
boundaries, SDRAM bandwidth and physical BSRAM budget. Preserve all feasible
published caching modes and tune with the same time budget. Run its policy through
the same independent legality checker; repair infeasible mappings with a
documented common fallback and count that repair cost. Distinguish unchanged
published code, cost-model adaptation and new mechanisms. No baseline performance
has been measured yet.

## Candidate claim and falsifiable hypothesis

Candidate: a compiler for one reusable tiny FPGA engine that emits bit-exact
integer schedules together with physically realizable scratchpad allocations,
jointly choosing tile boundaries, live INT32 state, BSRAM width/depth modes and
transfer overlap. The intended contribution is the demonstrable effect of those
combined executable constraints and a verifiable schedule certificate. It is not
the first fusion search, memory-aware scheduler, or low-cost CNN accelerator.

Hypothesis: at matched accuracy, eight MAC lanes, routed clock, SDRAM bandwidth
and bank budget, the proposed method reduces the geometric mean of complete KWS
and VWW accelerator latency by at least **15%** versus the strongest feasible
B1/B2/B3 schedule, with neither primary workload regressing more than **5%**.
The thresholds are prospective go/no-go criteria, not predictions. Energy is an
additional measured objective once supply instrumentation is available.

Eligibility for a scoped SOTA claim requires complete pinned accuracy sets, all
operator/host/DMA costs within identical boundaries, reproducible tuned baselines,
routed resources/timing, actual board executions, and uncertainty for energy.
Different FPGA processes, precision, model weights or host boundaries go in a
separate contextual table without frequency-only normalization. Test all feasible
points and report failures. Review newly published work again before submission.

## Overlap/pivot decision

Reject the original broad novelty framing: joint scheduling, fusion, physical
memory fragmentation and port models already exist. Continue engineering because
correctness and a real board implementation are prerequisites regardless of the
paper claim. Keep research novelty **provisional** until R05 constructs a concrete
counterexample that survives DeFiNES and a tile-expanded COSMA interpretation.
If merely supplying accurate Gowin costs to existing algorithms explains the
benefit, pivot to an implementation/methodology paper or develop a new mechanism;
do not describe the cost-model port as a new scheduling algorithm.

R01 is complete as a claim/baseline decision gate; the [September 26 audit](PRIOR_WORK_AUDIT.md) refreshes the prior-work boundary and pins inspected source files. Baseline execution/tuning remains B03, and the final submission-time review remains E04. Neither a bounded literature review nor source inspection certifies SOTA.

## Phase-2 R05 decision — 2026-09-23

The [one-port service bound](R05_PIVOT.md) and synchronous SRAM test confirm an
implementation constraint, but not a missing capability in DeFiNES. R05 chooses
the explicit **pivot/continue-with-engineering** branch of its done criterion. A
new scheduler claim remains provisional and must beat an adapted port-aware
DeFiNES and tile-expanded COSMA; no SOTA outcome is inferred from phase-2 fit.

The September 26 audit also includes [MATCHA](https://arxiv.org/html/2604.09124v1) and a [RISC-V depthwise accelerator](https://arxiv.org/html/2511.21232v1). Their heterogeneous-FP16 and layer-level execution boundaries make them contextual comparisons, not interchangeable Tang20K measurements.
