# Claim-freeze review — 2026-09-26

R01 freezes a falsifiable research direction and the closest comparison. It does
not certify novelty, reproduce every published accelerator, or prove SOTA. The
review builds on the original [matrix](novelty_matrix.md) and the
[R05 pivot decision](R05_PIVOT.md). B3 implementation and fair tuning remain B03;
the final literature refresh and independent reproduction remain E04.

## Search and inclusion record

The September 26 refresh searched primary papers and author repositories using:
`FPGA TinyML accelerator memory scheduling fusion 2025 2026 arxiv`,
`DeFiNES COSMA memory allocation quantization scheduling FPGA 2025 2026`, and
`Tang Nano 20K neural network accelerator INT8 TinyML paper`. It followed the
primary sources already named in the roadmap and their implementation links.
Include work on fusion/tiling, memory placement, physical resource costs and
quantized FPGA execution. Exclude unrelated COSMA HPC libraries, hobby forum
throughput claims and comparisons without a defined execution boundary.
This is a bounded claim review, not an exhaustive systematic review.

## Feature and eligibility findings

“Established” means supported by the linked paper or inspected source;
“unknown” means no verified implementation evidence in this audit. A blank
Gowin result is not evidence that another method cannot express the constraint.

| Primary work | Established overlap | Comparison status / unknowns |
|---|---|---|
| [FINN-R](https://arxiv.org/html/1809.04570), §3.2 | Folding and BRAM resource costs, including physical allocation granularity. | Contextual FPGA baseline; different quantization/streaming choices require matching. A Gowin eight-lane backend is not verified. |
| [SAMO](https://arxiv.org/html/2112.00170v2), §III | Partition/folding optimization with resource, memory-bandwidth and reconfiguration costs. | Author [repository](https://github.com/AlexMontgomerie/samo) exposes FINN, hls4ml and fpgaConvNet integration; backend reuse is established, a Tang20K run is unknown. |
| [DNNExplorer](https://arxiv.org/html/2008.12745), §§5–7 | Pipeline/layer-reuse split, buffering alternatives, height partitioning and traffic costs. | Architecture-level context; published performance uses different devices/models. No reproduced backend result here. |
| [Fused-Layer CNN](https://compas.cs.stonybrook.edu/%7Emferdman/downloads.php/MICRO16_Fused_Layer_CNN_Accelerators.pdf), §III | Dependency pyramids and overlap storage/recomputation tradeoffs. | Author manuscript reviewed in the earlier September 9 record; September 26 refetch timed out. No code or board reproduction claimed. |
| [MCUNetV2](https://arxiv.org/html/2110.15352), §§3–4 | Patch inference plus architecture/schedule co-design; measured versus analytic SRAM distinction. | Model changes require separate accuracy accounting; MCU deployment is contextual. |
| [msf-CNN](https://arxiv.org/html/2505.11483v3), §§4–6 | Fusion-block graph edges encode memory and MAC cost; constrained search and caching. | Secondary policy comparator. A Gowin physical memory certificate is unknown; do not infer absence from the MCU scope. |
| [DeFiNES](https://arxiv.org/html/2212.05344v1), §§III–V | Tile/fusion/caching modes, memory hierarchy placement and latency/energy models. | Closest B3; source explicitly models memory ports. Must retain feasible caching modes on the common engine. |
| [COSMA](https://arxiv.org/html/2311.18246v1), §III | Joint scheduling, contiguous placement and tensor replacement. | Mandatory counterargument or comparison: expand operations into a tile graph before claiming it lacks tile awareness. |
| [AccML 2026 depth-first fusion](https://accml.dcs.gla.ac.uk/papers/2026/8th_AccML_paper_9.pdf), §III | Fusion boundaries, TVM producer/consumer tiling, liveness and circular-buffer bounds. | Shrinking tile buffers and checking overwrite safety are already established ideas. No Gowin adaptation verified. |
| [RISC-V TinyML depthwise accelerator](https://arxiv.org/html/2511.21232v1), Tables II–IV | Fused expansion/depthwise/projection pipelines. | Its headline speedup is for a bottleneck layer; resource/power table is Vivado-based. Not an eligible full-model Tang20K or measured-board-energy comparison. |
| [MATCHA](https://arxiv.org/html/2604.09124v1), §§3–4 | Tile/device assignment and joint scheduling/address planning with spill costs; emitted execution plans. | FP16 heterogeneous Carfield/VCU118 context. Its DS-CNN/MobileNet tests show no latency gain from the evaluated tiling, a useful warning against assuming every model benefits. |

## Implementation inspection and B3 decision

DeFiNES remains the strongest directly relevant schedule/caching baseline.
At commit `7097d6090dc22321e44ce91434e7cc23b065864f`, inspection of
`classes/stages/DepthFirstStage.py` confirmed `backpropagate_tilesize`, horizontal
and vertical cache handling, cache-memory selection, and retained weights across
tiles. `classes/hardware/architecture/memory_level.py::port_allocation` constructs
read/write/read-write ports and maps operand movement directions to them.
File hashes and immutable retrieval URLs are saved in
[`evidence/phase0/prior-code.json`](evidence/phase0/prior-code.json).
This is source inspection, not execution of DeFiNES or a performance result.

The current accelerator command format cannot yet implement every DeFiNES
caching choice. B03 must extend that representation or label any restricted
adaptation, report the lost choices, and avoid treating a deliberately weakened
policy as published DeFiNES. The [B3 gap record](PHASE_5_STATUS.md) governs
this work. Use the same model, precision, engine resources, clock, bandwidth,
legality checker and tuning budget. No externally reported speedup is copied
into this project's performance table.

## Frozen decision

Continue with a **verified tiny-FPGA implementation and executable schedule /
memory-certificate hypothesis**. Reject “first joint memory-aware scheduler,”
“first port-aware scheduler,” and novelty from low board cost alone. Candidate
novelty remains a hypothesis until an adapted DeFiNES and tile-expanded COSMA
fail to explain a useful measured benefit.

Keep the prospective acceptance target: at matched accuracy, reduce complete
KWS/VWW geometric-mean device latency by at least 15% against the strongest
feasible tuned B1/B2/B3, with neither workload more than 5% slower. Include every
DMA, dispatch and dependency wait inside that device boundary. Report transfer-
inclusive latency separately. Energy needs physical instrumentation. If accurate
Gowin cost parameters alone explain the result, choose an implementation or
methodology contribution rather than inventing a new scheduling mechanism.
