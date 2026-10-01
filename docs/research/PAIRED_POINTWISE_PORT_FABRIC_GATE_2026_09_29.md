# Paired pointwise and port-fabric research gate — 29 September 2026

This follow-up tests the next investment gate from [the parallel architecture screens](PARALLEL_ARCHITECTURE_EXPERIMENTS_2026_09_28.md). It separates exact native RTL timing, physical implementation, and modeled transactions. The selected control is the 27 MHz cached-weight co-issue engine archived in the referenced Phase 6 worktree (SHA256 `9f934dcc27ae891e4c7b8b84bcc4b65e82d903492c1867572784fbe1cd9484ee`). All candidate changes are isolated; the selected engine, model programs, arithmetic, eight MAC lanes, and scratchpad interface are otherwise unchanged.

## Ordinary two-output control

The first serial paired traversal reused the existing accumulator bank but re-entered parameter setup for each paired pixel tile. It passed eight exact native KWS/VWW full-model checks and focused edge tests, yet regressed relative to co-issue: KWS 1,099,170→1,306,018 cycles (+18.8%) and VWW 2,014,205→2,030,581 (+0.81%) at fixed abstract RAM. This is a diagnostic, **not** the strong paired baseline.

The v2 control holds two eight-lane 33-bit accumulator banks and two parameter sets. Each activation gather serves sequential even/odd weight-MAC microsteps. It preserves the original per-channel bias, requantization, zero point, output placement, and INT32 overflow boundaries. A v3 revision shares the eight adder/overflow paths between the two microsteps to reduce synthesized logic; it has the same native cycle counts as v2. Source: [v3 engine](../../work/phase6/paired-pw-rtl-v3/engine.sv), [native runner](../../tools/phase6/paired_pw_native.py).

| Workload, fixed abstract RAM | Co-issue cycles | Strong paired v3 cycles | Lower native latency |
| --- | ---: | ---: | ---: |
| KWS pinned | 1,099,170 | 944,034 | 14.114% |
| VWW pinned | 2,014,205 | 1,807,391 | 10.268% |

All eight frozen pinned/stress × two-stall-seed KWS/VWW cases pass exact full-model outputs and tensor checks in [the v3 native report](../../work/phase6/paired-pw-rtl-v3/native/report.json). The stalled-memory reductions are 13.995% KWS and 10.037% VWW on pinned inputs. [Directed tests](../../work/phase6/paired-pw-audit-v3/report.json) pass on selected co-issue, v2, and v3 engines for odd channel/pixel tails, transient sums that cancel at the original reduction boundary, and even/odd overflow error 5 under SRAM backpressure. Held-out AD falls back to its original GEMM path: four native runs pass 20 tensor checks each, with unchanged 295,640 fixed / 379,822 stalled cycles in [the AD report](../../work/phase6/paired-pw-rtl-v3/ad-native/report.json).

Neither strong paired implementation fits the Tang Nano 20K. Gowin synthesis of v2 reports RP0006, 23,308 logic primitives versus the 20,736 device limit. Sharing the eight arithmetic paths in v3 reduces that count by 1,396, but [its source-pinned unsandboxed build](../../work/phase6/paired-pw-rtl-v3/route27-escalated/report.json) and [raw route log](../../work/phase6/paired-pw-rtl-v3/route27-escalated/route.log) still show RP0006 at **21,912/20,736 logic**, 1,176 over. Its 18 active project sources are pinned; the build exits during synthesis, before placement or timing, and produces no bitstream. The first sandboxed v2 launch crashed before synthesis because of macOS services and is not an area result. Native cycles must not be presented as board latency.

## Split-bank fragment candidate

[The bounded model](../../tools/phase6/port_fabric_model.py) checks all 20 frozen pointwise descriptors against the observed one-word first/cross reads and MAC issue counts, plus accepted traffic counters. It then compares ordinary paired 8-pixel × 1-output traversal with a candidate 4-pixel × 2-output mapping on two independently addressed 32-bit groups of the existing eight byte banks. The candidate charges two scalar weights, one engine-or-DMA grant at a time, cache reads, SRAM reads, fixed weight/parameter/descriptor reads, and output byte writes; it gives no free DMA overlap. Results are optimistic serialized service scores, **not** simulated or measured cycles.

| Model | Paired all-in PW service score | Split-bank candidate | Reduction |
| --- | ---: | ---: | ---: |
| KWS | 415,296 | 397,888 | 4.19% |
| VWW | 464,318 | 448,666 | 3.37% |

The VWW candidate also removes 8,392 issue slots on short tails by ordinary 4×2 lane packing, which is a separate effect from split-bank delivery. Eight earlier aligned VWW pointwise layers see no activation-service benefit. The [per-descriptor report](../../work/phase6/port-fabric-model-v1/README.md) retains the assumptions, source hashes, and odd-tail/alignment stress. Existing full-width VWW 2D tiles are 532,360 measured cycles behind the selected strip schedule on the same prior image; removing only their extra DMA time leaves roughly 230,000 cycles of engine deficit. [Matched baseline evidence](PHASE_6_MATCHED_BASELINES_V1.md).

The model's small *incremental* opportunity, the v2 physical rejection, and the lack of demonstrated engine-time gain from a fragment unit do not justify a full split-bank RTL prototype at this gate. A nonfunctional scratchpad split-read primitive was prepared for a physical-only screen but has **not** been routed or used for any inference claim. The paired traversal is conventional output blocking and is a strengthened control, not a novelty claim.

## Decision boundary

The ordinary paired traversal is exact and faster in native RTL, but its physically faithful v2 and area-reduced v3 implementations exceed this FPGA's logic limit. No paired board latency, routed frequency, energy, or accuracy-set result exists. The split-bank fabric's incremental modeled service improvement is too small to warrant adding its interface logic to a design already failing fit. **Stop this architecture line at the present gate.** A renewed attempt needs a different area-saving engine design or a different physical target, plus a workload/shape trace showing a substantial gain beyond the paired control at equal MAC lanes, BSRAM and frequency. Residual-join INT8 ownership remains a separate, untested hypothesis.
