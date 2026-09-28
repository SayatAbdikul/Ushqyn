# Phase 6 non-streaming novelty screens

Date: 2026-09-28. Experiments 1, 2 and 4 from the follow-up research list
were run in parallel. The proposed continuous audio/video experiment 3 was
excluded at the user's request. These are bounded screens of mechanisms, not
a new SOTA claim or a completed Phase 6 evaluation.

The frozen physical reference is the selected eight-lane INT8 Tang Nano 20K
image at 27 MHz, SHA-256
`3e943639382e4efea62d4d2072c016fa40eacc7a97ef6bad7f5bfdc69d8dafce`,
with channel compaction and three exact VWW strip-fused pairs. Its 10/10 exact
short-board medians are **1,201,854 cycles / 44.513111 ms / 22.4653 KWS
inferences/s** and **2,441,608 cycles / 90.429926 ms / 11.0583 VWW FPS**.
The matched candidate checks use the frozen 27 MHz pooled engine and identical
model inputs. Device execution excludes host upload and readback. The existing
mechanism expansion gate is at least **1.03x dual-workload geometric-mean
throughput** on the matched comparison.

| Experiment | Strongest result | Decision |
| --- | --- | --- |
| 1. Producer-generated zero-block activity | The present schedule has 31,726 KWS and 38,120 VWW physically aligned, all-zero-point weighted channel iterations on pinned inputs. A one-cycle ideal skip is only 1.02147x dual-workload throughput; a two-cycle ideal is 1.04392x before metadata, correction, or timing cost. The existing line-cache two-channel fusion condition occurs zero times. | No executable speedup established. Do not expand the existing-cache path to the board. A separate producer-written sidecar remains unimplemented and unmeasured. |
| 2. Compiler-verified fast-path contracts | Ten full-model native RTL runs were exact with unchanged cycles; malformed hints were rejected. Gowin used 10,151 versus 10,095 CLS, while Fmax rose from 27.613 to 28.910 MHz. | Reject as an area or current-clock performance mechanism. No board run. |
| 4. Symbolic final constant channels | A 234-channel VWW constant slice was propagated through activation, pooling and classifier. Latest three-pair VWW native latency fell 0.2245% and 0.3782% at two deterministic memory seeds; KWS was unchanged. | Exact but far below the 1.03x expansion gate. No board run. |

## Experiment 1: quantized-zero activity

The census decodes actual pointwise descriptors in the selected three-pair
schedule and compares pinned and seeded-stress integer graph values. All
pointwise input zero points are `-128`. The current engine multiplies raw
stored bytes and starts from a corrected bias, so replacing an all-zero-point
input group with zero arithmetic is **incorrect**. Each skipped input channel
still needs its exact `zero_point * weight` accumulator contribution. Exhaustive
INT8 zero-point/weight arithmetic and a legal overflow counterexample are in
the feasibility probe; a centered-product rewrite cannot silently replace the
generic engine's eight-channel overflow behavior.

Pointwise descriptors account for 643,484/1,170,215 KWS native engine cycles
(55.0%) and 908,698/2,133,486 VWW cycles (42.6%) in the unchanged seed-0
engine. However, no full eight-input-channel weight word has all eligible
zero-point groups. Physically aligned tags cover 31,726 of 262,144 KWS and
38,120 of 333,798 VWW channel/output-block iterations on the pinned inputs.
The native trace attributes 33,322 and 40,028 cycles to those iterations'
input-fetch states, respectively. These are native trace observations, not
measured board savings or strict bounds on physical SDRAM stalls.

The best free-tag, free-correction **one-cycle** extrapolation from the board
medians reaches only 1.02147x dual-workload throughput. A hypothetical
**two-cycle** skip reaches 1.04392x; to retain 1.03x, all extra work must
average below about 0.615 cycle per certified iteration across the two
workloads. This is an optimistic economic target, not an implemented result.
The tested route to two-cycle fusion using the existing prefetched line word
has zero qualifying events: KWS alignment eliminates them and VWW's 27,364
otherwise suitable next-channel groups are not line-cache hits. A separate
producer/DMA-maintained metadata RAM, an early read port, and a compensation
update fused with an existing state would be required. That sidecar has not
been built, routed, or tested on the FPGA. It remains a future research option,
not a result of this screen.

Evidence: `work/phase6/novelty-zero-block-v1/report.json`,
`work/phase6/novelty-zero-block-v1/native-profile/report.json` and
`work/phase6/novelty-zero-bypass-v1/report.json`. Recreate with
`python3 tools/phase6/novelty_zero_block.py`,
`.venv/bin/python3 tools/phase6/novelty_zero_block_profile.py`, and
`python3 tools/phase6/novelty_zero_bypass.py` in that order.

## Experiment 2: compiler control contracts

The isolated compiler marks proven 1x1 pointwise and 3x3 depthwise
descriptors with formerly unused header bits. Unmarked descriptors retain a
generic path; the RTL validates the marked geometry and rejects inconsistent
or contradictory hints before output writes. Ten pinned/stress/timed
full-model native cases passed exact tensor checks with zero cycle difference.
Adversarial RTL checks covered generic unmarked execution and four bad-hint
classes. The integrated 27 MHz Gowin route used the same frozen host, PLL and
build script as the selected image and reported zero setup/hold violations.

| Routed quantity | Selected | Hinted contract | Change |
| --- | ---: | ---: | ---: |
| CLS / 10,368 | 10,095 | 10,151 | +56 |
| Logic / 20,736 | 18,882 | 18,932 | +50 |
| BSRAM / 46 | 38 | 38 | 0 |
| DSP / 24 | 9.5 | 9.5 | 0 |
| Fmax | 27.613 MHz | 28.910 MHz | +1.297 MHz |

The hint validator and changed control erase the intended area recovery.
At 27 MHz the cycles are unchanged. Under the existing direct rPLL and 27 MHz
input, 28 MHz is not a legal exact output because the PFD must remain at least
3 MHz; the next direct step is 30 MHz, above the candidate's routed Fmax.
Thus the timing margin does not yield an attainable current-architecture
clock improvement. The generated bitstream was not programmed.

Evidence: `work/phase6/novelty-contracts-v1/report.json`,
`edges/report.json`, `route27/report.json` and `decision.json`. Recreate the
isolated checks with `.venv/bin/python3 tools/phase6/novelty_contract.py
native`, `edges`, `route`, then `decision`; the Gowin route requires the
installed vendor tools and access to their macOS runtime.

## Experiment 4: final constant propagation

The compacted VWW graph still materialized 234 known-constant channels in
its final convolution, activation, pool and reshape. Exact symbolic
propagation reduces that convolution's 256 outputs to 22 and the classifier's
input count from 256 to 22. It eliminates 455 nonzero classifier weights and
13,808 bytes of schedule DMA loads. All removed pooled constants equal the
classifier input zero point, so the classifier's mathematical bias delta is
exactly zero; integer partial-sum bounds stay within INT32.

Independent original-versus-folded integer checks passed all 58 VWW layers
for pinned and stress inputs. Three local strip replays and full-model native
checks passed two memory-stall seeds; check fixtures compared 30 intermediate
tensors per seed. Against the latest matched three-pair schedule, VWW native
cycles fell **2,244,277→2,239,238** (seed 0) and
**2,304,047→2,295,334** (seed 6063). With KWS unchanged, geometric-mean
throughput is only **1.00112x–1.00190x**. A board run would not change the
mechanism's expansion decision.

Evidence: `work/phase6/novelty-constants-v1/certificate.json`, `report.json`,
`threepair/report.json` and `decision.json`. Recreate with the `prepare` and
`native` stages of `tools/phase6/novelty_constants.py`, followed by
`tools/phase6/novelty_constants_threepair.py` and
`tools/phase6/novelty_constants_decision.py`.

## Research and hardware decision

The reconnected Tang Nano 20K was detected over JTAG, but these candidates
provide no new routed and cycle-improving image to test. None was programmed;
the prior physical reference remains the last measured result. These screens
do not establish a novel architecture or SOTA. The next valuable research
test is still a general executable cross-layer tile/INT32/bank-port mechanism
against tuned DeFiNES, COSMA and direct-schedule baselines, with matched
accuracy, device latency, resource, and energy boundaries. The symbolic
constant pass may be retained as a compiler cleanup, but its small benefit
must remain separate from a central architecture claim.

The [reproducibility archive](evidence/phase6/novelty-nonstream-v1/manifest.json)
seals 93 source and evidence files and hash-pins 17 generated files. Run
`python3 tools/phase6/archive_novelty_nonstream.py --verify
--verify-workspace` to check the archive against this checkout, including the
frozen physical baseline and matched Gowin source sets.
