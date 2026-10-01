# Phase 6: matched engine profile and focused prototype

Date: 2026-09-28. This experiment starts from the strongest executable KWS
and VWW B3 programs, keeps their exact models and INT8/INT32 arithmetic, and
tests one hardware change against the existing eight-lane engine. The broader
Phase 6 research and novelty gates remain open.

## Where cycles go

The source-pinned native RTL profiler replays both original programs with
fixed and randomized external-memory stalls. All four profiled runs reproduce
the original output checks and elapsed/engine/DMA/overlap counters. At seed 0,
activation gathering/cache handling occupies **47.84% of KWS engine cycles**
and **46.40% of VWW engine cycles**. MAC issue accounts for 30.75% and 24.38%;
weight access/cache for 7.57% and 8.09%. These are measured state occupancies,
not removable-cycle or speedup estimates.

KWS's four 64-to-64 pointwise layers have 125-byte channel planes. Their
second-word request/wait states total **237,568 cycles**, or 20.10% of complete
native device latency. VWW's pointwise first-word waits total **302,479
cycles**, or 13.92% of complete native latency. Additional request
backpressure is only 22 KWS and 8 VWW cycles at seed 0: fixed activation
delivery states, rather than external-memory arbitration, dominate these
counts. The different KWS/VWW shapes mean a crossing-only change cannot be
assumed to help both.

The companion compiler screen finds no identity activation map to remove
among 9 KWS and 40 VWW fused descriptors. Inserting software eight-pixel
pointwise packing into the existing dense-layout path loses at least 29,440
KWS cycles even under optimistic read-removal assumptions; DMA packing exceeds
the 32 KiB command store. This is a negative result for that insertion path,
not a proof against a redesigned persistent layout.

## Controlled RTL screens

The first RTL experiment overlaps cache clearing with LUT loading. It passes
its full-model and focused edge checks, but saves just 0.19% KWS and 0.47% VWW
native latency, so it was not routed.

The second experiment loads a **valid cached weight on the edge that finishes
its activation operand**. The ordinary separate weight-request state is
skipped on a hit. Misses retain the original request/wait protocol; MAC width,
accumulation, requantization, commands and model programs are unchanged. The
candidate engine SHA256 is
`9f934dcc27ae891e4c7b8b84bcc4b65e82d903492c1867572784fbe1cd9484ee`.

The paired native matrix executes both engines on KWS and VWW, pinned and
seeded-stress inputs, and external-stall seeds 0 and 6063: **16 full-model
executions pass exact output checks**. Three focused RTL edge-test groups also
pass. The control engine reproduces its archived counters. On fixed-memory
pinned runs:

| Model | Control cycles | Candidate cycles | Lower latency | Throughput speedup |
| --- | ---: | ---: | ---: | ---: |
| KWS | 1,181,858 | 1,099,170 | 7.00% | 1.0752× |
| VWW | 2,173,551 | 2,014,205 | 7.33% | 1.0791× |

The 82,688 and 159,346 saved engine cycles exactly match the profiled cached
weight-hit counts for these two programs. The geometric-mean native latency
falls **7.16%**, below the project's prospective 15% matched-device-latency
gate. Native timing does not establish FPGA fit, device latency, energy or
architectural novelty.

## Physical and research decision

The separately measured generic B3 program on the existing 27 MHz image has
21/21 exact short-screen executions and medians of **1,201,854 KWS** and
**2,317,026 VWW** device cycles. The latter is 4.66% lower latency than the
matched current VWW program on the same image (2,430,351 cycles). KWS uses
identical program bytes in both policies. This is a real B3 scheduling gain,
and the same B3 programs are the engine candidate's controls.

The isolated Gowin route replaces exactly one of the selected project's 18
active sources and preserves the same 27 MHz constraints. It passes with
**28.826 MHz routed core Fmax**, zero setup/hold violations, **10,090/10,368
CLS**, 38/46 BSRAM and 9.5/24 DSP. Relative to the previous image, that is
five fewer CLS and 19 more logic cells, within the same device limits. Both
images are measured at 27 MHz; the candidate's bitstream SHA256 is
`eaf74e99519d3dfac876bf21e9087dd005071bb8b8682d3f8ae6fb23bb76644d`.
The first bounded candidate-board screen passes **10/10 exact inferences**,
including seeded stress outputs, with the same strongest B3 commands, payloads,
inputs and oracle bytes. Its medians are 1,119,158 KWS and 2,157,424 VWW
cycles. Because the original B3 screen reloaded and warmed up before each timed
inference, a second **14/14 exact** screen repeats that protocol: each of the
three timed runs/model follows a separate fixture reload and immediate warmup.
The protocol-matched medians are:

| Model | Sealed B3 control cycles | Candidate median cycles | Candidate latency | Lower device latency |
| --- | ---: | ---: | ---: | ---: |
| KWS | 1,201,854 | 1,119,142 | 41.450 ms (24.13/s) | 6.882% |
| VWW | 2,317,026 | 2,157,255 | 79.898 ms (12.52 FPS) | 6.896% |

The geometric-mean device latency falls from **61.806 to 57.548 ms**, or
**6.889%**, relative to the sealed B3 medians. The first and protocol-matched
screens agree within 169 VWW and 16 KWS cycles. The candidate and control were
measured in separate sessions, so this remains an exploratory screen, not an
interleaved A/B statistical campaign. The device counter excludes UART model
upload and readback, as in the control. The screen is not an accuracy-set,
energy or endurance result. An independent checker verifies raw records,
warmup/load adjacency, exact outputs, source/bitstream hashes, route and seals.

A source-pinned native gain is an engineering result;
cache-hit co-issue is not yet a defensible novel architecture. Related FPGA
work already [overlaps operand movement and computation](https://arxiv.org/abs/1912.07284).
[DeFiNES](https://arxiv.org/abs/2212.05344) also models word length, port
conflicts and data copies, so our earlier weak proxy is not evidence of a
published DeFiNES limitation. [FINN-R](https://doi.org/10.1145/3242897)
and [D-SWIM](https://doi.org/10.3390/jimaging5030034) establish neighboring
streaming, interleaved-layout and misaligned-window mechanisms. This bounded
primary-source review is not an exhaustive novelty search.

The next architectural hypothesis is a no-pack activation-stationary traversal
with one/two-word fragment retention and a runtime-selectable fallback under
the same SRAM/port budget. It must beat ordinary loop interchange, simple
two-word buffering, their combination, a resource-matched streaming/window
baseline and the corrected DeFiNES adaptation. It must improve both KWS and
VWW with exact outputs and physically routed latency; the current profile does
not predict its gain. From the measured 6.889% geometric-mean latency reduction,
reaching the prospective 15% gate would require at least another **8.71%**
reduction relative to this candidate, with no workload regression beyond the
gate. Even that numeric result would not establish architectural novelty by
itself.

## Evidence

The [verified durable archive](evidence/phase6/engine-candidate-v2/manifest.json)
embeds 3,503 referenced files and retains 20 generated/external identities by
hash. It also includes two portable physical-screen archives with the tested
bitstream and native executable. Workspace verification passes. Its four
reported path limitations all refer to the local Gowin executable outside the
repository; they do not concern missing fixtures or experiment records.

- Full native state profile: `work/phase6/engine-profile-v1/final/report.json`
  and `summary.md`; rerun with `tools/phase6/profile_engine.py`.
- Compiler rejection: `work/phase6/engine-candidate-compiler-v1/report.json`;
  rerun with `tools/phase6/engine_candidate_compiler.py`.
- RTL v1/v2 identity, full-model matrix and edge reports:
  `work/phase6/engine-candidate-rtl-v{1,2}/`; rerun with
  `tools/phase6/hypothesis_engine_overlap.py` and
  `tools/phase6/hypothesis_engine_weight_issue.py`.
- B3 physical short screen and seal:
  `work/phase6/generic-b3-board-v2/physical/`; its verified dependency closure
  is `evidence/phase6/generic-b3-board-v1/archive/manifest.json`.
- Bounded primary-source review and its limitations:
  `work/phase6/engine-prior-work-v1/NOTES.md`.
- Candidate route project, raw reports and source coverage:
  `work/phase6/engine-candidate-rtl-v2/route27/report.json`; rerun with
  `tools/phase6/hypothesis_engine_route.py`.
- Candidate physical plan, records, seal and embedded dependency archive:
  `work/phase6/engine-candidate-rtl-v2/physical/{screen-v1,matched-v1}/`;
  rerun with `tools/phase6/physical_engine_candidate.py` and
  `tools/phase6/physical_engine_candidate_matched.py`.
- Independent audit of both physical screens:
  `work/phase6/engine-candidate-rtl-v2/independent-audit/`.
