# Matched baseline campaign on the selected FPGA image

This record compares executable policies on the **same** Tang Nano 20K
accelerator, rather than comparing different FPGA builds. It is a short
full-model policy screen. It does not certify B03 or G6 while the published
DeFiNES policy space remains only partly executable. The complete B1/B2 tuning
record and signed 30-run primary and 45-run finalist physical campaigns are
archived under
[`evidence/phase6/matched-baselines-v1`](evidence/phase6/matched-baselines-v1).

## Frozen comparison contract

- Target: Tang Nano 20K (`GW2AR-LV18`), one eight-lane INT8 engine and the
  27 MHz routed image at
  `work/phase6/pool-timing-v1/route27/phase6_uart_burst/impl/pnr/phase6_uart_burst.fs`
  (SHA-256 `3e943639382e4efea62d4d2072c016fa40eacc7a97ef6bad7f5bfdc69d8dafce`).
  The [exact bitstream](evidence/phase6/matched-baselines-v1/route27-phase6_uart_burst.fs.gz)
  and [route report](evidence/phase6/matched-baselines-v1/route27-report.json)
  are archived. The route report records 27.613 MHz Fmax, no setup/hold violations, 38/46
  BSRAM, 9.5/24 DSP, and 10,095/10,368 CLS. Every policy uses this bitstream;
  no independent frequency or resource normalization is applied.
- Models: the pinned KWS and VWW ONNX models, calibration, public INT8 input
  format, and exact output bytes. The same grouping and dead-channel compaction
  are applied before policy lowering. Each candidate must match the selected
  implementation's pinned **and stress** input/output hashes, pass independent
  integer command replay, and pass full-model native RTL simulation at memory
  stall seeds 0 and 6063. Intermediate tensor checks are enabled in diagnostic
  fixtures; timed fixtures have only the final output check.
- Common optimizations: constant-filter lowering, exact tagged activation
  lookup, legal post-fusion tail prefetch, selected engine timing fixes, and the
  same 32 KiB scratchpad/8 MiB external address capacities. Policies may choose
  different legal placement, tiling, caching and fusion; those choices are the
  subject of the comparison, not a hardware advantage.
- Device latency is the FPGA cycle counter from command launch to completion,
  divided by 27 MHz. It includes all in-schedule DMA and engine work. UART
  fixture loading, input upload/readback and host polling are recorded
  separately. The physical campaign remeasures B1, B2 and selected B4 in
  randomized policy/model blocks on the same flashed image, with one stress
  execution, one pinned warmup and three pinned timed executions per block.

## Policies and tuning

**B1** is a conventional tiled schedule that materializes each completed
tensor in external memory, while retaining producer-to-activation epilogues
allowed by the common lowering. **B2** changes only physical live-range
placement: when a full tensor or identical sibling input fits in SRAM, it is
retained across the adjacent stages. Both enumerate the same four points:
full/half tiles × safe post-fusion prefetch off/on. Prefetch includes immutable
descriptor/parameter loads and legal same-layer next-input loads; the latter
matter for B1's half-tile VWW winner but have zero legal moves in the selected
full-tile schedules. Selection minimizes the
worst native cycle count over seeds 0 and 6063; the full tuning record is
[archived](evidence/phase6/matched-baselines-v1/b1b2-report.json). Native
selection alone does not prove the fastest setting on physical SDRAM, so a
separate VWW finalist screen covers the closest alternate setting for each
baseline.

The [DeFiNES paper](https://arxiv.org/html/2212.05344v1) covers spatial tile
size, fusion depth and overlap cache/recompute modes, including layer-wise
endpoints. Its pinned source revision is `7097d6090dc22321e44ce91434e7cc23b065864f`
and is an analytical geometry/cost explorer, not an
FPGA command compiler. The current executable backend covers only fixed
full-width VWW strip pairs. The restricted adaptation calls the pinned author's
halo recurrence, independently enumerates all eight on/off combinations of
the three executable depth-first cuts (layers 3–6, 7–10 and 11–14), and ranks
each exact native program by the same worst-of-two-stall-seed objective. The
result is [archived](evidence/phase6/matched-baselines-v1/b3/catalogue-selection.json).
The best combination is `111`, byte-identical to B4's VWW program. KWS endpoint
selection chooses B2, also byte-identical to B4. All eight VWW combinations,
the selected KWS/VWW pinned and stress fixtures, and the cache variant pass
independent full-model symbolic command replay and two-seed native RTL checks.
Thus this restricted comparison is a **tie**, not evidence of superiority over
DeFiNES.
A distinct vertical-halo SRAM cache using legal COPY descriptors saves 3,072
external bytes but needs 112 COPY descriptors and is slower in both native
seeds (2,268,261/2,331,540 versus 2,244,277/2,304,047 cycles). Its stress
fixture also passes native intermediate checks. An interior 8×16 output tile
of VWW layers 11–14 exposes the remaining spatial-search gap. Direct aligned
DMA cannot pack its off-axis halo, and direct output-row scatter across all six
tiles would need 1,536 DMA commands plus 1,536 waits before input or compute
commands, beyond the 2,048-command store. An isolated alternative **does** run
on the current image: aligned overfetch followed by chained SRAM COPY packs an
exact interior 8×16 tile in 105 sequencer commands, peaking at SRAM address
30,784; its 4,096-byte output matches the independent tile oracle in native
RTL at both stall seeds ([block evidence](evidence/phase6/matched-baselines-v1/b3-x-tiles/block-probe.json)).
A separate [six-tile full-model lowering](evidence/phase6/matched-baselines-v1/b3-x-tiles/full-model-native.json)
then assembles
three 8-row output bands in SRAM and emits 1,471 commands, within the same
2,048-command and 32 KiB bounds. Its segment has independent byte-level
DMA/COPY replay, and the composed VWW program passes exact final logits in
native RTL at both stall seeds. It takes 2,582,977/2,719,494 cycles versus
2,244,277/2,304,047 for selected B4, so this particular 2D tile is slower
in simulation. Separate [pinned and stress diagnostics](evidence/phase6/matched-baselines-v1/b3-x-tiles/diagnostics.json)
pass native RTL at both seeds with eight exact tensor checks: six depthwise
tiles covering the complete layer-12 tensor, the assembled layer-14 tensor,
and final logits. The symbolic proof is compositional: it replays the frozen
full-model source and every new x-tile segment independently, then checks the
splice; a single transformed-program replay in the general verifier remains
open. Its physical comparison appears below. Other tile shapes, stack cuts and
cross-layer placement remain untested; the restricted B3 row cannot close B03.

An [additional exact VWW screen](evidence/phase6/matched-baselines-v1/b3-spatial/report.json)
changes the layer-11–14 full-width strip height to 12, 8, 6 or 4 rows while
preserving the other layers and the common arithmetic and compiler
optimizations. Every candidate passes independent full-model replay and native
RTL at both stall seeds. Their worst-seed cycles are 2,305,104, 2,310,567,
2,316,558 and 2,327,174, respectively, versus 2,304,047 for selected B4.
Thus this additional tile-height search does not improve on B4 in simulation;
the matched physical h12/h8/B4 screen below preserves that ranking, although
h12 is close. For KWS, a
[bounded ABI audit](evidence/phase6/matched-baselines-v1/b3/kws-abi-audit.json)
finds the existing selected schedule already retains all relevant 64×25×5
intermediates in SRAM with no intermediate external stores or reloads. The
direct compact per-channel strip recipe requires each height to be divisible
by eight for aligned DMA/COPY, but the total height is 25. This does not rule
out a new packing algorithm or a changed hardware interface.

## Native screening, before physical comparison

Selected timed fixtures below use the same native executable and omit
diagnostic snapshot stores. Numbers are device cycles, not board measurements.

| Policy | KWS seed 0 | KWS seed 6063 | VWW seed 0 | VWW seed 6063 |
| --- | ---: | ---: | ---: | ---: |
| B1 layer-wise | 1,235,655 | 1,266,467 | 2,369,080 | 2,468,323 |
| B2 physical liveness | 1,181,858 | 1,190,362 | 2,321,487 | 2,392,682 |
| Selected B4 | 1,181,858 | 1,190,362 | 2,244,277 | 2,304,047 |

The selected B4 therefore ties B2 on KWS and improves VWW seed-0 cycles by
3.4%; the two-model native geometric-mean improvement over B2 is only 1.7%.

## Matched physical result

The board campaign completed all 30 planned executions with exact final
outputs, no mismatches and a sealed plan/report/record chain. The routed image
and 27 MHz clock were common to every row. Each policy/model had one stress
check, one pinned warmup and three pinned timed executions. Median device
latency excludes UART fixture loading and host input transfer; these are
recorded separately in the full board report.

| Policy | KWS cycles | KWS ms | KWS/s | VWW cycles | VWW ms | VWW FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 layer-wise | 1,347,760 | 49.917 | 20.033 | 2,711,182 | 100.414 | 9.959 |
| B2 physical liveness | 1,201,853 | 44.513 | 22.465 | 2,563,427 | 94.942 | 10.533 |
| Selected B4 | 1,201,859 | 44.513 | 22.465 | 2,441,571 | 90.429 | 11.058 |

B4 is 1.116× faster than B1 by two-model geometric mean, and 1.025× faster
than B2. KWS B2 and B4 have byte-identical command/payload programs, so the
KWS part of the latter comparison is a tie, not independent mechanism evidence.
VWW B4 is 1.050× faster than B2. The largest within-cell spread of the three
timed cycles is below 0.03%; these short runs establish repeatable device
cycles for the pinned inputs, not a full-dataset latency distribution.

The [one-flash finalist preflight](evidence/phase6/matched-baselines-v1/combined-finalists-preflight.json)
combines the closest B1/B2 settings, the DeFiNES-derived strip heights, the 2D
candidate and the halo-cache ablation on the same image. Its read-only gate
passes with canonical plan SHA-256
`12732ec1388f76317243002cb00f78c2bbf1ca3a871ab3cae8ccb32c65c00aa2`.
The [physical finalist report](evidence/phase6/matched-baselines-v1/combined-finalists-board-x-after-reconnect/report.json)
and [seal](evidence/phase6/matched-baselines-v1/combined-finalists-board-x-after-reconnect/seal.json)
pass **45/45 exact executions** with no output mismatches, one programming
operation and one UART session. Nine distinct VWW command streams each received
one stress check, one pinned warmup and three pinned timed runs. Byte-identical
`recompute` and `full_width` controls alias B4 and were run only once. Per-row
signatures and the plan/report/record/runner SHA-256 seal verify; the saved
plan-file hash differs from the canonical preflight hash because of JSON
serialization. Median device cycles exclude UART loading and host transfer.

| VWW schedule | Median cycles | Device ms | FPS |
| --- | ---: | ---: | ---: |
| B4 selected; recompute/full-width controls | 2,441,449 | 90.424 | 11.059 |
| DeFiNES-derived strip height 12 | 2,442,659 | 90.469 | 11.054 |
| DeFiNES-derived strip height 8 | 2,451,104 | 90.782 | 11.015 |
| Vertical-halo cache | 2,478,740 | 91.805 | 10.893 |
| B2 selected physical liveness | 2,563,037 | 94.927 | 10.534 |
| B2 half-tile plus prefetch | 2,620,613 | 97.060 | 10.303 |
| B1 selected layer-wise | 2,711,238 | 100.416 | 9.959 |
| B1 full-tile plus prefetch | 2,777,283 | 102.862 | 9.722 |
| Six-tile 2D spatial schedule | 2,973,809 | 110.141 | 9.079 |

B4 is the fastest of these nine. The selected B1/B2 settings beat their
measured alternate tile/prefetch settings. B4 is 1.050× faster than B2 for VWW, but
the KWS B2/B4 programs are byte-identical. Combining the KWS primary campaign
with this VWW finalist screen gives a 1.025× two-model geometric-mean B4/B2
speedup, still below the prospective 15% threshold. The 2D schedule takes
1.218× B4's cycles; this tests one feasible spatial tile, not every published
DeFiNES policy. Each timed cell has only three pinned repetitions, so this is
a repeatable short-screen ranking, not a full-dataset latency distribution or
a certified global policy optimum.

Three earlier cache-only starts programmed the FPGA successfully but timed out
at the UART capability exchange **before any inference**; [their failed
records](evidence/phase6/matched-baselines-v1/b3-cache-failed-attempts) are
preserved. A [failed combined start](evidence/phase6/matched-baselines-v1/combined-finalists-failed-attempt/report.json)
likewise completed zero inferences. The [minimal UART loopback diagnostic](evidence/phase6/matched-baselines-v1/uart-diagnostic/report.json)
failed before the USB reconnection and [passed afterward](evidence/phase6/matched-baselines-v1/uart-diagnostic/after-reconnect-report.json)
following a USB reconnection and S1 reset. The successful 45-case campaign
then ran without UART errors.

The restricted adaptation does **not** certify B03 or a full DeFiNES
comparison. A complete published-policy backend, complete accuracy/energy
studies and the remaining G6 causal ablations are still needed for a
publication claim.
