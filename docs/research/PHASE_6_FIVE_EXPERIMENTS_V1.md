# Five follow-up accelerator experiments

Date: 2026-09-27. This ledger records a bounded screen of five proposed
mechanisms on the selected eight-lane INT8 Tang Nano 20K accelerator. The
reference is the exact fused-activation, safe-tail-prefetch, UART256 image at
24 MHz: 1,201,917 cycles / 50.079875 ms for KWS and 3,086,061 cycles /
128.585875 ms for VWW. Its bitstream SHA-256 is
`03e57521885f1a9d6b4b4cef7e374122ba948028ed379d963c97f794efc9f195`.
All latency and throughput below are **device execution only**, excluding
input upload and host readback. The pinned image results and acceptance rule
are in [the preceding campaign](PHASE_6_FINAL_CAMPAIGN_V2.md).

| Experiment | Most decisive result | Decision |
| --- | --- | --- |
| Exact channel compaction | VWW 128.586→107.091 ms at 24 MHz, 10/10 exact | Keep |
| Balanced pooling and 27 MHz clock | Grouped VWW 128.586→114.031 ms, 10/10 exact | Keep provisionally |
| Narrow accumulators | 10,137→10,141 CLS; Fmax 26.285→25.570 MHz | Reject |
| Cross-layer spatial fusion | Three full-width VWW pairs: 94.932→90.430 ms, each 10/10 exact | Keep bounded prototype |
| Restricted joint selection | 16 exact native cases; both greedy orders have zero regret | No scheduler novelty shown |

Combining compaction and the 27 MHz core yields **44.513 ms KWS and
94.932 ms VWW** on the board. Adding three strip-fused VWW pairs reduces
VWW to **90.430 ms** with KWS unchanged, for a **1.26482×** dual-workload
geometric-mean device throughput gain over the original selected image.

## 1. Exact internal channel compaction

`tools/phase6/channel_compaction.py` retains a channel unless all retained
consumers have zero weights for it. It preserves the public model interface
and the original INT8 weights, quantizers, and integer layer boundaries on
retained paths. KWS is unchanged; 33 VWW layers are affected. VWW aligned
weight payload drops from **216,832 to 42,240 bytes** and estimated MACs
after existing constant-filter lowering from **4,358,528 to 3,733,541**.
The original and compacted programs
match independent integer oracles at every layer for pinned and stress inputs,
pass command replay, and pass 12 full-model exact native RTL cases on the
same fused engine executable.

The unchanged 24 MHz selected image passed **10/10 exact board inferences**
with the compacted schedule. VWW median is **2,570,173 cycles / 107.090542 ms /
9.33789 FPS**, a **1.20072×** device throughput gain over the prior VWW
median; KWS is **1,201,921 cycles / 50.080042 ms / 19.96803/s**, effectively
unchanged. Dual-workload geometric-mean throughput improves **1.09577×**.
This isolates a compiler change, although these are short screens rather than
a full accuracy or endurance campaign. Source and board evidence are under
`work/phase6/channel-compaction-v1/`.

## 2. Balanced, narrower pooling arithmetic

`tools/phase6/pool_timing.py` changes the pooling sum tree without changing
cycles or INT8 output semantics. The accepted maximum pool area is 961,
so the centered-sample total is bounded by 245,055 and fits signed 19 bits;
four-sample subtotals fit signed 11 bits. Focused edge, engine, UART bridge,
27 MHz PHY, and 24 full-model native cases pass; native cycles match the
original engine exactly. The 24 MHz candidate route uses **10,085 CLS** and
has **26.601 MHz Fmax**, versus 10,137 CLS and 26.285 MHz for the selected
engine. The 27 MHz route passes timing at **27.613 MHz Fmax**, with zero
setup/hold violations and **10,095/10,368 CLS, 38/46 BSRAM, 9.5/24 DSP**.
The 27 MHz bitstream SHA-256 is
`3e943639382e4efea62d4d2072c016fa40eacc7a97ef6bad7f5bfdc69d8dafce`.
It uses the supported 27 MHz PLL configuration and a 36-clock UART divisor
for 750 kbaud. A 12.5% clock-only throughput gain was a pre-board prediction,
not a measurement. Route and RTL evidence are under
`work/phase6/pool-timing-v1/`.

The grouped schedule on this 27 MHz image passed **10/10 exact board
inferences**. Its medians are **1,201,843 cycles / 44.512704 ms / 22.46550/s
KWS** and **3,078,838 cycles / 114.031037 ms / 8.76954 FPS VWW**. Against
the matched grouped 24 MHz screen, device latency improves **1.12507× KWS**
and **1.12764× VWW**. The near-equal cycle counts show the measured gain is
primarily from the higher clock.

The compacted schedule on the **same 27 MHz bitstream** also passed **10/10
exact board inferences**. Its medians are **1,201,848 cycles / 44.512889 ms /
22.46540/s KWS** and **2,563,167 cycles / 94.932111 ms / 10.53384 FPS
VWW**. This is **1.12507× KWS** and **1.12808× VWW** versus the matched
compacted 24 MHz screen. Compaction alone on the 27 MHz image improves VWW
device throughput **1.20119×** versus the grouped 27 MHz screen. Relative to
the original selected 24 MHz image, the combined result is **1.12506× KWS**,
**1.35450× VWW**, and **1.23447× dual-workload geometric-mean device
throughput**. The 27 MHz route has only **0.613 MHz Fmax margin**, so this is
the best **short-screen** result, pending longer qualification.

## 3. Proof-selected narrow accumulators

The pinned models' nonconstant prefixes admit narrower arithmetic: signed
21 bits for KWS and signed 28 bits for VWW after 971 large-bias zero filters
are lowered as constants. A 28-bit spatial accumulator candidate passed 12
exact native full-model runs and retained identical cycles, but its 24 MHz
route used **10,141 CLS instead of 10,137** and fell to **25.570 MHz Fmax
instead of 26.285 MHz**. Moreover, a legal arbitrary descriptor with bias
2³¹−1 and a subsequent product of 1 must reach 2³¹ and raise the existing
overflow error; a universal 28-bit replacement wraps. This candidate is
**rejected** for both performance and generic compatibility. The proof and
route comparison are in `work/phase6/narrow-accum-v1/`.

## 4. Depthwise-to-pointwise cross-layer spatial fusion

`tools/phase6/cross_layer_screen.py` and
`tools/phase6/cross_layer_compaction_screen.py` execute exact software
depthwise–activation–pointwise–activation tiles on pinned and stress inputs.
They compare against independent integer oracles for 8×8 and 16×16 tile
requests with cache on and off. The pre-epilogue grouped VWW schedule
accounts for **99,072 bytes** of bridge transfers across these pairs; KWS
has none.
After compaction, VWW has **92,160 bridge bytes**. This accounting uses the
pre-epilogue schedule stage list; it does not independently parse the final
bytecode.

Compaction makes **eight nominal byte-count configurations in two late VWW
blocks** fall below a 32 KiB scratch budget. For the first block, the
cache-off count is 53,120→7,043 bytes; for the second, 89,856→21,681 bytes.
The 8×8 and 16×16 requests clip to the same small spatial extent in those
blocks, so these are not eight distinct spatial shapes. Those two bridges
already have zero transfer bytes after compaction. A traffic-only sensitivity
calculation using historical DMA fit gives 1.01715× dual-workload throughput
if all fitted bridge time vanished with no added work; this is **not a bound**
on an implemented fusion architecture.

The 8×8/16×16 demand-driven software tiles still have **no direct FPGA
lowering or address/port timing certificate**. A separate **executable
full-width strip** candidate now tests one of these pairs with the existing
hardware ABI. For VWW layers 3–6, two 24-row strips retain the depthwise
activation in SRAM, feed the pointwise layer, and store the original NCHW
output. Before the first output overwrites the external source slot, the
second strip's source is loaded into freed SRAM. The four original INT8
requantization/activation boundaries remain exact; the selected 27 MHz
bitstream is unchanged. The standalone block uses 137 commands and has a
28,416-byte SRAM high-water mark. Native RTL checks pass pinned and stress
inputs with both memory-stall seeds; the full-model timed seed-0 cycles fall
from **2,321,487 to 2,257,570**. Check and stress fixtures verify 30
intermediate tensors at each seed.

The strip schedule passed a **10/10 exact short board screen**. VWW median
falls from **2,563,167 cycles / 94.932111 ms / 10.53384 FPS** to
**2,478,506 cycles / 91.796519 ms / 10.89366 FPS** on the same image, a
**1.03416×** VWW device throughput improvement. KWS control remains
**1,201,840 cycles / 44.512593 ms**. The paired dual-workload geometric
mean improves **1.01694×**, below the 1.03× mechanism-expansion gate.
This is a real measured cross-layer gain for one VWW pair, not proof that
arbitrary 8×8 fusion or the full joint search is physically implemented.

A second 12-row-strip schedule for VWW layers 11–14 composes with the first.
Full-model native pinned and stress checks again pass both memory seeds with
30 intermediate tensors checked. The combined schedule passes **10/10 exact
board inferences** on the same bitstream, with VWW **2,451,737 cycles /
90.805074 ms / 11.01260 FPS** and KWS **1,201,840 cycles / 44.512593 ms**.
Relative to the one-pair physical result, the second pair adds a **1.01092×**
VWW gain. Together, the two pairs improve dual-workload geometric-mean
throughput **1.02248×** against the compacted 27 MHz schedule, still below
the 1.03× expansion gate. The mechanism is physically promising but the
current scope does not establish a general fused-tile architecture.

A third pair, VWW layers 7–10, uses two 12-row full-width strips with
asymmetric bottom/right padding. Its independent local INT8 oracle, command
replay, and full-model native RTL checks pass pinned and stress inputs at two
memory-stall seeds. The timed seed-0 native schedule improves from
**2,247,781 to 2,244,277 cycles**; 30 intermediate tensors are exact in
each check run. The short board screen then passes **10/10 exact** on the
same 27 MHz image. VWW median improves from **2,451,737 cycles /
90.805074 ms / 11.01260 FPS** to **2,441,608 cycles / 90.429926 ms /
11.05829 FPS**. KWS is **1,201,854 cycles / 44.513111 ms**. The third
pair adds only a **1.00415×** VWW speedup. All three pairs together improve
VWW device throughput **1.04979×**, and dual-workload geometric-mean
throughput **1.02459×**, against compacted 27 MHz. Against the original
24 MHz selected image, the gains are **1.12506× KWS**, **1.42194× VWW**,
and **1.26482× geometric mean**. This remains below the 1.03×
mechanism-expansion gate when evaluated incrementally against the compacted
27 MHz schedule. A larger or more general fused architecture remains
unproven.

## 5. Restricted joint selection

`tools/phase6/joint_selection_screen.py` executes a 2×2 catalogue on one
frozen RTL executable, with two deterministic memory-stall seeds. The factors
are exact channel compaction and **activation epilogue fusion plus safe-tail
parameter prefetch**; the latter is not an isolated activation-fusion
ablation. All 16 pinned full-model native cases pass exact final-output
checks. For seed 0, VWW cycles are **2,933,853** (neither), **2,479,454**
(compaction), **2,725,926** (fusion/prefetch), and **2,321,487** (both).
KWS is 1,260,399 without fusion/prefetch and 1,181,858 with it, unaffected
by compaction. The paired VWW gain overlaps the separate effects by 49,960
cycles. Seed 6063 has the same ordering and 54,110 cycles of overlap.

Both greedy factor orders and independent one-pass decisions select a best
cell for both models and seeds with **zero regret**; KWS ties between cells
01 and 11. This restricted catalogue
therefore gives **no evidence that a novel joint search is needed**. It does
not test the unimplemented cross-layer hardware or full bank-aware physical
search. Evidence is under `work/phase6/joint-selection-v2/`.

## Research decision

Compaction is a measured compiler win, and the narrowed pooling datapath
has a timing-closed, board-verified higher-clock implementation. A restricted
cross-layer strip schedule is also exact and physically faster. These results
are not yet sufficient evidence for a top-tier architecture contribution.
Narrow accumulation is a
negative result. The restricted joint catalogue does not beat simple choices.
The broader cross-layer candidate still needs a general physical lowering,
matched B1/B2/B3 comparisons, held-out cost prediction, full accuracy,
power/energy measurement, and causal ablations before G6 can close.

The [evidence manifest](evidence/phase6/five-experiments-v1/manifest.json)
pins 119 compressed source, simulation, route, and physical artifacts by SHA-256;
the [matched physical comparison](evidence/phase6/five-experiments-v1/matched-comparison.json)
recomputes medians and speedups from signed records with identical timed inputs
and outputs. The archive excludes bitstreams and executables, but validates
their hashes against the local generated files. Recreate and verify it with
`.venv/bin/python3 tools/phase6/archive_five_experiments.py --run` after
reproducing the work artifacts.
