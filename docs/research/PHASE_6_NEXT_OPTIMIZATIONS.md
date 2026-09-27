# Optimization brainstorm after direct writeback — 2026-09-27

The later [follow-on optimization results](PHASE_6_FOLLOWON_OPTIMIZATIONS.md)
record which ideas were implemented, measured, or rejected after this brainstorm.

This is a read-only architecture analysis. No new RTL, image, model, clock or
board campaign was changed. The VWW accuracy campaign remains paused. The
starting point is the physically screened direct-writeback candidate:
125.447 ms KWS and 363.630 ms VWW at 20.25 MHz, eight INT8 multipliers,
32 KiB SRAM and 8 MiB SDRAM. Full accuracy on that candidate is still pending.

The [measured baseline](evidence/phase6/writeback/summary.json) is authoritative.
The accompanying [calculation record](evidence/phase6/next-optimization-analysis.json)
pins the sources and separates structural cycle counts from physical counters.
Every proposed saving below is unimplemented. Percentages use current measured
total cycles as the denominator, not a claim that the saved engine states
translate unchanged through arbitration. Alternatives and overlapping savings
must not be added.

## What limits the current design

| Observation | KWS | VWW |
| --- | ---: | ---: |
| Measured total cycles | 2,540,307.5 | 7,363,506.5 |
| Measured engine cycles | 2,385,818.5 | 6,644,183 |
| Scheduled convolution/classifier MACs | 2,656,768 | 7,489,664 |
| MACs divided by eight times total cycles | 13.07% | 12.71% |
| Ideal MAC-only time at current clock | 16.40 ms | 46.23 ms |
| Pointwise structural engine cycles | 1,340,984 | 3,673,198 |
| Depthwise structural engine cycles | 455,032 | 1,206,626 |
| Activation structural engine cycles | 360,697 | 1,160,220 |
| Initial-convolution structural cycles | 216,103 | 581,228 |

Scheduled MACs include padded convolution positions. This utilization ratio is
an end-to-end scheduled-work metric, not a sampled measure of multiplier
activity. The ideal MAC-only times omit data movement, requantization,
activations and control; they are not achievable latency forecasts.

Routing uses 9,678/10,368 CLS sites (93.3%), 35/46 BSRAM blocks and 14.5/24 DSP
equivalents. Only 690 CLS sites remain. The current Fmax is 20.334 MHz with
0.205 ns worst setup slack. Free DSPs alone do not establish space for more
lanes, buffers, muxes and routing.

## Ranked experiments

1. **Overlap the next pointwise input request with the current MAC.** The
   scratchpad port is idle in `PW_MAC`; the engine then enters `PW_X_REQ`.
   Present the next channel's address during the current MAC and skip that
   request state only if the arbiter accepts it. Preserve a stalled request and
   execute the current MAC exactly once. There are 258,048 removable request
   states in KWS and 888,832 in VWW: 10.16%/12.07% of current total cycles as
   no-contention ceilings. This uses the existing port and arithmetic units.
   Address-path timing and read-response ownership are the principal risks.

2. **Register the selected broadcast weight during existing fetch states.**
   `whex[col[2:0]*8+:8]` currently selects a byte before all eight multipliers.
   The worst routed path starts at `col_2_s0/Q` and ends at a mapped requantizer
   DSP input, with 48.922 ns data delay and 41 logic levels. Latch the selected
   byte during existing input/weight fetch states, including cache-hit and
   first-word cases. First seek timing margin at the existing clock without
   new MAC states. Only a new route and board tests can justify a higher clock.

3. **Reuse spatial pointwise input words in the existing line-cache BSRAM.**
   `PW_X*` bypasses the 256-entry tagged line cache used by other paths. A
   channel's second word from one pixel group becomes the first word of its
   next group. Descriptor-level address walks find 215,040 reusable first
   words for KWS and 266,240 for VWW. A synchronous cache-hit state replacing
   request-plus-wait saves one cycle per hit: 8.47%/3.62%. Prefetching the cache
   read into a preceding state could remove both states: 16.93%/7.23% ceilings.
   Validate full tags, descriptor changes, abort/reset, SRAM DMA lifetimes,
   channel boundaries and partial groups. Existing storage is available, but
   steering it still adds control logic. This overlaps experiment 1.

4. **Reuse each scalar activation input word across its eight bytes.**
   ReLU currently reloads an entire 64-bit word for each scalar byte. There are
   63,000/202,608 redundant reads. Bypassing both request and response states
   removes 126,000/405,216 cycles, or 6.22/20.01 ms at the current clock:
   4.96%/5.50% ceilings. Exact in-place operation can retain the original word;
   arbitrary partial overlaps need a proved implementation or a fallback.
   This is a small experiment and an alternative baseline for experiment 5.

5. **Replace separate activation passes with exact LUT epilogues.** Each
   frozen INT8 activation maps 256 possible input codes to 256 output codes.
   Preserve the producer's original requantization, then apply a table of the
   original activation's complete clipping, requantization and saturation.
   A one-cycle table lookup instead of the current five-cycle standalone pass
   suggests about 288,000/926,208 saved cycles, 11.34%/12.58% of total time,
   before setup/control and contention changes. The table is only 256 bytes
   per distinct activation mapping and can use a synchronous BSRAM port.
   Exhaustively compare all 256 table entries with the independent oracle.
   There are no identity quantizers among the current activation descriptors;
   a bare clamp is incorrect. This subsumes most of experiment 4's benefit.

6. **Separate physical channel stride from logical tensor shape.** KWS uses
   125-byte planes, causing 223,232 split-word steps in 262,144 pointwise MAC
   steps. A physical stride of 128 avoids those splits while preserving the
   logical 25×5 geometry. It adds only 192 bytes per full 64-channel activation.
   Producers should emit the padded layout directly; standalone repacking can
   consume the saved time. Removing every extra split read has a ceiling of
   17.58% KWS/7.23% VWW total cycles. VWW needs layer-specific decisions: 36→40
   bytes costs 11.1% space, while 9→16 costs 77.8%. Compiler descriptors,
   scratchpad sizing and the independent verifier must distinguish logical
   geometry from physical strides. This overlaps caching in experiment 3.

7. **Use spatial SIMD for depthwise convolution.** Reuse the eight multipliers
   across neighboring output pixels instead of reducing one window at a time.
   If depthwise execution became twice as fast, the structural opportunity is
   about 9% KWS/8% VWW total latency reduction. This is a larger redesign:
   row/word alignment, padding, stride two, tails, per-channel parameters and
   overflow boundaries all need coverage. Avoid duplicating an entire engine
   when routing is already nearly full.

8. **Keep more intermediate activations in SRAM.** Every full KWS compute
   stage fits within 21,248 bytes today; with padded pointwise planes the
   largest estimated working set is about 21,632 bytes. A lifetime allocator
   could keep the activation chain on chip and avoid 176,128 bytes of
   intermediate activation traffic. Many later VWW stages individually fit
   within 27,776 bytes, while early stages require spatial tiling. Pure DMA
   savings alone are bounded by the relatively small time outside engine
   execution (6.08% KWS/9.77% VWW if engine behavior stays fixed). Lost prefetch
   overlap and changed arbitration matter. The larger opportunity is to pair
   residency with aligned layouts and exact activation epilogues.

9. **Recover logic through statically proved specialization.** Investigate
   narrower accumulators and internal address/geometry counters while keeping
   descriptor validation and overflow behavior explicit. For frozen weights,
   use bounds covering every legal INT8 input and every partial sum, including
   folded bias; dataset-observed ranges do not suffice. Move selected small
   distributed memories into spare BSRAM where its read latency can be hidden.
   Quantify recovered CLS and Fmax before deciding whether 16 MACs or another
   pipeline stage is feasible. No area or speedup number is assigned yet.

10. **Choose the eight-lane mapping per layer.** Compare 8 pixels × 1 output
    channel with 4 pixels × 2 channels and channel-reduction modes. Small
    spatial planes, tails and alignment may favor different mappings. This
    needs new weight organization, buffering and an executable cost model;
    an eightfold SIMD label does not imply eight useful results every cycle.
    No defensible whole-model gain is available before prototyping.

11. **Remove repeated parameter-check overhead.** Already validated unchanged
    channel parameters still pass through `P_REQ` and `P_CHECK`. A fast path
    has an estimated ceiling of only about 3.4% KWS/3.3% VWW. It can lengthen
    the output-advance path, so it ranks below the input-delivery changes.

12. **Change model precision or structure only in a separate branch.** INT4
    weights, mixed precision, structured channel pruning and distillation may
    reduce storage and arithmetic, but require new quality measurements.
    INT4 does not automatically double Gowin DSP throughput, and unstructured
    sparsity does not save cycles in the present dense engine. First prove an
    exact Gowin-specific packed multiply and route its correction logic;
    then evaluate the quality/latency frontier. First and last layers can
    remain INT8 if a measured mixed-precision search favors that choice.

## What performance to target

The next engineering target should be approximately **1.3× overall speedup**
from the current candidate, using exact INT8 transformations. This means about
96.5 ms KWS and 279.7 ms VWW. It is a target for the combined experiments,
not the sum of their individual upper bounds.

A more ambitious architecture target is **1.5–2×** the current throughput:
approximately 83.6–62.7 ms KWS and 242.4–181.8 ms VWW. Reaching it likely needs
both better data delivery and either a higher clock or another substantial
operator optimization. A further 3× improvement is a stretch hypothesis, not
an evidence-backed forecast at current resource occupancy.

For clock scale alone, with unchanged cycle counts, 27 MHz would give
94.09/272.72 ms and 32 MHz would give 79.38/230.11 ms. These are arithmetic
scenarios. New pipeline bubbles, SDRAM timing, refresh, UART divisors and route
closure must be included; the current image cannot simply run at those clocks.

The host path is a separate large opportunity. In the short timing screen,
VWW's median input upload alone was 2.595 seconds; complete input/run/readback
was 2.979 seconds, versus 0.364 seconds of FPGA execution. Batch larger UART
transactions or pipeline transfers before considering an external SPI host
adapter or native camera input. Any protocol change must be measured on the
actual USB bridge/host. It improves application throughput and experiment
turnaround, not the device-only compute figure. FPGA-only replay of resident
inputs is useful for timing/stability, but cannot replace full unique-sample
accuracy evaluation.

## Order and acceptance

Start with weight-selection registration, the next-input request during MAC,
and scalar word reuse as separately switchable variants. Then add pointwise
word caching and compare it with a padded-layout implementation. Explore exact
activation LUT fusion next. Only then commit to wider lanes, general spatial
fusion, or reduced precision.

For each change, predict removed states/traffic before measurement, validate
corner cases and complete-model tensors in RTL, route at the same clock, and
use a short paired physical comparison. Keep a candidate if it improves both
workloads or has a clearly explained useful tradeoff. A practical trigger for
further testing is ≥3% geometric-mean improvement with no unexplained >1%
regression, exact outputs and passing timing. Small timing-margin passes should
prompt margin work rather than immediate overclocking. Use separate evidence
for ablations and combined candidates. Defer long accuracy runs until the
candidate stabilizes; final acceptance still needs full accuracy and repeated
programming-session checks.

The most credible research hypothesis is a compiler that jointly selects
physical layout, SIMD mapping, exact activation representation and retained
tiles under a routed resource/timing constraint. A simple retention policy
already nearly matches the existing search, so the executable design space
must become richer before claiming a new scheduling contribution. Evaluate
held-out layer shapes/models and competitive published-policy adaptations;
the individual tricks below are established techniques, not novelty by
themselves.

## Primary-source context

- [Gowin DSP user guide](https://www.gowinsemi.com/upload/database_doc/8/document/5a0e570a988c0.pdf)
  documents multiplier configurations and register/bypass stages. Its DSP
  structure differs from the Xilinx structure in many packing papers.
- [FINN folding constraints](https://finn.readthedocs.io/en/latest/reference/folding-constraints.html)
  document per-layer parallelism choices and their shape constraints.
- [A2Q+, ICML 2024](https://proceedings.mlr.press/v235/colbert24a.html)
  studies accumulator-width constraints and overflow avoidance. It motivates
  width analysis, not a transferable area/speedup promise for this design.
- [DeFiNES, HPCA 2023](https://arxiv.org/abs/2212.05344)
  models fusion schedules, all memory levels and computation/copy costs. A
  credible comparison must include those costs and an executable adaptation.
- [DSP-Packing, FPL 2022](https://arxiv.org/abs/2203.11028)
  distinguishes exact correction from approximate overpacking on Xilinx DSPs.
  Those packing factors must not be assumed for this Gowin device.
