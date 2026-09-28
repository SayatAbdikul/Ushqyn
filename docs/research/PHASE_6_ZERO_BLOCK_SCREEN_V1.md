# Zero-point block sideband screen

Date: 2026-09-28. This is a bounded screen of producer-generated zero-block
metadata for the exact selected 27 MHz KWS and VWW schedules. It is **not** a
new routed accelerator or a measured speedup. The baseline board medians are
1,201,854 cycles KWS and 2,441,608 cycles VWW, with exact ten-case short
screens on the Tang Nano 20K.

The census decodes every executed pointwise descriptor from the frozen
three-pair schedule, including VWW strips and compacted output channels.
It applies pinned and seeded-stress values from the independent INT8 graph
oracle. A group is eight spatial values from one input channel; partial
groups are in the iteration denominator but are not marked skippable.

| Pinned workload | Executed pointwise iterations | Logical all-zero-point, nonzero-weight iterations | Certified by 64-bit physical-word tags | Pointwise descriptor cycles / engine cycles |
| --- | ---: | ---: | ---: | ---: |
| KWS | 262,144 | 40,535 (15.46%) | 31,726 (12.10%) | 643,484 / 1,170,215 (55.0%) |
| VWW | 333,798 | 40,763 (12.21%) | 38,120 (11.42%) | 908,698 / 2,133,486 (42.6%) |

Stress values give 41,421 logical and 31,854 physical-tag eligible KWS
iterations, and 40,192 logical and 37,231 physical-tag eligible VWW
iterations. The selected pointwise input zero point is **−128** in every
layer. No input has a complete eight-value group equal to raw integer zero.
Across all executed pointwise descriptors on both inputs, there are **zero
aligned full eight-input-channel zero-point runs** that would let the engine
skip an entire eight-weight word. The largest contiguous zero-channel run in
the dominant KWS layer 17 is five; in VWW layer 21 it is three.

The physical-word producer tag is attractive because it could be generated
while an aligned SRAM word is written. It is conservative when the consumer's
eight-pixel group spans two physical words: both tags must be set. This
removes 8,809 of 40,535 otherwise eligible KWS iterations and 2,643 of
40,763 VWW iterations. The source regions require at most 1,152 sideband
bits per simultaneously live producer tensor in this schedule, although a
generic scratchpad-wide tag array would need 4,096 bits.

## Exact arithmetic and cycle limits

The current pointwise engine starts from a corrected bias and multiplies raw
INT8 bytes. For input zero point `z`, its accumulator is
`b_corrected + sum(x*w)`. A skipped all-`z` channel still needs to add
`z*w` to each output accumulator lane. Simply omitting the MAC changes the
answer; shifting to a centered product also changes the intermediate
overflow contract unless separately proved. The software candidate applies
the exact compensation for every pointwise pixel and output channel in both
models, on pinned and stress data, and matches the independent centered
integer oracle. The physical-word conservative mask is checked separately.
An independent arithmetic probe exhaustively checked all **65,536** INT8
zero-point/weight pairs at seven accumulator boundary values; it also found
a 16-channel case where centered-product replacement changes intermediate
overflow behavior.

An impossible free-lookup, free-compensation upper bound of **two cycles per
physical-tag eligible iteration** would remove 63,452 KWS and 76,240 VWW
cycles. Applied mechanically to the latest board medians, that corresponds
to a **1.04392× dual-workload geometric-mean throughput ceiling**. This is
an opportunity bound, not a candidate estimate. Charging one cycle for a
metadata lookup on every iteration already costs 262,144 KWS and 333,798
VWW cycles; even the optimistic two-cycle benefit breaks even only below
0.309 and 0.244 lookup cycles per iteration, respectively, before producer
metadata costs.

An instrumented copy of the unchanged native harness records each engine
descriptor and every pointwise input-fetch state, with outputs checked under
stall seeds 0 and 6063. Its per-descriptor cycle totals exactly equal the
normal native engine-cycle counters. For the physically certified pinned
iterations, the native engine spends **33,322 KWS** and **40,028 VWW** cycles
in explicit `PW_X_REQ/WAIT/X2` input-fetch states, just 1.05 per eligible
iteration. These are native state counts, not physical-board latency bounds;
the SDRAM stall pattern on the board can differ. If exact compensation
occupies one state, eliminating all these input-fetch states suggests about
**1.0226×** geometric-mean throughput when mechanically translated to the
latest board cycle medians. This is a sensitivity calculation, not a
physical performance bound. A faster design must also absorb the
compensation update into an existing state and suppress speculative input
prefetch. The optimistic two-cycle bound could clear the **1.03×
dual-workload gate** only if correction and metadata together cost less
than about **0.615 cycles per certified event** under a common-overhead
model; the stress figure is 0.606.

The first fused-next-channel candidate required a full aligned next input,
a nonzero next weight within the current eight-weight word, and a next input
cache hit. In the exact native trace, the geometric conditions identify
**0 KWS** and **27,364 VWW** opportunities, but the required cache hit
reduces VWW coverage to **0**. A separate producer sideband with earlier
prefetch would be needed. It must also invalidate tags when DMA or another
engine overwrites SRAM and preserve a dense fallback for arbitrary
descriptors. Those changes have not passed RTL, route, or board tests.

Evidence is generated by `tools/phase6/novelty_zero_block.py`,
`tools/phase6/novelty_zero_block_profile.py`, and the independent
`tools/phase6/novelty_zero_bypass.py`, with JSON reports under
`work/phase6/novelty-zero-block-v1/` and
`work/phase6/novelty-zero-bypass-v1/`. The native profiler links the
unmodified Verilated design archive and fixture hashes, then checks its
event counts against the independent census. The tested direct cache-hit
fusion is a **screened no-go** because it has zero matching events; the
producer sideband remains an unimplemented hypothesis. A hardware speed
claim requires a routed, full-model exact native candidate and a matched
short board measurement.
