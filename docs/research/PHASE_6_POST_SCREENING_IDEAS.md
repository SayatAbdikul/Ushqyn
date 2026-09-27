# Further optimization after the combined short screen

The subsequent implementations and measured short-screen outcomes are recorded
in [Phase 6 follow-on optimization results](PHASE_6_FOLLOWON_OPTIMIZATIONS.md).
This document preserves the pre-experiment hypotheses and targets.

Analysis date: 2026-09-27. This document proposes the next experiments; it does
not change RTL, compiler behavior, model weights or the board. It supersedes
the performance targets in `PHASE_6_NEXT_OPTIMIZATIONS.md`, which used the older
direct-writeback baseline. The selected image still needs full qualification.

## Current reference point

Source: `work/phase6/experiments-v1/quick-chain/records.jsonl`, timed medians.
The image is `all-exact`, with chain retention, eight INT8 lanes at 20.25 MHz.

| Measurement | KWS | VWW |
| --- | ---: | ---: |
| FPGA execution | 68.840 ms | 250.975 ms |
| Device-only throughput | 14.526/s | 3.984 FPS |
| Elapsed cycles | 1,394,008 | 5,082,251 |
| Engine cycles | 1,354,866 | 4,318,084 |
| DMA cycles | 38,561 | 761,538 |
| DMA fraction of elapsed | 2.77% | 14.98% |
| Input upload | 0.0481 s | 2.5918 s |
| Host upload/run/readback wall time | 0.1381 s | 2.8618 s |

The chain uses serialized DMA. Perfect hiding of its present DMA, with engine
behavior unchanged, would cap throughput improvement at approximately 1.0285x
KWS and 1.1763x VWW. Traffic elimination and DMA overlap address overlapping
costs; their gains must not be added independently.

The selected route uses 10,022/10,368 CLS, 36/46 BSRAM and 14.5/24 DSP
equivalents. Only 346 CLS sites remain. Additional DSP capacity does not imply
that a wider engine and its steering logic will fit. Routed Fmax is 21.714 MHz.

## Highest-value new opportunities

### 1. Compile out exactly constant VWW filters

The current VWW weights contain all-zero output filters accounting for
**3,131,136 of 7,489,664 dense MACs, or 41.806%**. This is substantially more
promising than adding multipliers to execute those products. KWS has no
all-zero output filters and only about 0.8% zero weights.

These VWW zeros are present in the upstream INT8 model too: its fourteen
standard convolution weight tensors match the compiled weights byte for byte
after OHWI-to-OIHW conversion. The pinned upstream file is
`work/upstream/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite`,
SHA256 `597a384c8c2c8a1276f04702f25013b7838f2f814f1ca7c174d295b73e3d6b7b`.
Thus this opportunity does not require new pruning or quantization.

An all-zero filter produces a constant determined by its bias, multiplier,
shift and output zero point. **The correct replacement is not necessarily a
zero-filled tensor.** First prototype compiler-generated constant tiles and
existing DMA operations, preserving output layout. Then assess a compact FILL
operation if transfers erase the benefit. Retain ordinary computation for
nonzero filters; account for extra descriptors and fragmented channel ranges.
Constant propagation into later layers is a subsequent experiment, requiring
proof for padding, both quantization boundaries and accumulator behavior.

41.806% is removable dense arithmetic, not a latency prediction. Input reads,
parameter handling, output writes and other operators remain. Use this model
alongside dense cases when evaluating the research contribution; report actual
executed work as well as dense-equivalent MAC counts.

### 2. Simplify exact requantization and repair the next timing path

The selected route's longest path is `mult[15]` to `requant.result[6]`:
46.018 ns data delay and 40 logic levels. `rtl/v2/requantizer.sv` currently
combines a wide multiplication, absolute value, rounding, variable shift,
negation, zero-point addition and saturation.

Prototype signed arithmetic shift with guard/sticky rounding and early
saturation, narrowing the final zero-point addition where proved safe. For
positive shift, arithmetic-floor quotient plus a nonnegative remainder can
implement the same ties-away-from-zero result: increment for remainder at
least half for nonnegative products, strictly greater than half for negative
products. Handle shift zero separately. Prove the complete signed operation,
including extreme values and saturation, against the current implementation.

The next independent path is 41.534 ns, from input zero point through a DSP,
overflow detection, the prefetch request and arbiter to `row_origin` enable.
Decouple memory-request timing from same-cycle overflow detection. Any
speculative SRAM read must have a legal address and must be discarded safely
on error or abort.

These changes could improve timing without adding a cycle per output. A naive
extra pipeline stage can lose much of its benefit in the current serialized
output controller. If pipelining is required, make quantization/writeback a
stream with a small queue and account for its storage and control.

### 3. Reuse depthwise windows, then remove remaining pointwise bubbles

The selected spatial-depthwise path fills the word cache but cannot hit it:
`pw_cache_hit` requires `spatial_pw`. Adjacent horizontal taps consequently
fetch overlapping words repeatedly. Reuse a roughly ten-byte input row span
for the three horizontal taps of eight stride-one outputs, or extend the
existing tagged cache with correct depthwise indexing. Reuse across rows is
a later extension. Overlap tap address/mask preparation with the preceding
MAC when possible; removing `DW_PREP` saves at most nine cycles per 3x3 output
group, before accounting for stalls and other changes.

Pointwise has a smaller missed fast path: a prefetched noncrossing cache hit
still traverses `PW_X_REQ`. Directly use the cached word when the next weight
is ready. This removes one cycle per qualifying hit. Count those hits in
native simulation before assigning a percentage.

Both changes reuse the eight existing multipliers. Include KWS width-five
rows, padding, cross-word windows, tails and backpressure in short RTL checks.

### 4. Preserve VWW inputs across sibling output-channel tiles

The current chain retains the immediately preceding output. Several tiled
operators reload the same input after an activation's 16-byte parameter write
clobbers its beginning. Moving that parameter into unused space and extending
input lifetimes can avoid **144,000 transfer bytes**:

| Layer | Repeated input bytes |
| --- | ---: |
| 1 | 82,944 |
| 5 | 36,864 |
| 13 | 18,432 |
| 49 | 1,152 |
| 53 | 4,608 |

The first convolution has 288 unused SRAM bytes, enough for its activation
parameter. The proposed traffic reduction is 19.57% of the current 735,778
transferred bytes. Scaling present DMA cycles proportionally suggests about
7.36 ms, or 2.93% total VWW latency, as a rough hypothesis. Actual transfer
lengths, setup costs and placement require replay and measurement. KWS does
not have this duplicate-input opportunity. This experiment may need only
compiler changes.

Then prefetch next-stage weights/parameters into safe unused SRAM tails while
retaining activations. The present single live-interval contract limits which
holes can be used. Start outside that interval before expanding the contract.

### 5. Process activation LUT outputs in words

The current standalone LUT path reads a 64-bit word but writes individual
bytes. Parallel lookups and one masked word write could remove about seven
write cycles per eight outputs: approximately **3.11 ms KWS /9.97 ms VWW**,
or **4.5% /4.0% of current elapsed time**, before implementation overhead.
Four dual-port table copies are one possible implementation; actual RAM mode,
lookup latency, extra CLS and byte-alignment handling need synthesis.

True producer-to-activation fusion remains useful, but preserve producer INT8
requantization followed by the exact activation mapping. Present standalone
activation passes have a minimum structural cost around 96,912 cycles KWS and
316,686 VWW (about 7.0% /6.2% of elapsed time); fusion cannot claim removal of
all that work if it adds serialized lookups to the producer.

Caching the most recent activation table alone is low priority: ten repeated
VWW builds save only 7,680 cycles, 0.379 ms or 0.15% overall. KWS has nine
distinct activation quantizers.

### 6. Recover logic before expanding the architecture

Narrow geometry and bounds arithmetic only after validating discarded upper
bits; retain rejection of malformed descriptors. Alternatively share a
multiplier across descriptor-setup states. Current wide geometry arithmetic
uses resources despite operating only at setup. Move suitable distributed
memories into spare BSRAM when the extra read latency can be hidden.

Then compare padded physical channel strides and four pixels by two output
channels. Existing software prototypes establish arithmetic correctness, not
routed speed. Their read-count savings overlap the selected word cache. An
executable layer cost model should choose between layouts and lane mappings,
including tails, weights, SRAM occupancy and measured control overhead.

There is also a model-specialization opportunity: a conservative KWS bound
`abs(corrected_bias) + 128 * sum(abs(weight))` reaches at most 708,324 across
its Conv/Gemm channels, fitting signed 21-bit accumulation for every legal
INT8 input and every partial sum. This does not authorize narrowing the shared
engine globally: VWW has much larger bounds. Reassess VWW after constant
propagation and compare specialized hardware with the cost of a wide fallback.

INT4, structured pruning, additional MAC lanes and higher clocks beyond the
repaired timing limit remain separate, larger experiments. INT4 needs quality
evaluation and Gowin-specific mapping; reducing bit width alone does not
double the throughput of the existing 9x9 multiplier mapping.

## Application throughput is a separate optimization target

VWW currently spends 2.592 seconds uploading a 27,648-byte input and 0.251
seconds executing it. At 750 kbaud with 8N1, payload-only wire time is at least
0.36864 seconds, before headers, acknowledgements and other work. There is a
large protocol-efficiency gap.

Build a proper BRAM-backed receive queue or larger CRC-checked frames, with
explicit credits/backpressure, then burst complete words into memory. The
previous guarded batching trial failed; sending batches to the unchanged
parser is not an accepted optimization. Packet counters, response correlation
and bounded recovery must be retained. This work improves host-fed throughput
and evaluation time; it does not improve the device-only latency above.

## Targets and short-screen sequence

A useful next target is **1.2-1.5x throughput from this new baseline**, meaning
57.4-45.9 ms KWS and 209.1-167.3 ms VWW. This is an engineering target, not a
forecast or the sum of estimated opportunities. VWW constant-filter removal
could alter its attainable range; measure its complete schedule first.

Pure clock arithmetic with unchanged cycles gives 61.96/225.88 ms at22.5 MHz
and 58.08/211.76 ms at24 MHz. The current selected image is not qualified at
those clocks. Retiming, PLL choice, memory timing, refresh and routing must be
included. A further 2x overall gain remains a stretch target.

Recommended sequence:

1. Model constant-filter elimination and input retention independently in the
   compiler; predict full schedule cost and replay every intermediate tensor.
2. Screen exact requantizer simplification and request-path decoupling at the
   current clock, then assess additional timing margin.
3. Add DW input reuse and pointwise cache-hit bypass as isolated variants.
4. Compare word-wise LUT processing with pipelined exact epilogue fusion.
5. Recover area, then evaluate aligned layouts, alternative lane mappings and
   wider compute only when they fit and the model predicts a benefit.

Each candidate should pass focused edge cases, complete-model RTL checks,
route constraints and a short paired board screen. Keep the existing >=3%
geometric-mean improvement gate, or document a useful workload-specific
tradeoff. Defer full accuracy and long stability until the implementation is
stable. No additional board run was started for this analysis.

## Evidence

- `evidence/phase6/post-screening-analysis.json` records model source hashes,
  per-layer zero-filter counts and MAC totals, and current physical medians.
- `work/phase6/experiments-v1/all-exact/route/report.json` and the corresponding
  routed timing report supply resource and critical-path evidence.
- `work/phase6/experiments-v1/chain/fixtures/*-pinned-chain-timed/schedule.json`
  supplies descriptors, tensor lifetimes and transfers.
- `work/phase6/experiments-v1/algorithms/report.json` contains the existing
  layout, lane-mapping and reduced-precision arithmetic prototypes.

## Primary-source context and research direction

The [Gowin DSP guide](https://cdn.gowinsemi.com.cn/UG287E.pdf) documents the
registered multiplier and ALU configurations to investigate after area
recovery. [DSP-Packing](https://arxiv.org/abs/2203.11028) is useful context for
packed low-precision arithmetic, but its mapping results cannot be transferred
directly to Gowin. [A2Q+](https://proceedings.mlr.press/v235/colbert24a.html)
provides a later accumulator-aware training direction if exact specialization
is insufficient. [DeFiNES](https://arxiv.org/abs/2212.05344) is a relevant
baseline for fusion, scheduling and memory-cost modeling.

The strongest research direction is a verified compiler jointly selecting
constant elimination, accumulator widths, layouts and SRAM lifetimes under
actual routed FPGA constraints. Constant folding and the other individual
techniques are established. A paper needs a defensible combined method,
held-out audio and vision cases, competitive baselines and ablations; faster
execution of these two pinned models alone does not establish novelty or SOTA.
