# Exact internal channel grouping screen

The existing VWW constant-filter lowering identified 971 all-zero output
filters, but they were interleaved with live filters. Each small alternating
run produced another command, descriptor load, or aligned constant fill.
This experiment permutes each internal dense Conv output to put live channels
first and constants last. The same mapping follows Relu/depthwise/pooling
outputs; the next Conv's input weight columns and the final Gemm columns are
permuted correspondingly. All quantizers and the public input/output order
stay unchanged. The compiler also omits operand DMA loads on a tile whose
outputs are all constants.

The isolated implementation is
[`tools/phase6/followup_graph.py`](../../tools/phase6/followup_graph.py).
It does not modify the production compiler. Its output is in
`work/phase6/followup_graph/`. `prepare` compared every transformed layer to
the corresponding original independent integer-oracle layer on pinned and
seeded stress inputs, then replayed every generated command list. The selected
native engine passed 12/12 full-model runs, including every intermediate
tensor for the snapshot fixtures and two memory-stall seeds. No board run is
claimed here.

| Native timed schedule | KWS fixed cycles | VWW fixed cycles | VWW stalled cycles | VWW commands |
| --- | ---: | ---: | ---: | ---: |
| Constant+sibling reference | 1,313,903 | 3,164,674 | 3,307,037 | 1,671 |
| Grouped, redundant loads removed | 1,313,903 | 3,083,068 | 3,205,039 | 639 |

VWW fixed/stalled cycle reductions are 2.58%/3.08%. In the fixed native run,
engine cycles fell 2,918,529 to 2,868,379 and DMA cycles fell 239,460 to
212,132. These counters explain why removing 61.8% of commands gives only a
modest total latency gain: most cycles remain inside the engine. The
constant-filter lowering still requires twelve output fills and nine live
Conv runs in its affected tiles. KWS has no all-zero
filters and its schedule is unchanged.

The separately tested `--fold` variant is in
`work/phase6/followup_graph_fold/`. It propagates known INT8 channel values
through depthwise and activation operators and folds uniform constants into
later Conv/Gemm biases only after proving the new INT32 accumulator bounds.
It passed the same 12/12 native runs with identical cycle counts. Every
weight attached to a symbolically known input channel in the eight following
VWW pointwise Convs is already zero, including the two known spatial patterns
that are not uniform. Thus there are **zero nonzero pointwise weight values**
for scalar constant folding to remove. The final Gemm does contain 455 such
nonzero weights; folding them is exact, but its descriptor still traverses
the same input count. Any latency gain from those zero pointwise columns
requires an explicit sparse/packed lowering and engine schedule change.

This corrects the earlier 612,000-MAC opportunity estimate: counting
input-independent channels as if their associated weights are useful
multiplications overstates the available algebraic work. Zero weights still
consume cycles in the current dense loop, so a sparse representation remains
a distinct candidate. Its saved cycles must be measured against packing and
index overhead.

Evidence: `work/phase6/followup_graph/fixtures.json`,
`work/phase6/followup_graph/native-combined-spec-scalar-v1/report.json`, and
the corresponding two files under `work/phase6/followup_graph_fold/`.
