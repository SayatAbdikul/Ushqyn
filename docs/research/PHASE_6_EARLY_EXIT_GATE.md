# Certified early-exit screen

The read-only profitability gate in `tools/phase6/early_exit_gate.py` tests
whether a convolution output can stop after 25%, 50%, or 75% of its products.
For each partial sum, it bounds all possible remaining centered INT8 products
using the remaining weights and the complete legal INT8 input range. It counts
an exit only when the entire resulting interval remains in INT32 and both
endpoints quantize to the same INT8 code as the independent integer oracle.
All-zero output filters already removed by the selected compiler are excluded.
The gate evaluates the pinned input and one seeded full-range stress input for
both KWS and VWW; it does not access the FPGA.

| Workload | Sample | Certified skipped MAC slots | Remaining dense MAC slots | Fraction |
| --- | --- | ---: | ---: | ---: |
| KWS | Pinned | 390 | 2,656,000 | 0.0147% |
| KWS | Stress | 348 | 2,656,000 | 0.0131% |
| VWW | Pinned | 1,514 | 4,358,016 | 0.0347% |
| VWW | Stress | 1,574 | 4,358,016 | 0.0361% |

These are arithmetic-slot savings before hardware comparators, control,
divergent SIMD lanes, and state overhead. A controller for this exact gate
would have a negligible whole-model ceiling, so no RTL or board expansion is
justified. The result is specific to the three tested checkpoints and this
conservative full-range bound; it does not rule out another proof or model.

Machine-readable evidence: `work/phase6/early-exit-gate-v1/report.json`.
