# Phase 6 direct writeback — 2026-09-27

**The physical screen passed:** 48/48 inferences, zero output mismatches,
and lower measured latency on both workloads. Full G6 remains open.

The user requested pausing full VWW accuracy to prioritize further optimization.
The accuracy campaign stopped with 4,890 checked KWS samples and 212 checked
VWW samples. KWS classification accuracy is 4,514/4,890 (92.31%). The record
hashes and resume prefixes were independently verified after interruption.
Both datasets belong to the previous spatial image, whose complete source and
artifact identities remain unchanged. A later continuation reloads that image
and starts VWW at sample index 212; it must not mix outputs from a newer image.

## Change and causal prediction

`rtl/phase6/writeback_engine.sv` is an isolated derivative of the screened
spatial engine. The registered requantizer result directly drives the output
write. The engine consumes that result and advances its output indices only
when SRAM accepts the write. Backpressure holds the same byte, address and
strobe until acceptance. COPY uses its existing result register and advances
on the accepted write as well.

This removes two states per quantized output and one state per COPY output.
It preserves the original quantizer, overflow checks, descriptor ABI, model
data, command schedule, eight INT8 multipliers and 20.25 MHz operating clock.
No higher-clock result is claimed.

For the selected retention schedules, the exact uncontended reduction is
288,152 cycles for KWS and 926,724 cycles for VWW. Every serialized fixed-RAM
test reproduces that delta exactly. Overlapped fixed-RAM timing improves from
2,725,879 to 2,441,799 cycles on KWS (10.42% reduction) and from 7,789,276 to
6,885,071 cycles on VWW (11.61%). These are RTL simulation results; physical
SDRAM arbitration can change the savings.

## Verification

Three engine regression suites pass, including existing randomized GEMM,
convolution, pooling, cache, tail and overflow tests. The added suite checks
30 combinations of RELU/COPY/CLIP, short and unaligned output tails, and
in-place/disjoint buffers. It stalls every write for five cycles, checks exact
write counts and counter accounting, then tests abort/reset/clear while a
quantized output is pending and verifies a fresh run after cancellation.

The full-model matrix passes 24 RTL runs and 632 tensor comparisons: both
models, pinned and stress inputs, serialized and overlapped schedules, fixed
and seeded variable-latency external memory. Simulation uses the unchanged
sequencer, DMA, SRAM and independent integer-oracle fixtures.

Gowin fits the device with zero setup and hold violations:

| Resource | Used | Available |
| --- | ---: | ---: |
| Logic | 17,626 | 20,736 |
| Registers | 5,451 | 15,915 |
| CLS sites | 9,678 | 10,368 |
| BSRAM | 35 | 46 |
| DSP equivalents | 14.5 | 24 |

Routed Fmax is 20.334 MHz at the unchanged 20.25 MHz operating constraint.
The worst setup slack is 0.205 ns. This passes, but leaves narrow timing margin;
it is one route, not evidence of robust frequency improvement. The worst path
starts at the input-channel index and crosses weight selection/MAC logic into
the requantizer's mapped DSP input. Registering the selected broadcast weight
before multiplication is a specific follow-up to investigate without adding
MAC lanes. It has not been implemented or credited as a speedup here.

Two further opportunities remain unimplemented: reuse the already fetched
eight-byte word across adjacent scalar activation outputs, and investigate a
pipelined broadcast-weight selection path. The former must preserve overlapping
buffer semantics; the latter must be routed before making a frequency claim.
An inspection of every activation descriptor found zero identity quantizers
among 72,000 KWS and 231,552 VWW activation outputs. Replacing their exact
requantization with a bare clamp would therefore change the arithmetic.

## Physical screen

The campaign completed at 14:41:05 Almaty time. The independently reproduced
results use the same retention schedules, frozen INT8 models and clock:

| Model | Previous spatial image | Direct writeback | Latency reduction | Device-only throughput |
| --- | ---: | ---: | ---: | ---: |
| KWS | 139.573 ms | 125.447 ms | 10.12% | 7.97 inferences/s |
| VWW | 407.782 ms | 363.630 ms | 10.83% | 2.75 frames/s |

Values are medians of ten timed inferences after one warmup per model/image;
they exclude UART transfers, preprocessing and initial model loading. The
largest within-configuration range divided by the median is 0.0138%.
All 156 intermediate/final tensor comparisons in the four diagnostic runs
passed. The screen took 12 minutes 37 seconds including programming, complete
payload readback and diagnostic tensor transfers. The board was left idle with
the previous spatial image after its final timing group; no accuracy queue
was restarted.

The bounded screen first checks four complete diagnostic runs: pinned and
stress input for both models, with all supplied intermediate and final tensors.
It then compares the previous spatial image and the new writeback image under
the same retention schedule, ten timings plus one warmup per model/image.
This is 48 inferences in total. It stops on any mismatch, protocol failure,
timeout, incomplete sequence or inconsistent counters; there are no automatic
retries or queued full-dataset tests.

The source/image/fixture hashes, program logs, atomic report and fsynced
records are under `work/phase6/writeback-v1/`. The audit reproduced the summary,
verified every record hash and tensor identity, and checked that the paused
accuracy evidence and all 109 frozen Phase 5 files are unchanged.

- [Audited summary and archive hashes](evidence/phase6/writeback/summary.json)
- [Physical report](evidence/phase6/writeback/physical-report.json.gz)
- [Physical records](evidence/phase6/writeback/physical-records.jsonl.gz)
- [Full-model RTL results](evidence/phase6/writeback/native-report.json.gz)
- [Candidate bitstream](../../hardware/releases/phase6/writeback_candidate_750k.fs.gz)

The uncompressed candidate SHA256 is
`1170b51dd41c6677a171a1fa4cc2623ea2052c5939499b9efb89c11096d1b682`.

## Reproduction and limits

```sh
venv/bin/python3 tools/phase6/run_writeback.py engine
.venv/bin/python3 tools/phase6/run_writeback.py native
.venv/bin/python3 tools/phase6/run_writeback.py route
.venv/bin/python3 tools/phase6/screen_writeback.py
.venv/bin/python3 tools/phase6/screen_writeback.py --run
.venv/bin/python3 tools/phase6/audit_writeback.py --archive
```

The first four commands are boardless. The board runner refuses an existing
evidence directory; preserve prior results and select a fresh directory for a
new campaign. The audit requires a completed successful physical screen.

This is a conventional control-path optimization, not a new scheduling claim.
A short screen cannot establish complete accuracy on the new image, independent
programming-session repeatability, long-duration stability, energy efficiency,
SOTA or G6 closure. Existing KWS accuracy remains valid for its original image;
adopting the newer image requires its own final accuracy qualification.
