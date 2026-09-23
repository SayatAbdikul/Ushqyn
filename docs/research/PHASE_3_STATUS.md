# Phase 3 implementation record — 2026-09-23

The same board hierarchy now runs a complete eight-layer on-chip SmallCNN:
Conv–ReLU–MaxPool–Conv–ReLU–MaxPool–Reshape–Gemm. It uses the existing
eight-lane signed-INT8 MAC/requantization path, one synchronous 32-KiB SRAM
port, a register window gather, an eight-word tagged activation cache and a
64-byte filter cache. The target ID is
**8195**; the archived phase-2 candidate reports 8194 and must not be used
with this image. The numerical contract and 64-byte descriptor format remain
version 2.

| Roadmap item | Current result |
|---|---|
| M01 | Complete for this one physical mode: eight byte lanes, one 64-bit synchronous port, 32 KiB addressable SRAM, alignment and reserved descriptor bytes. Independent allocation verification checks live overlap, bounds and peak bytes. Its 16-block count matches the new Gowin route. Other width/depth modes remain later research work. |
| M02 | Complete for the on-chip subset: live tensor regions reuse addresses, while all loaded weights/parameters persist across repeated RUN commands. This rule was caught by the 1,000-job test. There is no active duplicate full tensor array. |
| C01 | Functional ordinary-Conv/FC subset passes; performance work remains. One MAC array accumulates eight reduction terms into a bounded INT32 output-stationary accumulator. 1×1, rectangular, 3×3, asymmetric padding, stride 1/2, signed extremes, channel and reduction tails pass with randomized SRAM backpressure. Small filters are fetched once per output channel. Activation broadcast and larger-filter reuse are not implemented. |
| C02 | Functional window/pooling subset passes; streaming work remains. Synchronous scratchpad reads gather each eight-element window with valid/ready, zero-point padding and odd/asymmetric boundaries. An eight-word tagged buffer reuses SRAM words across adjacent windows. MaxPool excludes padding and requantizes at the declared boundary. There is no dedicated multi-row line buffer; gathering remains serialized. |
| C03 | Simulation and routed build complete; physical gate open. Full MNIST float/INT8 quality and 1,000 exact board-system RTL jobs are archived. Real-board programming, full readback, 1,000 physical runs and measured latency/power remain pending. |

**G3 remains open** because C01/C02 performance features and the real-board
part of C03 are unfinished; G2 also remains open for the same physical access.
The routed result is a candidate, not a measured board result or SOTA claim.

## Reproducible results

The tracked `compiler/small_cnn_weights.pth` is exported afresh to ONNX.
Calibration uses 64 MNIST training images; evaluation uses the separate full
10,000-image test set. The float source predicts **9,647/10,000 (96.47%)**;
the independent static-INT8 oracle predicts **9,640/10,000 (96.40%)**.
The program uses **9,568 of 32,768 SRAM bytes** (12,014 logical segment bytes,
8,208 peak live bytes) and performs **61,184 useful MACs** per inference. The
[quality record](evidence/phase3/quality-summary.json)
pins weights, model, calibration IDs and dataset hashes.

Native Verilator executes the exact board `v2_system` including protocol and
synchronous inferred SRAM. **1,000/1,000** successive image changes produce
identical final INT8 outputs to the independent oracle; one input is also
checked at every one of the eight layer boundaries by stopping after that
layer. All jobs reconcile elapsed = compute + wait + control cycles and report
61,184 useful MACs. Each job records **182,535 simulated core cycles**:
14,370 compute, 64,914 wait, 103,251 control, and 225,936 physical SRAM read
bytes. The tagged window cache cuts cycles by 34.5% and read traffic by 63.0%
against the otherwise identical uncached controller. This is about 6.76 ms
only if the board actually runs at 27 MHz;
it is **not measured board latency**. UART model loading and per-image host
traffic are excluded from that core-cycle conversion. See the
[per-job raw counters](evidence/phase3/smallcnn-1000-rtl.json).

Strict Verilator lint, 121 compiler tests, randomized-latency engine tests,
system commands and board-top UART tests pass. One RTL system also runs
MLP→SmallCNN→MLP by replacing only the model image, with exact outputs and
readback at each switch. The [evidence directory](evidence/phase3/)
contains test XML, source hashes, SRAM layout, quality and route summaries.

## Routed candidate

Gowin Education V1.9.11.03, `GW2AR-LV18QN88C8/I7` revision C, 27-MHz constraint:

| Metric | Post-route value |
|---|---:|
| Logic | 7,525 / 20,736 |
| Registers | 2,341 / 15,750 |
| BSRAM | 16 / 46 |
| SSRAM RAM16 | 37 used for the two small caches |
| DSP equivalent | 19.75 / 24 |
| Routed Fmax | 27.473 MHz |
| Worst setup slack | +0.638 ns |
| Setup total negative slack | 0 |

The [candidate bitstream](../../hardware/releases/phase3/tinyml_v3.fs) has
SHA256 `0ea86b65fd31a5d253c1c0e40b7b49b8b26404bac300e1a02a80a564d50f133c`.
Gowin still reports generic routing for the board clock (`PR1014`). The
post-route power estimate is not a measured energy result. The modest timing
margin and clock warning both require physical bring-up before a release claim.

## Remaining work

Add an actual synchronous row/line-buffer path and activation reuse, then
re-profile end-to-end memory traffic and the port-service lower bound. Test
larger Conv reductions and memory budgets, preserve one shared MAC array,
and reroute after every RTL change. With an identified Tang Nano 20K, program
the candidate, verify CAPS target 8195 and full SRAM readback, and run
`tools/phase2/verify_board.py --jobs 1000` on the SmallCNN fixture. Archive
board revision, programmer/serial identity, exact outputs, counters, clock
and voltage/current traces. Rebuild if the confirmed board pinout differs.
