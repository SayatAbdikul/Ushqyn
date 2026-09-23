# Phase 3 implementation record — 2026-09-23

**Latest board-tested candidate:** the shared engine now has a four-row,
single-read synchronous activation buffer (eight tagged 64-bit words per row)
with lookahead reads and a 128-byte filter cache. The buffer maps to two
additional BSRAM blocks and the filter cache to SSRAM; the main scratchpad
remains one synchronous 32-KiB port. This source-matched target **8196**
passes 10,000/10,000 exact SmallCNN RTL jobs, all eight layer stops, the
MLP→SmallCNN→MLP switch test, and directed wide-row/81-term/261-term kernel
cases with randomized SRAM stalls. It routes at **27.376 MHz** against a
27-MHz constraint with **+0.509 ns** worst setup slack. The connected board
then passed 61 physical diagnostics, 1,000 exact jobs per stage in an
MLP → SmallCNN → MLP switch, and all 10,000 full-set SmallCNN jobs with zero
integer mismatches and 96.40% classification accuracy. Physical SmallCNN
counters are **164,165 cycles** and **152,456 SRAM read bytes** per job.
The [physical result](PHASE_3_LINEBUFFER_PHYSICAL.md),
[raw evidence](evidence/phase3-linebuffer-physical/summary.json),
[boardless candidate evidence](evidence/phase3-linebuffer/summary.json) and
[bitstream](../../hardware/releases/phase3-linebuffer/tinyml_v5_candidate.fs)
remain separate from the earlier physical baseline.

**Physical baseline:** the reset-corrected target **8196** has completed
1,000 exact SmallCNN jobs in an MLP → SmallCNN → MLP switch test, with full
SRAM readback, physical counters and all eight layer outputs checked on three
inputs. The separate 10,000-image board evaluation also passed: every raw INT8
output matched the independent reference, and 9,640 classifications were correct
(96.40%). Physical core cycles are fixed at 182,535; median host job time for
the full set was 185.850 ms, including UART transfers and status polling. See the
[physical record](PHYSICAL_BOARD_STATUS.md). The original target-8195 source,
simulation and route results below remain historical evidence.

The board hierarchy runs a complete eight-layer on-chip SmallCNN:
Conv–ReLU–MaxPool–Conv–ReLU–MaxPool–Reshape–Gemm. It uses the existing
eight-lane signed-INT8 MAC/requantization path, one synchronous 32-KiB SRAM
port and serialized window gather. The previous physical image has an
eight-word activation cache and a 64-byte filter cache; the latest candidate
has the four-row buffer and 128-byte filter cache above. The archived
phase-2/3 candidates reported older target IDs and remain historical. The
numerical contract and 64-byte descriptor format remain version 2.

| Roadmap item | Current result |
|---|---|
| M01 | Complete for this one physical mode: eight byte lanes, one 64-bit synchronous port, 32 KiB addressable SRAM, alignment and reserved descriptor bytes. Independent allocation verification checks live overlap, bounds and peak bytes. The scratchpad uses 16 BSRAM blocks; the candidate line buffer adds two. Other width/depth modes remain later research work. |
| M02 | Complete for the on-chip subset: live tensor regions reuse addresses, while all loaded weights/parameters persist across repeated RUN commands. This rule was caught by the 1,000-job test. There is no active duplicate full tensor array. |
| C01 | Functional ordinary-Conv/FC subset passes. One MAC array accumulates eight reduction terms into a bounded INT32 output-stationary accumulator. 1×1, rectangular, 3×3, asymmetric padding, stride 1/2, signed extremes, channel and reduction tails pass with randomized SRAM backpressure. The new 128-byte cache reuses up to 128 filter terms across spatial outputs; a directed 261-term case verifies the uncached multi-tile fallback. Cross-output-channel activation broadcast and external-memory partial-sum tiling remain. |
| C02 | A four-row, 256-byte synchronous tagged line buffer is implemented and inferred as two BSRAM blocks. Lookahead hides its read cycle on hits. Zero-point padding, odd/asymmetric boundaries, wide-row tag aliasing and pooling pass with backpressure. The byte-wise window gather still consumes substantial control cycles; a wider streaming datapath remains. |
| C03 | Both the earlier release and improved candidate pass the complete 10,000-image physical comparison, 1,000 switch-stage jobs, full readback and isolated layer checks. Full-set accuracy is 96.40% with zero integer mismatches. The improved image reduces physical cycles 10.064% and SRAM reads 32.522%; host timing is recorded. Power is unmeasured. |

**G3 remains open** because cross-output-channel broadcast and efficient
streaming are unfinished. The improved image has now completed the C03
physical correctness and quality checks, including the full 10,000-image
comparison. This does not establish a SOTA claim.

## Boardless candidate profile

The candidate's 10,000 exact full-MNIST RTL jobs each take **164,165 simulated core cycles**:
14,370 compute, 46,544 memory wait and 103,251 control cycles. They perform
61,184 useful MACs and read 152,456 physical SRAM bytes. Relative to the
earlier physical image's matching RTL run, cycles fall **10.064%** and SRAM reads
fall **32.522%**. The physical board now reports these same counters on every
SmallCNN job. At the nominal 27-MHz core clock this is **6.080 ms**; clock
frequency was not independently measured. The control fraction
is still **62.9%**, making serialized gather the next performance target.
The one-port trace requires at least **27,487** SRAM service cycles
(19,057 reads plus 8,430 byte writes), only **16.7%** of elapsed cycles;
this lower bound quantifies how much time the controller spends outside
port service.
The independent oracle scores **9,640/10,000 (96.40%)**, and the candidate
matches all 10,000 raw INT8 outputs in both RTL and the separate physical run.
The [per-layer counters](evidence/phase3-linebuffer/summary.json), complete
[10,000-job trace](evidence/phase3-linebuffer/smallcnn-10000-rtl.json.gz),
[randomized-stall kernels](evidence/phase3-linebuffer/kernel-profiles.json),
[source manifest](evidence/phase3-linebuffer/source-manifest.json) and
[Gowin route](evidence/phase3-linebuffer/gowin-route.txt) are archived.

| Candidate route metric | Value |
|---|---:|
| Logic | 8,192 / 20,736 |
| Registers | 2,525 / 15,750 |
| BSRAM | 18 / 46 |
| SSRAM RAM16 | 26 |
| DSP equivalent | 19.75 / 24 |
| Routed Fmax | 27.376 MHz |
| Worst setup slack | +0.509 ns |
| Setup total negative slack | 0 |

The routed candidate retains Gowin's generic-clock-routing warning
(`PR1014`). The bitstream SHA256 is
`fc6c90d3fb162eadfa042ecbfc1d89277c3a7a0af09fbc44a377c1fe90c81164`; this
candidate should be identified by that hash rather than target ID alone,
since the earlier board-tested release also uses target 8196.

## Board-tested baseline

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

## Historical routed candidate

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
This archived artifact has the old incorrect KEY1 polarity. Use the
[current physical release](../../hardware/PHYSICAL.md), which has target ID
8196 and its own source/route/bitstream evidence.
Gowin still reports generic routing for the board clock (`PR1014`). The
post-route power estimate is not a measured energy result. The modest timing
margin and clock warning both require physical bring-up before a release claim.

## Remaining work

The improved candidate's board tests are complete. Cross-output-channel
activation broadcast, less serialized gather and reduction tiles beyond
on-chip SRAM remain architectural work before closing G3. Subsequent designs
will need the same physical regression. A broader latency/energy comparison
remains research work; energy cannot yet be measured because no
voltage/current instrument is available. The PCB revision is unknown.
