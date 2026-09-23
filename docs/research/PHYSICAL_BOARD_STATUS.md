# Physical board baseline validation — 2026-09-23

This page records the earlier reset-corrected baseline image. The improved
four-row line-buffer image has since passed its own full physical run; see the
[Phase 3 line-buffer board record](PHASE_3_LINEBUFFER_PHYSICAL.md).

The connected Tang Nano 20K now executes the accelerator correctly. A board
reset-polarity defect was fixed, then **MLP → SmallCNN → MLP passed 1,000 jobs
per stage**, with exact raw INT8 outputs, full image readback at each model
load and no FPGA reprogramming between stages. All **59 diagnostic checks**
also passed. A separate **10,000-image SmallCNN evaluation passed with zero
integer mismatches and 9,640 correct classifications (96.40%)**. In total,
13,000 model inferences matched their independent integer references.

## Hardware and release identity

- User-reported chip markings: `GW2AR-LV18`, `QN88C8/I7`, `2537C`, `NCWS02.00`.
  The first two lines match the build part `GW2AR-LV18QN88C8/I7`; the other
  markings are retained verbatim without interpretation.
- JTAG ID `0x0000081B`, detected as `GW2A(R)-18(C)`; target device revision C.
- USB debugger serial `2025030317`, SIPEED, VID:PID `0403:6010`.
  FPGA UART is `/dev/cu.usbserial-20250303171`; channel `...170` is JTAG.
- PCB revision is unknown. The user confirmed that no voltage/current meter
  or oscilloscope is available. Power, energy and instrumented clock frequency
  remain unmeasured.
- Gowin Education V1.9.11.03; openFPGALoader 1.1.1; pyserial 3.5;
  UART 115200 8N1; declared board clock 27 MHz.
- Programming used **temporary SRAM configuration only**, with requested
  2.5-MHz JTAG and actual 2-MHz JTAG. Flash was not changed. The programmer
  reported final status `0x00006020` after each successful load.
- Current accelerator target ID 8196, SHA256
  `9fe4967eb5ea0ad290f0f16d80d7b154375521143da06379862e29b5e3591524`.

The [physical workflow](../../hardware/PHYSICAL.md) gives reproducible
build/program/run commands. The [new release](../../hardware/releases/physical/tinyml_v4_tang20k.fs) is
archived separately; original Phase 2–4 candidates remain historical. The
[summary and hash manifest](evidence/physical/summary.json),
[switching outputs](evidence/physical/model-switch.json.gz),
[full-set outputs](evidence/physical/smallcnn-full.json.gz) and
[diagnostic results](evidence/physical/checks.json) preserve the raw evidence.
Large JSON/timing files use deterministic gzip compression.
The UART verifies the ABI, not the configuration hash. Programmer logs and
artifact hashes establish which image was supplied to each run.

## Defect found by real-board testing

Initial JTAG detection was followed by a USB disconnect. After reconnection,
SRAM programming succeeded but both clocked test images were silent. A
clock-free wire loopback passed all 1,280 bytes, establishing the USB bridge,
UART port and RX/TX wiring. The cause was the board wrapper: it treated KEY1
at pin 88 as active low, holding the design in reset when the button was
released. Sipeed's [UART reference](https://github.com/sipeed/TangNano-20K-example/blob/main/uart/src/uart_top.v)
inverts that input before its internal active-low reset.

The corrected wrapper uses an explicit active-high `reset_button`, pull-down,
and initialized three-stage reset synchronizer. Reset asserts asynchronously
and releases synchronously. Configuration startup works without a button press.
Board-top simulation verifies startup and active-high button recovery. The
corrected minimal UART, rebuilt from tracked RTL/constraints, passed all
**1,280 bytes** physically; its source manifest and programming log are retained.

## Completed model and diagnostic runs

| Run | Physical jobs | Integer mismatches | Correct classifications | Core cycles per job |
|---|---:|---:|---:|---:|
| MLP, first load | 1,000 | 0 | 948 / 1,000 | 7,316 |
| SmallCNN, switch load | 1,000 | 0 | 941 / 1,000 | 182,535 |
| MLP, switch back | 1,000 | 0 | 948 / 1,000 | 7,316 |
| SmallCNN, full MNIST test set | 10,000 | 0 | 9,640 / 10,000 | 182,535 |

The switch stages cover the first 1,000 MNIST test images; the final row covers
the entire separate 10,000-image test set. Classification accuracy is distinct
from integer implementation correctness. Full-set physical SmallCNN accuracy
equals the frozen static-INT8 oracle result (96.40%); the source float result
was 96.47%. Model, dataset, calibration, image and oracle hashes are archived.
Every job changed the input, ran the loaded program, compared every output
byte with the independent oracle, reconciled counters, and checked for protocol
errors. Each stage first wrote and read back the entire 32-KiB image.

The separate diagnostic suite passed:

- Six full-range 32-KiB write/read patterns: zero, all-one, walking-one,
  walking-zero, address-dependent and seeded random; **393,216 aggregate
  transferred bytes**. This tests on-chip SRAM, not the external SDRAM.
- Seven protocol checks: CRC rejection without mutation; oversized-frame
  draining with an embedded command; version/bounds/alignment/opcode rejection;
  incomplete-frame timeout recovery; busy ownership plus ABORT; RESET counter
  clearing with SRAM preservation; descriptor-version error and next-RUN recovery.
- Seven directed scalar-reference kernels, each repeated three times: KWS-shaped
  10×4 stride-2 Conv, pointwise 4→5, stride-2 depthwise, 256-channel depthwise,
  25×5 average pool, 3×3 average pool, and Clip. These are synthetic boundary
  cases, not complete KWS/VWW models or all real-model tensor values.
- Every MLP layer (5) and SmallCNN layer (8) on three inputs: **39 exact layer
  comparisons**, reading each intermediate before its SRAM region was reused.
  Ordinary complete-model execution was checked again after restoring descriptors.

## Measured counters and latency boundary

| Model | Compute cycles | Wait cycles | Control cycles | SRAM bytes read / written | Nominal core time | Median host job time |
|---|---:|---:|---:|---:|---:|---:|
| MLP, first 1,000 | 1,324 | 5,626 | 366 | 22,112 / 98 | 0.271 ms | 173.863 ms |
| SmallCNN, first 1,000 | 14,370 | 64,914 | 103,251 | 225,936 / 8,430 | 6.761 ms | 185.815 ms |
| SmallCNN, full 10,000 | 14,370 | 64,914 | 103,251 | 225,936 / 8,430 | 6.761 ms | 185.850 ms |

The counters are read from the physical FPGA. Core time is **derived using the
nominal 27-MHz clock**, without an independent frequency instrument. Host times
are measured wall times for quantized-input upload, RUN/status polling and output
readback; they exclude preprocessing, argmax, report serialization and model
loading. Loading plus full readback took approximately 12.3 seconds per stage.
There was no explicit warm-up exclusion. UART overhead dominates the host result;
these are bring-up measurements, not a tuned system or comparative SOTA benchmark.

SmallCNN's measured cycle split is approximately 7.9% compute, 35.6% waiting
and 56.6% control. Its 61,184 MACs over 182,535 cycles use about 4.19% of
the eight-lane peak when averaged across the whole model. These observations
motivate the still-open reuse/streaming/controller work; they do not establish
an advantage over another accelerator.

SmallCNN per-layer physical profiles for the first input:

| Layer | Cycles | SRAM read bytes | Output bytes |
|---|---:|---:|---:|
| Conv1 | 61,567 | 51,584 | 2,704 |
| ReLU1 | 18,969 | 21,776 | 2,704 |
| MaxPool1 | 10,179 | 13,648 | 676 |
| Conv2 | 79,639 | 121,216 | 968 |
| ReLU2 | 6,817 | 7,888 | 968 |
| MaxPool2 | 3,071 | 4,256 | 200 |
| Reshape/copy | 1,036 | 1,728 | 200 |
| FC | 1,376 | 4,288 | 10 |

Each isolated layer includes RUN/HALT overhead, so these cycles must not be
summed as an uninterrupted model measurement. Full counters for all three
inputs, kernel profiles, actual outputs and hashes are retained in the raw report.
The on-chip SmallCNN image uses 9,568 bytes; MLP uses 12,416 bytes of 32,768.

## Routed implementation and remaining gates

The new route uses **8,025 / 20,736 logic**, **2,421 / 15,750 registers**,
**16 / 46 BSRAM**, and **19.75 / 24 DSP equivalents**. Routed Fmax is
**27.233 MHz**, worst setup slack **+0.318 ns** at 27 MHz, with zero setup/hold
TNS. Gowin still reports generic clock routing (`PR1014`). Board execution
passes at the declared configuration; no frequency sweep or temperature/voltage
qualification has been performed.

H05/G2's on-chip physical execution/readback/counter/route requirements pass.
G0 remains partial because instrument access, PCB/second-target details, AD
provenance and research audit items remain open. C03's repeated-board and layer
checks and complete 10,000-image board comparison pass. For this baseline,
G3's C01/C02 architecture work (broader reuse, multi-tile accumulation and
streaming line buffers) was unfinished; the later physical image adds a
synchronous four-row buffer but still has open broadcast/streaming work.
G4/G5 remain open: this bitstream has **no integrated SDRAM controller
or DMA**, complete KWS/VWW deployment or tuned B1/B2/B3 comparison. Connecting the
board makes those future tests possible; it does not supply the missing designs.

No energy-efficiency or SOTA claim follows from these results. Gowin power is an
estimate and does not replace the unavailable physical power instrumentation.

Board-top startup/reset/UART simulation and the shared kernel-vector RTL suite
also pass. Six physical-runner/archive tests and four Phase 5 audit tests pass,
including deliberate output/hash tampering. The evidence collector independently
rechecks every saved model logit and layer hash against the pinned fixtures;
pass flags alone are insufficient. The Phase 5 readiness audit recognizes this
on-chip release while leaving its complete-model and SDRAM gates false.
