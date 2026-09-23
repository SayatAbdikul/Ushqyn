# Phase 3 line-buffer candidate on Tang Nano 20K — 2026-09-23

The routed line-buffer image ran successfully on the connected Tang Nano 20K.
The physical test completed **3,000 exact MLP → SmallCNN → MLP jobs** without
reconfiguring the FPGA between model loads, **61 diagnostics**, and a separate
**10,000-image SmallCNN evaluation with zero INT8 output mismatches**. The
SmallCNN classified 9,640/10,000 MNIST test images correctly (96.40%), matching
the pinned independent static-INT8 oracle. Each model load included full
32-KiB image write/readback; the full-set run compared every output byte and
reconciled physical counters on every image.

The supplied bitstream is
[`tinyml_v5_candidate.fs`](../../hardware/releases/phase3-linebuffer/tinyml_v5_candidate.fs),
SHA256 `fc6c90d3fb162eadfa042ecbfc1d89277c3a7a0af09fbc44a377c1fe90c81164`.
It was built from source commit `aa518a488270f6103feffab0b4030705048b2394`
for `GW2AR-LV18QN88C8/I7` revision C, target ID 8196. The USB debugger serial
was `2025030317`; FPGA UART was `/dev/cu.usbserial-20250303171` at 115200 baud.
Programming used temporary SRAM configuration, leaving flash untouched. The
[programmer log](evidence/phase3-linebuffer-physical/program.log) reports JTAG
ID `0x0000081B`, successful SRAM load and final DONE status `0x00006020`.
`openFPGALoader -v` is verbose output, not configuration readback; the UART
checks the target ABI but cannot read back the bitstream hash. The supplied
file hash and programmer log establish which image was loaded.

| SmallCNN metric | Earlier physical image | Line-buffer image | Change |
|---|---:|---:|---:|
| Exact full-set outputs | 10,000/10,000 | 10,000/10,000 | unchanged |
| Correct classifications | 9,640/10,000 | 9,640/10,000 | unchanged |
| Physical core cycles / image | 182,535 | 164,165 | −10.064% |
| Physical SRAM read bytes / image | 225,936 | 152,456 | −32.522% |
| Nominal core time at 27 MHz | 6.761 ms | 6.080 ms | −0.680 ms |
| Median host job time | 185.850 ms | 179.893 ms | −5.956 ms |

The core counter split for the new image is 14,370 compute, 46,544 memory-wait
and 103,251 control cycles, totaling 164,165; it performs 61,184 useful MACs
and 8,430 SRAM byte writes per image. Physical counters match the board-system
RTL result on all 11,000 SmallCNN jobs in these runs. The largest isolated-layer
gain is Conv2: **79,639 → 62,141 cycles** and **121,216 → 51,224 read bytes**.
Pooling gets slightly slower with this buffer, so the whole-model figures above
are the fair net comparison. Isolated layer runs include their own RUN/HALT
overhead and are not summed as an uninterrupted model time.

The 61 diagnostics comprise six full-range 32-KiB SRAM patterns, seven protocol
and recovery checks, nine directed kernels repeated three times, and all five
MLP plus eight SmallCNN layers on three inputs (39 exact intermediate checks).
The new directed cases include 81-term wide-row Conv and a 261-term fallback
beyond the 128-byte filter cache. The [archiver](../../tools/physical/archive_phase3_linebuffer.py)
rechecks raw outputs against pinned fixtures, each kernel's expected bytes,
all intermediate hashes and counters, model identity, source hashes, bitstream
identity and the older baseline artifact hash before writing the
[summary](evidence/phase3-linebuffer-physical/summary.json). Its raw
[switch jobs](evidence/phase3-linebuffer-physical/model-switch.json.gz),
[full-set jobs](evidence/phase3-linebuffer-physical/smallcnn-full.json.gz)
and [diagnostics](evidence/phase3-linebuffer-physical/checks.json) are retained.

One immediate post-configuration CAPS attempt received a malformed serial
response before any inference was issued. A later raw probe returned a valid
CAPS frame, and the complete diagnostic and model runs then passed. The
initial attempt is not counted as a hardware test pass.

Core milliseconds divide FPGA cycle counters by the **nominal** 27-MHz board
clock; no separate clock instrument was available. Host job time covers UART
input upload, RUN/status polling and output readback, excluding preprocessing,
argmax, report writing and the roughly 12.3-second model load/readback. Power
and energy remain unmeasured because no voltage/current instrument is
available. This image uses on-chip SRAM only; it does not validate SDRAM, DMA
or complete audio/vision models. Physical correctness and the measured
within-project performance gain do not establish a SOTA accelerator claim.
