# 512-byte UART burst candidate: measured no-go

The stop-and-wait UART upload uses a CRC-protected 256-byte payload per frame.
A fresh VWW image is 27,648 bytes, so it needs 108 frame/response round trips
before inference can start. The earlier scalar UART image measured a
0.647999625 s upload and roughly 159 ms FPGA inference at 750 kbaud; the
selected fused core now takes about 128.586 ms for device inference. A
separate 8 KiB screen measured a 256-byte burst median of 0.190656354 s.

An isolated 512-byte variant doubles only the CRC-committed payload buffer,
widens its BRAM index, and advertises 512 in the read-only feature response.
It retains the legacy 64-byte commands, frame CRC, SDRAM boundary checks,
single-frame commit, per-frame acknowledged status, and no automatic retry
after an ambiguous response. Sources are
`rtl/phase6/uart_burst_512_command.sv`,
`rtl/phase6/uart_burst_512_bridge.sv`,
`tools/phase6/uart_burst_512.py`, and their isolated test/host/build files.

Boardless gates passed:

| Gate | Result |
| --- | --- |
| Host framing/probe/sequence/ambiguous-response tests | 3/3 passed |
| Parser Cocotb: 512-byte CRC, pre-CRC noncommit, bounds, oversize, legacy | 1/1 passed |
| Full bridge Cocotb with stalled external memory, plus tiled-host regression | 2/2 passed |

Each request sends 12 framing bytes plus its payload, and each acknowledged
response sends 13 bytes. The serial wire minimum for a 27,648-byte upload is
0.40464 s at 256 bytes/frame and 0.38664 s at 512 bytes/frame. The measured
256-byte transfer therefore has 2.253 ms/frame beyond wire time. The
independent 8 KiB measurement gives 2.211 ms/frame, supporting a useful
first-order model. Holding that residual constant had predicted a 512-byte
VWW upload of **about 0.508 s**, a **0.140 s** improvement. This projection
is now **retracted as a performance estimate**: it assumed the per-frame
residual would stay constant after changing frame size, and the same-board
paired measurement below contradicts it.

The candidate changes host transfer time only. Device inference cycles and
device FPS are unchanged. Host upload time matters whenever a new input or
other payload is sent; running the already resident input again avoids an
input upload altogether.

The 512-byte bridge was integrated with the exact fused-activation core. Its
24 MHz Gowin route passed with 24.226 MHz Fmax, 10,224 CLS, 38 BSRAM, and
9.5 DSP equivalents. The fused core with the existing 256-byte bridge routed
at 26.285 MHz Fmax, 10,137 CLS, 38 BSRAM, and 9.5 DSP equivalents. Both
images passed 10/10 exact short board screens. The 512-byte image's KWS and
VWW medians were 1,201,917 and 3,085,977 cycles; the selected 256-byte
image's were 1,201,917 and 3,086,061 cycles. The tiny difference is run
variation, not evidence of a core improvement from UART framing.

A paired upload experiment used the **same 512-capable bitstream** and
alternated 256- and 512-byte frames for the same 27,648-byte VWW input,
three uploads each. Every upload was verified by readback. Median input
upload times were **0.646962875 s (256)** and **0.651045375 s (512)**:
512-byte frames were **4.083 ms slower**, a **0.993729× speedup** relative
to 256-byte frames. The signed six-record report and seal are in
`work/phase6/fused-uart512-paired-transfer-v1/`. The 18 ms theoretical
wire-time advantage did not translate into measured end-to-end upload time.
We therefore retain 256-byte framing for the selected fused image, which
also has more route timing margin and uses fewer CLS resources.

The combined boardless candidate now uses the byte-identical `stream-mask-v1`
engine (`5a7c89729f9694ae23b4d254bd718c6179219edfce1299f74a06ada684290311`)
in `work/phase6/experiments-v1/stream-mask-uart512-v1/`. Its integrated
bridge passed 2/2 Cocotb checks, and the grouped KWS/VWW fixtures passed
12/12 native full-model checks. The timed fixed native cycles are 1,260,399
KWS and 2,933,853 VWW. The 27 MHz PLL, host, and Gowin build inputs are
prepared in `build27p0.tcl` with source hashes in `route-inputs-27p0.json`.
This stream-specific 27 MHz route was not pursued after the fused-core
comparison; its boardless checks remain useful as an independent bridge test.

The grouped plus tail-prefetch fixtures also passed 12/12 native checks on
the same combined engine (`native-grouped-tail-prefetch/report.json`). Their
timed fixed cycles are 1,251,026 KWS and 2,891,293 VWW. The prepared PLL
uses `.FBDIV_SEL(7), .IDIV_SEL(7)` for 27 MHz from the board's 27 MHz input;
both UART instances use `CLK_FREQ=27000000` and `BAUD_RATE=750000` (exactly
36 core clocks per UART bit); the SDRAM port uses `CLOCK_HZ=27000000`. The
isolated Gowin build list names the pinned 512-byte parser and bridge sources.
