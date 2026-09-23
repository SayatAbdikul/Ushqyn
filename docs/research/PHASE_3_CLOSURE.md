# Phase 3 gate closure on Tang Nano 20K — 2026-09-23

**G3 passed for the on-chip SmallCNN.** The final image completed 3,000 exact
MLP → SmallCNN → MLP jobs on one programmed board, 61 physical diagnostics, and
a separate 10,000-image SmallCNN run. Every raw INT8 output matched the pinned
independent oracle; 9,640/10,000 classifications were correct (96.40%). Each
model load included a full 32-KiB write/readback. All 11,000 physical SmallCNN
jobs reported the same counters as source-matched board-system RTL. The
[validated summary](evidence/phase3-closure/summary.json) and
[raw switch](evidence/phase3-closure/model-switch.json.gz),
[full-set](evidence/phase3-closure/smallcnn-full.json.gz), and
[diagnostic](evidence/phase3-closure/checks.json) reports are archived.

The shared eight-lane MAC engine now visits ordinary-Conv output channels in
pairs when a reduction has at most 128 terms. Its first channel gathers one
activation tile; the second consumes that same tile, while separate 128-byte
filter banks retain both channels' weights across spatial positions. Longer
reductions use the proven output-stationary fallback and still accumulate all
tiles before applying the corrected bias and requantization. The four-row
single-read synchronous line buffer remains in BSRAM. A line-buffer hit or
SRAM response can feed up to eight contiguous Conv bytes or four pooling
bytes per cycle without crossing a memory word, kernel row, reduction tile,
image boundary, or pool window. Zero-point padding and intermediate
requantization remain unchanged. The single engine instance is in
`rtl/v2/system.sv`; the route still uses 19.75/24 DSP equivalents.

| SmallCNN metric per image | Prior board-tested line-buffer image | G3 closure image | Change |
|---|---:|---:|---:|
| Exact full-set INT8 outputs | 10,000/10,000 | 10,000/10,000 | unchanged |
| Correct classifications | 9,640/10,000 | 9,640/10,000 | unchanged |
| Physical core cycles | 164,165 | 118,267 | −27.958% |
| Physical SRAM read bytes | 152,456 | 133,032 | −12.741% |
| Compute / wait / control cycles | 14,370 / 46,544 / 103,251 | 14,370 / 41,688 / 62,209 | control −39.750% |
| Nominal core time at 27 MHz | 6.080 ms | 4.380 ms | −1.700 ms |
| Median host job time, full set | 179.893 ms | 173.832 ms | −6.062 ms |

The host time includes UART input upload, RUN/status polling and output
readback. It excludes preprocessing, argmax, report writing and the roughly
12.3-second full model-image load/readback. Host timing is subject to USB and
operating-system variation; the core cycle counter is the cleaner
within-project hardware comparison. The physical Conv2 layer stop fell from
62,141 to 33,837 cycles and from 51,224 to 33,592 SRAM read bytes. Conv1's
read traffic fell from 47,040 to 45,248 bytes. The comparison above uses
complete model runs because isolated stops include their own RUN/HALT overhead.

The frozen [bitstream](../../hardware/releases/phase3-closure/tinyml_v6_tang20k.fs)
has SHA256
`4e3d6e9a47d872bfb65134fff9cdbb25c11dfc22bcbbf39db7d2c1ac9b61c0da`.
The [source manifest](evidence/phase3-closure/source-manifest.json) hashes
every routed RTL source. Gowin Education V1.9.11.03 routed the
`GW2AR-LV18QN88C8/I7` revision-C target at **27.023 MHz Fmax** for the
27-MHz constraint, with **+0.032 ns** worst setup slack and zero setup TNS.
The [route report](evidence/phase3-closure/gowin-route.txt.gz) records
10,903/20,736 logic, 2,561/15,750 registers, 18/46 BSRAM blocks,
58 distributed RAM16 units and 19.75/24 DSP equivalents. Gowin still warns
`PR1014` about generic clock routing. The positive setup margin is only
32 ps, so this is a timing-closed gate result with little PVT margin, not a
claim of robust higher-frequency operation.

The board's USB debugger serial was `2025030317`; JTAG identified device
`0x0000081B`, and the [programmer log](evidence/phase3-closure/program.log.gz)
records successful DONE status `0x00006020`. The image was loaded into
temporary FPGA SRAM; flash was untouched. The UART ABI has no configuration
hash readback, so identity is established by the supplied file hash, source
manifest and programmer log. The [archiver](../../tools/physical/archive_phase3_closure.py)
independently rechecks every saved output against the pinned fixtures, each
job's counters, all 39 layer outputs, nine directed kernels repeated three
times, six full-range SRAM patterns, seven protocol checks, route limits and
the bitstream/source hashes. Strict Verilator lint, 123 compiler tests,
randomized-stall engine tests, directed kernel tests, board/system protocol
tests and the RTL MLP → SmallCNN → MLP switch passed. Separate source-matched
RTL runs compared 1,000 and 10,000 SmallCNN jobs exactly, including all eight
layer stops. The raw RTL traces and test XML are linked from the
[summary](evidence/phase3-closure/summary.json).

The nominal core-time calculation divides cycles by the declared 27-MHz
clock; no separate clock measurement was available. The routed Gowin
[power report](evidence/phase3-closure/gowin-power.html.gz) estimates
**161.506 mW total FPGA power**: 122.800 mW quiescent and 38.706 mW
dynamic, for a typical process at 25 °C. Its default I/O and remaining-net
toggle settings are both `0.125`; no VCD or SAIF workload activity was
provided. Gowin's [Power Analyzer guide](https://www.gowinsemi.com/upload/database_doc/36/document/5bfcfe278fe50.pdf)
explains how toggle assumptions enter the calculation. The estimate is not
a physical board measurement and should not be used as measured energy per
inference. Actual power and energy are unmeasured because no voltage/current
instrument was available. This gate
covers an on-chip SmallCNN and MLP, not complete KWS/VWW models, SDRAM/DMA,
or a SOTA energy/latency claim. Those remain later roadmap work.
