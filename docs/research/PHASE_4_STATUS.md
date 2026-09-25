# Phase 4 working record — updated 2026-09-24

**Physical addendum:** the reset-corrected target 8196 now runs on the actual
Tang Nano 20K. MLP → SmallCNN → MLP passed 1,000 jobs per stage, and all seven
directed kernel cases passed three times each with exact scalar-reference
outputs and reconciled counters. Full scratchpad patterns and protocol recovery
also pass. The [physical record](PHYSICAL_BOARD_STATUS.md) and
[reproduction commands](../../hardware/PHYSICAL.md) identify the new release.
This closes on-chip bring-up gaps; it does not integrate SDRAM,
tiled full-model, comprehensive node-coverage or fitted cost-model work.

**SDRAM addendum:** isolated board images now pass full-range 8-MiB,
64-sweep/1-GiB write-read tests with both the Apache-2.0 open controller
([evidence](evidence/phase4/physical-sdram-open.json), 288.503 s) and the
target-configured Gowin HS controller
([evidence](evidence/phase4/physical-sdram-hs.json), 537.826 s). Both include
80-ms retention waits per pass and continuous refresh. A separate routed
two-word HS burst image passed 73 directed write/read pairs twice, including
all 64 walking-one bits and address boundaries
([evidence](evidence/phase4/physical-sdram-burst.json)). This closes the
isolated D01 physical-memory gate and proves a basic burst command path.
All three standalone routes met their 27-MHz controller-clock constraint with
zero setup TNS; Gowin reported core-clock Fmax of 125.097 MHz (open BIST),
119.863 MHz (HS BIST) and 89.398 MHz (two-word burst). The archived
[open](evidence/phase4/physical-sdram-open-route.txt),
[HS](evidence/phase4/physical-sdram-hs-route.txt) and
[burst](evidence/phase4/physical-sdram-burst-route.txt) route summaries apply to
diagnostic images only; they are not accelerator timing, SDRAM bandwidth or
inference throughput measurements.

**Integrated DMA addendum:** a separate board image now joins the actual
`v2_tiled_core` scratchpad and tile DMA to the two-word HS SDRAM port. It
passed three fresh-program runs of a 32-KiB SRAM→SDRAM→SRAM round trip ending
at the last physical SDRAM byte, followed by a 13-byte transfer across a
2-MiB bank boundary with a five-byte final strobe. The host verified the
board's exact-data pass frame in every run; the archived
[physical report](evidence/phase4/physical-sdram-dma.json) and
[two repeats](evidence/phase4/physical-sdram-dma-repeat-a.json)
([second](evidence/phase4/physical-sdram-dma-repeat-b.json)) pin the tested
[bitstream](../../hardware/releases/phase4-sdram/hs_dma_full_tile.fs) and all
RTL/IP hashes. The full-tile DMA reports 114,189 write and 114,188 read core
cycles at the nominal 20.25-MHz PLL setting, or 5.542 MiB/s in either
direction. These cycle counts repeated exactly; they exclude SRAM fill,
clear and verification and are **DMA-path throughput**, not model inference
throughput. The integrated route uses 9,910/20,736 logic, 18/46 BSRAM and
19.75/24 DSP equivalents. Its [timing report](evidence/phase4/physical-sdram-dma-timing.html)
shows 23.728-MHz core Fmax against the 20.25-MHz generated-clock constraint
and zero setup TNS; the [place-and-route report](evidence/phase4/physical-sdram-dma-route.txt)
records the resource usage. The engine is present but held idle in this image.
Its DMA success closes a board-integration step within D02, while ping-pong
buffers and useful compute/transfer overlap remain open. A transient USB UART
silence was isolated with the minimal loopback image and cleared by a USB
power cycle; the release image then passed three fresh-program runs.

After the MaxPool RTL fix below, a fresh Gowin route of the **diagnostic**
SDRAM/DMA hierarchy passed its 20.25-MHz generated-core constraint:
[22.552-MHz routed Fmax and zero setup TNS](evidence/phase4/boardless-reroute-dma.json),
with 9,920/20,736 logic, 18/46 BSRAM and 19.75/24 DSP equivalents. The
[route](evidence/phase4/boardless-reroute-dma-route.txt.gz) and
[timing report](evidence/phase4/boardless-reroute-dma-timing.html.gz) pin this
updated build. Its engine is still idle in the diagnostic state machine, and
the newly routed bitstream has **not** run on the board. These post-route
numbers do not replace the three-run physical DMA evidence for the earlier
source revision.

The measured 5.542 MiB/s is only 7.2% of the SDRAM's simple 32-bit ×
20.25-MHz payload ceiling (77.25 MiB/s). This ratio is a diagnostic, not a
claim that the memory can sustain the ceiling: each DMA beat currently incurs
its own activate/read-or-write/settle sequence. Dividing the geometry-only
planned DMA payloads by the observed full-tile rate gives optimistic
transfer-only floors of about 55 ms for KWS and 234 ms for VWW. Their many
short transfers and all compute are excluded, so these are **not** measured
inference times or FPS. Long streaming bursts and overlapping tile staging
are now critical performance work for G4.

**P4 is in progress; G4 is open.** Target ID 8196 is a routed **kernel-only**
candidate for Tang Nano 20K. The active board hierarchy still has a single
32-KiB SRAM and no SDRAM connection. The standalone tile DMA is not in this
bitstream. None of the KWS/VWW end-to-end, integrated SDRAM,
sustained-bandwidth or energy gates are claimed by the active accelerator image.

**Boardless tile execution addendum:** the calibrated `Program` can now be
materialized into a parameter image and per-tile descriptors/DMA transfers by
[`compiler/phase4_compile.py`](../../compiler/phase4_compile.py). The packer
matches the existing on-chip weight/parameter ABI, rejects branched graphs
and runtime constants, and materializes all 22 KWS and 58 VWW geometries. The
[`compile_tiled.py` CLI](../../tools/phase4/compile_tiled.py) accepts a
converted ONNX graph and its schema-2 calibration report, verifies the model
hash before writing outputs, and emits the parameter image and tile plan. A new
[RTL regression](evidence/phase4/boardless-tiled-program.json) uses the real
engine, tile DMA and scratchpad with randomized external-memory stalls. It
matches an independent integer oracle after each layer of an eight-kernel
Conv→depthwise→pointwise→ReLU→MaxPool→AveragePool→Clip→FC chain, crosses a
two-tile ReLU boundary, and executes a two-tile FC whose weight matrix alone
occupies 32 KiB. This synthetic regression uses an abstract external-memory
port; it is not physical SDRAM inference. This
regression found and fixed an RTL MaxPool parameter validation mismatch for
non-−128 input zero points. The archived physical releases predate that RTL
fix and remain source-hash pinned to their tested revision.

The same RTL path has now executed real weights from the SHA-verified MLCommons
KWS and VWW source artifacts. A fresh conversion passed source-framework
parity, but its canonical ONNX bytes differ from the frozen conversion because
of generated constant names. A strict
[calibration rebase](PHASE_4_REAL_MODEL.md) checked the original source hashes,
conversion parity, every operator, attribute, activation tensor and constant
geometry; only two KWS and one VWW constant input names changed. The resulting
[KWS](evidence/phase4/boardless-real-kws.json) and
[VWW](evidence/phase4/boardless-real-vww.json) tests compared **every** node
output byte with an independent integer oracle for one deterministic INT8
input per model: 22/22 KWS nodes (20 compute tiles) and 58/58 VWW nodes
(75 compute tiles) passed. The external parameter images contain 49,376 KWS
bytes and 334,816 VWW bytes. Their SHA-256s, the converted-model provenance,
the fixture hashes and the simulation transcripts are archived alongside the
reports. This proves one-input boardless tensor-value coverage of the two real
source models, not complete-set accuracy, a byte-identical rerun of the frozen
ONNX binaries, or physical model execution. Reported simulation clocks include
test-host descriptor writes and randomized abstract-memory stalls and must not
be converted to board FPS.

## Kernel progress (C04)

The existing eight-lane MAC and requantizer now execute ordinary Conv,
depthwise Conv with channel multiplier one, pointwise Conv, MaxPool,
unpadded AveragePool/GlobalAveragePool, ReLU, Clip, FC and copy/reshape.
Depthwise shares the same MAC array; it changes the input-channel address and
reduction length rather than adding a second array. The descriptor handles
the pinned KWS 10×4 stride-2 asymmetric first Conv, VWW stride-2 depthwise
boundaries, 256-channel depthwise tiles, KWS 25×5 average pooling and VWW 3×3
average pooling. Average pooling accumulates centered INT8 values and uses a
compiler-packed per-window fixed-point divisor; its accepted large-stride
form has one unpadded full window. Clip clamps in the input quantization
domain before one requantization.

The [frozen inventory audit](evidence/phase4/inventory-audit.json) identifies
0 geometries outside this kernel subset among KWS's 22 and VWW's 58
operators. VWW's initial NHWC→NCHW transpose is explicitly the declared host
layout boundary. This is a **shape/attribute audit**; the separate real-model
boardless run above supplies one-input tensor-value execution of all nodes.
Independent scalar RTL tests cover the named boundary
geometries under randomized SRAM stalls; an exact 1,000-job SmallCNN
board-system regression also passes on the new target. Complete pinned
KWS/VWW physical tensor-value execution remains outstanding.

The audit also shows why external memory is mandatory under the current
32-KiB scratchpad: KWS has 33,216 packed persistent weight/parameter bytes
before descriptors or activations; VWW has 260,640. Six VWW layers alone
have an input/output live pair over 32 KiB, and two weight/parameter sets
individually exceed it. These are lower-bound storage observations, not a
complete tiled placement schedule.

## Off-chip progress (D01–D03)

The [tile DMA](../../rtl/v2/tile_dma.sv) copies aligned 64-bit
source/destination beats between a 32-KiB SRAM port and an abstract 8-MiB
external-memory port. It handles 1–32,768-byte transfers, partial final
strobes, 24-bit upper addresses, randomized backpressure and delayed reads,
invalid bounds and abort drain. Its randomized RTL test uses an 8-MiB
behavioral array, including a transfer ending at the final external byte.
It is **single-outstanding**. The integrated board diagnostic now connects it
to a real two-word-burst SDRAM port, but it does not yet double-buffer tiles
or overlap useful compute with transfers.

An isolated [tiled core](../../rtl/v2/tiled_core.sv) now connects the existing
engine and DMA to the same real RTL scratchpad through a fair single-port
arbiter. Its regression starts a ReLU kernel and DMA concurrently in both
transfer directions under randomized external stalls; it verifies exact
compute/DMA data, read-response ownership, partial tails, host exclusion
while either client is busy, and actual overlap of busy intervals. This is
functional SRAM arbitration only. The physical DMA diagnostic instantiates
this same hierarchy with the SDRAM controller but keeps its engine idle; it
does not demonstrate useful compute/transfer throughput overlap.

Gowin Education includes SDRAM Controller HS IP, now physically validated in
isolated board tests. The [vendor datasheet](https://cdn.gowinsemi.com.cn/DS226E.pdf)
specifies this device's embedded SDRAM as 8 MiB, 32 bits, four banks, 2048
rows and 256 columns per bank, with 4096 refresh cycles per 64 ms. Those
facts are pinned in the [geometry manifest](../../hardware/sdram_tang_nano_20k.json).
The controller IP configuration found in Gowin's installed example tree is
**not suitable**: it is 16-bit, 13-row-bit, 9-column-bit and describes 32 MiB.
The new [IP guard](../../tools/phase4/audit_sdram_ip.py) rejects it. This
example is outside the project, and it was never part of the board image.
An isolated `gw_sh` Tcl probe using `create_ipc` and `read_ipc` also rejected
`sdram_controller_hs` with `ERROR (IP1001): The ip is not exist`; Gowin's
[Tcl IPFlow guide](https://cdn.gowinsemi.com.cn/SUG1220E.pdf) does not list
this controller among supported scripted IPs. The IDE GUI generated the
correct [target-specific configuration](../../hardware/phase4_sdram/sdram_controller_hs.ipc).
Its encrypted Verilog remains a local build artifact under `work/`; the routed
HS full-memory and two-word burst images are archived under
[hardware/releases/phase4-sdram](../../hardware/releases/phase4-sdram/).
The active accelerator bitstream still has no SDRAM controller.
The [Gowin HS controller guide](https://www.gowinsemi.com/upload/database_doc/2265/document/68f697b42e222.pdf)
requires user-issued auto-refresh commands. A standalone
[refresh scheduler](../../rtl/v2/sdram_refresh.sv) now issues requests every
421 core clocks at 27 MHz (within the 4096/64-ms requirement), accumulates
deferred requests while the controller is busy, catches up after stalls and
flags a full-window backlog. Its reset/cadence/stall RTL test passes with a
behavioral acknowledgement. It is now wired to the standalone Gowin HS port
and both physical memory-test hierarchies. The full-memory write/hold/read
sweeps demonstrate retention and refresh on this board at 27 MHz.

The [boardless tiling planner](../../compiler/phase4_tiling.py) now emits one
legal v2 descriptor and aligned DMA transfer sequence per tile across all
frozen KWS/VWW nodes. It reserves two nonoverlapping external activation
slots, packs per-node weight and parameter regions, and refuses a tile that
cannot fit the 32-KiB SRAM or 8-MiB external address range. The
[machine-readable plans](evidence/phase4/boardless-tiling-summary.json) are
reproducible with `make p4-plan`. KWS uses 20 compute tiles across 20 compute
nodes, peaks at 21,248 SRAM bytes and plans 322,006 DMA payload bytes. VWW
uses 75 compute tiles across 56 compute nodes, peaks at 32,768 SRAM bytes and
plans 1,359,570 DMA payload bytes. Alias/layout nodes do not execute on the
engine. A host-layout VWW transpose is assumed at its declared boundary.
These counts are geometry and traffic estimates, **not** physical timing or
SDRAM bandwidth measurements. Real source-model parameter images and one-input
full-graph RTL execution now pass as described above; the model payloads remain
outside Git and are reproducible from their pinned source manifests. The planner uses a single
tile at a time and does not implement burst commands, compute/DMA overlap or
ping-pong scratchpad tiles. The randomized abstract-port RTL DMA test now
also replays selected planner-produced high-address, short and tail transfers.

D01's isolated physical full-range/1-GiB refresh gate is met. D02's basic
two-word burst command and integrated full-tile DMA are physically demonstrated,
but double buffering and useful compute/transfer overlap remain open. D03's
physical SDRAM tensor-value tiled inference gate and P01's fitted measurement
gate remain open.

`make p4-test PYTHON=/absolute/path/to/.venv/bin/python` now passes 135
compiler tests, target/ISA checks, Verilator lint, the existing kernel RTL
regression, the abstract DMA RTL regression with planner-derived transfers,
the shared-SRAM arbitration, packed tiled-program and refresh-scheduler RTL regressions, the frozen inventory audit and
deterministic plan reproduction. The new
schedule tests independently check descriptor legality, all tile output-byte
coverage, disjoint live activation slots, SRAM region separation and exact
byte movement through each declared transfer. The separate one-input
real-model RTL runs prove complete-node arithmetic under the abstract-memory
test conditions, but not the integrated board hierarchy. The
[suite log](evidence/phase4/boardless-suite-log.txt) records this run.

## Historical routed candidate and evidence

Gowin Education V1.9.11.03, `GW2AR-LV18QN88C8/I7` revision C, 27-MHz
constraint:

| Metric | Post-route value |
|---|---:|
| Logic | 7,954 / 20,736 |
| Registers | 2,421 / 15,750 |
| BSRAM | 16 / 46 |
| DSP equivalent | 19.75 / 24 |
| Routed Fmax | 27.611 MHz |
| Worst setup slack | +0.820 ns |
| Setup total negative slack | 0 |

The [kernel candidate bitstream](../../hardware/releases/phase4/tinyml_v4_kernels.fs)
has SHA256 `a214798c00b86a71940dd67f1a943403f15b3a642b9ddbb8cb9f715b172d0b33`.
It preserves the original pre-bring-up build with incorrect KEY1 polarity.
Use the physical release for board execution; its new route uses 8,025 logic,
2,421 registers, 16 BSRAM and 19.75 DSP equivalents, with 27.233-MHz Fmax
and +0.318-ns setup slack at 27 MHz. Historical source hashes and this old
candidate remain frozen rather than being relabeled as physical evidence.
Gowin still warns that the board clock uses generic routing (`PR1014`), so
physical timing must be checked. The [evidence summary](evidence/phase4/summary.json)
pins the exact route, bitstream, source hashes, 123 passing compiler tests,
kernel/DMA/system/board RTL tests and the 1,000 exact SmallCNN runs.
The model-switch test also passes MLP→SmallCNN→MLP under target ID 8196.
The [seven kernel profiles](evidence/phase4/kernel-profiles.json) record
simulated compute/wait/control cycles and physical SRAM traffic under a fixed
randomized-stall seed. They are an initial P01 data collection, not a fitted
held-out latency model or measured energy database.

## Next work to close G4

Carry the physically tested HS port and tile DMA from the diagnostic image
into the host-command accelerator board hierarchy. Implement ping-pong
scratchpad tiles with explicit live-region protection and measure useful
compute/transfer overlap. Run several independent inputs from each real model
through that routed hierarchy, then build a held-out kernel/DMA cost database
and collect physical latency, accuracy and power results. The corrected MaxPool
RTL must be rebuilt and revalidated on the board before any new physical claim.
