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

**P4 is in progress; G4 is open.** Target ID 8196 is a routed **kernel-only**
candidate for Tang Nano 20K. The active board hierarchy still has a single
32-KiB SRAM and no SDRAM connection. The standalone tile DMA is not in this
bitstream. None of the KWS/VWW end-to-end, integrated SDRAM,
sustained-bandwidth or energy gates are claimed by the active accelerator image.

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
layout boundary. This is a **shape/attribute audit**, not execution of all
real model nodes. Independent scalar RTL tests cover the named boundary
geometries under randomized SRAM stalls; an exact 1,000-job SmallCNN
board-system regression also passes on the new target. Complete pinned
KWS/VWW tensor-value RTL coverage remains outstanding.

The audit also shows why external memory is mandatory under the current
32-KiB scratchpad: KWS has 33,216 packed persistent weight/parameter bytes
before descriptors or activations; VWW has 260,640. Six VWW layers alone
have an input/output live pair over 32 KiB, and two weight/parameter sets
individually exceed it. These are lower-bound storage observations, not a
complete tiled placement schedule.

## Off-chip progress (D01–D03)

The standalone [tile DMA](../../rtl/v2/tile_dma.sv) copies aligned 64-bit
source/destination beats between a 32-KiB SRAM port and an abstract 8-MiB
external-memory port. It handles 1–32,768-byte transfers, partial final
strobes, 24-bit upper addresses, randomized backpressure and delayed reads,
invalid bounds and abort drain. Its randomized RTL test uses an 8-MiB
behavioral array, including a transfer ending at the final external byte.
It is **single-outstanding** and does not yet issue SDRAM bursts, double-buffer
tiles or overlap transfers from a real memory controller.

An isolated [tiled core](../../rtl/v2/tiled_core.sv) now connects the existing
engine and DMA to the same real RTL scratchpad through a fair single-port
arbiter. Its regression starts a ReLU kernel and DMA concurrently in both
transfer directions under randomized external stalls; it verifies exact
compute/DMA data, read-response ownership, partial tails, host exclusion
while either client is busy, and actual overlap of busy intervals. This is
functional SRAM arbitration only. It is not in the board hierarchy, has no
SDRAM controller or burst command path, and does not demonstrate useful
compute/transfer throughput overlap.

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
These counts are geometry and traffic estimates, **not** full-model RTL
execution, timing or SDRAM bandwidth measurements. The planner uses a single
tile at a time and does not implement burst commands, compute/DMA overlap or
ping-pong scratchpad tiles. The randomized abstract-port RTL DMA test now
also replays selected planner-produced high-address, short and tail transfers.

D01's isolated physical full-range/1-GiB refresh gate is met. D02's basic
two-word burst command is physically demonstrated, but integrated DMA,
double buffering and useful compute/transfer overlap remain open. D03's
tensor-value tiled inference gate and P01's fitted measurement gate remain
open.

`make p4-test PYTHON=/absolute/path/to/.venv/bin/python` now passes 127
compiler tests, target/ISA checks, Verilator lint, the existing kernel RTL
regression, the abstract DMA RTL regression with planner-derived transfers,
the shared-SRAM arbitration and refresh-scheduler RTL regressions, the frozen inventory audit and
deterministic plan reproduction. The new
schedule tests independently check descriptor legality, all tile output-byte
coverage, disjoint live activation slots, SRAM region separation and exact
byte movement through each declared transfer. These boardless checks do not
prove arithmetic for complete model nodes or the integrated board hierarchy.

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

Integrate the physically tested HS port with the tile DMA, shared-SRAM
arbiter and board hierarchy, then demonstrate actual DMA transfers to SDRAM.
Implement ping-pong scratchpad tiles with explicit live-region protection.
Connect the geometry-only tile plan to real packed
weights, quantization parameters and tensor values; run all pinned audio and
vision nodes through RTL with an independent exact oracle. Then build a
held-out kernel/DMA cost database, reroute the integrated hierarchy and
collect physical board results.
