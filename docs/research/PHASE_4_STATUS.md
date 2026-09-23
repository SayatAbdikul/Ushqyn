# Phase 4 working record — updated 2026-09-24

**Physical addendum:** the reset-corrected target 8196 now runs on the actual
Tang Nano 20K. MLP → SmallCNN → MLP passed 1,000 jobs per stage, and all seven
directed kernel cases passed three times each with exact scalar-reference
outputs and reconciled counters. Full scratchpad patterns and protocol recovery
also pass. The [physical record](PHYSICAL_BOARD_STATUS.md) and
[reproduction commands](../../hardware/PHYSICAL.md) identify the new release.
This closes on-chip bring-up gaps; it does not implement the missing SDRAM,
tiled full-model, comprehensive node-coverage or fitted cost-model work.

**P4 is in progress; G4 is open.** Target ID 8196 is a routed **kernel-only**
candidate for Tang Nano 20K. The active board hierarchy still has a single
32-KiB SRAM and no SDRAM connection. The standalone tile DMA is not in this
bitstream. None of the KWS/VWW end-to-end, physical SDRAM, latency, bandwidth
or energy gates are claimed by this result.

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
It is **single-outstanding**: it does not yet issue SDRAM bursts, arbitrate
with the compute engine, double-buffer tiles or overlap compute and transfer.

Gowin Education includes SDRAM Controller HS IP, which may be a suitable
integration path. The [vendor datasheet](https://cdn.gowinsemi.com.cn/DS226E.pdf)
specifies this device's embedded SDRAM as 8 MiB, 32 bits, four banks, 2048
rows and 256 columns per bank, with 4096 refresh cycles per 64 ms. Those
facts are pinned in the [geometry manifest](../../hardware/sdram_tang_nano_20k.json).
The controller IP configuration found in Gowin's installed example tree is
**not suitable**: it is 16-bit, 13-row-bit, 9-column-bit and describes 32 MiB.
The new [IP guard](../../tools/phase4/audit_sdram_ip.py) rejects it. This
example is outside the project, and it was never part of the board image.
The target-specific controller still needs generation, clocking, route,
initialization and physical refresh validation. There is no controller in the
current source manifest or bitstream.

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

D01's physical full-range/1-GiB refresh gate, D02's burst/overlap gate,
D03's tensor-value tiled inference gate and P01's fitted measurement gate
remain open.

`make p4-test PYTHON=/absolute/path/to/.venv/bin/python` now passes 127
compiler tests, target/ISA checks, Verilator lint, the existing kernel RTL
regression, the abstract DMA RTL regression with planner-derived transfers,
the frozen inventory audit and deterministic plan reproduction. The new
schedule tests independently check descriptor legality, all tile output-byte
coverage, disjoint live activation slots, SRAM region separation and exact
byte movement through each declared transfer. They do not prove arithmetic
for complete model nodes, controller behavior, refresh or board timing.

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

Generate the **32-bit, 2-bank-bit, 11-row-bit, 8-column-bit** embedded SDRAM
IP for this exact part and pass the configuration guard; integrate it into a
new board hierarchy without changing the archived kernel release. Verify
initialization, refresh, full address reach, bank/row boundaries and at least
1 GiB aggregate read/write traffic independently of inference. Add a burst
adapter, compute/DMA arbitration and ping-pong scratchpad tiles with explicit
live-region protection. Connect the geometry-only tile plan to real packed
weights, quantization parameters and tensor values; run all pinned audio and
vision nodes through RTL with an independent exact oracle. Then build a
held-out kernel/DMA cost database, reroute the integrated hierarchy and
collect physical board results.
