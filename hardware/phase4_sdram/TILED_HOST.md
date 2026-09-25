# Phase 4 host-addressable tiled board image

The current release adds an [autonomous command sequencer](AUTONOMOUS.md) and
64-byte SDRAM bursts. The diagnostic UART/DMA interface below is retained.
Older bitstream hashes, route numbers and two-word-adapter limitations in this
document describe the preceding `hs_tiled_host_overlap.fs` release; use the
[Phase 4 closure](../../docs/research/PHASE_4_STATUS.md) for current results.

`phase4_tiled_host` joins the UART command parser, v2 engine, tile DMA,
32-KiB scratchpad, Gowin HS SDRAM controller and refresh scheduler. It is a
**physically tested image**. The host
can upload a compiled parameter image and quantized input to the device's own
8-MiB SDRAM, schedule tile transfers, execute each descriptor, and read
intermediate/final tensors. The computer is the controller and data source,
not the accelerator's SDRAM.

The existing v2 stop-and-wait UART frame and CRC-16 remain unchanged. The
`CAPS` response adds four data bytes after its original ten: feature flag `1`
and a little-endian 24-bit external-memory capacity (`0x800000`). The original
on-chip `v2_command` mode still has its prior memory map and CAPS length.

| Host address | Access | Meaning |
|---|---|---|
| `0x000000..0x007fff` | READ/WRITE | 32-KiB engine scratchpad |
| `0x400000..0x40001f` | READ/WRITE | Tile-DMA control/status registers |
| `0x800000..0xffffff` | READ/WRITE | 8-MiB physical SDRAM window |

READ/WRITE frames may transfer 1–64 bytes and may not cross a window edge.
RUN starts the engine from a 64-byte-aligned descriptor in scratchpad; RESET
or ABORT cancels active work. A command issued before SDRAM initialization or
while DMA is busy receives BUSY status. During engine execution, MMIO writes
may stage and start a DMA; scratchpad and external host accesses remain BUSY.
STATUS reports combined engine/DMA busy and error; its counters remain the
engine's counters. The live-region limit is a software declaration of the
current descriptor's scratchpad footprint. The host must set it truthfully
before a concurrent transfer. An overlapping incoming DMA is rejected with
error code 9.

DMA register offsets from `0x400000` are little-endian bytes:

| Offset | Bytes | Meaning |
|---:|---:|---|
| `0x00` | 3 | SDRAM byte offset, aligned to 8 |
| `0x04` | 3 | SRAM byte offset, aligned to 8 |
| `0x08` | 4 | Transfer length in bytes, 1–32768 |
| `0x0c` | 1 | Direction: `1` SDRAM→SRAM, `0` SRAM→SDRAM |
| `0x0d` | 1 | Write `1` to start DMA |
| `0x0e` | 2 | Engine live-region limit; concurrent DMA must start at/above it |
| `0x10` | 1 | Busy bits: bit 0 DMA, bit 1 engine |
| `0x11` | 1 | Last engine/DMA error code |
| `0x12` | 4 | Bytes copied by the last DMA command |
| `0x16` | 4 | Last engine elapsed-cycle counter |
| `0x1a` | 4 | Last DMA elapsed-cycle counter, through memory-port idle |
| `0x1e` | 2 | Saturating cycles when engine and DMA were both busy |

The host sequence is: write parameter image and input through the SDRAM
window; for each tile, configure/start each planned DMA-to-SRAM transfer,
write its descriptor to SRAM address zero, RUN, then DMA the output back to
SDRAM; finally READ outputs through the SDRAM window. The
[`tiled_host.py` runner](../../tools/phase4/tiled_host.py) performs this
sequence for a prepared KWS or VWW fixture and checks every node's exact
quantized tensor against the independent oracle. It verifies SDRAM readback
of the uploaded parameter image and input before execution. The report records
per-node engine/DMA cycles and DMA payload bytes separately from host wall time, which
includes stop-and-wait UART traffic. It deliberately does not flash a
bitstream or label host wall time as accelerator inference latency.

For RTL validation, `make p4-tiled-host` drives the actual packet
parser, engine, DMA and scratchpad RTL against an 8-MiB behavioral external
array with randomized stalls; it checks CRC-framed commands, invalid window
crossings, last-byte SRAM/SDRAM accesses, a round trip DMA at the final SDRAM
address, guarded engine/DMA overlap, tile upload/run/readback, and exact
output. The extended `make p4-real-host` regression, after fixture preparation,
passes every output node of the real KWS and VWW models through the same
framed path; see the [KWS](../../docs/research/evidence/phase4/boardless-host-kws.json)
and [VWW](../../docs/research/evidence/phase4/boardless-host-vww.json) reports.
`make p4-test` also
checks the existing on-chip command mode, kernel arithmetic, DMA, tiling and
refresh regressions. The behavioral array is not a model of the proprietary
SDRAM controller's signal timing.

To route a fresh image, generate the target-specific encrypted HS IP as
described in [README.md](README.md), create an empty output directory, then
run Gowin `gw_sh` against [`build_tiled_host.tcl`](build_tiled_host.tcl) with
the installed IDE library paths set. The current
[physical route record](../../docs/research/evidence/phase4/physical-tiled-host-overlap-route.json)
pins the source/IP/bitstream hashes and reports 21.458-MHz core Fmax at a
20.25-MHz generated clock, zero setup TNS, 12,405 logic, 18 BSRAM and 19.75
DSP equivalents. The [physical result](../../docs/research/PHASE_4_STATUS.md)
includes full KWS/VWW node-value checks and one guarded overlap experiment.
That preceding adapter used single-outstanding two-word commands. The current
autonomous release adds full-model hybrid ping-pong schedules and 64-byte bursts.

Gowin's [power estimate](../../docs/research/evidence/phase4/physical-tiled-host-overlap-power.html.gz)
for this route is 162.572 mW total (122.800 mW quiescent, 39.772 mW dynamic)
using its default 0.125 toggle rates and no VCD/SAIF activity trace. This is
not measured board power, SDRAM-chip power or energy per inference.

The routed [physical bitstream](../releases/phase4-sdram/hs_tiled_host_overlap.fs)
is archived with its tested SHA-256. When the board is available, use the
programming procedure in
[PHYSICAL.md](../PHYSICAL.md), then run the prepared fixture with:

```sh
.venv/bin/python3 tools/phase4/tiled_host.py \
  --port /dev/cu.usbserial-20250303171 \
  --fixture work/phase4/rtl-kws \
  --report work/phase4/physical-host-kws.json
```

Repeat with `rtl-vww` after KWS. The port example is the previously observed
UART interface; discover the current device path before use. Preserve the
generated reports and the programmed bitstream hash. Run
`tools/phase4/run_physical_overlap.py` to repeat the directed concurrent
Conv/DMA test and live-region rejection check.

The [half-scratchpad feasibility audit](../../docs/research/evidence/phase4/boardless-pingpong-feasibility.json)
checks the current output-channel tiler under two 16-KiB regions. All 22 KWS
operators fit individually, but VWW operators 1, 5 and 13 (zero-based Conv
indices) cannot fit even their minimum channel tile. A uniform 16-KiB bank
split is therefore not a legal full-VWW schedule. A future overlap design
must retain a full-SRAM fallback or add finer spatial/input-channel tiling,
and must prove live-region ownership and actual net speedup.
