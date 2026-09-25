# Autonomous Phase 4 execution

The Tang Nano image retains the CRC-framed UART diagnostic protocol and adds
a command sequencer with 32 KiB of dedicated BSRAM. Weights, descriptors and
activation tensors remain in the FPGA's own 8 MiB SDRAM. Upload the command
list and model once, then write one start byte; no host tile scheduling or
intermediate tensor readback is required during normal execution.

## Address map and command format

The prior scratchpad, DMA MMIO and external-memory windows are unchanged.

| Address | Access | Purpose |
|---|---|---|
| `0x500000..0x507fff` | READ/WRITE while idle | 2,048 command records, 16 bytes each |
| `0x410000` | WRITE byte `1` | Start command record zero |
| `0x410000..0x41001f` | READ while idle | Sequencer status and counters |

Each command is little-endian `opcode:u8, flags:u8, reserved:u16, arg0:u32,
arg1:u32, arg2:u32`; reserved bits must be zero.

| Opcode | Meaning |
|---|---|
| 0 | HALT after both units and buffered SDRAM writes finish |
| 1 | DMA: flags bit 0 selects SDRAM→SRAM; arguments are external address, scratch address, byte count |
| 2 | RUN: arg0 is descriptor PC; arg1 packs live-region base in bits 15:0 and exclusive end in bits 31:16 |
| 3 | WAIT: flags bit 0 waits for compute, bit 1 for DMA including memory-port drain |

DMA and RUN launch asynchronously. WAIT enforces dependencies. A RUN waits
for any previous DMA to finish. During compute, either DMA direction must
be disjoint from the compiler-declared live region; violation returns error
9 and drains active work. Commands cannot access beyond the declared SRAM
or SDRAM bounds. Program exhaustion and a cycle watchdog fail closed.
The existing ABORT/RESET command cancels the sequencer and both units;
outstanding DMA reads drain before the interface becomes idle.
The existing DMA abort error remains latched until a new transfer starts;
RESET clears sequencer/protocol errors, and recovery is verified with a
fresh transfer and output readback rather than an empty HALT program.

Status offsets are 0: busy byte; 1: error byte; 4: ASCII `SEQ4`; 8: elapsed
cycles; 12: engine-busy cycles; 16: DMA-busy cycles including write drain;
20: simultaneous busy cycles; 24: current command index. Counters are
little-endian 32-bit values, and elapsed time includes command dispatch and
dependency waits. UART STATUS remains available while the sequence runs.

## Compiler and memory behavior

`compile_tiled(program, prefer_half=True)` tries 16 KiB tiles and retains a
32 KiB fallback when the full input cannot fit a half-bank. The autonomous
compiler alternates half-bank tiles and serializes transitions involving a
full-scratchpad tile. It prefetches the next tile's descriptors/parameters,
and its input only when both tiles belong to the same layer. A dependent
layer's input waits for the previous layer's output to reach SDRAM.

The SDRAM adapter combines writes and reads ahead in aligned 64-byte lines,
using sixteen 32-bit words per physical burst. Byte masks preserve partial
tails. Reads, address changes, idle timeout and refresh force dirty writes
to commit. The controller's busy signal includes buffered writes, preventing
premature DMA completion. Refresh can force a drain even when the same
partial line is written continuously. The adapter contract simulation is
not a substitute for the encrypted controller or physical SDRAM timing.

## Reproduction and measurement boundaries

Build with `build_tiled_host.tcl` in a fresh directory, then program the exact
generated `.fs` with the established temporary-SRAM programming procedure.
Run `tools/phase4/run_physical_sequence.py --model kws` and then `--model vww`,
passing the programmed `--bitstream` and distinct `--report` paths. The
runner verifies source-model/fixture hashes and SRAM/SDRAM uploads.

First it executes a snapshot schedule and compares every node to the saved
independent integer oracle. It then compares sequential and overlapping
schedules with identical tile placement and three repeats. Snapshot copies
are excluded from those performance schedules. Reports retain device
elapsed/overlap counters, input-upload-through-output-readback wall time,
and model-load-and-verification time separately. Program upload/readback
occurs before the reported inference wall interval.

`compiler/phase4_cost.py` predicts isolated engine cycles from descriptor
geometry and cache-tag walks without weights, inputs or measured counters.
Contention in overlapping runs is measured separately. The physical kernel
profiler's `--holdout` option writes its predictions before executing 28
fresh configurations. `summarize_sequence_costs.py` validates them and fits
the DMA model using an explicit held-out set of transfer lengths.

No physical power or inference-energy measurement is provided by this path.
Complete-set KWS/VWW accuracy and the balanced 10,000-job evaluation remain
Phase 5 work; one-input node exactness establishes a different boundary.
