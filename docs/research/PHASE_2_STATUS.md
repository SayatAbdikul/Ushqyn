# Phase 2 implementation record — 2026-09-23

The corrected eight-lane MLP accelerator has a single active numerical-v2 board
hierarchy. It passes strict integer simulation and fits/routs on the Tang Nano
20K target. **G2 is open:** no Tang Nano or serial device is connected to this
computer, so programming, physical readback and 1,000 real-board inferences have
not been performed. This is a routed candidate bitstream, not a measured board
result. Phase 0 physical access also remains open.

| Roadmap item | Result | Evidence and limit |
|---|---|---|
| H01 | Complete for v2 target | One [target manifest](../../hardware/targets/tang_nano_20k_v2.json) drives the compiler ABI, generated `target_pkg.sv`, Verilator source list and Gowin build. Legacy v1 modules remain only for historical tests and are excluded from the v2 source list. Strict lint and target regeneration pass. |
| H02 | Complete for FC subset | 64-byte v2 descriptors encode 24-bit addresses and independent 16-bit convolution geometry; all unsupported spatial geometry rejects. Encoding/decoding, alignment, bounds and 32-bit counts/strides are tested. One synchronous SRAM request may be accepted per cycle, with explicit ready, read-valid, abort and error semantics. |
| H03 | Complete for FC/ReLU subset | Eight signed INT8 MAC lanes, per-channel corrected INT32 bias and static multiplier/shift/zero point, one elastic requantizer. Directed/random 64-bit rounding tests and independent FC tests include extreme zero points, tails, stalls and aborts. Accumulator overflow rejects. CNN kernels remain in phases 3–4. |
| H04 | Complete in RTL simulation | Framed UART 115200 8N1; `CAPS`, `READ`, `WRITE`, `RUN`, `STATUS`, `ABORT`, `RESET`. CRC, sequence, size, timeout, oversized-frame drain, busy ownership, version/descriptor rejection and repeated jobs pass. Board UART pin wrapper is simulated; physical transport awaits the board. |
| H05 | Partial | Counters reconcile exactly; the actual board hierarchy uses synchronous inferred BSRAM and passes 1,000 MLP jobs with zero integer-output mismatches in two simulation harnesses. Final Gowin place-and-route/bitstream meet 27 MHz. Physical 1,000-job execution, board identity/programming/readback and power are pending. |
| R05 | Pivot decision recorded | The [single-port service bound](evidence/phase2/memory-service-bound.json) shows why the present fixed engine cannot issue activation and weight reads simultaneously. DeFiNES already models physical ports, so this is **not** a novelty counterexample. The proposed scheduling novelty remains provisional until a tuned legal DeFiNES/COSMA comparison or a stronger mechanism is demonstrated. |

## Reproduced numerical result

The original trained MLP weight fixture SHA256 is
`3e4319307a8ccb9831361cb5183d4848d1b619ca9de628ea6ecd44afd224fe79`.
It is ignored by Git and not redistributed. Calibration uses the first 64 MNIST
training images; the complete 10,000-image test has no identical raw contents in
that set. Original float PyTorch predicts **9,490/10,000 (94.90%)**; static v2
software predicts **9,484/10,000 (94.84%)**. Both use the same trained weights,
with no accuracy-set tuning. The model/board image is 12,416 used bytes in a
32-KiB addressable SRAM; it performs 10,112 useful MACs/inference.

Native Verilator executes the exact `v2_system` hierarchy over 1,000 MNIST test
inputs, changing the input in SRAM and running the same program each time:
**zero integer-output mismatches**, three jobs with every intermediate checked,
and every job's useful MAC and cycle counters checked. A separate Cocotb run of
the same system also recorded 1,000/1,000 exact jobs; subsequent final-source
regression reran its directed protocol test and one full MLP job. The native
result records every final source hash and per-job counters. Its final job used
7,311 core cycles, split into 1,324 compute, 5,626 wait and 361 control cycles;
that is a simulated core count, not measured host or board latency. The board-top
Cocotb test exercises reset synchronization and serial UART wires; randomized
engine memory latency and requantizer stall tests pass separately.

The [evidence directory](evidence/phase2/) contains CI (119 tests), strict lint,
RTL test logs, the 1,000-job raw counter record, model-quality record and exact
source manifest. Dataset and generated trained-model payloads remain under
ignored `work/`; [phase 2 reproduction commands](../../hardware/PHASE_2.md)
regenerate them from the locally available trained weight fixture and MNIST data.

## Routed candidate

Gowin Education V1.9.11.03, `GW2AR-LV18QN88C8/I7` revision C, 27-MHz constraint:

| Metric | Post-route value |
|---|---:|
| Logic | 5,050 / 20,736 |
| Registers | 2,017 / 15,750 |
| BSRAM | 16 / 46 |
| DSP equivalent | 11.25 / 24 |
| Routed Fmax | 27.555 MHz |
| Worst setup slack | +0.747 ns |
| Setup total negative slack | 0 |

The previous hierarchy exceeded the register budget; this one uses synchronous
inferred BSRAM. The tool reports generic clock routing (`PR1014`), and UART/reset
are asynchronous; board clock behavior remains a physical check. The tool's
power estimate is **not** an energy measurement. The generated
[bitstream](../../hardware/releases/phase2/tinyml_v2.fs) has SHA256 in
[gowin-summary.json](evidence/phase2/gowin-summary.json) and is ready for
programming after board identity and pinout are confirmed. A rebuild should
produce an independently verified route and bitstream; Gowin output bytes can
vary across tool installations.

## Remaining gate

Connect the target board, identify its PCB revision and programmer/serial path,
program the archived `.fs`, verify `CAPS` and full SRAM readback, then run
`tools/phase2/verify_board.py` for 1,000 exact physical jobs. Archive serial logs,
board/bitstream identity, output/counter results and supply instrumentation.
G2 becomes complete only after those checks and the open phase-0 lab items pass.
The current v2 hardware accepts FC, ReLU and shape-preserving copy only; KWS,
VWW, AD and CNN output claims remain software or future hardware work.
