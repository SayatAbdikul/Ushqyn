# Phase 2 implementation record — 2026-09-23

**G2's hardware checks now pass on the connected Tang Nano 20K.** The
reset-corrected target 8196 completed MLP → SmallCNN → MLP, with 1,000 exact
inferences and full SRAM image readback at each stage, without reprogramming
between stages. Both MLP stages had zero integer mismatches or protocol errors.
See the [physical record](PHYSICAL_BOARD_STATUS.md) for raw evidence and the
new routed release. [G0 is now closed](PHASE_0_CLOSURE.md) for its research-contract scope;
board power and energy are unmeasured. The original Phase 2 simulation and
route results below remain historical evidence for target 8194.

| Roadmap item | Result | Evidence and limit |
|---|---|---|
| H01 | Complete for v2 target | One [target manifest](../../hardware/targets/tang_nano_20k_v2.json) drives the compiler ABI, generated `target_pkg.sv`, Verilator source list and Gowin build. Legacy v1 modules remain only for historical tests and are excluded from the v2 source list. Strict lint and target regeneration pass. |
| H02 | Complete for FC subset | 64-byte v2 descriptors encode 24-bit addresses and independent 16-bit convolution geometry; all unsupported spatial geometry rejects. Encoding/decoding, alignment, bounds and 32-bit counts/strides are tested. One synchronous SRAM request may be accepted per cycle, with explicit ready, read-valid, abort and error semantics. |
| H03 | Complete for FC/ReLU subset | Eight signed INT8 MAC lanes, per-channel corrected INT32 bias and static multiplier/shift/zero point, one elastic requantizer. Directed/random 64-bit rounding tests and independent FC tests include extreme zero points, tails, stalls and aborts. Accumulator overflow rejects. CNN kernels remain in phases 3–4. |
| H04 | Complete | Framed UART 115200 8N1; `CAPS`, `READ`, `WRITE`, `RUN`, `STATUS`, `ABORT`, `RESET`. Protocol regressions pass in RTL. Physical load/change-input/run/read and model switching pass on the corrected board wrapper. |
| H05 | Complete for the on-chip MLP gate | Two 1,000-job physical MLP stages match every INT8 logit, with full image readback and reconciled counters. Current target 8196 records 7,316 core cycles and 10,112 MACs per MLP job. New route: 27.233 MHz Fmax, +0.318 ns setup slack at 27 MHz. Power is outside this completed functional gate and remains unmeasured. |
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
[gowin-summary.json](evidence/phase2/gowin-summary.json). It is retained for
provenance and has the old incorrect reset polarity; use the
[reset-corrected physical release](../../hardware/PHYSICAL.md). A rebuild should
produce an independently verified route and bitstream; Gowin output bytes can
vary across tool installations.

## Physical closure and remaining scope

The H05 execution/readback/counter/route requirements are satisfied by the
current target's physical MLP results. Its broader kernels retain the same
numerical-v2 contract. The initial silent UART was caused by treating KEY1
as active low; the corrected wrapper follows the board's active-high button.
Configuration startup and button reset are covered in board-top simulation.

The FPGA top marking supplied by the user matches `GW2AR-LV18QN88C8/I7`;
JTAG identifies revision C. The PCB revision is unknown and the user has no
power instrument. Instrument access, AD provenance and novelty/baseline work
remain open under G0/later phases. Full KWS/VWW/AD deployment still requires
the unfinished external-memory and tiling work.
