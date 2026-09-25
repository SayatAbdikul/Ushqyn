# Phase 4 closure — 2026-09-25

**G4 passes its engineering acceptance gate on the physical Tang Nano 20K.**
The same release executes complete KWS and VWW schedules autonomously from
on-board SDRAM, including compiler-managed hybrid ping-pong transfers. Every
node matches the independent integer oracle for the pinned deterministic
input. Complete-set accuracy, the balanced 10,000-job campaign, matched
research baselines and measured energy remain separate work.

Reproduce the [closure audit](evidence/phase4/physical-sequence-closure.json)
with `.venv/bin/python3 tools/phase4/audit_phase4_closure.py`. It checks the
bitstream, current RTL/IP hashes, timing, physical reports, net overlap benefit
and held-out prediction thresholds.

## Measured execution

The [autonomous release](../../hardware/releases/phase4-sdram/hs_tiled_autonomous.fs)
has SHA256 `ea4eed105933e87d308d42b4bfa581010e843c9c9b9f94e9c9e8e852c19b98fb`.
The [route record](evidence/phase4/physical-sequence-route.json) reports
21.897-MHz core Fmax at the **20.25-MHz operating clock**, zero setup TNS,
14,182/20,736 logic, 4,876/15,915 registers, 34/46 BSRAM and 19.75/24 DSP
equivalents. Additional BSRAM stores commands; engine scratchpad remains 32 KiB.

| Workload | Exact nodes | Tiles | Sequential | Overlap | Matched speedup | Input → result wall time |
|---|---:|---:|---:|---:|---:|---:|
| KWS | 22/22 | 28 | 284.479 ms | 280.940 ms | 1.0126× | 0.396 s |
| VWW | 58/58 | 107 | 928.209 ms | 903.750 ms | 1.0271× | 6.119 s |

Numbers are medians of three physical repeats per mode. Device execution
includes command dispatch, compute, DMA, refresh stalls and dependency waits.
Both modes use identical tile placement and one bitstream; their observed
latency ranges do not overlap. Diagnostic snapshot copies are excluded from
both timed modes. Wall time includes input upload, launch/status traffic and
final readback; it excludes preprocessing and model/program upload plus their
verification. These are repeated single-input latencies, not sustained FPS.

[KWS](evidence/phase4/physical-sequence-kws.json) and
[VWW](evidence/phase4/physical-sequence-vww.json) retain node hashes, placement
and raw counters. VWW node checks came from an
[initial run](evidence/phase4/physical-sequence-vww-initial.json) whose later
timing repeat was interrupted by macOS USB. Its resumed run reverified every
immutable model/descriptor byte before collecting all six timings. An old
serial diagnostic process was removed; the final resumed run records zero
USB recoveries. The original interruption remains in the evidence.

The [DMA profile](evidence/phase4/physical-sequence-dma.json) passes 78 transfers:
13 lengths/addresses, both directions, three repeats. Full 32-KiB transfers
sustain median **19.00 MiB/s toward SRAM** and **18.12 MiB/s toward SDRAM**,
versus about 5.54 MiB/s previously. The adapter uses aligned 64-byte bursts,
read-ahead and masked write combining. Short transfers have proportionally
larger fixed and refresh overhead.

## Gate evidence

| Item | State | Evidence |
|---|---|---|
| C04, complete frozen arithmetic | Passed | All primary node snapshots exact; original and 28 new directed geometries pass randomized-stall RTL and physical tests. |
| D01, SDRAM initialization/addressing/refresh | Passed | Retained full-8-MiB and ≥1-GiB randomized [HS-controller evidence](evidence/phase4/physical-sdram-hs.json), plus new burst boundary, mask/coherence and refresh-contract checks. |
| D02, burst DMA and double buffering | Passed | Compiler-generated half-bank schedules with full-SRAM fallback; full-model net overlap benefit; physical live-region rejection and abort/reset/transfer recovery. |
| D03, tiled inference | Passed at its defined fixture/shape boundary | KWS/VWW, known MLP/SmallCNN fixtures, larger-than-tile FC and frozen ToyCar AD shapes. AD uses synthetic weights, not trained-model quality evidence. |
| P01, measured cost database | Passed for latency and memory feasibility | Static prediction, prospective holdouts, measured DMA fit, routed resources and exact placement checks. Physical energy is unavailable. |

The [cost report](evidence/phase4/physical-sequence-costs.json) replaces the
counter-dependent fit with a descriptor/cache-address walk. All **28 prospective
configurations × 3 physical repeats** have exactly predicted isolated engine
cycles. Both models' aggregate uncontended engine cycles are also exact.
Prediction consumes no weights, input values, measured engine counters or
fitted per-model coefficients. Independently counted randomized SRAM stalls
reconcile with this prediction in RTL tests.

DMA fitting holds out lengths 7, 9, 65, 256 and 4,096 bytes before fitting.
Held-out median/p95 errors are **0.09%/3.98%** toward SRAM and **2.35%/10.41%**
toward SDRAM, meeting the ≤10% median and ≤20% p95 development targets.
Concurrent SRAM contention is separately measured; the isolated engine model
does not automatically predict overlap stalls.

The [physical recovery checks](evidence/phase4/physical-sequence-checks.json)
cover 25 masked writes, 125 neighboring-line readbacks, bank/end boundaries,
live-region error 9, ABORT while busy and a correct fresh transfer after RESET.
DMA abort errors retain the existing sticky-until-next-transfer behavior.
The [simulation transcript](evidence/phase4/physical-sequence-suite.txt.gz)
includes compiler, arithmetic, DMA, sequencer, refresh and burst-adapter tests.
The test-only HS command model validates adapter logic, not electrical timing.

The final image also passes fresh [MLP/SmallCNN replay](evidence/phase4/physical-sequence-legacy.json)
and [synthetic/AD-shape checks](evidence/phase4/physical-sequence-synthetic.json),
including every frozen ToyCar node and weights larger than a scratchpad tile.

## Remaining scope and reproduction

See [the autonomous interface](../../hardware/phase4_sdram/AUTONOMOUS.md) for
registers, command records, compiler behavior and reproduction commands.
`make p4-test PYTHON=/absolute/path/to/.venv/bin/python3` runs boardless checks.
The programmer log and archived bitstream identify the temporary SRAM image
used for all new physical reports.

The current 20.25-MHz clock and 281/904-ms execution remain below the broader
roadmap aspirations of 27/54 MHz and 20/100 ms. These are development targets,
not fixed G4 FPS requirements or publication acceptance thresholds. Phase 4
closure establishes the memory/scheduling foundation; it does not establish
SOTA performance. Gowin power analysis is an estimate. No physical meter was
available, so measured energy and an energy cost model are explicitly deferred.

Earlier [host-controlled results](evidence/phase4/physical-host-overlap.json)
and the [previous route](evidence/phase4/physical-tiled-host-overlap-route.json)
remain historical evidence. Their two-word DMA and UART-per-tile timings do
not describe this autonomous release.
