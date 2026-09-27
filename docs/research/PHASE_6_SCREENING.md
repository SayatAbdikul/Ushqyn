# Phase 6 physical screening — 2026-09-26

The user authorized this bounded campaign after the 10,000-job Phase 5 switch
campaign passed its independent audit. The original Phase 5 full-accuracy and
B1/B2 queues are paused. This supersedes the earlier physical plan's requirement
to finish every Phase 5 campaign before reprogramming, for this screening only.
The full 17,850-timing matrix is not being run.

## Scope and counts

Use the original Phase 5 image, the `c256p1` cache candidate, and the `spatial`
SIMD candidate. All run at 20.25 MHz, with the same frozen INT8 KWS/VWW models,
32-KiB scratchpad, SDRAM and 750,000-baud protocol. Compare original
preferred-half/overlapped scheduling and simple SRAM retention on every image.
The searched scheduling policies, INT4 and wider MAC engines are excluded.

1. Correctness: three images × two models × two schedules × pinned/stress inputs
   = **24 inferences**. Check final outputs and every supplied intermediate
   tensor against the frozen independent integer oracle.
2. Timing: three images × two models × two schedules × ten repeats = **120
   timed inferences**, plus **12 warmups**. No diagnostic snapshots in timing.

All 24 correctness checks must pass before timing starts. Correctness bring-up
uses original, cache, then spatial hardware. Timing hardware order and fixture
order are randomized with a fixed seed of 20260926. Each stage programs the
verified image into volatile FPGA SRAM and verifies uploaded payloads/commands.

Each successful inference is flushed and fsynced to JSONL. The summary is
replaced atomically after each result. Programming logs, exact artifact hashes,
model upload/readback time, UART input/run/output time, device counters and
checked tensor hashes are saved separately. Any mismatch, protocol failure,
timeout, incomplete sequence or inconsistent counter stops the campaign;
launches and programming are never automatically retried.

## Screening decisions

- Correctness requires zero mismatches and no execution/protocol failures.
- Timing spread `(maximum - minimum) / median` should be at most 2% within each
  configuration. Investigate larger variation before adding repetitions.
- A candidate merits independent confirmation if its geometric-mean speedup
  across both models is at least 1.3× under the same schedule, and neither model
  slows by more than 5%. Report individual model results and both schedules.
- Compare measured cycles with the already-frozen predictions. Median absolute
  prediction error should be at most 10%, and maximum at most 20% for this small
  screen. Larger error motivates cost-model investigation, not deletion of an
  otherwise correct physical hardware improvement.
- Possible confirmation: original and winning hardware, both models, common
  retention schedule, two additional independently programmed sessions, ten
  timings and one warmup per configuration (**80 timings + 8 warmups**). Require
  persistent speedup and investigate session median variation above 3%.

The runner does **not** start confirmation or expansion automatically. Screening
cannot establish full-dataset accuracy, long-duration stability, measured energy,
SOTA, strong-baseline eligibility or G6 closure. It does not change the compiler
or quantization contract.

## Execution and evidence

Runner: `tools/phase6/run_screening.py`. Thirteen boardless safeguard tests pass
in `test/phase6/test_screening_runner.py`, including corrupted artifacts,
incomplete execution, impossible counters, output mismatch, protocol error,
ambiguous-launch non-retry and exact campaign counts/stage ordering.

```sh
.venv/bin/python3 tools/phase6/run_screening.py
.venv/bin/python3 tools/phase6/run_screening.py --run
```

The first command is read-only preflight, including the Phase 5 switch audit.
The second accesses JTAG/UART and requires exclusive board ownership. Evidence
is under `work/phase6/screening-v1/` and console output under
`work/phase6/screening-v1.log`. The runner refuses an existing output directory.
It preserves copies of the completed Phase 5 report and all 10,000 job records
under `phase5-switch/`. Those results remain valid after later reprogramming.

The authoritative live status is `report.json`; creating this document does
not mean that the physical campaign has passed. If interrupted, preserve its
directory. There is no automatic resume/skip implementation in this first
runner; inspect the completed evidence before defining a continuation.
