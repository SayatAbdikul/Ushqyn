# Phase 6 priority campaign — 2026-09-27

**Current state:** confirmation passed all 88 runs; KWS completed with
4,514/4,890 correct classifications and zero output mismatches. At the user's
request, VWW was interrupted after 212 checked samples to prioritize the
[direct-writeback optimization](PHASE_6_WRITEBACK.md). Saved evidence is intact.
Resuming this campaign reloads its original spatial image and starts VWW at
zero-based sample index 212. Results from a newer image require a separate
accuracy campaign and must not be appended here.

The completed screening found spatial SIMD with simple SRAM retention to be
the fastest tested configuration. This campaign confirms that improvement
across independent programming sessions, then checks all frozen KWS and VWW
accuracy samples on that configuration. The user authorized selection and
execution of the highest-priority tests after reconnecting the board.

## Scope and expected duration

| Stage | Coverage | Initial estimate |
| --- | --- | --- |
| Independent confirmation | Original and spatial hardware, both models, common retention schedule, two additional sessions; 80 timings and 8 warmups | 10–15 minutes |
| Full KWS accuracy | 4,890 distinct samples | 20–30 minutes |
| Full VWW accuracy | 10,961 distinct samples | 9.5–11 hours |

Allow approximately **10–12 hours total**. These estimates include host work
and UART transfers; device-only latency is substantially lower. Reassess the
estimate from saved sample durations once full accuracy starts. All runs use
the existing 20.25 MHz clock and 750,000-baud UART. Both models remain resident
in on-board SDRAM throughout accuracy evaluation.

Confirmation uses ten timings and one warmup per model/hardware/session, with
deterministically randomized hardware and model order. Together with the
completed screening, this provides three programming sessions. Each session
must retain at least 1.3× geometric-mean speedup across the models and neither
model may regress by more than 5%. Within-configuration timing spread must be
at most 2%; variation among session medians must be at most 3%.

Each accuracy sample is compared byte-for-byte against the independent
centered-integer evaluator. Final classification totals must reproduce the
frozen software results: **4,514/4,890 KWS** and **9,239/10,961 VWW**. Any
numerical mismatch, protocol error, failed counter check or ambiguous launch
stops the campaign without automatic retry. The full accuracy stages run
only after confirmation passes.

## Execution and evidence

Runner: `tools/phase6/run_priority.py`. Its safeguards and the screening
runner's safeguards pass 26 boardless tests together.

```sh
# Read-only preflight; no board access.
.venv/bin/python3 tools/phase6/run_priority.py

# Fresh campaign; requires exclusive JTAG/UART ownership.
.venv/bin/python3 tools/phase6/run_priority.py --run

# After inspecting an interrupted accuracy run and reconnecting the board.
.venv/bin/python3 tools/phase6/run_priority.py --run --resume
```

Live evidence is in `work/phase6/priority-v1/`:

- `plan.json`: image, source, fixture and dataset identities and test order.
- `report.json`: atomic live progress, stage results and failures.
- `confirmation.jsonl`: timing and correctness records for confirmation.
- `kws-accuracy.jsonl` and `vww-accuracy.jsonl`: checked sample outputs,
  expected outputs, sample identities, predictions and timing counters.
- `program-*.log`: programming records.

Console output is in `work/phase6/priority-v1.log`. The launched process uses
`caffeinate -i` to prevent idle sleep. Keep the Mac powered and the board
connected; closing a laptop lid can still suspend the host.

Every completed sample is flushed and fsynced before its progress update.
Resume requires complete passed confirmation, unchanged artifact identities,
valid contiguous sample records and no prior numerical mismatch. It reloads
the same image and both models, then starts at the next unrecorded sample.
Incomplete confirmation requires investigation and a fresh output directory.
Corrupted or incomplete record lines are rejected rather than silently removed.

## Limits

This campaign tests the winning Phase 6 image. It does not close original-image
Phase 5 accuracy, the complete G6 research gate, measured energy, SOTA or a new
10,000-job alternating stability test. Existing Phase 5 results remain intact
and its older queues remain paused. No further board tests start automatically
after these three stages. The authoritative outcome is the live report;
creating this document does not indicate that the campaign has passed.
