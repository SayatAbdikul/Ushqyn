# Phase 5 working record — updated 2026-09-25

**Physical addendum:** the SDRAM-connected board has now passed one complete,
byte-exact KWS input and one complete, byte-exact VWW input on the same image,
with every intermediate node checked. It also passed the prior on-chip
MLP/SmallCNN tensors through SDRAM and a directed compute/DMA overlap. See
the [Phase 4 physical evidence](PHASE_4_STATUS.md). This is one input per
primary model, not the complete accuracy or 10,000-job switch workload below.

**P5 is in progress; G5 is open.** This repository pins the primary-model
switch workload, but its complete accuracy sets and 10,000 alternating board
jobs have not run. There is no AD full-set result or measured B1/B2/B3
comparison. The physical Phase 4 image proves model execution; it does not
establish a Phase 5 or SOTA accelerator claim.

## Reproducible B01 switch workload

`make p5-plan` reads the frozen accuracy-record manifests, rejects duplicate
IDs and calibration overlap, and creates a deterministic 10,000-job plan.
The jobs strictly alternate KWS and VWW. KWS has only 4,890 accuracy samples,
so its 5,000 stress jobs include every sample once and 110 repeated samples;
VWW uses 5,000 distinct samples from its 10,961-sample accuracy split. A
fixed pseudorandom permutation avoids an index-order subset. Every job pins
the sample ID, source index and preprocessed-feature SHA256. The compact
[plan evidence](evidence/phase5/switch-plan.json) records the JSONL SHA256,
manifest hashes, seed and selection counts. Generate the complete plan with:

```sh
python3 tools/phase5/schedule.py --output work/phase5/balanced-jobs.jsonl
```

This is a **planned** switch stress sequence, not 10,000 executions. Full
accuracy must separately process all 4,890 KWS and all 10,961 VWW samples;
neither the stress subset nor software-v2 scores certify board quality. When
the board path exists, run the *same bitstream* throughout, record model-load
and reflash events, compare every raw INT8 logit with the independent integer
reference, and keep classification correctness separate from implementation
mismatches. Preprocessing and host argmax must follow the frozen boundaries.

## Evidence inventory

`make p5-audit` reruns the plan regression tests and the
[readiness audit](evidence/phase5/readiness.json). The audit checks the
kernel bitstream and canonical inventory SHA256 values against Phase 4's
frozen evidence. The later on-chip physical release and its raw artifact hashes are checked separately; that clears board-access readiness without claiming complete primary-model execution. It also checks that the existing full-set **software**
accuracy reports match the pinned split counts and hashes. Current software
results are KWS 4,514/4,890 (92.31%) and VWW 9,239/10,961 (84.29%). They
are not hardware measurements; the VWW split is research validation, not an
official MLPerf certification.

The archived readiness audit predates the new Phase 4 physical image and
therefore still lists integrated SDRAM/DMA and primary-model board execution
as missing. Those two one-input checks now pass, but the audit must be updated
before it is used as a current inventory. Full-set model images/accuracy
payloads, AD and baseline reports remain missing. File presence alone cannot
certify the contents of future results. The `g5_certified` field is false.
The current 32-KiB scratchpad is below even KWS's 33,216 bytes of packed
persistent parameters; VWW needs 260,640 bytes of such parameters and has
six full-layer input/output pairs that exceed the scratchpad. Tiling and real
external memory are necessary for these pinned graphs.

## B02 and B03 comparison contract

AD remains a planned third workload. The frozen ToyCar model has a 640-element
input and reconstruction output. Its model/data manifest says the referenced
sample contents and full ROC-AUC are unavailable locally. B02 requires the
exact linear reconstruction output, the pinned anomaly score and scoring-time
boundary, streamed weights, independent integer comparisons and complete-set
ROC-AUC ≥0.85. Do not replace this with classification argmax or a synthetic
subset result.

For B03, use one frozen model, weight set, INT8 arithmetic, eight-lane engine,
host boundary, scratchpad/SDRAM capacity, clock and board setup for all three
policies. B1 is a tuned conventional layer-wise tiled schedule; B2 adds only
physical live-range/bank placement to that baseline; B3 is an adaptation of
the pinned DeFiNES policy described in the
[novelty decision](novelty_matrix.md). Preserve feasible cache/recompute
modes, record any legality repair and its cost, and give each policy the same
declared wall-clock tuning budget. Publish all feasible candidates and
failures. Report complete-model latency, host and DMA time, bytes transferred,
routed resources, achieved common clock, exact output mismatches and full-set
accuracy for both KWS and VWW. No comparative speedup or energy number exists
yet. A post-synthesis report supplies resource/timing estimates, not live
SDRAM behavior, end-to-end latency or power.

## Work required to close G5

1. Close G4's remaining compiler-managed ping-pong/streaming DMA and reliable
   static cost-model gates. The integrated SDRAM image and one-input KWS/VWW
   node-value checks are complete.
2. Materialize the frozen KWS/VWW model images and accuracy payloads using
   [the pinned benchmark recipes](../../benchmarks/README.md). Validate all
   source, preprocessing and split hashes before using them.
3. Run the two complete primary accuracy sets and the pinned 10,000-job
   alternating sequence on one release bitstream, with exact integer output
   comparison, no reflashing and repeatability evidence.
4. Implement and tune B1/B2/B3 fairly on that same hardware; report all
   complete-model results and infeasible cases. This is mandatory for G5.
5. Run AD with reconstruction scoring and full ROC-AUC to complete the planned
   suite, then assess the G5 gate. Physical board measurements remain necessary
   for deployment, latency and energy claims.
