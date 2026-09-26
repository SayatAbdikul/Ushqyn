# Phase 0 closure — 2026-09-26

**G0 is complete for the declared research contract.** R01–R04 now have the
claim decision, pinned benchmark evidence, preserved starting baseline and a
concrete lab/measurement plan required by their definitions of done. This gate
does not certify SOTA, official MLPerf submission eligibility, AD INT8 accuracy,
measured energy, a second FPGA, or completion of Phase 5.

| Item | Completed evidence |
|---|---|
| R01 claim and prior work | [Candidate, hypothesis and eligibility](novelty_matrix.md), [September 26 review and source inspection](PRIOR_WORK_AUDIT.md), and R05's explicit pivot. DeFiNES is B3; broad fusion/memory/port-awareness novelty is rejected. |
| R02 benchmark freeze | KWS/VWW/AD source, evaluator, operator inventories and conversion parity; complete declared evaluation/calibration provenance in `benchmarks/manifests/`. AD's missing raw and feature hashes are now materialized. |
| R03 starting evidence | Original commit `3aa5fe8`, dirty-path/environment records, 58-test starting result, five semantic defects and historical Gowin evidence remain preserved under `evidence/`. No historical estimate has been relabeled as a measurement. |
| R04 lab readiness and plan | Physical minimal UART passes 1,280 bytes; programmer, specimen, device, constraints and tool flow identified; [measurement plan](LAB_MEASUREMENT_PLAN.md) specifies the acquisition route, supplies, marker, experiment and uncertainty. |

## What was added for AD

The fixed upstream model/evaluator revision remains
`4addd0fa08d216e20637637874e084895f289da4`. All **248 recordings** in the pinned
AD evaluation CSV were reconstructed from public ToyCar WAVs. Each supplies 196
overlapping 640-feature windows: **48,608 model inputs**. Calibration is a
prospectively fixed **112-recording research subset** of the upstream 1,400-item
list: 16 evenly spaced entries per machine ID, giving **21,952 windows**.
Selection was fixed before quality evaluation, with no accuracy-based tuning.
It is not a claim that all 1,400 upstream calibration recordings were used.

The [raw manifest](../../benchmarks/manifests/ad.raw-data.json) records both
public archive URLs/DOIs, publisher MD5 values, per-member CRC, and the SHA256 of
every retrieved WAV. Partial downloads verify each member; the whole-archive
MD5 is explicitly **not** verified. The dataset's CC-BY-NC-SA-4.0 license is
recorded; source audio is downloaded locally, not committed or redistributed.

The [feature manifest](../../benchmarks/manifests/ad.data.json) fixes librosa
0.6.0, preprocessing source, environment lock, calibration selection, labels,
sample IDs, raw/feature content, and NPZ hashes. Every generated histogram was
checked against all 196 upstream feature windows. Calibration/evaluation raw
and feature overlap checks pass. The archived official preprocessed `.bin`
files were not available for equality testing: this is a reproducible research
reconstruction of the pinned recording list, not official MLPerf certification.

The original H5→ONNX conversion reproduced its original SHA256 and passed the
16 synthetic probes at `atol=1e-5`, `rtol=1e-4`; canonical inventory is identical
to the frozen 19-node, **264,192-MAC** inventory. An additional 32 real-input
probes pass source-H5 versus ONNX and canonical-ONNX checks at the same tolerance.
The source float model scores **0.880423 pooled ROC-AUC** across the 248
recordings; per-machine-ID macro ROC-AUC is **0.882275**. Scores average squared
reconstruction error over all 196 windows per recording. These are software
source-float results, not new INT8 or FPGA results. [Raw quality/parity record](evidence/phase0/ad-real-input-parity.json).

## Explicit gate boundaries

The original roadmap defines G0 as a benchmark/claim contract with correctly
labeled evidence, and R04's done condition asks for an instrument acquisition /
borrow **plan** and actual capability list. The plan is now concrete, but no
instrument has been obtained or reserved. E02 cannot close without measured
traces. Second-target access is also unconfirmed and remains an E03 dependency.
The PCB silkscreen revision stays unknown; FPGA revision C is not substituted
for it. The tested specimen is identified by its chip, USB serial, JTAG ID,
constraints and release hashes. No additional board access is needed to repeat
this Phase 0 audit, and active Phase 5 campaigns were left undisturbed.

Baseline implementation/tuning belongs to B03, AD quantized/physical evaluation
to B02, scheduler evidence to Phase 6, and independent reproduction / the final
literature refresh to E04. R01 is a reviewed decision, not an assertion of
exclusive novelty. VWW remains the declared training-recipe validation split.
These limitations remain visible after G0 closes.

## Reproduction and checks

Follow the [AD reproduction instructions](../../benchmarks/README.md), then run:

```sh
.venv/bin/python3 -m pytest test/research/test_phase0.py -q
.venv/bin/python3 tools/research/audit_phase0.py
```

The nine focused regression cases cover corrupt ZIP members, changed input
indices, malformed HTTP ranges, histogram framing, truncation and nonfinite
values. The [closure audit](evidence/phase0/closure.json) rechecks source hashes,
model/data links, counts/disjointness, local accuracy archives, AD calibration
and WAVs, real-input parity provenance and the physical UART release. Primary
calibration preparation is preserved Phase 1 evidence; it is not rerun by this
audit. It does not access UART or reprogram the FPGA.
