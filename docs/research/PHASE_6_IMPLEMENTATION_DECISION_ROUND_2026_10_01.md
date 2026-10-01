# Phase 6 implementation decision round — 1 October 2026

The authorized boardless round is complete. The frozen VWW factor candidate gives only a 0.21% cycle reduction against the strongest optimized dense control. The tested selective interruption policy does not outperform both uniform controls across the frozen sensitivity suite. The autonomous resident controller passes native correctness checks but fails FPGA placement. These results do not justify advancing these specific candidates as the central architecture contribution of a Q1 paper.

Work ran in parallel across factor compilation, policy evaluation and controller implementation, followed by root integration and an independent raw-evidence audit. All changes are isolated research drivers, fixtures, copied RTL and reports. The audit confirms that 165 production/compiler/RTL/target/benchmark-manifest files and nine frozen model-selection files retain their start-of-round hashes. Existing user edits were preserved.

| Question | Completed evidence | Decision |
| --- | --- | --- |
| Does the frozen factor graph retain a useful gain after equally strong dense optimization? | Same 12 compiler configurations per graph; exact native validation and bundled-source binary regeneration | Stop this VWW candidate as the main architecture thesis |
| Does selective splitting improve deadline behavior over ordinary splitting with the same checkpoint optimizer? | 828 frozen arrival cases; 52,992 schedules with calibrated metadata lookup costs | No overall advantage over both uniform controls; retain as an overhead/storage tradeoff |
| Can resident interruption run without host service and fit the current FPGA? | 64 native checks and matched baseline/candidate Gowin implementation | Correct in tested native cases; current implementation fails placement |

The factor comparison applies the existing exact channel grouping, final constant folding and channel compaction to both graphs. It keeps the incumbent four spatial region cuts and gives each graph the same 12 final-region height/width/cache/residency settings. Three settings per graph fit and tie; the other nine fail explicit SRAM or command capacity checks. This restricted search is fair within its declared space, but does not establish global schedule optimality.

| Selected VWW execution | Dense | Frozen factor |
| --- | ---: | ---: |
| Fixed RAM cycles | 2,014,205 | 2,009,959 |
| Stalled RAM cycles, seed 6063 | 2,059,395 | 2,055,195 |
| Useful MACs after exact transformations | 3,733,073 | 3,707,612 |
| Engine RUNs | 42 | 49 |
| Command bytes | 15,568 | 16,464 |
| Scheduled DMA bytes | 212,558 | 213,334 |

The reductions are 4,246 cycles (0.2108%) with fixed RAM and 4,200 cycles (0.2039%) with sampled stalls. The dense control reproduces the strongest existing B3 commands and payload byte for byte. After existing transformations, factor MAC savings are only 0.682%; the earlier large logical saving was measured before the strong dense transformations. Extra descriptors, intermediate materialization and transport absorb most of the remaining benefit.

Six feasible candidate timing runs, 16 selected validation runs and two identity-LUT ablation runs pass. Selected validation crosses a real pinned input and a distinct full-range INT8 stress input with both RAM modes and timed/diagnostic streams. Diagnostics check every engine RUN output plus the final tensor. Signed factor latents keep their exact INT8 requantization; proved identity clips let the existing backend lower them, and identity LUT tags and unused LUT DMA are removed. The arithmetic RTL is unchanged. Both winning command/payload binaries regenerate exactly from the bundled compiler sources. A missing lazy compiler dependency was found by this portability check, added to the source inventory, and the matched comparison rerun with unchanged results.

This round does not retrain factors or reuse observed held-out examples for selection. Earlier frozen accuracy evidence remains separate. It does not advance the older generic AD factor result through this VWW spatial backend, evaluate all ranks or architectures, or measure board latency and energy. See the [factor report](/Users/sayat/Documents/GitHub/tinyML_accelerator/work/phase6/representation-optimized-v1/report.json) and [experiment notes](/Users/sayat/Documents/GitHub/tinyML_accelerator/work/phase6/representation-optimized-v1/README.md).

The policy protocol froze 828 single, periodic and burst arrival cases across all six ordered KWS/VWW/AD pairs before new candidate measurement. Original, uniform 25k, uniform 50k and selective-long-25k streams receive the same finite-catalogue checkpoint optimizer and conservative live 8-byte granule save/restore rule. Selective splitting applies only to eligible independent output channels/rows with original RUN duration above 75k cycles. The optimizer minimizes a maximum blocking-plus-save surrogate within 32 KiB resident program capacity, with minimum checkpoint code as a tie break. Its optimality concerns that finite catalogue and surrogate only.

The final replay preserves every drained WAIT and charges the five-state program-memory metadata read sequence, including unsuccessful full table scans. Lookup-cycle counts match all eight native normal-urgent witnesses. DMA costs of 4/8/16/32 cycles per 8 bytes and fixed dispatch allowances of 128/512 cycles remain sensitivity assumptions. The full grid has not been executed in resident RTL.

| Calibrated trace policy | Total missed deadlines | Zero-miss schedules | VWW/AD resident bytes |
| --- | ---: | ---: | ---: |
| Original | 16,713 | 9,609 | 21,872 |
| Uniform 25k | 14,837 | 10,458 | 32,720 |
| Uniform 50k | 15,012 | 10,344 | 30,816 |
| Selective long 25k | 15,282 | 10,250 | 26,896 |

Each policy has 13,248 schedules. Selective has fewer/more misses in 199/461 cases against uniform 25k and 167/321 against uniform 50k. It rescues 732 original schedules to zero misses, but loses 91 previously zero-miss original schedules. Its lower average overhead and smaller VWW/AD resident footprint are useful tradeoffs, not dominance. Cases are correlated sensitivity coverage rather than deployment frequencies or independent statistical trials.

The selectively split VWW/KWS streams pass four new standalone native output checks; AD reuses an identical baseline in both RAM modes. Fixed-RAM VWW overhead is 0.6233% and KWS overhead is 1.9465%. All resident sizes include two model programs, sparse save/restore DMA programs, 16-byte header/site records and a 32-byte stress-only clobber prefix. Unrestricted checkpoint code at all drained boundaries would require 133,840–238,832 bytes for three jobs; budgeted three-job allocation and controller execution remain untested. See the [policy report](/Users/sayat/Documents/GitHub/tinyML_accelerator/work/phase6/deadline-policy-v1/scan-calibrated-report.json) and [assumptions](/Users/sayat/Documents/GitHub/tinyML_accelerator/work/phase6/deadline-policy-v1/README.md).

The trace model maintains an EDF urgent queue. The tested controller admits one outstanding urgent request and rejects notifications during save/urgent/restore. No queue/admission adapter or its hardware and timing cost is implemented. Repeated jobs also reuse immutable inputs. Periodic/burst trace results therefore describe the declared model and cannot be presented as measured behavior of this RTL.

The resident controller uses copied sequencer/control RTL and the unchanged selected arithmetic engine. One CRC-validated notification triggers a precompiled save → urgent AD → restore → VWW resume chain at an allowed fully drained WAIT. There is no host UART activity after admission. A late-arrival bug was repaired so an accepted request after the last checkpoint executes the urgent job after background HALT, including the admission/completion race.

The original 32 native checks cover four policies and two RAM modes, each with uninterrupted execution, normal urgent execution, destructive whole-scratchpad clobber, and omitted-restore negative control. All 24 positive cases are exact and all eight negative controls detect corruption or an execution error. Four unsafe busy writes are rejected in every run. Another 32 tests cover early, middle, after-last-site and completion-race arrivals; all produce exact VWW and AD outputs. Sixteen directly exercise late completion fallback. The completion-race release is 20 cycles before the same harness's uninterrupted end; the after-last-site release uses a scaled trace estimate and asserts actual fallback. The initial mismatched stalled timing-reference failure is preserved separately from the successful campaign.

At the original selected arrival, normal urgent fixed-RAM responses are 374,822 / 315,639 / 339,550 / 301,034 cycles for original / uniform 25k / uniform 50k / selective. The latter three meet the 353,303-cycle relative deadline. All four miss it in the sampled stalled RAM mode. These are particular native witnesses, not worst-case bounds. Artificial clobber adds separate latency and is excluded from the trace cost model.

| Matched 27 MHz Gowin implementation | Baseline | Resident candidate |
| --- | ---: | ---: |
| Logic | 18,901 / 20,736 | 20,483 / 20,736 |
| Registers | 5,230 / 15,915 | 5,445 / 15,915 |
| Clusters (CLS) | 10,090 / 10,368 | 10,368 / 10,368 |
| BSRAM | 38 / 46 | 38 / 46 |
| DSP | 9.5 / 24 | 9.5 / 24 |
| Routed core Fmax | 28.826 MHz | Unavailable |
| Result | Route passes; zero setup/hold violations | PR0003: 48 equivalent LUTs unplaced |

The current prototype cannot proceed to board measurement. The first baseline run crashed in the filesystem sandbox; its log is retained separately, and the approved rerun succeeded. Native tests use the v2 UART path; the burst UART copies receive synthesis/placement coverage only. Shared engine/core sources match across the native and physical builds. Source snapshots, fixtures, reports and logs are pinned in the [resident report](/Users/sayat/Documents/GitHub/tinyML_accelerator/work/phase6/deadline-resident-v1/report.json).

Prior art substantially narrows the possible claim. [DERCA, RTSS 2025](https://peipeizhou-eecs.github.io/publication/2025_rtss_clare/2025_rtss_clare.pdf) already studies intra-layer preemption, on-chip EDF, persisting versus recomputing state and joint point/dataflow selection. Its metadata storage accounting also makes a resident program-capacity constraint alone insufficient evidence of novelty. [ART, GLSVLSI 2025](https://peipeizhou-eecs.github.io/publication/2025_glsvlsi_art/) covers FPGA transformation and preemption-point placement. The [focused eight-source comparison](/Users/sayat/Documents/GitHub/tinyML_accelerator/docs/research/PHASE_6_DECISION_ROUND_PRIOR_ART_2026_10_01.md) records which full papers, author slides and abstracts were accessible. It is not an exhaustive literature review.

The next research goal should be a different measurable mechanism that survives a comparison with the optimized dense baseline and the closest prior art. Further rank sweeps or checkpoint-density tuning alone have weak justification from these results. A small controller-area redesign could pursue engineering feasibility, but fitting the FPGA would still not establish novelty. Any new central hypothesis should identify what existing work cannot do, specify a controlled comparison, and demonstrate a meaningful effect across more than one workload before committing to board/energy evaluation. The tested candidates and settings are exhausted for this round; the entire research space is not exhausted.

The independent [audit](/Users/sayat/Documents/GitHub/tinyML_accelerator/work/phase6/decision-round-v1/audit.json) reconstructs every raw completion's misses, lateness, responses and work conservation in both 52,992-record replays; checks 124 serialized checkpoint records and their DMA programs; compares native scan witnesses; validates the 64 controller reports and effective negative controls; and checks compiler regeneration, source hashes and physical logs. The policy's own audit compares its dynamic program with 120 small brute-force cases. Python processes stay below 1 GiB RSS: final factor comparison about 95 MiB, policy below 346 MiB and root audit about 416 MiB. The largest reported physical-tool child RSS is about 661 MiB. Numeric threads are one, native build jobs two and physical runs serialized.

Reproduction entry points are [representation_optimized.py](/Users/sayat/Documents/GitHub/tinyML_accelerator/tools/phase6/representation_optimized.py), [deadline_policy.py](/Users/sayat/Documents/GitHub/tinyML_accelerator/tools/phase6/deadline_policy.py), [deadline_resident.py](/Users/sayat/Documents/GitHub/tinyML_accelerator/tools/phase6/deadline_resident.py), [finalize_decision_round.py](/Users/sayat/Documents/GitHub/tinyML_accelerator/tools/phase6/finalize_decision_round.py) and [decision_round_audit.py](/Users/sayat/Documents/GitHub/tinyML_accelerator/tools/phase6/decision_round_audit.py). Their README commands specify required fixtures and stages. Toolchain versions and original benchmark inputs remain external prerequisites for fresh native compilation/model regeneration. No production deployment, board programming, commit or PR was performed.

The [durable evidence pack](/Users/sayat/Documents/GitHub/tinyML_accelerator/docs/research/evidence/phase6/decision-round-v1/README.md) contains raw records, fixtures, source snapshots, logs, frozen selections and reports with a streamed SHA-256 manifest and archive verification. It excludes original dataset maps and generated build binaries. It supports evidence inspection; it is not a complete redistribution of datasets and proprietary FPGA tools.
