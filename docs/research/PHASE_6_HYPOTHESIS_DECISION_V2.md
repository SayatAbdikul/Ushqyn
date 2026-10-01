# Task 2: hypothesis tests after matched baselines

Date: 2026-09-28. **The bounded hypothesis screen is complete; the novelty gate
remains open.** The work produced a validated cost-model improvement and a safe
way to avoid redundant compilation. It did not produce a new best accelerator
result or demonstrate a novel architecture. No FPGA programming or new physical
inferences were performed in this screen.

## Evidence received from Task 1

The completed Task 1 branch at commit `53428af` contains a 30-run primary
campaign and a 45-run VWW finalist campaign, both with zero output mismatches.
Their plans, records, reports and seals were imported unchanged into
[`hypothesis-v2/task1`](evidence/phase6/hypothesis-v2/task1). Every file digest,
individual record digest, output and common image/clock identity was checked.
This is evidence reuse, not 75 new executions.

The primary campaign measured 44.513 ms KWS and 90.429 ms VWW for selected B4.
The later finalist campaign measured 90.424 ms VWW. B4 versus B2 was 1.02465×
geometric-mean throughput, or about 2.41% lower geometric-mean latency; their
KWS programs were identical. The applicable restricted depth-first baseline
can also select B4's program, so these measurements do not show an exclusive
scheduling capability. Task 1's physical exploration remains bounded.

The broader generic DeFiNES adaptation developed in this worktree has separate
native evidence. It must not inherit Task 1's board validation: its exact graph
transforms, command programs and exploration space differ. Its current native
incumbents remain KWS 1,181,858/1,190,362 and VWW 2,173,551/2,218,945 cycles
at memory stall seeds 0/6063. These are native RTL timings, not board timings.

## Tests and decisions

| Candidate | Test actually performed | Result and decision |
| --- | --- | --- |
| INT32 scratchpad live-state mechanism | Independent exact toy scheduling/arithmetic check plus inspection of the selected RTL | Reject for the current hardware: accumulators already remain in registers. The scratchpad constraint invented by the toy is inapplicable. |
| Finite command store as novelty | Primary-source audit and inspection of the existing emitted-byte constraints | Reject as a standalone claim. Finite microcode storage and command compression have prior art; the current baseline already charges actual command bytes. |
| Reuse equivalent compilation states | All 33,544 catalogue outcomes and top 24 proxy paths/model compared; 96 fresh real lowerings, 258 synthetic integer replays, all 4,968 avoided lowerings rerun | Correct bounded equivalence; 15.32% fewer KWS and 14.72% fewer VWW lowerings. Keep as compiler engineering, not an inference speedup or established novelty. |
| Cache-aware command cost | Predictions frozen before four new VWW heights; eight whole-model native executions at two stall seeds | Corrected a real proxy misranking. All four fixed-memory holdouts had zero-cycle prediction error within the declared descriptor family. Keep as a cost-model improvement available to every baseline. |
| Uniform aligned activation-base coloring | Functional cache traces for offsets 0, 8, …, 56 bytes | No change in hits, misses, gathering or padding. These shifts permute cache sets; no performance claim or executable allocation change follows. |

The [primary-source audit](PHASE_6_HYPOTHESIS_PRIOR_ART.md) covers DeFiNES,
COSMA, VTA, SERENITY, TASO, Exo and additional compiler work. In particular,
DeFiNES already caches tile evaluations and cache-level searches. Reusable
state descriptions and validated compiler transformations also have substantial
prior work. This bounded review does not prove the absence of all possible
novel contributions.

## Search experiment: what improved

The key includes graph topology, quantizers, parameter identity, compiler
sources, alignment, guards and the standalone entry ABI. It only merges cache
modes whose actual horizontal/vertical cache actions are absent. A counterexample
with equal tensor shapes but different valid quantized weights produces different
executable programs; shape-only reuse is rejected.

| Model | Original lowering calls | Avoided | Measured avoided work | Key construction | Estimated net saving |
| --- | ---: | ---: | ---: | ---: | ---: |
| KWS | 5,040 | 772 (15.32%) | 9.827 s | 0.855 s | 8.972 s |
| VWW | 28,504 | 4,196 (14.72%) | 37.269 s | 6.728 s | 30.541 s |

The last column subtracts measured key cost from measured removed work. It is
**not** an end-to-end search speedup measurement. Two-round synthetic search
timings ranged from 1.25× to 3.55×; they do not demonstrate the proposed 5×
real-model search target. Only empty-arena standalone fragments are supported.
Arbitrary retained SRAM state, placements and asynchronous boundaries have not
been proven equivalent. Preserving proxy paths does not prove preservation of
the physically fastest schedule.

## Cost experiment: why the old proxy chose poorly

The experiment changes only the first VWW Conv/activation stack's height in
one fixed complete-model path. MAC count, arithmetic, hardware and policy mode
stay the same. Height changes affect conflicts in the engine's input line
cache; MAC and total-traffic counts alone miss this effect.

The relative model uses address-derived cache misses, DMA beats, descriptor
runs and sequencer commands. One per-RUN constant (671 cycles) is fitted using
the previously observed heights 24 and 12. Heights 16, 8, 4 and 1 are newly
held out; height 6 was already observed and is explicitly not a holdout.

| Selection | Height | Fixed-memory native cycles | Stalled native cycles |
| --- | ---: | ---: | ---: |
| Structural proxy | 24 | 2,188,464 | 2,233,339 |
| Cache-aware relative cost | 6 | 2,173,818 | 2,219,905 |
| Stalled-memory winner in this grid | 8 | 2,174,188 | 2,219,800 |
| Existing B3 incumbent, with resident parameters | 6 | **2,173,551** | **2,218,945** |

The cache-aware choice removes 14,646 cycles compared with the structural
proxy (0.669% lower latency; equivalently 0.674% throughput gain). Its stalled
regret is only 105 cycles relative to the tested grid. It still loses to the
existing B3 incumbent. Therefore this is an explanation of a tuning error,
not a new best result or superiority over DeFiNES.

The zero prediction error applies only to relative fixed-memory timing for
this single descriptor family; the unaffected path provides a measured anchor.
It is not a general cycle model. The random-stall result and physical SDRAM
latencies are not predicted. All eight new runs checked exact pinned-input
logits, not a complete accuracy dataset. Common overlap remained unchanged,
so the screen also does not validate a new overlap model.

## Remaining latency headroom

[`headroom.json`](evidence/phase6/hypothesis-v2/headroom.json) partitions the
measured counters into engine-busy, DMA-only and neither-busy time. Holding
the observed engine-busy count fixed, erasing *all* other time gives the
following deliberately optimistic counterfactual:

| Evidence | KWS latency reduction | VWW latency reduction | Geometric mean reduction |
| --- | ---: | ---: | ---: |
| Task 1 physical B4 traces | 2.63% | 12.62% | 7.76% |
| Generic B3, fixed native memory | 0.99% | 3.83% | 2.42% |
| Generic B3, stalled native memory | 1.69% | 5.80% | 3.77% |

These are not universal bounds or attainable predictions. Engine busy includes
stalls, copies, control and requantization, and can change with a new layout,
kernel or schedule. The implication is narrower: improving transfer overlap
while preserving current engine work is insufficient for the project's target.

The existing target is **15% lower geometric-mean latency**, equivalent to
1.17647× geometric-mean throughput, with neither workload more than 5% slower.
It is not 1.15× throughput. If KWS stays unchanged, VWW needs 27.75% lower
latency to meet that target. This is our prospective research gate, not a
publication acceptance rule.

## Next research decision

Do not spend an endurance campaign validating a novelty claim from these two
compiler improvements. They should strengthen all applicable baselines. Keep
the existing best implementation as the reference.

The remaining concrete **H2** is a search-method hypothesis: a sound physical
boundary summary or lower bound that can skip *distinct* executable lowerings,
achieving at least 5× lower cold compile/search/check time than the fastest
ordinary memoized control at the same ≤1% bounded-optimum gap and zero invalid
compositions. Charge fragment generation and certificate work to both methods;
use the same legal catalogue and finite memory/control budgets. The current
inactive-mode canonicalization does not fulfill H2. The
[audit](PHASE_6_HYPOTHESIS_PRIOR_ART.md#one-defensible-hypothesis-to-test) gives
the controls and rejection conditions.

A device-performance contribution additionally needs to reduce engine work on
both workloads and survive the strongest matched baseline. A larger LUT/cache
or a faster clock alone would change the comparison contract; record its routed
resources and matched controls rather than treating it as a scheduler gain.
**Task 2's bounded falsification is complete; a positive novelty result and
Phase 6 closure are not.**

## Reproduction and archive

The scripts and source-pinned reports are preserved in the
[compressed evidence manifest](evidence/phase6/hypothesis-v2/archive/manifest.json).
Declared fixture binaries are embedded; generated native executables, models
and bitstreams are hash-pinned. The imported Task 1 data above is a separate
unchanged record, checked by the headroom script.

```sh
.venv/bin/python3 tools/phase6/hypothesis_headroom.py
.venv/bin/python3 tools/phase6/hypothesis_cost.py --output work/phase6/hypothesis-cost-repeat
.venv/bin/python3 tools/phase6/hypothesis_search.py --output work/phase6/hypothesis-search-repeat
.venv/bin/python3 tools/phase6/hypothesis_search_cost.py --report work/phase6/hypothesis-search-repeat/report.json --output work/phase6/hypothesis-search-repeat/cost.json
.venv/bin/python3 tools/phase6/hypothesis_search_winners.py --report work/phase6/hypothesis-search-repeat/report.json --output work/phase6/hypothesis-search-repeat/winners.json
.venv/bin/python3 tools/phase6/hypothesis_search_validate.py --directory work/phase6/hypothesis-search-repeat
.venv/bin/python3 tools/phase6/archive_matched_baselines.py --verify --verify-workspace --output docs/research/evidence/phase6/hypothesis-v2/archive
```

Use fresh output directories. The scripts require the declared frozen Task 1
catalogues, model files and native executable; the archive manifest records
their identities and storage policy. Reproduction is a bounded correctness and
hypothesis screen, not a replacement for held-out board or accuracy evaluation.
