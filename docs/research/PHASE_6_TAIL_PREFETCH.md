# Phase 6: compiler-proved tail parameter prefetch

The existing sequencer can run an SRAM-bound engine stage concurrently with a host-to-SRAM DMA when the DMA destination lies outside that stage's declared live SRAM interval. The resident schedules previously loaded all next-stage parameters after `WAIT_ENGINE`; this experiment moves only eligible, byte-identical `DMA`/`WAIT_DMA` pairs between `RUN` and `WAIT_ENGINE`. The engine, payloads, descriptors, and SRAM addresses do not change.

`tools/phase6/prefetch_tail.py` checks the current RUN's live interval, the exact destination of each immutable next-stage descriptor or parameter transfer, and intervening overlapping writes before moving a pair. No descriptor transfer qualifies at its existing address. The grouped combination in `tools/phase6/prefetch_grouped.py` applies the same rule to the immutable grouped fixtures, remaps command-indexed contracts, and checks the original/grouped integer oracles plus resident replay. The grouped source fixtures in `work/phase6/followup_graph` are untouched.

On the ungrouped constant+sibling schedule, the same transformation prefetched 28,688 KWS and 154,336 VWW parameter bytes. Six independent resident replays and 12 native cases passed; `work/phase6/prefetch-tail-v1/native-all-exact/report.json` records the exact cycles and source hashes. This isolates the prefetch effect before combining it with grouping.

| Grouped model | Eligible parameter bytes | Fixed cycles, grouped → prefetch | Fixed speedup | Stalled cycles, grouped → prefetch | Stalled speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| KWS | 28,688 | 1,313,903 → 1,304,530 | 1.00718× | 1,324,057 → 1,307,354 | 1.01278× |
| VWW | 151,008 | 3,083,068 → 3,040,508 | 1.01400× | 3,205,315 → 3,142,213 | 1.02008× |

The incremental geometric mean is **1.01059× fixed** and **1.01642× stalled** relative to the grouped schedule. Against the selected constant+sibling baseline, the full grouped-plus-prefetch candidate reaches **1.05948× fixed** and **1.06772× stalled** in native cycle count. This cross-image result needs physical confirmation at a matched clock.

All six grouped fixtures pass independent resident replay; all 12 complete native cases pass with seeds 0 and 6063. The manifest, candidate executable, and source hashes are recorded in `work/phase6/followup_graph_tail_prefetch/native-combined-spec-scalar-v1/report.json`. `screen_engine_schedule.py` preflight passes with the same manifest SHA-256 (`cbffdb4a1a9c800b16ef888ba92a9454138ad9c5455bdfb1bcfd33f14c310294`) and plans 10 short board runs. Preflight does not execute those runs.

The compiler-only combination reuses the already routed `combined-spec-scalar-v1` engine; it has no new timing or resource cost over that routed image. Prefetch alone misses the campaign's **≥3% geometric-mean** expansion trigger, while the complete grouped-plus-prefetch candidate clears it against the selected baseline. The short board screen should measure that combined candidate before any long verification.
