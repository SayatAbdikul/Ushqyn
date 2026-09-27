# Phase 6 optimization screens — 2026-09-27

This campaign uses short tests, as requested. Every candidate has separate
generated RTL, simulation, routing and programming evidence. Full accuracy and
long stability remain paused; the completed KWS records and VWW's 212-sample
prefix remain byte-identical. This is not G6 closure or a SOTA claim.

## Physical ablations

Medians of three measured repetitions after a warmup, plus one stress inference
per model/image. All final logits must match. The table uses identical pinned
models and the original resident/overlap schedule, at 20.25 MHz except where
marked. Later template edits do not rewrite the generated source of any row.

| Candidate | KWS ms | VWW ms | Routed Fmax MHz | CLS |
| --- | ---: | ---: | ---: | ---: |
| Direct-writeback reference | 125.446 | 363.624 | 20.334 | 9,678 |
| Registered pointwise weight | 125.453 | 363.627 | 20.478 | 9,701 |
| + next-input request during MAC | 112.711 | 319.731 | 20.927 | 9,655 |
| + scalar word reuse | 106.372 | 301.262 | 20.386 | 9,664 |
| + pointwise word cache | 85.737 | 287.918 | 20.278 | 10,107 |
| Scalar reuse + compact arithmetic | 106.376 | 301.264 | 21.634 | 9,689 |
| Cache + compact arithmetic | 85.738 | 287.928 | 22.460 | 10,091 |
| Scalar reuse + parameter fast path | 104.219 | 295.303 | 20.292 | 10,076 |
| Compact arithmetic + activation LUT | 100.780 | 284.347 | 23.682 | 9,793 |
| Cache + compact arithmetic + LUT | 80.135 | 270.993 | 20.262 | 10,352 |
| Above + cache valid bits in BRAM | 80.376 | 271.774 | 23.049 | 9,889 |
| Compact arithmetic + depthwise SIMD | 101.943 | 285.528 | 22.054 | 9,880 |
| Cache + compact arithmetic at **22.5 MHz** | 77.119 | 258.933 | 22.505 | 10,073 |
| All exact changes, original schedule | 75.927 | 256.033 | 21.714 | 10,022 |
| **All exact changes, full-tensor chain schedule** | **68.840** | **250.975** | **21.714** | **10,022** |

The selected candidate is `all-exact` at **20.25 MHz**, with the chain schedule
for both workloads. It reaches **14.526 KWS inferences/s** and **3.984 VWW FPS**,
**1.822× / 1.449×** the paired writeback throughput (45.12% / 30.98% less latency).
The geometric-mean speedup is **1.625×**. These are FPGA execution times and
exclude feature preparation, input uploads and output readback. The chain
schedule adds 10.30% KWS / 2.02% VWW throughput versus the same image with the
original schedule; both improve and the combined gain passes the expansion gate.

All **150/150** physical inferences across seven short campaigns passed with
exact final outputs. This is not a full-set accuracy or long-stability result.
The FPGA is left idle with the selected image and the VWW chain program loaded.
It uses **36/46 BSRAM**, **14.5/24 DSP equivalents**, 18,645/20,736 logic and
10,022/10,368 CLS, with unchanged eight-MAC parallelism and a 20.25 MHz clock.

The 22.5 MHz image passes the short check but has only 0.005 MHz of
routed margin. The 27 MHz build fails timing at 23.240 MHz Fmax and was not
programmed. Clock variants update PLL, UART divisor and refresh frequency
together; the pinned SDRAM IP delay cycles and their time equivalents are recorded.

## Implemented branches and decisions

- **Registered weights, input overlap, scalar reuse:** separately checked on
  hardware. Stalled requests remain stable and consume each MAC once. Scalar
  reuse supports disjoint and exact in-place buffers; partial overlaps fall back.
- **Pointwise word cache:** the first RAM interface failed synthesis at
  21,928/20,736 logic elements. An explicit single RAM read/write interface
  fixed the fit; both attempts are preserved.
- **Balanced reduction and 33-bit temporary sums:** exact for all legal INT8
  inputs. A product has magnitude at most 16,384; each eight-product group
  starts in INT32 and deviates by at most 131,072 before the original overflow
  check. Signed 33-bit intermediates suffice. No dataset-derived bound is used.
- **Activation LUT:** generates all 256 codes through the existing requantizer,
  preserving clipping, signed rounding, zero points and saturation. For outputs
  of at least 512 bytes, 768 setup cycles replace two states per byte. The first
  unbalanced variant failed STA at 19.145 MHz with 38 setup violations; compact
  arithmetic fixed it. This is a standalone LUT pass, a precursor to producer
  epilogue fusion; it does not yet eliminate the pass or its descriptor.
- **Valid bits in BRAM:** clears the cache before each spatial descriptor and
  reads validity with tag/data. It recovers 463 CLS sites in the combination and
  improves Fmax from 20.262 to 23.049 MHz for about 0.3% extra physical latency.
  Every new spatial descriptor clears first, including after abort/reset.
- **Depthwise SIMD:** eight existing multipliers process neighboring pixels for
  3×3 stride-one depthwise convolution. Padding masks, short rows, unaligned
  words, channel changes and overflow boundaries remain exact. Other geometries
  retain the original walker. Both workloads improve in the physical ablation.
- **Parameter fast path:** checked on hardware, excluded from the combination.
  Its approximately 2% gain costs 412 CLS sites versus scalar reuse, below the
  proposed 3% expansion trigger.
- **Full-tensor SRAM retention:** a separate compiler experiment on the existing
  ABI. It removes 144,128 KWS and 171,008 VWW transfer bytes relative to its
  serialized full-tile seed, not relative to the original overlapped schedule.
  Allocation failures retain legal placement and required external stores.
  Independent bytecode replay checks provenance/lifetimes; full-model RTL checks
  actual arithmetic and includes lost-overlap costs.

## Architecture prototypes, not completed FPGA implementations

These branches were executed in software to screen the next RTL/ABI work.
They do not have routed or physical speedup claims.

| Idea | Executed screen | Result and remaining work |
| --- | --- | --- |
| Padded physical planes | Every pointwise operator on pinned and stress inputs against the independent oracle | Exact. Uncached word reads fall 485,376→262,144 KWS and 1,171,456→905,216 VWW. These savings overlap caching. Producers, consumers, allocator and verifier still need a physical-stride ABI. |
| Four pixels × two output channels | Exact eight-product-slot arithmetic on padded inputs | Exact. No KWS read-count benefit versus padded 8×1; VWW reads and scheduled slots fall about 10.4%. Dual weight/parameter delivery and FPGA control remain unimplemented. |
| INT4 / mixed precision | Per-channel 15-value weight codebooks, all layers versus first/last preserved; two fixtures per model | Logits change; KWS stress prediction changes. This is not an accuracy estimate. Decoder, packed storage, training and held-out quality evaluation remain separate work. |
| Exact packed multiply | All 65,536 signed INT8 activation / two signed INT4 weight combinations | Integer identity passes with a 17-bit packed weight operand and sign correction. It does not establish two products per existing Gowin 9×9 multiplier or a DSP-area win. |
| UART batching | Ordinary 8 KiB uploads followed by guarded batches | Three ordinary uploads passed readback at about 10.7 kB/s. The first guarded batch failed response validation and stopped. No transfer speedup is accepted; subsequent programming restored a clean UART state. |

There are 136 whole-operator checks for layout/lane arithmetic and modeled
transactions. They do not verify FPGA controller cycles, routing or accuracy.
Pruning, distillation, full QAT and wider-lane hardware were not implemented.
Free DSP count alone does not establish that their control/buffers/routing fit.

## Verification and reproduction

Each distinct engine runs three engine suites and 24 complete-model RTL cases
(fixed/stalled external memory, pinned/stress inputs, serial/overlapped
schedules), checking 632 intermediate/final tensors. LUT variants additionally
exercise all 256 codes under multiple quantizers, threshold lengths, in-place
and partial-overlap writes, backpressure, cancellation and descriptor changes.
The first edge-test cancellation failure was a testbench settling bug; its
failed XML is retained. Chain schedules add 12 full-model RTL runs per engine.

Physical screens use five inferences per model/image: stress, warmup and three
timings. Upload readback, logits, status, protocol errors and counters are
checked. Errors stop without retry. Four runner tests cover accounting,
duplicates, corrupted responses, sequence wrap and batched control-write rejection.

`tools/phase6/experiments.py` provides `prepare`, `engine`, `edges`, `native` and
`route` stages. Use `venv/bin/python3` for cocotb (`engine`/`edges`), and
`.venv/bin/python3` for other stages. `--label` preserves revised candidates.
`chain_resident.py` prepares independent fixtures; `--native all-exact` checks
them in that engine's actual compiled RTL. `algorithm_experiments.py` reproduces
software prototypes. `test/phase6/test_experiment_runners.py` runs the four checks.

`quick_experiments.py` defaults to read-only preflight. Device access requires
`--run`, explicit labels and a fresh output directory. `--schedule chain`
selects the separate retention experiment. Evidence is under
`work/phase6/experiments-v1` and is archived by `archive_experiments.py`.
All 109 frozen Phase 5 sources remain intact.

The [machine-readable archive](evidence/phase6/experiments/summary.json) pins
generated engines, reports, raw signed physical records, historical runner
versions, chain fixtures and the compressed candidate image. It records failed
builds/prototypes alongside passing candidates. Rehydrate the archived generated
engine to its recorded workspace path when reproducing a historical label;
creating that label from the newest mutable template is not equivalent.
