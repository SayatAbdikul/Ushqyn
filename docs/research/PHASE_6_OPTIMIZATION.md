# Phase 6 — executable cache, spatial SIMD and retention milestone

The isolated engines and exact producer/activation retention compiler now
run both primary models in RTL. Gowin fits both candidates at the existing
20.25 MHz clock. **No Phase 6 image has been programmed onto the board. G6 is
still open; a complete published-policy comparison is not claimed.**

The original Phase 5 RTL, compiler, runners and release remain byte-identical
to the 109-file preservation manifest. The only production entry-point edit
is additional, explicitly invoked Phase 6 Make targets. The physical Phase 5
campaign continues on its original image.

## Implemented and checked

* `rtl/phase6/engine.sv`: 64/128/256-entry channel-indexed input caching for
  stride-one unpadded 1×1 convolution. A full SRAM-word address tag accompanies
  the synchronously read data. Other kernels retain the row-indexed policy.
  Valid bits are cleared on descriptor changes, reset, abort and clear.
* Two tagged parameter records retain the current output-channel pair. This
  replaces four parameter request/response states with one cache-hit state:
  three cycles saved per reuse, rather than the earlier four-cycle ideal.
* `rtl/phase6/spatial_engine.sv`: use the same eight multipliers on eight
  neighboring pixels for unpadded stride-one 1×1 convolutions with at most
  256 input channels. Broadcast weights, retain eight accumulators, handle
  unaligned channel planes and tails, and use the original requantizer for
  every output. Other operators use the cache engine path. INT32 overflow is
  checked at the original eight-channel boundaries; temporarily wider sums
  must not change the accepted arithmetic. Kernel mode is registered during
  descriptor validation to remove a long combinational decode path.
* `scheduler/resident.py`: execute each producer tile and its original ReLU
  or Clip consecutively, retaining the INT8 output in SRAM. The activation
  runs in place; its parameter record replaces a dead producer parameter
  record. Both original quantization/rounding boundaries execute. There is no
  clamp-only approximation, recomputation or unreported additional SRAM.
* `resident_verify.py`: independently replay raw bytecode plus untrusted
  graph-coordinate claims. Check geometry, exact parameter/input bytes, tensor
  versions, coordinates, output coverage, immutable storage, waits, DMA/live
  conflicts and every live operand. Permit only the proven exact in-place
  elementwise case. Diagnostic snapshots check original intermediate tensors.
* `engine_cost.py` and `event_cost.py`: structural engine timing and independent
  engine/DMA timelines, including command fetch/dispatch/waits and overlap.
  Contended SRAM service remains an approximation in this analytical model;
  actual RTL arbitrates accesses. Prediction error is reported separately.
* `resident_search.py`: bounded deterministic local search over actual emitted
  channel-tiling, activation-retention and overlap choices. Keep eight strong
  simple configurations, including the current overlapped schedule, as
  fallbacks. The bound relaxes compatibility and DMA cost while retaining the
  minimum required serial engine work for every original layer. It applies
  only to this declared catalogue and the selected engine.

The selected bitstream supports **every** compared compiler policy. Cache
improvements are a common hardware improvement, not a privileged B4 feature.

## Results and interpretation

For the original preferred-half, overlapped schedule, fixed-RAM RTL simulation
gives the following. These counters include the command sequencer, DMA, SRAM
and engine; the external port is an abstract RAM model, not physical SDRAM.

| Model | Original sequence cycles | Cache + parameter reuse | Spatial SIMD | Spatial speedup vs original |
|---|---:|---:|---:|---:|
| KWS | 5,495,354 | 3,502,921 | 2,780,369 | 1.977× |
| VWW | 17,648,780 | 11,180,639 | 7,919,803 | 2.228× |

The similarly named `c0p0` control disables both changes. Its counters must
exactly match the original RTL under both external-memory stall settings.
Cache-only sizes and parameter-only configurations are retained as ablations.

| Routed configuration | Core Fmax, MHz | Operating clock, MHz |
|---|---:|---:|
| Parameter reuse, original index policy | 22.451 | 20.25 |
| 64-entry channel cache | 23.324 | 20.25 |
| 128-entry channel cache | 23.625 | 20.25 |
| 256-entry channel cache | 23.714 | 20.25 |
| 256-entry channel cache + parameter reuse | 24.123 | 20.25 |
| Spatial SIMD + caches, registered mode | 20.798 | 20.25 |

All six routes pass setup and hold checks under the unchanged constraints.
The cache fallback uses 14,911/20,736 logic cells, 5,136/15,915 registers,
9,220/10,368 CLS sites, 35/46 BSRAM blocks and 19.75/24 DSPs. The spatial
candidate uses 17,242 logic cells, 5,451 registers, 9,585 CLS sites, 35 BSRAM
blocks and 14.5 DSPs in Gowin's accounting. The original release used 34 BSRAM
blocks. These are routed resources, not estimates from array byte counts.
One route per configuration does not establish a statistical frequency
advantage; the spatial design has modest timing margin. The operating clock
has not been increased.

The first spatial route failed: 18.336 MHz, 154 setup violations and zero hold
violations. Registering mode decode resolved that failure without changing the
cycle count. The rejected route, original engine source and timing report are
archived alongside the passing revision; its image is not a board candidate.

The new search completed within 60 seconds per model on this Mac, with a bound
gap of **4.61% KWS / 6.07% VWW** on the cache engine, and **5.85% / 8.41%**
on the spatial engine, for the declared catalogue. Exact measured host
times and settings are in the archived search records. The independent small
fixture exhaustively enumerates all 64 combinations and checks the selected
result and bound.

Using the old, measured physical DMA fit with the new structural engine gives
the following **projections**, not board measurements:

| Candidate | Model | Selected predicted cycles | Device latency at 20.25 MHz | Reciprocal throughput |
|---|---|---:|---:|---:|
| Cache + retention | KWS | 3,549,938 | 175.31 ms | 5.70 inferences/s |
| Cache + retention | VWW | 11,452,881 | 565.57 ms | 1.77 frames/s |
| Spatial SIMD + retention | KWS | 2,825,458 | 139.53 ms | 7.17 inferences/s |
| Spatial SIMD + retention | VWW | 8,185,553 | 404.22 ms | 2.47 frames/s |

These projections exclude UART, preprocessing and initial loading. Against
the previous measured 281/904 ms, the spatial candidate suggests approximately
2.0× / 2.2× improvement.
New physical arbitration/timing effects remain uncalibrated. Random-stall RTL
is a robustness test, not a calibrated replacement for the SDRAM controller.

The search ties the simple all-retained/half/overlapped baseline on KWS and
improves its predicted VWW latency by only **0.58%** on the cache engine and
**0.81%** on the spatial engine. Relative to the best non-retained schedule on
the **same improved hardware**, the predicted benefit is about 4.0% KWS / 3.2%
VWW for caches and 4.9% / 4.4% for spatial SIMD. The hardware improvement must not be presented as
a novel scheduler improvement or as SOTA versus prior publications.

## Validation and evidence

The matrix includes the unchanged engine, a disabled-feature control, three
cache capacities, parameter-only reuse, the combined implementation and spatial SIMD.
The full models run with fixed RAM and seeded backpressure/variable response
latency. Snapshot configurations compare every materialized compute-layer
tensor with an independent integer oracle on the pinned input and a second
seeded INT8 input. Timing configurations exclude diagnostic snapshots.

Microtests exercise odd channel planes, every cache-capacity boundary, more
channels than cache entries, odd output-channel groups, extreme zero points,
different per-channel quantizers, repeated addresses with changed inputs,
reset/abort, delayed responses and INT32 overflow boundaries. The original GEMM/spatial/pooling regression
also runs on the experimental engine. Unit tests reject damaged coordinates,
parameters, lifetimes, waits and duplicate/missing output coverage.

The machine-readable audit is authoritative for counts and source hashes:

The completed matrix contains **392 full-model RTL runs and 7,688 checked
tensors**, with no mismatches, plus 159 passing unit tests and eight passing
engine regression cases across four configurations. Six current builds pass
Gowin routing/timing. All serialized fixed-RAM timing predictions match exactly;
the largest overlapped prediction error is **0.0883%**. These errors concern
the simulated memory interface, not physical SDRAM.

* [Compact optimization summary](evidence/phase6/optimization/summary.json)
* [Complete audited results](evidence/phase6/optimization/summary.json.gz)
* [Pinned fixtures](evidence/phase6/optimization/fixtures.json.gz)
* [159-test JUnit record](evidence/phase6/optimization/tests.xml.gz)
* [KWS search](evidence/phase6/optimization/kws-search.json.gz)
* [VWW search](evidence/phase6/optimization/vww-search.json.gz)
* [Spatial KWS search](evidence/phase6/optimization/kws-spatial-search.json.gz)
* [Spatial VWW search](evidence/phase6/optimization/vww-spatial-search.json.gz)
* [Pending physical comparison plan](evidence/phase6/optimization/board-plan.json)

The exact passing cache and spatial images are preserved as compressed
candidates under `hardware/releases/phase6/`. They have not been programmed or
qualified by a physical campaign. The pending plan preserves policy aliases,
hashes, randomized order and independent sessions; it must wait for the
original Phase 5 campaigns to finish before any programming change.

The audit requires exact structural predictions for every fixed-RAM serialized
timing case, records overlap errors without refitting, checks routed setup and
hold results, and rechecks all 109 frozen files. Simulation coverage is not the
complete accuracy dataset or a physical stability campaign.

## Published baseline status and remaining research

The adapter executes the author's unmodified, pinned DeFiNES in-regime
tile/halo/cache geometry routine, including recompute, horizontal caching and
horizontal-plus-vertical caching. An independent recurrence checks the returned
shapes on both primary networks. Its input graph retains explicit activation
and quantization barriers. The source archive and imported Python files are
pinned in [the geometry record](evidence/phase6/optimization/defines-geometry.json.gz).
The revision is the same one inspected in the prior-work audit:
[upstream DeFiNES](https://github.com/KULeuven-MICAS/DeFiNES/tree/7097d6090dc22321e44ce91434e7cc23b065864f).

This is **not a full B3 performance reproduction**. Our executable catalogue
still lacks general multi-convolution spatial fusion, halo cache placement,
strided packing and retained-weight alternatives. The same limitations apply
to the proposed scheduler. A restricted retention policy must not be relabeled
as DeFiNES. A full backend/adaptation and tile-expanded COSMA comparison remain
boardless research work if the project continues that scheduling claim.

The current evidence does not justify extending the scheduling novelty claim:
a much simpler feasible policy nearly matches it. Before expanding that
research direction, validate the selected engineering changes physically and
reassess the hypothesis. Spatial SIMD is now implemented, but is a conventional
hardware technique rather than a new scheduling contribution. An exact
activation LUT epilogue and higher-clock pipelining remain exploratory
alternatives; they were not implemented or credited in these results.

G6 still needs matched physical correctness/performance results, held-out
physical prediction errors, complete strong-baseline eligibility and an
explained benefit beyond those baselines. Connecting the board will not by
itself resolve the remaining research novelty/backend gap.

## Reproduction

Use the existing compiler Python environment for fixtures/search/native runs;
use the cocotb environment for engine tests. All generated files stay under
`work/phase6/optimization`. No command below accesses JTAG or UART.

```sh
.venv/bin/python3 -m pytest compiler/test_scheduler_*.py -q --junitxml=work/phase6/optimization/tests.xml
make p6-opt-fixtures PYTHON="$PWD/.venv/bin/python3"
make p6-opt-native PYTHON="$PWD/.venv/bin/python3"
venv/bin/python3 tools/phase6/run_engine.py --variant c64p0
venv/bin/python3 tools/phase6/run_engine.py --variant c128p0
venv/bin/python3 tools/phase6/run_engine.py --variant c256p1
venv/bin/python3 tools/phase6/run_engine.py --variant spatial
.venv/bin/python3 tools/phase6/run_gowin.py --variants c0p1 c64p0 c128p0 c256p0 c256p1 spatial
.venv/bin/python3 tools/phase6/run_defines_geometry.py
make p6-opt-audit PYTHON="$PWD/.venv/bin/python3"
```

The DeFiNES runner expects the revision-pinned source extracted beneath
`work/phase6/defines-source/DeFiNES-7097d6090dc22321e44ce91434e7cc23b065864f`.
The source tar URL is
`https://codeload.github.com/KULeuven-MICAS/DeFiNES/tar.gz/7097d6090dc22321e44ce91434e7cc23b065864f`.
It requires the existing NumPy, NetworkX and SymPy packages; it does not change
the active Phase 5 environment. Preserve the upstream BSD license with any
redistribution. Gowin uses the pinned previously generated SDRAM IP and the
unchanged board constraints; its macOS services may require execution outside
the filesystem sandbox.
