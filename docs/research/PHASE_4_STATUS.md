# Phase 4 working record — updated 2026-09-25

**G4 remains open.** The connected Tang Nano 20K now executes one complete,
byte-exact KWS input and one complete, byte-exact VWW input through its own
8-MiB SDRAM. A separately measured Conv tile overlaps a protected DMA transfer
with compute. The remaining Phase 4 work is a compiler-generated, full-model
ping-pong/streaming schedule with sustained burst bandwidth and a model cost
database whose features are available before execution. Complete-set accuracy,
board energy and defensible FPS have not been established by these experiments.

## Current physical result

The [overlap release](../../hardware/releases/phase4-sdram/hs_tiled_host_overlap.fs)
connects CRC-framed UART control, the v2 tiled engine, 32-KiB scratchpad, DMA,
Gowin HS SDRAM controller and refresh scheduler. The computer uploads and
schedules tiles; all parameters, inputs and intermediate tensors reside in the
board's SDRAM during execution. It is not using computer RAM as the FPGA's
working SDRAM. The [route record](evidence/phase4/physical-tiled-host-overlap-route.json)
pins the bitstream (`c8eb28fadf68d14fac8c9592c608b73b3bb8ddff0a23c215b94f3d1d442ad206`),
RTL/IP hashes and raw Gowin reports. Its core Fmax is 21.458 MHz against the
20.25-MHz generated-clock constraint, with zero setup TNS. It uses 12,405 of
20,736 logic, 3,366 of 15,915 registers, 18 of 46 BSRAM and 19.75 of 24 DSP
equivalents. The [release audit](evidence/phase4/physical-release-audit.json)
checks the bitstream, current RTL/IP hashes, successful programming transcript
and ten same-image physical reports. This narrow timing margin should be
revisited after any DMA redesign.

The same image passed exact independent-oracle checks at every KWS and VWW
node, with no reflash between models. The [physical model summary](evidence/phase4/physical-host-overlap.json)
links per-node reports and fixture hashes. These are **one deterministic input
per model**, not model accuracy-set measurements. KWS has 22 nodes and 20
compute tiles; VWW has 58 nodes and 75 compute tiles. Their parameter images
contain 49,376 and 334,816 bytes respectively. The VWW initial transpose is
the declared host preprocessing boundary. The compiler's real source-model
rebase and frozen-name caveat are recorded in [the model record](PHASE_4_REAL_MODEL.md).

The [physical overlap report](evidence/phase4/physical-overlap.json) uses the
first KWS Conv and an 8,192-byte SDRAM-to-SRAM transfer into a disjoint
scratchpad region. Both outputs match expected bytes. The engine and DMA were
busy together for 28,533 core cycles; their combined active interval fell from
284,459 sequential cycle-counts to 256,036, a **1.111× directed cycle-level
speedup**. A transfer aimed inside the live tile region was rejected with
error 9, and the DMA cycle count did not advance. This is a directed overlap
proof, not a full KWS/VWW ping-pong speedup or wall-clock FPS measurement. The
host still stages most tiles sequentially with stop-and-wait UART commands.

The same release also has [78 directed DMA transfers](evidence/phase4/physical-dma-profile-overlap.json),
[directed kernel repetitions](evidence/phase4/physical-kernel-profiles-overlap.json),
[synthetic multi-kernel/large-FC tests](evidence/phase4/physical-synthetic-overlap.json)
and [the earlier pinned MLP/SmallCNN tensors](evidence/phase4/physical-legacy-overlap.json).
The MLP's 5/5 and SmallCNN's 8/8 nodes agree with their saved on-chip
independent-reference fixtures through the new SDRAM tile path. The synthetic
tests include a mixed nine-node chain, split ReLU, a two-tile FC with 32 KiB
of weights, and all 19 frozen ToyCar AD layer shapes with deterministic
synthetic weights. This tests AD geometry, not anomaly-detection quality.
Their physical input/output and bitstream hashes are archived. The KWS/VWW
sequential runs used 5,385,783 and 17,230,984 engine
cycles, plus 1,122,254 and 4,738,426 DMA cycles. The post-upload host
schedules took about 30 and 107 seconds respectively,
including UART commands and per-node readback. Sums of engine and DMA cycles
are lower bounds for this sequential schedule; neither they nor the host
times are autonomous inference throughput. A
[KWS→VWW→KWS switchback](evidence/phase4/physical-host-kws-switchback.json)
on the previous measured-host image also reconfirmed the KWS hash without
reflashing; its different bitstream SHA is recorded separately.

The DMA profile verifies byte-exact partial tails, 2-MiB bank
crossings, the final SDRAM byte and a full 32-KiB tile, in both directions.
Its full-tile path sustains about **5.54 MiB/s at the nominal 20.25-MHz core
clock**. The present two-word, single-outstanding SDRAM command path incurs
an activate/operation/settle sequence per 8-byte DMA beat. Streaming longer
bursts is a material performance task, especially with VWW's 1,359,570
planned DMA payload bytes.

The [initial physical cost fit](evidence/phase4/physical-cost-model-overlap.json)
splits directed DMA lengths and common kernel configurations before fitting.
Its DMA held-out mean absolute percent errors are 0.75% toward SRAM and
1.52% from SRAM; the common-kernel in-distribution holdout mean is 1.39%.
Cross-model held-out worst errors exceed 100%, and rare kinds lack independent
validation. Engine features currently include MAC/SRAM-traffic counters read
from the device. A useful compiler cost model must derive them statically and
be revalidated across both workloads. An
[exploratory static-feature fit](evidence/phase4/physical-static-cost-model.json)
does use only compiler-visible tile geometry, but has 25.62% in-distribution
and 99.43% KWS / 33.81% VWW whole-model mean absolute errors; it is not
ready for scheduler decisions. Gowin's 162.572-mW total figure for
the overlap route assumes a default 0.125 toggle factor without activity data;
it is **not measured board power or energy per inference**. The user has no
current/power meter, so no physical energy result is claimed.

## Gate accounting

| Item | State | Evidence and remaining work |
|---|---|---|
| C04, pinned audio/vision kernel arithmetic | Passed for one physical input/model | All KWS/VWW nodes exact; directed boundary kernels and RTL regressions pass. More inputs belong to complete-model validation. |
| D01, physical SDRAM bring-up | Passed | [Open-controller](evidence/phase4/physical-sdram-open.json) and [Gowin-HS](evidence/phase4/physical-sdram-hs.json) full-range 1-GiB sweeps with refresh; [two-word burst](evidence/phase4/physical-sdram-burst.json) boundary checks. |
| D02, burst DMA and double-buffer overlap | In progress | Exact partial-tail/full-range DMA and guarded physical overlap pass. Compiler-managed ping-pong tiles, sustained longer bursts and a full-model net benefit remain. A simple 16-KiB/16-KiB split cannot fit three VWW Conv nodes, so a hybrid or finer tiler is required. |
| D03, off-chip tiled inference | Passed for the pinned shape/one-input gate | Full KWS/VWW, the named MLP/SmallCNN on-chip fixtures through SDRAM, a synthetic chain/two-tile 32-KiB-weight FC, and all 19 frozen ToyCar AD layer shapes pass physically. The [MLP/SmallCNN](evidence/phase4/physical-legacy-overlap.json) and [synthetic/AD-shape](evidence/phase4/physical-synthetic-overlap.json) reports include host time from upload through all tile transfers and output readback. AD uses synthetic weights because the source model/data are unavailable locally; this is not AD quality validation. |
| P01, physical kernel/cost database | In progress | Kernel/DMA cycle and traffic records plus held-out fits exist. The compiler-static fit has large cross-model errors, and bandwidth variants and measured energy where instrumentation becomes available remain. |

The current schedule emits legal sequential tiles for all frozen KWS/VWW
nodes, with two external activation slots and checked 32-KiB SRAM regions.
[KWS/VWW tiling evidence](evidence/phase4/boardless-tiling-summary.json) records
20/75 compute tiles and 322,006/1,359,570 DMA payload bytes. The
[half-scratchpad audit](evidence/phase4/boardless-pingpong-feasibility.json)
shows why a uniform 16-KiB bank split fails three VWW Conv layers. Host-facing
registers and reproduction steps are documented in
[TILED_HOST.md](../../hardware/phase4_sdram/TILED_HOST.md).

Run `make p4-test PYTHON=/absolute/path/to/.venv/bin/python3` for compiler,
protocol, numerical, DMA, tiling and RTL regression. The packet-level test
now checks simultaneous engine/DMA work and live-region rejection. Earlier
[abstract-memory KWS](evidence/phase4/boardless-host-kws.json) and
[VWW](evidence/phase4/boardless-host-vww.json) runs checked every node before
physical deployment. The older on-chip kernel milestone and historical
routes remain in [the Phase 3 closure](PHASE_3_CLOSURE.md); their resource and
timing numbers do not describe this SDRAM-connected image.
