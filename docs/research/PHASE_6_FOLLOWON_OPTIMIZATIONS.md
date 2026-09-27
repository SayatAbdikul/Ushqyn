# Phase 6 follow-on optimization results

This document records the **2026-09-27 short-screen campaign** following the
selected eight-MAC `all-exact` engine and full-tensor chain schedule. The
[machine-readable ledger](evidence/phase6/followon/summary.json) is regenerated
by `python3 tools/phase6/archive_followon.py`. It verifies signed physical
records, program and fixture identities, generated RTL, and routed bitstream
hashes, then stores deterministic compressed copies of the evidence. Re-run the
archiver after a screen completes; it never uses the board.

An optimization advances only with exact outputs, a passing route, at least
**3% geometric-mean throughput improvement** across KWS and VWW, and no
unexplained single-model regression above **1%**. A smaller ablation may still
be retained as a component or a documented workload-specific tradeoff. Short
screens contain one stress inference, one warmup, and three timed inferences
per model. They do not establish full-set accuracy, energy, long stability,
temperature tolerance, or SOTA.

## FPGA execution measured on the Tang Nano 20K

Times are medians of the three timed device counters, excluding host feature
preparation, UART uploads, and readback. Every completed row below passed
**10/10 physical inferences**, exact final logits, upload checks where the
runner provided them, and a routed clock constraint. Speedups compare each
row with the same pinned models on the selected `all-exact` chain schedule:
68.839901 ms KWS and 250.975358 ms VWW at 20.25 MHz.

| Change | Clock MHz | KWS ms | VWW ms | KWS × | VWW × | Geomean × | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Selected `all-exact` chain baseline | 20.25 | 68.840 | 250.975 | 1.000 | 1.000 | 1.000 | Reference |
| Exact constant-filter lowering | 20.25 | 68.840 | 188.651 | 1.000 | 1.330 | 1.153 | Keep |
| Sibling input retention | 20.25 | 68.840 | 243.734 | 1.000 | 1.030 | 1.015 | Keep with constant lowering; weak alone |
| Constant filters + sibling retention | 20.25 | 68.839 | 181.424 | 1.000 | 1.383 | 1.176 | Keep |
| Depthwise word reuse, otherwise baseline | 20.25 | 66.109 | 246.671 | 1.041 | 1.017 | 1.029 | Useful component; just below 3% alone |
| Word-wide activation LUT, otherwise baseline | 20.25 | 65.727 | 241.003 | 1.047 | 1.041 | 1.044 | Keep |
| Fast exact requantizer at higher clock | 22.50 | 61.948 | 225.648 | 1.111 | 1.112 | 1.112 | Keep |
| Constant/sibling compiler + DW reuse, vector LUT, fast RQ, narrowed geometry | 20.25 | 62.999 | 167.128 | 1.093 | 1.502 | 1.281 | Keep, but timing margin narrow |
| Above with overflow-safe speculative prefetch | 20.25 | 62.999 | 167.138 | 1.093 | 1.502 | 1.281 | More timing margin, no measured speedup at this clock |
| Above with scalar LUT and overflow-safe speculative prefetch | 22.50 | **59.491** | **159.199** | **1.157** | **1.576** | **1.351** | Fastest compute-only image |
| Above with integrated UART burst bridge | 22.50 | **59.491** | **159.188** | **1.157** | **1.577** | **1.351** | Same compute throughput, burst upload support |

The strongest completed integrated row reaches **16.81 KWS inferences/s** and
**6.28 VWW FPS** for device execution. Its route uses 37/46 BSRAM, 9.5/24 DSP
equivalents, and 10,096/10,368 CLS sites; Gowin reports 24.319 MHz Fmax at a
22.5 MHz target. Its measured compute time is effectively the same as the
scalar image without burst UART. The preceding combined vector-LUT image has only 0.028 MHz
Fmax margin at 20.25 MHz and uses 44/46 BSRAM. Passing one route and one short
screen does not justify assuming equivalent margin across builds or boards.

The constant-filter pass proves 971 VWW all-zero convolution output filters
and skips 3,131,136 dense MACs. It emits the exact bias/requantized constant
codes; all KWS command and payload bytes remain unchanged. Sibling retention
keeps repeated VWW tile input in the existing scratchpad, eliminating 144,000
external transfer bytes. Their combined physical result improves VWW beyond
either change alone. All complete-model native checks include pinned and seeded
stress inputs with fixed and stalled RAM and exact intermediate tensors; these
simulation cycles were **not** substituted for the physical times above.

## Other checked candidates

The pointwise cache-hit bypass was repaired after a stalled-request stability
failure; the corrected `pw-hit-v3` passes engine, edge, native and route checks.
Its native whole-model improvement was below the 3% two-model expansion gate,
so it was not added to the strongest combination. The word-wide activation LUT
passes all 256 code values and overlap/backpressure edge cases and routes at
20.25 MHz, but uses 44 BSRAM and 10,213 CLS sites. Its isolated physical screen
passed with a 4.44% geometric-mean gain. Exact fast requantization passes 27,176 signed boundary and shift
cases, and its separate 22.5 MHz image passes both routing and a physical
screen. The higher clock, not requantizer cycle count, creates that measured
speedup.

Overflow-safe speculative pointwise prefetch raised an isolated 20.25 MHz
route's Fmax to 27.331 MHz, but Fmax alone is not application speed. A vector
LUT combination failed the **22.5 MHz** route at 22.020 MHz Fmax with 25 setup
violations; its separate **20.25 MHz** route passes at 25.118 MHz and passed a
10/10 physical short screen. Its measured latency is effectively unchanged
from the earlier vector combination at the same clock. Removing the word-wide LUT from the speculative design
gave the completed 22.5 MHz scalar route and short-screen result above. Narrow
geometry alone recovered CLS and DSP equivalents without changing cycles.

Padded physical strides progressed beyond a read-count model to an exact native
full-model experiment. The best implementable in-place PACK policy saved only
0.64% KWS and 0.31% VWW fixed-RAM cycles, or **1.00477× geometric mean**.
It adds commands and physical-stride ABI complexity; the declared expansion
gate therefore rejects further routing and physical screening. The live-filter 4×2
dual-output mapping's optimistic MAC-step-only gain is **1.00233× geometric
mean** and requires another weight/parameter path; it is also a no-go for this
campaign. Neither result rules out a different model or memory organization.

The 256-byte CRC-checked UART burst protocol passes parser, full bridge, host
and routing tests with the selected compute engine. Its 20.25 MHz route uses
37/46 BSRAM and has 22.649 MHz Fmax. Four alternating physical 8-KiB uploads
all passed exact SDRAM readback; median upload time was **0.767 s legacy** versus
**0.191 s burst**, a **4.02× host-transfer speedup**. This is separate from FPGA
inference FPS. The bridge is also integrated with the fastest 22.5 MHz scalar
compute image. That image routes at **24.319 MHz Fmax** using 37/46 BSRAM,
9.5/24 DSP equivalents, and 10,096/10,368 CLS sites. Its paired inference
short screen passed 10/10 exact checks with the latencies above. On that same
integrated bitstream, four alternating 8-KiB transfers passed exact readback;
the medians were **0.7940 s legacy** and **0.1907 s burst**, a **4.165×
host-transfer speedup**. A separate frozen-source burst-loaded inference run
passed **2/2 exact pinned KWS/VWW outputs** on the already programmed image,
including payload, command and input readback. The signed audit records 214
payload plus 2 input burst frames for KWS, and 1,572 plus 108 for VWW, each
with 256-byte frame accounting. Its first attempt also produced two exact
outputs but failed the runner-source stability check because that source
changed during the run; that attempt is preserved as failed evidence and
excluded. The two pinned checks confirm the burst path works with inference;
the timed 10/10 screen above remains the device execution measurement.

## Evidence boundary and reproduction

The ledger distinguishes `board-short-screen-passed`, `passed-route`,
`failed-timing`, `running`, and `pending` states. A running screen contributes
no median or speedup. Generated engine files and routed bitstreams are pinned
by SHA-256; physical rows are signed and checked against the bitstream in their
plan. The early constant-filter boardless memo records its then-current source;
the compiler gained sibling support afterward. The ledger records that
historical mismatch and pins the current compiler source separately. The
[original optimization log](PHASE_6_EXPERIMENT_LOG.md) remains the
historical record of earlier direct-writeback-to-`all-exact` work; these
follow-on measurements use that selected chain image as their baseline.

```sh
python3 tools/phase6/archive_followon.py
```

All remaining full VWW accuracy, independent repeatability, energy and strong
external-baseline comparisons must be run on a stable final image before a
research-paper performance or SOTA claim.
