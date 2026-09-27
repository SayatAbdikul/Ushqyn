# Phase 6 second optimization campaign

This ledger covers the exact compiler, engine, and host-transfer experiments
after the selected 22.5 MHz scalar UART image. Its previously measured device
medians are 59.4912 ms KWS and 159.187956 ms VWW. The current campaign's
machine-readable, SHA-pinned evidence is in
[`evidence/phase6/final-campaign-v2/summary.json`](evidence/phase6/final-campaign-v2/summary.json).
The final short-screen selection is the **24 MHz fused-activation core with
the existing 256-byte CRC UART bridge**. It passed 10/10 exact board checks
and has the best practical combination of measured device latency, route
margin, FPGA area, and fresh-image upload time in this campaign.
Regenerate it with:

```sh
.venv/bin/python3 tools/phase6/archive_final_campaign_v2.py
```

The archive stores deterministic compressed copies of small source, fixture
manifest, test XML, route report, and physical report files. It records routed
bitstream SHA-256 but excludes large bitstreams and model payloads. Native and
route source hashes are checked against the current files. A running or failed
physical screen has no median or speedup entry.

| Candidate | Exactness and native gate | Route gate | Decision for this campaign |
| --- | --- | --- | --- |
| Internal channel grouping | 12/12 full-model native; VWW commands 1,671→639; fixed native cycles 3,164,674→3,083,068 | Reuses selected engine | Combine with tail prefetch |
| Compiler tail parameter prefetch | 12/12 grouped native; fixed native cycles 1,304,530 KWS and 3,040,508 VWW on the selected engine | Reuses selected engine | Combine with grouping and output stream |
| One-result-per-cycle output stream plus registered lane mask | Full-model native and integrated bridge pass; grouped-prefetch fixed native cycles 1,251,026 KWS and 2,891,293 VWW | 24 MHz passes at 26.635 MHz Fmax; 27 MHz fails at 26.312 MHz | 10/10 exact short board screen passes; superseded by fused core's lower measured latency |
| Registered tail mask variant | Native and integration pass | 24 MHz passes at 27.721 MHz Fmax | No physical screen; remains unselected under the board gate |
| Exact activation epilogue fusion | Full-model native checks pass; grouped fixed native cycles 1,181,858 KWS and 2,725,926 VWW | 24 MHz route passes at 26.285 MHz Fmax, 10,137 CLS, 38 BSRAM, 9.5 DSP equivalents | Selected with existing 256-byte UART after two separate 10/10 exact short board screens |
| Producer-native padded planes | Exact PACK precursor tested; optimistic complete-model geometric mean ≤1.024× | No route for native producer design | Stop below the 1.03× expansion gate |
| Certified early exit | Pinned and stress bounds proved; skipped slots <0.04% on either workload | No RTL or route | Stop; negligible possible gain for this proof |
| 512-byte CRC UART transfer | Host tests 3/3, parser Cocotb 1/1, integrated bridge 2/2; combined stream-mask core and both grouped schedules pass 12/12 native | Fused-core/512-byte host route passes at 24 MHz with 24.226 MHz Fmax, 10,224 CLS, 38 BSRAM, 9.5 DSP equivalents; 10/10 exact board screen | Paired board upload is 0.63% slower than 256-byte frames; no-go for the selected fused core |

The compiler-only grouping and prefetch changes retain the exact INT8
quantization boundaries. Their native results use a stalled abstract RAM and
do not substitute for SDRAM measurements. The stream route uses 10,122/10,368
CLS, 37/46 BSRAM, and 9.5/24 DSP equivalents at 24 MHz. Output-pipeline-only
routing passed at 22.502 MHz Fmax for a 22.5 MHz target; the registered stream
variant has more margin at 24 MHz.

The accepted 24 MHz stream-mask board screen completed **10/10 exact
inferences**, including stress and warmup cases. Its timed device medians are
**1,257,742 cycles / 52.405917 ms KWS** and **3,230,045 cycles / 134.585208 ms
VWW**. Relative to the previously selected scalar UART image, device
throughput improves **1.1352× KWS**, **1.1828× VWW**, and **1.15876× geometric
mean**. The signed physical records have SHA-256
`a0de8a4874ae0c44a3905e21a6ea1466f9edbff2b9bc45fcb409b300fb265312`.
The first fused 24 MHz image also completed **10/10 exact inferences**. Its device
medians are **1,201,916 cycles / 50.079833 ms KWS** and **3,085,976 cycles /
128.582333 ms VWW**. The signed physical records have SHA-256
`7454c2266f39aaad066d3cec62d716d3c81ff9620d5ff0cf9d526093bc1a50ea`.
The baseline and fused timed inputs and final output bytes have identical
hashes for each model; their stress fixtures differ, so stress latency is not
used for the comparison. Relative to the selected 22.5 MHz image, fusion has
**1.18793× KWS**, **1.23802× VWW**, and **1.21272× geometric-mean** device
throughput. Relative to the 24 MHz stream-mask result, fusion lowers KWS and
VWW cycles by 4.44% and 4.46%, respectively.

The final independent fused+256 board confirmation passed **10/10 exact**.
Its medians are **1,201,917 cycles / 50.079875 ms / 19.96810 inferences/s
KWS** and **3,086,061 cycles / 128.585875 ms / 7.77690 FPS VWW**. Relative
to the selected prior 22.5 MHz image, those are **1.18793× KWS**,
**1.23799× VWW**, and **1.21270× geometric-mean** device throughput. The
27,648-byte fresh VWW input had median **0.647937 s upload** and **3.389532 s
host wall time**, which includes a **2.591859 s verified input readback**.
The selected routed bitstream SHA-256 is
`03e57521885f1a9d6b4b4cef7e374122ba948028ed379d963c97f794efc9f195`;
the final signed physical records SHA-256 is
`b8844f964becdbd04e95dc24366f38dd838310c6f78a31b73d451b586ea885d1`.

| Model | Selected cycles at 22.5 MHz | Fused cycles at 24 MHz | Cycle-count speedup | Clock speedup | Combined latency speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| KWS | 1,338,552 | 1,201,917 | 1.11368× | 1.06667× | 1.18793× |
| VWW | 3,581,729 | 3,086,061 | 1.16062× | 1.06667× | 1.23799× |

The cycle-count gain and clock increase multiply to produce the reported
latency speedup. The fused+512 image also passed **10/10 exact** and its device
medians were **1,201,917 cycles / 50.079875 ms KWS** and **3,085,977 cycles /
128.582375 ms VWW**. KWS matches the final fused+256 median; VWW is 84
cycles lower, a negligible run-to-run difference at 24 MHz.
Its signed records have SHA-256
`0457c7f8bbf4f4a5a507b22628cfa0d3cb1a702b848ef295873d877adcb07555`.
This validates composition without claiming a core speedup from the UART change.
The selected 256-byte image has **2.285 MHz** routed Fmax margin over its
24 MHz operating clock and uses 10,137 CLS; the 512-byte image has only
**0.226 MHz** margin and uses 10,224 CLS.

A direct 25.5 MHz clock step was rejected before routing. With the current
27 MHz input and direct rPLL `CLKOUT`/`CLKOUTP` architecture, it requires
`FBDIV/IDIV = 17/18`, making the phase-frequency detector 1.5 MHz, below
the GW2AR C8/I7 specified 3 MHz minimum. Under that limit, there is no
supported direct clock strictly between 24 and 27 MHz; fusion was not routed
at 27 MHz. The bounded arithmetic and device specification are recorded in
[`PHASE_6_FUSED_CLOCK_FEASIBILITY.md`](PHASE_6_FUSED_CLOCK_FEASIBILITY.md).

The accepted expansion gate requires exact pinned and stress full-model
checks, a routed image with zero setup/hold violations and all resources in
budget, then a 10/10 short board screen with three timed inferences per model.
The physical geometric mean of KWS and VWW device throughput must improve by
at least 1.03× over the selected scalar UART image before a longer campaign.
Physical rows must match the routed bitstream SHA-256, signed record hashes,
final outputs, and the report medians. The final image selection is for this
short-screen campaign and does not replace longer qualification.

The 512-byte transfer model had projected a roughly 0.140 s saving for a fresh
27,648-byte VWW input upload. A paired same-bitstream board comparison
overturned that projection: three 256-byte and three 512-byte uploads, all
verified by readback, had medians **0.646963 s** and **0.651045 s**,
respectively. The 512-byte mode was **0.63% slower** (speedup 0.993729×), so
it is a no-go for this image. This is an upload-only comparison, separate from
FPGA device inference FPS. Its signed records and plan are sealed in the
machine-readable evidence archive.

Short physical correctness and latency cannot establish held-out VWW
accuracy, full stability, power or energy, or a SOTA research claim. Those
remain separate final qualification and paper-evaluation gates.
