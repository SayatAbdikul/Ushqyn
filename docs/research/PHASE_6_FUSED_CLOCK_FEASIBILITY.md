# Fused engine 25.5 MHz clock feasibility

The proposed 25.5 MHz step is rejected for the existing single-rPLL clock
architecture. No Gowin run, route input generation, or board run is needed to
establish this limitation. The frozen 24 MHz fused engine, its 256-byte UART
bridge, and its model fixtures remain unchanged.

The official [GW2AR data sheet](https://cdn.gowinsemi.com.cn/DS226E.pdf),
DS226-2.7E, Table 3-20 on printed page 38, specifies a minimum PFD frequency
of 3 MHz and a VCO range of 500–1250 MHz for C8/I7. The equations in the
[Gowin clock guide](https://cdn.gowinsemi.com.cn/UG286E.pdf), section 5.1,
give `CLKOUT = CLKIN * FBDIV / IDIV`, `PFD = CLKIN / IDIV`, and
`VCO = CLKOUT * ODIV`.

| Quantity | Current 24 MHz | Proposed 25.5 MHz |
|---|---:|---:|
| Input clock | 27 MHz | 27 MHz |
| Actual FBDIV / IDIV | 8 / 9 | 17 / 18 |
| FBDIV_SEL / IDIV_SEL | 7 / 8 | 16 / 17 |
| ODIV_SEL | 32 | 32 |
| PFD | 3 MHz | **1.5 MHz — out of specification** |
| VCO | 768 MHz | 816 MHz |
| Integer divisor for 750 kbaud | 32 | 34 |
| SDRAM CLOCK_HZ | 24000000 | 25500000 |
| Refresh period, CLOCK_HZ / 64000 | 375 cycles | 398 cycles |

The target ratio 17/18 is irreducible. Any other exact static divider pair
has a denominator at least 18 and an equal or lower PFD. A different ODIV
can change the VCO but cannot change this PFD bound. A finite search of all
64 input divisors, 64 feedback divisors, and the supported output divisors
confirms that none produces a compliant direct 25.5 MHz output.

More generally, the 3 MHz PFD minimum and 27 MHz input force `IDIV <= 9`.
The largest ratio below one with those denominators is 8/9, so there is
**no legal direct CLKOUT frequency strictly between 24 and 27 MHz** in
this architecture. This includes 24.75 and 26.25 MHz, even before UART
considerations. The next legal direct clock is 27 MHz. The current 24 MHz
route reports a 26.285 MHz core Fmax; that result does not demonstrate
27 MHz closure.

The hypothetical 25.5 MHz UART divisor is exact and its 398-cycle refresh
period is 15.607843 microseconds, within the existing 15.625-microsecond
refresh interval. These calculations do not independently certify all
SDRAM setup/hold, initialization, and phase constraints. The PFD violation
already rejects this candidate, so no new SDRAM timing claim is made.

Using divided PLL outputs, cascaded PLLs, or another reference clock would
be a separate architecture change. It would need proof of a suitable
phase relationship for the SDRAM clock, generated-clock constraints,
reset/lock behavior, routing, and board screening. Those changes are not
part of this bounded clock-only assessment.

Run `.venv/bin/python3 tools/phase6/fused_clock_feasibility.py` to reproduce
the exact-rational divider enumeration and audit the preserved source,
fixture, native-test, UART-integration, and bitstream identities. The
immutable result is written to
`work/phase6/experiments-v1/fused-clock-25p5-feasibility-v1/report.json`.
