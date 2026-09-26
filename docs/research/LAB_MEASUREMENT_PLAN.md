# Phase 0 lab capability and measurement plan — 2026-09-26

This completes R04's planning and capability record. It does **not** report
measured power, an instrument reservation, or access to a second FPGA.

## Available and verified

| Capability | Evidence and boundary |
|---|---|
| Tang Nano 20K identification | User supplied `GW2AR-LV18`, `QN88C8/I7`, `2537C`, `NCWS02.00`; JTAG ID `0x0000081B`, FPGA revision C. USB serial `2025030317`. |
| PCB revision | Unknown: the FPGA package marking is not a PCB silkscreen revision. Preserve this field as null; do not infer it from revision C. The device, serial, actual constraints and bitstream identify the tested specimen. |
| Programming | openFPGALoader 1.1.1; temporary SRAM configuration. Unplugging loses the configuration. Flash programming is not needed for the research runs. |
| Minimal build and communication | Gowin Education V1.9.11.03; reset-corrected UART release; all 1,280 stop-and-wait echo bytes passed. [Raw readback](evidence/physical/uart-readback.json), [device/build record](evidence/physical/summary.json), [reproduction](../../hardware/PHYSICAL.md). |
| Compiler / route flow | Custom compiler/RTL and Gowin synthesis/place-and-route. openFPGALoader programs the board. Neither LiteX nor OSS CAD Suite is required by this measured release. |
| Working accelerator | Phase 4 passes complete KWS/VWW node checks using on-board SDRAM. Phase 5 has a separate dual-resident release and ongoing campaigns; see their individual gates. |
| Clock | On-board nominal 27 MHz reference; current Phase 4/5 core operates at 20.25 MHz using its declared clock configuration. Post-route timing and FPGA cycle counters are available. No external frequency measurement has been performed. |
| Power / instrumentation | USB-powered board. User reports no meter or oscilloscope. No physical power or energy result exists. Gowin power reports are estimates. |
| Second target | No second FPGA has been identified or reserved. Cross-target generalization remains an E03 dependency. |

Board capacities and board-level connections must be checked against the
[manufacturer's documentation](https://wiki.sipeed.com/hardware/en/tang/tang-nano-20k/nano-20k.html)
and the checked-in constraints for each release. Do not substitute the host
computer for the FPGA's SDRAM in performance claims.

## Instrument acquisition route and acceptance

Owner: the researcher. Before E01/E02, request a loan from an accessible
university or electronics lab of a calibrated voltage/current acquisition
instrument with digital triggering, or a scope with a characterized shunt and
suitable differential measurement. This is a concrete borrowing route to pursue,
not a confirmed booking; no lab has been contacted on the user's behalf.
If a loan is unavailable, obtain a rental/purchase quotation for user approval.
No particular instrument or expenditure is committed by this plan.

Accept an instrument only after recording model/serial, calibration date and
specification, current/voltage ranges, effective bandwidth, sample rate, trigger
support, shunt value/tolerance if used, and exported raw-data format. Require
simultaneous voltage/current capture at at least 10 ksample/s for these
hundreds-of-milliseconds inferences, or demonstrate with longer batches that a
slower instrument gives converged energy estimates. Choose ranges after a pilot
measurement; neither expected current nor uncertainty is known today. Reject
clipping, unrecorded auto-ranging gaps and samples without timestamps. Compare
integrated energy at two capture rates/bandwidth settings before the final runs.
A display-only USB meter is insufficient for short-inference energy claims.

## Supply boundary and timing setup

1. Map every supply path before wiring the instrument. Prefer one measured
   5-V board-input path upstream of the board regulator and USB bridge. Preserve
   USB data but prevent an unmeasured USB power path or a second supply from
   bypassing the measurement. Confirm the actual board connection against its
   schematic before using an adapter. Do not improvise simultaneous supplies.
2. Include the FPGA, SDRAM, regulators and powered board peripherals in **gross
   board energy**. Record peripherals/LED state and supply voltage. Host computer
   energy is excluded and must be labeled. FPGA-core-rail energy would require a
   separate verified rail measurement and cannot be inferred from board input.
3. In E02, add a routed GPIO timing marker on a verified free header pin. Assert
   it when the accepted inference command begins and deassert after the final
   device result is committed. It must include command execution, DMA and waits.
   Verify marker edges against the cycle counter in RTL and then on the scope.
   The LED is an activity indicator, not a calibrated timing marker. No pin or
   already-implemented marker is asserted here.
4. Freeze two windows: device execution with models/inputs resident, and
   input-upload-to-result transfer-inclusive execution. Count actual completed
   jobs in each window. Report model-load energy separately; do not amortize it
   invisibly. State whether preprocessing is included in each result.
5. Check the generated clock on a suitable routed test output when a scope is
   available. Until then, milliseconds derived from counters use the nominal
   clock, with that limitation stated.

## E02 experiment and release conditions

Use at least five independent sessions in randomized configuration order, with
warm-up and repeated windows of at least 10 seconds (or longer if required by
the instrument). Keep clock, supply, bitstream, input sequence and host traffic
matched for baseline comparisons. Record ambient conditions and warm-up policy.
Archive V(t), I(t), marker samples, job count, hashes and complete setup metadata.

Integrate `E = integral(V(t) * I(t) dt)` over each window and divide by completed
jobs. Report gross board energy first. If reporting idle-subtracted energy,
capture matched idle traces separately and show both values; this difference is
not automatically FPGA-core energy. Report 95% statistical intervals and
instrument/shunt/timing systematic uncertainty separately. Check repeatability
and the effect of capture rate. A Gowin estimate never substitutes for a trace.

If instrumentation cannot be obtained, E02 stays open and the paper must omit
measured energy/energy-SOTA claims. If a second board cannot be obtained, E03's
cross-target experiment stays open; simulation or a second route is not a
physical second-target result. These are explicit later-phase dependencies,
not hidden assumptions needed to claim that Phase 0 planning is complete.
