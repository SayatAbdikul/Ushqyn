# Phase 4 embedded SDRAM diagnostics

The Tang Nano 20K has 8 MiB of embedded 32-bit SDR SDRAM. These are isolated
memory diagnostics; the released accelerator at target ID 8196 remains the
on-chip kernel-only image until the SDRAM port and DMA are included in its
board hierarchy.

`bist.sv` uses the Apache-2.0 NESTang controller copied under `third_party/`.
It writes and reads every byte in the 8-MiB address space in each of 64 sweeps,
for 1 GiB of aggregate traffic. The first sweep is sequential; the rest use a
seeded, invertible xorshift permutation of 23-bit addresses. The four pattern
families expose aliasing in low, middle and high address bits. Each write sweep
is followed by an 80-ms hold while `v2_sdram_refresh` continues to issue
refresh commands. The board sends a pass marker after each sweep and a final
frame with its 64-sweep count and byte count. The physical runner verifies all
markers and retains a partial report on failure.

`sdram_controller_hs.ipc` pins the exact Gowin SDRAM HS IP configuration:
32-bit data, 2 bank bits, 11 row bits, 8 column bits, CL=3, tRP=3, tRFC=9,
tMRD=3, tRCD=3 and tWR=3. Gowin Education V1.9.11.03's `gw_sh` IPFlow does not
generate this IP. Use the IDE's **Tools → IP Core Generator → Memory Control →
SDR SDRAM Memory Interface → SDRAM Controller HS**, select
`GW2AR-LV18QN88C8/I7` revision C, enter the pinned values, and create it in
`work/phase4/ip-generated`. Then check:

```sh
python3 tools/phase4/audit_sdram_ip.py work/phase4/ip-generated/sdram_controller_hs.ipc
```

The generated encrypted Verilog remains local under `work/`; it is a Gowin
build artifact, not project source. `bist_hs.sv` and `hs_byte_adapter.sv`
independently validate its initialization, single-word read/write and refresh.
`rtl/v2/hs_sdram_port.sv` converts one 64-bit external-memory beat into a
two-word SDRAM burst; `burst_smoke.sv` exercises a distinct-half pattern, all
64 walking-one patterns, eight row/bank/address-boundary cases and continuous
refresh. Its clean routed image passed twice on the board. The write-data
changeover and four-clock read offset follow a [working Tang Nano 20K Gowin
HS implementation](https://github.com/calint/tang-nano-20k--riscv--cache-sdram/blob/main/src/cache.sv).
The IP command port is single outstanding and the adapter waits for command
acknowledgment and precharge before accepting another beat. These pieces are
not yet wired to the active accelerator bitstream.

For either build, make a fresh output directory and run Gowin's `gw_sh` with
`DYLD_FRAMEWORK_PATH` and `DYLD_LIBRARY_PATH` set to the installed IDE's `lib`
directory, as shown in [the physical build guide](../PHYSICAL.md). Use
`build.tcl`, `build_hs.tcl`, or `build_burst.tcl`, respectively. Program only
the resulting `.fs` to temporary FPGA SRAM (`openFPGALoader -m`); this leaves
the board flash unchanged. The physical test runners under `tools/physical/`
can program and capture the UART result with exact bitstream/source hashes.

The controller and SDRAM clock are generated from the board's 27-MHz clock by
a Gowin rPLL. The constraint reports cover the controller-clock logic; the
physical SDRAM tests are needed because post-route Fmax alone does not prove
phase alignment or data retention. No physical power meter is available.

Archived bitstreams are in `hardware/releases/phase4-sdram/`; the corresponding
board reports are `physical-sdram-open.json`, `physical-sdram-hs.json`, and
`physical-sdram-burst.json` under `docs/research/evidence/phase4/`. Both full
memory tests completed 64 sweeps over the entire 8-MiB address space, with
1 GiB aggregate traffic each. The burst report covers 73 directed write/read
pairs; it does not establish sustained DMA bandwidth or model inference FPS.
