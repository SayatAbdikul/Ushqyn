# Phase 6 UART burst transfer candidate

This is an isolated host-transfer experiment. It uses the already selected
`all-exact` engine and changes only the command parser, tiled bridge, and board
top. The frozen Phase 5 implementation and its 64-byte protocol remain intact.

## Design

- Legacy commands 1–7, their CRC framing, and their 64-byte transfer limit are
  unchanged. `CAPS` still reports 64 bytes.
- Read-only command 8 (`FEATURES`) returns `BW 00 01`, identifying a 256-byte
  burst WRITE limit. An older image returns the existing unsupported-command
  status, so the host does not send burst writes to it.
- Command 9 (`BURST_WRITE`) accepts 1–256 bytes only in the external SDRAM
  window `0x800000..0xffffff`, while the accelerator is idle. It buffers the
  whole payload in one BSRAM, verifies the frame CRC, then commits the bytes.
  An oversized frame is drained before the parser accepts another command.
- The host client checks the feature response, response command, sequence,
  address, status and CRC. It does not retry an ambiguous write.

## Boardless checks

Run:

```sh
.venv/bin/python3 tools/phase6/run_uart_burst.py
.venv/bin/python3 tools/phase6/run_uart_burst_integration.py
.venv/bin/python3 test/phase6/test_uart_burst_host.py -v
```

The standalone RTL test passed one test covering 256-byte writes, CRC-before-
commit, busy/range rejection, oversized nested-frame drain, legacy 64-byte
read/write, and sequence wrap. The complete tiled bridge passed two tests:
the existing UART/DMA/engine/SDRAM regression and a new 256-byte burst write
through randomized external-port stalls with full readback. The host client
passed three unit tests for framing/chunking, bounds, and no retry after a
truncated response.

The isolated Gowin build used
`work/phase6/experiments-v1/all-exact/engine.sv` (SHA256
`36465f2b3cd6609ac21ecd194d9c5796fc41c8c29562fcafecfe71c651d76e47`)
and the target `GW2AR-LV18QN88C8/I7`. Its bitstream is
`work/phase6/uart-burst/route-blockram/phase6_uart_burst/impl/pnr/phase6_uart_burst.fs`
(SHA256 `9d8d711e9afecbb25c2672f95f668c954a99fb1dc1e4aada8b731d1a433c0f4d`).

| Routed result | Value |
|---|---:|
| Core clock | 20.25 MHz |
| Routed core Fmax | 22.649 MHz |
| Setup / hold violated endpoints | 0 / 0 |
| Logic | 18,317 / 20,736 |
| CLS | 9,908 / 10,368 |
| Registers | 5,228 / 15,915 |
| BSRAM | 37 / 46 |
| DSP | 14.5 / 24 |

The first register-mapped buffer version failed timing at 14.321 MHz. Mapping
the 256-byte payload to one BSRAM removed about 2,056 registers and produced
the routed result above. Route reports and the rejected attempts remain under
`work/phase6/uart-burst/`.

## Short physical comparison

When the board is free, program the bitstream in volatile SRAM and run the
checked upload comparison:

```sh
openFPGALoader -b tangnano20k --ftdi-serial 2025030317 --freq 2500000 -m -v work/phase6/uart-burst/route-blockram/phase6_uart_burst/impl/pnr/phase6_uart_burst.fs
.venv/bin/python3 tools/phase6/measure_uart_burst.py --run --bitstream work/phase6/uart-burst/route-blockram/phase6_uart_burst/impl/pnr/phase6_uart_burst.fs --report work/phase6/uart-burst/physical-transfer.json
```

The runner obtains the shared board lock, checks UART ownership and frozen
sources, probes both capabilities, then alternates four 8-KiB legacy/burst
uploads. It reads every upload back and records median wall time. It does not
program the board or launch the accelerator.

The physical comparison passed on the Tang Nano 20K with four exact SDRAM
readbacks. Median legacy upload time was **0.766668 s** (10.44 KiB/s), and
median burst upload time was **0.190772 s** (41.93 KiB/s): **4.0188× faster**.
The signed run report is
`work/phase6/uart-burst/physical-transfer.json`, archived in the
[follow-on evidence ledger](evidence/phase6/followon/summary.json).
This improves host transfer, not FPGA execution time. Combining the burst
bridge with the fastest compute image still requires a separate route and
board check.
