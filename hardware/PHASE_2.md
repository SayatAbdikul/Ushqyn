# Tang Nano 20K phase-2 candidate (historical release)

These reproduction steps describe commit `f39e67e` and target ID 8194.
Current development sources implement the [Phase 4 kernel target](PHASE_4.md)
with target ID 8196. Check out `f39e67e` to reproduce the archived Phase 2 `.fs`.

The [status record](../docs/research/PHASE_2_STATUS.md) distinguishes routed
results from pending physical checks. Active sources are exactly those in
`targets/tang_nano_20k_v2.json`; `rtl/v2/` owns compute, memory, commands and
UART. `tang_nano_v2.sv` is the board clock/reset/pin wrapper. Legacy `src/` and
`rtl/` v1 execution modules are not part of this build.

## Rebuild

Use the tested Python environment and Gowin Education V1.9.11.03 or a compatible
installation. From repository root:

```sh
make ci PYTHON=/absolute/path/to/python
make v2-lint PYTHON=/absolute/path/to/python
make v2-fixture PYTHON=/absolute/path/to/python
/absolute/path/to/python tools/phase2/run_native.py
/absolute/path/to/python tools/phase2/build_gowin.py work/phase2/gowin
/absolute/path/to/python tools/phase2/generate_target.py --check --manifest work/phase2/gowin/source-manifest.json
```

Run `gw_sh build.tcl` from `work/phase2/gowin`, with the Gowin `IDE/lib` directory
in `DYLD_FRAMEWORK_PATH` and `DYLD_LIBRARY_PATH` on this macOS installation.
The resulting `tinyml_v2.fs` is for the `GW2AR-LV18QN88C8/I7` revision-C
configuration. Verify its part and PCB pinout before programming. The archived
candidate is [tinyml_v2.fs](releases/phase2/tinyml_v2.fs); its hash and routed
resource/timing evidence are in `docs/research/evidence/phase2/`.

`make v2-test PYTHON=/absolute/path/to/python` runs Verilator requantizer,
randomized-memory engine, board-UART and system/MLP suites. It expects the
ignored `compiler/digit_model_weights.pth` fixture and local MNIST IDX files.
The fixture generator verifies/traces their hashes. `run_native.py` provides a
faster 1,000-job regression and records every intermediate for three jobs.

To convert another calibrated software-v2 image, run
`python tools/phase2/compile_board.py software.uq2 board.bin`. This rejects
unsupported operators or networks beyond the 32-KiB SRAM budget. The host must
check `CAPS` and image metadata before writing the board image. The current
hardware ABI supports batch-one Gemm, ReLU and quantization-preserving
Flatten/Reshape/Identity. No spatial Conv/pool or off-chip DMA is implemented.

## Descriptor and memory contract

All addresses are byte addresses within a 24-bit field. The current physical SRAM
range is `[0,32768)`; a larger logical field does not create more memory. The
64-byte descriptor is little-endian:

| Offset | Field |
|---:|---|
| 0 | `USH2` magic (32 bits), version 2, opcode, zero flags/reserved bytes |
| 8 | input, output, weight, parameter byte addresses (four 32-bit words; upper eight bits must be zero) |
| 24 | reduction count, output count, weight-row stride, next descriptor address (four 32-bit words) |
| 40 | 12 independent 16-bit geometry fields: kernel H/W, stride H/W, four pads, input H/W/C, output C |

HALT opcode 0 terminates. GEMM=1 uses row-padded INT8 weights (eight-byte
alignment) and one 16-byte parameter record per output channel. Its parameter
record holds corrected INT32 bias, signed Q31 multiplier, shift 0–62, output and
input zero points, clamp bounds and two zero reserved bytes. ReLU=2 has one
parameter record; COPY=3 requires unchanged quantization. Spatial descriptor
fields are checked but any nontrivial geometry rejects in this release. No
output is emitted for unsupported graphs in `compile_board.py`.

The SRAM offers one synchronous eight-byte request per cycle. `req && ready`
accepts a request. Reads respond with `rvalid` on a later clock; writes use one
byte strobe per bank. Requests must remain stable until accepted. Engine and host
share this port; host reads/writes while busy return a busy error. RUN clears
engine state/counters, while SRAM weights persist. ABORT stops the job and
invalidates the pending result. No reset loop clears inferred block RAM.

## Host framing and physical validation

Wire format is `A5 5A | version | command | sequence | 24-bit address |
16-bit length | optional WRITE payload | CRC16-CCITT`, little-endian multi-byte
fields and CRC bytes, CRC initialized to `FFFF` over all bytes after sync through
payload. Responses echo version, command with bit 7 set, sequence and address;
length includes one status byte, followed by data and CRC. Maximum READ/WRITE
payload is 64 bytes. Invalid CRC, version, range, length, descriptor and busy
states return distinct status codes. An oversized WRITE drains its declared
payload before returning an error; an incomplete frame times out. Host clients
must not blindly retry an ambiguous RUN.

`CAPS` reports numerical/descriptor versions, lanes, transfer size, SRAM size,
address width and target ID. `STATUS` reports busy/error plus elapsed, compute,
wait and control cycles; useful MACs, SRAM read/write bytes, completed layers
and protocol errors. Elapsed equals compute+wait+control; wait counts requests
and delayed responses, not just external memory stalls. LED status gives busy,
layer marker and error indication (active low). The UART uses 115200 8N1.

With a connected, identified board and its programming cable, use Gowin
Programmer to load the `.fs` for the verified part. The serial device must be
explicitly supplied; no serial path is guessed. Then run:

```sh
python tools/phase2/verify_board.py --port /dev/cu.YOUR_DEVICE --fixture work/phase2/mlp --jobs 1000 --report work/phase2/board-measurement.json
```

The script resets protocol state, checks capabilities, writes the full model
image once, reads it back, changes input for each job, runs, reads output and
checks exact integer values and counters. It does not measure power. Save the
actual board revision, programmer/serial identities and measurement-instrument
metadata alongside the output before claiming physical completion.
