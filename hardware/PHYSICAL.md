# Tang Nano 20K physical validation

Use the active target ID **8196** and the reset-corrected release under
`hardware/releases/physical/`. The older Phase 2–4 bitstreams are historical
artifacts with the wrong KEY1 polarity. Their reports remain archived; use
the new release for board execution.

The board's KEY1 at pin 88 is **active high**. The wrapper initializes its
reset synchronizer at configuration, asserts reset while KEY1 is pressed,
and releases it after three clock edges. No button press is needed to begin
using the accelerator. This follows the inversion in Sipeed's
[UART reference](https://github.com/sipeed/TangNano-20K-example/blob/main/uart/src/uart_top.v).

## Build and program

Install the compiler/test environment and `tools/physical/requirements.txt`.
This execution used Gowin Education V1.9.11.03, Verilator 5.044,
openFPGALoader 1.1.1 and pyserial 3.5. From the repository root:

```sh
export USHQYN_ROOT="$PWD"
export DYLD_FRAMEWORK_PATH=/Applications/GowinIDE.app/Contents/Resources/Gowin_EDA/IDE/lib
export DYLD_LIBRARY_PATH="$DYLD_FRAMEWORK_PATH"
export GOWIN_SH=/Applications/GowinIDE.app/Contents/Resources/Gowin_EDA/IDE/bin/gw_sh
.venv/bin/python3 tools/phase2/build_gowin.py work/physical/new-build
.venv/bin/python3 tools/phase2/generate_target.py --check \
  --manifest work/physical/new-build/source-manifest.json
cd work/physical/new-build
"$GOWIN_SH" build.tcl
cd "$USHQYN_ROOT"
openFPGALoader -b tangnano20k --ftdi-serial 2025030317 --freq 2500000 -m -v \
  work/physical/new-build/tinyml_v2/impl/pnr/tinyml_v2.fs
```

For the frozen artifact, substitute
`hardware/releases/physical/tinyml_v4_tang20k.fs`. The test commands below
assume that frozen artifact was programmed. If you program a new build,
pass that exact new `.fs` to every `--bitstream` argument so the result records
its actual supplied hash. `-m` configures temporary
FPGA SRAM; the flash image is unchanged and returns after a power cycle.

The improved Phase 3 line-buffer image is separately archived at
`hardware/releases/phase3-linebuffer/tinyml_v5_candidate.fs` (SHA256
`fc6c90d3fb162eadfa042ecbfc1d89277c3a7a0af09fbc44a377c1fe90c81164`).
It has passed RTL simulation and Gowin routing, but has not been programmed
on a board. To validate it, substitute this path in the programming and
`--bitstream` commands below and write results to a new report directory.
Keep the physical release as the measured baseline until that rerun passes.

Close UART sessions before programming. Keep the successful programmer log
with the bitstream SHA256. The UART protocol reports the ABI, but cannot
read back a configuration hash.

The tested USB debugger serial is `2025030317`. On this Mac,
`/dev/cu.usbserial-20250303171` is the FPGA UART; channel `...170` is JTAG.
Other boards/hosts may enumerate differently. Pins are clock 4, KEY1 88,
UART RX 70 and TX 69. The FPGA part is `GW2AR-LV18QN88C8/I7`, revision C
as detected by JTAG; the PCB revision is unknown.

## Repeat the checks

Prepare the current-target model fixtures with the commands in
[Phase 2](PHASE_2.md) and [Phase 3](PHASE_3.md). The MLP preparation requires
the original locally held weight file. Then:

```sh
.venv/bin/python3 tools/physical/run_models.py \
  --port /dev/cu.usbserial-20250303171 \
  --fixture work/phase2/mlp --fixture work/phase3/smallcnn \
  --fixture work/phase2/mlp --jobs 1000 \
  --bitstream hardware/releases/physical/tinyml_v4_tang20k.fs \
  --report work/physical/model-switch.json

.venv/bin/python3 tools/physical/run_checks.py \
  --port /dev/cu.usbserial-20250303171 \
  --fixture work/phase2/mlp --fixture work/phase3/smallcnn \
  --bitstream hardware/releases/physical/tinyml_v4_tang20k.fs \
  --report work/physical/checks.json
```

`run_checks.py` verifies six patterns across all 32 KiB, malformed-frame
recovery, busy ownership, ABORT/RESET, seven directed kernel geometries
(three runs each), and every MLP/SmallCNN layer for three inputs. Individual
layer counters include that isolated RUN's startup/HALT overhead; they are
not additive measurements of uninterrupted whole-model execution.

For the complete SmallCNN accuracy set, either generate a new 10,000-case
fixture with `tools/phase3/prepare_smallcnn.py --jobs 10000`, or join the
already frozen full-set oracle to the current fixture:

```sh
.venv/bin/python3 tools/physical/prepare_smallcnn_quality.py
.venv/bin/python3 tools/physical/run_models.py \
  --port /dev/cu.usbserial-20250303171 \
  --fixture work/physical/smallcnn-full --jobs 10000 \
  --bitstream hardware/releases/physical/tinyml_v4_tang20k.fs \
  --report work/physical/smallcnn-full.json
```

The join checks software image, model, calibration and dataset identity,
all overlapping inputs/logits, and fresh independent oracle results for
the first, middle and final sample. It preserves cached-oracle hashes.
All runners save partial failure evidence and never automatically repeat
an ambiguous RUN. Use a separate report path for each new attempt.

For minimal transport bring-up, build `hardware/bringup/build.tcl` in a
fresh directory, program its `uart_loopback.fs`, then run
`tools/research/uart_readback.py --port DEVICE --report OUTPUT.json`.
It tests all byte values plus 1,024 seeded random bytes. Restore the
accelerator bitstream before running the model commands.

## Measurement boundary

The reports retain raw INT8 outputs, physical device counters and host
wall times. Core milliseconds derived from counters use the nominal
27-MHz clock; no independent clock instrument was available. Host job time
covers quantized-input upload, RUN/status polling and output readback;
it excludes preprocessing, argmax, report writing and model loading.
Model load plus full memory readback is reported separately. These bring-up
runs have no explicit warm-up exclusion and are not a tuned baseline comparison.

No power instrument is available, so board power/energy are unmeasured.
The present bitstream has no SDRAM controller or integrated DMA. Passing
these on-chip tests does not certify full KWS/VWW deployment, SDRAM, energy
efficiency or the remaining research novelty/baseline gates. See the
[physical results](../docs/research/PHYSICAL_BOARD_STATUS.md).
