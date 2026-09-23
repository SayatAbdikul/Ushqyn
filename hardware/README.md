**Current physical implementation:** the reset-corrected
[Phase 4 kernel hierarchy](PHASE_4.md) runs on Tang Nano 20K at the declared
27-MHz clock. MLP → SmallCNN → MLP passed 1,000 exact jobs per stage, and
all seven directed kernel cases pass on the board. SDRAM and DMA are not
integrated. Use the [physical build/program/run workflow](PHYSICAL.md) and
[results record](../docs/research/PHYSICAL_BOARD_STATUS.md).
The original Phase 2–4 bitstreams remain historical artifacts with incorrect
KEY1 polarity; use the new physical release for board execution.

# Historical Phase-0 Gowin and board checks

The following section records the initial September 9 investigation of the
old hierarchy. Its missing-board and failed-synthesis statements are historical;
the current physical results and reproducible commands are linked above.

Gowin V1.9.11.03 Education is installed at
`/Applications/GowinIDE.app/Contents/Resources/Gowin_EDA/IDE` on the development
machine. The target is `GW2AR-LV18QN88C8/I7`, device revision **C**. This identifies
the build target; the physical board revision has not been read from a connected
board. No programming or power measurement was performed.

Run the scripts in separate fresh output directories. Example for macOS:

```sh
export DYLD_FRAMEWORK_PATH=/Applications/GowinIDE.app/Contents/Resources/Gowin_EDA/IDE/lib
export DYLD_LIBRARY_PATH="$DYLD_FRAMEWORK_PATH"
export GOWIN_SH=/Applications/GowinIDE.app/Contents/Resources/Gowin_EDA/IDE/bin/gw_sh
mkdir -p /private/tmp/ushqyn-uart-build
cd /private/tmp/ushqyn-uart-build
"$GOWIN_SH" /absolute/path/to/tinyML_accelerator/hardware/bringup/build.tcl
```

For the accelerator, generate `rtl/accelerator_config_pkg.sv` with
`make config PYTHON=/absolute/path/to/python`, then run `hardware/build_current.tcl`
in a different fresh directory. This builds the **current** `src/` hierarchy and
generated simulation profile deliberately; it does not silently shrink buffers
or replace the source with the installed historical project. The source SHA256
manifest in `docs/research/evidence/gowin/` records the files actually used.

Archived text extracts normalize trailing whitespace; numerical report content
is retained.

## Results from this execution

| Build | Result | Evidence |
|---|---|---|
| Minimal UART echo | Synthesis, route, bitstream generation passed; 75 LUT, 55 registers. Requested 27 MHz; internal path Fmax 304.389 MHz. | `docs/research/evidence/gowin/minimal-*` |
| Current accelerator | Synthesis **failed**: inferred 273,847 DFF, target limit 15,750. No routed current-accelerator timing/bitstream. | `current-accelerator-console.txt` and current source hashes |
| Installed historical project | Archived existing reports, not a fresh build of this checkout. Synthesis Fmax 89.201 MHz; **routed Fmax 37.502 MHz** at 27-MHz constraint; 43 BSRAM primitives. | `historical-installed-*`, source comparison manifest |

Minimal build warning PR1014 reports generic clock routing; UART/reset paths are
asynchronous and I/O timing is not comprehensively constrained. The internal Fmax
is not a verified board operating frequency or a high-speed interface claim.
Clocking/reset synchronization and full I/O constraint review belong to board
integration. The current-accelerator DFF failure makes the earlier nominal BSRAM
byte estimate insufficient: inference/memory implementation must be corrected.

Gowin's minimal-design power estimate is **125.162 mW**, including 122.800 mW
quiescent power, using default toggle assumptions (0.125), without VCD/SAIF.
It is an **estimated FPGA power figure**, not board power or energy per inference.
Do not publish it as measured energy. The host/USB bridge, regulator losses,
SDRAM activity and actual inference switching are not established by that report.

## Physical completion procedure (pending)

1. Record board PCB revision and serial adapter identity. Program the generated
   minimal bitstream using Gowin Programmer; record bitstream SHA256.
2. Press S1 to reset. Run `python tools/research/uart_readback.py --port DEVICE
   --report uart-report.json` with pyserial installed. The tool uses stop-and-wait
   at 115200 baud, tests all byte values and 1,024 deterministic random bytes.
   It must receive all 1,280 echoes exactly. Continuous burst echo is not promised.
3. Agree an actual power instrument: borrow a calibrated current/voltage monitor
   or an oscilloscope plus characterized shunt. Record model, bandwidth, sample
   rate, calibration and all supply paths. Until obtained, energy is unavailable.
4. Later add a GPIO inference marker and integrate V(t)I(t) over repeated runs.
   Report gross board energy and idle-subtracted energy separately, including
   sample counts and uncertainty. Post-route power remains a separately labeled
   estimate even after hardware measurements exist.

No Tang Nano UART device was identified in the serial-device enumeration used
for this execution. No instrument availability or second FPGA access was supplied.
R04 is therefore partial despite successful minimal place-and-route.
