# Physical bring-up — 2026-09-23

The connected board was detected through Gowin Programmer, but its USB
interfaces disappeared before any bitstream was programmed. No physical
inference, UART loopback, or memory test has passed yet. Phase gates remain
open. A reconnection is required to continue.

## Observed identification

- Gowin Education Programmer V1.9.11.03 found one cable at USB location 1.
- JTAG scan found one device, ID `0x0000081B`, reported as
  `GW2A-18C GW2AR-18C (One of them)`, family GW2AR.
- Serial nodes initially appeared as `/dev/cu.usbserial-20250303170` and
  `/dev/cu.usbserial-20250303171`. Their UART function has not yet been tested.
- Subsequent device-code reads failed to open the cable, including an
  explicit channel and location. Both serial nodes then became unavailable;
  an independent openFPGALoader USB scan returned `No USB devices found`.
- The physical PCB revision and power instrument are still unverified.

The JTAG result supports the FPGA family/revision; it does not establish the
PCB revision, package pinout, power, or successful inference. No flash erase,
flash write, or SRAM configuration write was performed in this attempt.

## Prepared tools and artifacts

Gowin rebuilt the minimal UART loopback under
`work/physical/uart-build/uart_loopback/impl/pnr/uart_loopback.fs`.
Its SHA256 is
`79c2f7a824cdbc282c647a67b57cd15fe80dcf146971576a547634c7c56d065d`.
The current accelerator artifact remains
`hardware/releases/phase4/tinyml_v4_kernels.fs`, SHA256
`a214798c00b86a71940dd67f1a943403f15b3a642b9ddbb8cb9f715b172d0b33`.

Installed `pyserial==3.5` in `.venv` and openFPGALoader v1.1.1 through
Homebrew. The latter is an alternative programmer with documented
[macOS installation](https://trabucayre.github.io/openFPGALoader/guide/install.html)
and [Tang Nano 20K support](https://github.com/YosysHQ/apicula/wiki/openFPGALoader).
Sipeed's [board documentation](https://en.wiki.sipeed.com/hardware/en/tang/tang-nano-20k/nano-20k.html)
recommends direct USB connection and checking the cable when the onboard
debugger is not recognized.

`tools/physical/run_models.py` accepts multiple pinned fixtures so a single
UART session can run MLP, SmallCNN, then MLP without reprogramming the FPGA.
It checks target and image hashes, full image readback, each raw output byte,
MAC/cycle counters and protocol errors. It records per-job output/counters,
host timing, completed-job counts and partial failure evidence. Classification
accuracy remains separate from integer exactness. The supplied bitstream
hash records the chosen artifact; the UART cannot read the FPGA configuration
hash, so retain the successful programmer log alongside the test report.

The runner's four tests check normal collection, a wrong logit that leaves
the class unchanged, inconsistent counters, and a tampered model image:

```sh
.venv/bin/python3 -m unittest discover -s tools/physical -p 'test_*.py' -v
```

After programming and confirming the actual UART port, the initial repeatability
run is:

```sh
.venv/bin/python3 tools/physical/run_models.py \
  --port /dev/cu.CONFIRMED_BOARD_UART \
  --fixture work/phase2/mlp --fixture work/phase3/smallcnn \
  --fixture work/phase2/mlp --jobs 1000 \
  --bitstream hardware/releases/phase4/tinyml_v4_kernels.fs \
  --report work/physical/model-switch.json
```

Then collect full-set SmallCNN quality, per-layer/kernel profiles and protocol
recovery evidence. These can establish the currently implemented on-chip
path. SDRAM/DMA integration, complete KWS/VWW/AD models, tuned baselines and
instrumented energy remain additional work as recorded in the phase statuses.
