# Selected Tang Nano 20K accelerator

This is the current fastest board-tested KWS/VWW release: the cached-weight
co-issue engine, pooled arithmetic, 27 MHz core, 256-byte CRC UART bridge at
750,000 baud, 32 KiB scratchpad, 32 KiB command memory and 8 MiB SDRAM address
space. Models, exact INT8/INT32 arithmetic and selected programs are unchanged.

The [protocol-matched board screen](../../../../docs/research/PHASE_6_ENGINE_PROFILE_AND_PROTOTYPE_V1.md)
completed 14/14 exact executions, following an initial 10/10 screen. Median
device execution was 1,119,142 cycles / **41.450 ms KWS** and 2,157,255 cycles /
**79.898 ms VWW**. These are separate-session short screens, excluding UART
loading and output readback; they do not establish full-dataset accuracy,
endurance, energy or architectural novelty.

The tested image is [selected_27mhz.fs.gz](selected_27mhz.fs.gz). Its uncompressed
SHA-256 is `eaf74e99519d3dfac876bf21e9087dd005071bb8b8682d3f8ae6fb23bb76644d`.
The routed core Fmax is 28.826 MHz with no setup/hold violations, 10,090/10,368
CLS, 38/46 BSRAM and 9.5/24 DSP. [manifest.json](manifest.json) pins all 18
physical sources, exact fixture files, native expectations and the signed board
records. The [portable board evidence](../../../../docs/research/evidence/phase6/engine-candidate-v2/portable/matched-v1/manifest.json)
contains the original reports, image and source identities.

Four fixtures preserve the tested command/payload/input/oracle bytes. Pinned
KWS/VWW and stress VWW check final output; stress KWS additionally checks ten
intermediate tensors and repeats the final tensor check under its layer name.
The [native validation report](validation/native-report.json)
records eight new exact runs on the promoted engine at fixed RAM and stall
seed 6063, including source, driver, harness and executable identities.
The [relocation check](validation/relocation-report.json) also materializes and
verifies a copied release in an independent checkout path.

Run from the repository root:

```sh
.venv/bin/python3 tools/phase6/selected_release.py materialize
.venv/bin/python3 tools/phase6/selected_release.py verify
.venv/bin/python3 tools/phase6/selected_release.py native
```

`materialize` recovers the hash-pinned Gowin encrypted SDRAM build artifact and
decompresses the tested image under ignored `work/`; existing files with different
hashes fail instead of being overwritten. This IP remains a vendor build
artifact, not project source. Its generation settings are documented in
[the SDRAM hardware instructions](../../../phase4_sdram/README.md). Native
builds use two jobs and an abstract external RAM, without board access.

For a fresh Gowin route, run the relocation-safe
[build_selected27.tcl](../../../phase6/build_selected27.tcl) in a new directory
under `work/`, after verification. Promoted RTL bytes match the tested source
inventory; a newly routed image can differ from the archived tested bitstream
and needs its own identity and physical qualification. Historical experimental
build recipes and candidates remain available separately.
