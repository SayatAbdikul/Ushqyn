# Phase 4 kernel candidate (target ID 8196)

**Current board workflow:** use [PHYSICAL.md](PHYSICAL.md) for the corrected
KEY1 reset, physical release, build/program commands and measured results.
The build/archive commands below reproduce the original `667b948` snapshot,
whose bitstream has incorrect reset polarity. Preserve that frozen artifact;
do not overwrite it with a current build or rerun its collector as if the
historical source hashes described today's board image.

The current board image contains C04 kernel extensions, **not** SDRAM or DMA.
The Phase 3 bitstream and target ID 8195 remain archived. Use only an image
whose metadata reports target ID 8196 with this candidate.

From the repository root:

```sh
make p4-test PYTHON=/absolute/path/to/.venv/bin/python3
make p3-native PYTHON=/absolute/path/to/.venv/bin/python3
/absolute/path/to/.venv/bin/python3 tools/phase2/build_gowin.py work/phase4/gowin-kernels-wide
/absolute/path/to/.venv/bin/python3 tools/phase2/generate_target.py --check --manifest work/phase4/gowin-kernels-wide/source-manifest.json
```

Run `gw_sh build.tcl` from `work/phase4/gowin-kernels-wide` with the Gowin
`IDE/lib` directory on `DYLD_FRAMEWORK_PATH` and `DYLD_LIBRARY_PATH`, as in
[Phase 2](PHASE_2.md). Copy the routed `.fs` to
`hardware/releases/phase4/tinyml_v4_kernels.fs`, then run:

```sh
/absolute/path/to/.venv/bin/python3 tools/phase4/collect_evidence.py
```

The evidence collector refuses stale RTL, target, native SmallCNN result or
release bitstream. It requires passing compiler and RTL outputs plus the
frozen operator inventory audit. See [Phase 4 status](../docs/research/PHASE_4_STATUS.md)
for exact claims and open SDRAM/DMA gates.
