# Phase 3 SmallCNN candidate

This branch extends the numerical-v2 target with ordinary Conv and MaxPool
opcodes under target ID 8195. The previous phase-2 `.fs` remains archived at
`hardware/releases/phase2/` and advertises target ID 8194. Do not mix images
or bitstreams across those IDs.

From repository root, use the project's `.venv` and an installed Verilator:

```sh
make ci PYTHON=/absolute/path/to/.venv/bin/python3
make p3-native PYTHON=/absolute/path/to/.venv/bin/python3
make p3-switch PYTHON=/absolute/path/to/.venv/bin/python3
make p3-quality PYTHON=/absolute/path/to/.venv/bin/python3
/absolute/path/to/.venv/bin/python3 test/phase2/run.py engine
/absolute/path/to/.venv/bin/python3 test/phase2/run.py board
/absolute/path/to/.venv/bin/python3 tools/phase2/build_gowin.py work/phase3/gowin-cache-pipeline
```

Run `gw_sh build.tcl` in `work/phase3/gowin-cache-pipeline`, with the Gowin IDE library
path set as in [Phase 2](PHASE_2.md). Then regenerate the source manifest and
collect evidence:

```sh
/absolute/path/to/.venv/bin/python3 tools/phase2/generate_target.py --check --manifest work/phase3/gowin-cache-pipeline/source-manifest.json
cp work/phase3/gowin-cache-pipeline/tinyml_v2/impl/pnr/tinyml_v2.fs hardware/releases/phase3/tinyml_v3.fs
/absolute/path/to/.venv/bin/python3 tools/phase3/collect_evidence.py
```

The trained weight file is tracked, while MNIST raw files and generated
payloads under `work/` must be provided locally. The [Phase 3 status](../docs/research/PHASE_3_STATUS.md)
states which results are simulated, routed or still require physical access.
