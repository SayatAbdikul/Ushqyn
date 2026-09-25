# Phase 4 boardless real-model tile validation

The pinned MLCommons KWS and VWW source artifacts are retrieved with per-file
SHA-256 verification. A fresh conversion may produce a different ONNX binary
from the frozen September 2026 build because generated constant names can
change. `rebase_calibration.py` permits only constant-name drift after source
hash, source-framework conversion parity, operator, attribute, activation
tensor and constant-geometry checks. It records both canonical hashes. These
new tile images are development artifacts, not byte-identical copies of the
frozen software-v2 binaries or new accuracy measurements.

From the repository root, use the Python 3.11 conversion environment and the
Python 3.13 compiler environment:

```sh
python3.11 -m venv work/phase4/convert-venv
work/phase4/convert-venv/bin/python -m pip install -r tools/research/requirements-conversion.txt
.venv/bin/python3 tools/research/fetch_artifacts.py benchmarks/manifests/kws.json work/upstream
.venv/bin/python3 tools/research/fetch_artifacts.py benchmarks/manifests/vww.json work/upstream
work/phase4/convert-venv/bin/python tools/research/convert_savedmodel.py work/upstream/benchmark/training/keyword_spotting/trained_models/kws_ref_model work/phase4/kws.onnx work/phase4/kws-conversion.json
work/phase4/convert-venv/bin/python tools/research/convert_benchmark.py work/upstream/benchmark/training/visual_wake_words/trained_models/vww_96_float.tflite work/phase4/vww.onnx work/phase4/vww-conversion.json
```

For each workload, substitute `kws` or `vww` for `NAME`:

```sh
.venv/bin/python3 tools/phase4/prepare_logits.py work/phase4/NAME.onnx work/phase4/NAME-logits.onnx work/phase4/NAME-boundary.json
.venv/bin/python3 tools/research/inventory_onnx.py work/phase4/NAME.onnx work/phase4/NAME-inventory.json --classifier
.venv/bin/python3 tools/phase4/rebase_calibration.py NAME work/phase4/NAME.onnx work/phase4/NAME-inventory.json work/phase4/NAME-conversion.json work/phase4/NAME-calibration-rebased.json work/phase4/NAME-rebase-report.json
.venv/bin/python3 tools/phase4/compile_tiled.py work/phase4/NAME-logits.onnx work/phase4/NAME-calibration-rebased.json work/phase4/NAME-parameter-image.bin work/phase4/NAME-tiled-plan.json
.venv/bin/python3 tools/phase4/prepare_rtl_fixture.py NAME work/phase4/NAME-logits.onnx work/phase4/NAME-calibration-rebased.json work/phase4/NAME-tiled-plan.json work/phase4/NAME-parameter-image.bin work/phase4/rtl-NAME
.venv/bin/python3 test/phase4/run_tiled_program.py --fixture work/phase4/rtl-NAME
```

The fixture uses one deterministic quantized input. An independent centered
integer oracle saves every intermediate tensor. The Cocotb test uploads each
descriptor, performs every planned transfer through the real `v2_tile_dma`
and `v2_scratchpad` RTL, executes `v2_engine`, and compares each output byte.
External memory is an 8-MiB behavioral array with randomized response delay
and backpressure. The reported simulated clocks include test-host descriptor
loads and all behavioral memory stalls; they are not physical inference FPS.

The KWS graph has 22 operators and 20 compute tiles; its parameter image
through the final packed region is 49,376 bytes. VWW has 58 operators and 75
compute tiles; its image is 334,816 bytes. Actual board closure still needs an
physical output validation, useful compute/DMA overlap and measured bandwidth.
The SDRAM-connected host-command hierarchy now exists and meets post-route
timing, but has not been programmed on the board. Complete-set accuracy uses
the frozen quality fixtures and is a separate check.

Both one-input full-graph RTL runs passed on 2026-09-24. The archived
[KWS report](evidence/phase4/boardless-real-kws.json) records all 22 exact
node outputs, a 20-tile parameter-image hash of
`7157cfdbc8380d6283cebc1c3819c44c3532ae28508bbf07b7577d0f3b656139`,
and final output hash
`ce178213db2ef2aa27d0b532e4a17851cfef1772ad74b2fc7a95311b4a6714a7`.
The [VWW report](evidence/phase4/boardless-real-vww.json) records all 58
exact node outputs, 75 tiles, parameter-image hash
`6251628dcaf714c8690a3454126ee4a1ac6a7eebe5d7d044af812bead6bc4805`,
and final output hash
`6a0cfa0be8aa6d3a3137a6926b09aa569b91c44d1b845612779c2d55e019dbc2`.
The corresponding [KWS](evidence/phase4/real-kws-fixture.json) and
[VWW](evidence/phase4/real-vww-fixture.json) fixture manifests pin the input,
model, calibration, plan and expected-output hashes. The
[KWS](evidence/phase4/real-kws-simulation.txt) and
[VWW](evidence/phase4/real-vww-simulation.txt) transcripts retain node-level
pass events. The external-memory port is behavioral with randomized stalls;
the recorded 5,561,680 KWS and 18,008,277 VWW simulated clocks include host
activity and must not be read as physical inference latency or FPS.

The newer framed-host regression executes the same prepared fixture through
the packet parser, host-visible 8-MiB window, DMA registers and tiled engine.
It validates the upload/run/readback sequence intended for the routed
[board candidate](../../hardware/phase4_sdram/TILED_HOST.md), still using a
behavioral external-memory port. To reproduce it after preparing the fixtures:

```sh
.venv/bin/python3 test/phase4/run_tiled_host.py --fixture work/phase4/rtl-kws
.venv/bin/python3 test/phase4/run_tiled_host.py --fixture work/phase4/rtl-vww
```

The separate [physical host runner](../../tools/phase4/tiled_host.py) uses the
same tile plan and exact expected tensors once a board is available. It
records engine counters and DMA payload bytes independently of host wall time.
The framed-host boardless runs passed all [22 KWS nodes](evidence/phase4/boardless-host-kws.json)
and [58 VWW nodes](evidence/phase4/boardless-host-vww.json) on 2026-09-25.
Those reports pin fixture and transcript hashes. They do not exercise the
encrypted controller's physical SDRAM signaling or establish board FPS.
