#!/usr/bin/env python3
"""Held-out AD INT8/software and structural tile experiment.

The batched path is an evaluation accelerator, checked against the production
single-window VM and independent integer oracle on real held-out windows.
No FPGA timing or board correctness is inferred from this report.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import onnx
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler"))
from canonicalize import canonicalize
from integer_reference import evaluate as integer_oracle
from phase4_compile import compile_tiled
from phase4_sequence import compile_sequence
from program_image import load_image
from quantization import multiplier_shift
from static_pipeline import execute_layer, run


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(labels, scores):
    positive = scores[labels == 1][:, None]
    negative = scores[labels == 0][None, :]
    if not positive.size or not negative.size:
        raise ValueError("AUC needs both classes")
    return float(np.mean((positive > negative) + 0.5 * (positive == negative)))


def requantize_batch(acc, multipliers, shifts, zero_point):
    acc = np.asarray(acc, dtype=np.int64)
    if np.any(acc < -(1 << 31)) or np.any(acc > (1 << 31) - 1):
        raise ValueError("batched accumulator does not fit INT32")
    multipliers = np.asarray(multipliers, dtype=np.int64).reshape(1, -1)
    shifts = np.asarray(shifts, dtype=np.int64).reshape(1, -1)
    product = acc * multipliers
    rounding = np.where(shifts, np.left_shift(1, np.maximum(shifts - 1, 0)), 0)
    magnitude = (np.abs(product) + rounding) >> shifts
    signed = np.where(product < 0, -magnitude, magnitude) + zero_point
    return np.clip(signed, -128, 127).astype(np.int8)


def run_batch(program, features, save_layers=False):
    x = program.tensors[program.inputs[0]].quantization.encode(features).reshape(len(features), -1)
    outputs = []
    for layer in program.layers:
        if layer.op == "Gemm":
            p = layer.parameters
            acc = x.astype(np.int64) @ p["weight"].astype(np.int64).T
            acc += p["corrected_bias"].astype(np.int64)[None, :]
            x = requantize_batch(acc, p["multiplier"], p["shift"],
                                 program.tensors[layer.output].quantization.zero_point)
        elif layer.op == "Relu":
            iq = program.tensors[layer.inputs[0]].quantization
            oq = program.tensors[layer.output].quantization
            multiplier, shift = multiplier_shift(iq.scale / oq.scale)
            centered = np.maximum(x.astype(np.int64), iq.zero_point) - iq.zero_point
            x = requantize_batch(centered, [multiplier], [shift], oq.zero_point)
        else:
            raise ValueError(f"AD batch evaluator requires Gemm/Relu, found {layer.op}")
        if save_layers:
            outputs.append(x.copy())
    return x, outputs


def float_batch(model, features):
    constants = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    x = features.reshape(len(features), -1).astype(np.float32)
    for node in model.graph.node:
        if node.op_type == "Gemm":
            x = x @ constants[node.input[1]].T + constants[node.input[2]]
        elif node.op_type == "Relu":
            x = np.maximum(x, 0)
        else:
            raise ValueError(f"canonical AD batch evaluator cannot run {node.op_type}")
    return x


def check_probes(program, canonical, features, indices):
    runtime = ReferenceEvaluator(canonical)
    failures = []
    float_max_abs = 0.0
    for index in indices:
        sample = features[index]
        qinput = program.tensors[program.inputs[0]].quantization.encode(sample)
        oracle = integer_oracle(program, {program.inputs[0]: qinput})
        values = {program.inputs[0]: qinput}
        batched, stages = run_batch(program, sample[None], save_layers=True)
        for layer, stage in zip(program.layers, stages):
            values[layer.output] = execute_layer(program, layer, values)
            if not np.array_equal(values[layer.output], oracle[layer.output]):
                failures.append([int(index), layer.output, "VM vs oracle"])
            if not np.array_equal(stage[0], oracle[layer.output].reshape(-1)):
                failures.append([int(index), layer.output, "batch vs oracle"])
        direct = run(program, {program.inputs[0]: sample})[program.outputs[0]]
        if not np.array_equal(direct.reshape(-1), batched[0]):
            failures.append([int(index), program.outputs[0], "run vs batch"])
        expected = runtime.run(None, {canonical.graph.input[0].name: sample})[0]
        actual = float_batch(canonical, sample[None])[0]
        float_max_abs = max(float_max_abs, float(np.max(np.abs(expected.reshape(-1) - actual))))
        if not np.allclose(expected.reshape(-1), actual, rtol=1e-4, atol=1e-5):
            failures.append([int(index), canonical.graph.output[0].name, "float batch vs ONNX"])
    if failures:
        raise AssertionError(f"probe mismatch: {failures[:5]}")
    return {"count": len(indices), "indices": [int(i) for i in indices],
            "all_19_layers_exact": True, "float_max_abs_difference": float_max_abs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT / "work/phase6/parallel-ad-v1/ad.uq2")
    parser.add_argument("--model", type=Path, default=ROOT / "work/ad-float.onnx")
    parser.add_argument("--calibration", type=Path, default=ROOT / "work/phase0-ad/features/ad.calibration.npz")
    parser.add_argument("--accuracy", type=Path, default=ROOT / "work/phase0-ad/features/ad.accuracy.npz")
    parser.add_argument("--manifest", type=Path, default=ROOT / "benchmarks/manifests/ad.data.json")
    parser.add_argument("--report", type=Path, default=ROOT / "work/phase6/parallel-ad-v1/report.json")
    parser.add_argument("--chunk", type=int, default=256)
    args = parser.parse_args()
    if args.chunk < 1 or args.chunk > 2048:
        raise ValueError("chunk must be in [1, 2048]")
    manifest = json.loads(args.manifest.read_text())
    if digest(args.calibration) != manifest["splits"]["calibration"]["npz_sha256"]:
        raise ValueError("calibration archive hash differs from frozen manifest")
    if digest(args.accuracy) != manifest["splits"]["accuracy"]["npz_sha256"]:
        raise ValueError("accuracy archive hash differs from frozen manifest")
    program = load_image(args.image.read_bytes())
    canonical = canonicalize(onnx.load(args.model))
    if hashlib.sha256(canonical.SerializeToString()).hexdigest() != program.provenance["calibration"]["model_sha256"]:
        raise ValueError("image/model calibration hash mismatch")
    with np.load(args.calibration, allow_pickle=False) as data:
        calibration_ids = data["sample_ids"]
        if calibration_ids.tolist() != program.provenance["calibration"]["sample_ids"]:
            raise ValueError("image is not calibrated on the complete frozen AD split")
    with np.load(args.accuracy, allow_pickle=False) as data:
        features = data[manifest["input_name"]]
        ids = data["sample_ids"]
        recording_index = data["recording_index"]
        labels = data["recording_labels"]
    if len(features) != 48608 or len(labels) != 248 or len(ids) != len(features):
        raise ValueError("unexpected held-out AD dimensions")
    if set(ids.tolist()) & set(calibration_ids.tolist()):
        raise ValueError("calibration and evaluation IDs overlap")
    calibration_hashes = set(program.provenance["calibration"]["sample_sha256"])
    for sample in features:
        h = hashlib.sha256()
        h.update(program.inputs[0].encode())
        h.update(sample.astype("<f4").tobytes())
        if h.hexdigest() in calibration_hashes:
            raise ValueError("calibration and evaluation contents overlap")
    indices = np.linspace(0, len(features) - 1, 32, dtype=int)
    probes = check_probes(program, canonical, features, indices)
    plan, parameter_image = compile_tiled(program)
    sequence = {}
    for overlap in (False, True):
        command, payload, record = compile_sequence(plan, parameter_image, overlap=overlap)
        sequence["overlap" if overlap else "sequential"] = {
            "command_count": record["command_count"], "program_bytes": record["program_bytes"],
            "payload_bytes": len(payload), "prefetch_payload_bytes": record["prefetch_payload_bytes"],
            "final_output": record["final_output"], "command_sha256": hashlib.sha256(command).hexdigest(),
        }
    counts = np.bincount(recording_index, minlength=len(labels))
    if not np.all(counts == 196):
        raise ValueError("AD recordings must contain 196 windows")
    quant_error = np.empty(len(features), np.float64)
    float_error = np.empty(len(features), np.float64)
    start_time = time.perf_counter()
    oq = program.tensors[program.outputs[0]].quantization
    for start in range(0, len(features), args.chunk):
        sample = features[start:start + args.chunk]
        qout, _ = run_batch(program, sample)
        fout = float_batch(canonical, sample)
        original = sample[:, 0, :].astype(np.float64)
        quant_error[start:start + len(sample)] = np.mean((original - oq.decode(qout)) ** 2, axis=1)
        float_error[start:start + len(sample)] = np.mean((original - fout.astype(np.float64)) ** 2, axis=1)
        if (start // args.chunk) % 32 == 0:
            print(f"AD windows {start + len(sample)}/{len(features)}", flush=True)
    quant_scores = np.bincount(recording_index, weights=quant_error, minlength=len(labels)) / counts
    float_scores = np.bincount(recording_index, weights=float_error, minlength=len(labels)) / counts
    records = manifest["splits"]["accuracy"]["records"]
    machine_ids = np.asarray([record["path"].split("_id_")[1][:2] for record in records])
    per_id = {i: {"float_auc": auc(labels[machine_ids == i], float_scores[machine_ids == i]),
                  "int8_auc": auc(labels[machine_ids == i], quant_scores[machine_ids == i])}
              for i in sorted(set(machine_ids))}
    report = {
        "status": "passed", "scope": "full frozen AD split; software INT8 plus compiler structural feasibility; no FPGA execution",
        "model_sha256": digest(args.model), "canonical_sha256": hashlib.sha256(canonical.SerializeToString()).hexdigest(),
        "calibration_npz_sha256": digest(args.calibration), "accuracy_npz_sha256": digest(args.accuracy),
        "image_sha256": digest(args.image), "image_bytes": args.image.stat().st_size,
        "calibration_windows": len(calibration_ids), "evaluation_windows": len(features),
        "recordings": len(labels), "source_units": "one inference per five-frame 640-feature window; 196 per recording",
        "id_and_content_disjoint": True, "probes": probes,
        "float_pooled_roc_auc": auc(labels, float_scores),
        "int8_pooled_roc_auc": auc(labels, quant_scores),
        "float_macro_machine_id_roc_auc": float(np.mean([v["float_auc"] for v in per_id.values()])),
        "int8_macro_machine_id_roc_auc": float(np.mean([v["int8_auc"] for v in per_id.values()])),
        "per_machine_id": per_id,
        "eval_seconds": time.perf_counter() - start_time,
        "layers": len(plan["layers"]), "tiles": sum(len(layer["tiles"]) for layer in plan["layers"]),
        "layer_tiles": [{"index": layer["index"], "kind": layer["kind"],
                         "tiles": len(layer["tiles"]),
                         "peak_scratch_bytes": max((tile["scratch_bytes"] for tile in layer["tiles"]), default=0)}
                        for layer in plan["layers"]],
        "parameter_image_bytes": len(parameter_image),
        "activation_slot_bytes": plan["activation_slot_bytes"],
        "max_tile_scratch_bytes": max(tile["scratch_bytes"] for layer in plan["layers"] for tile in layer["tiles"]),
        "dma_payload_bytes_per_window": sum(transfer["bytes"] for layer in plan["layers"]
                                            for tile in layer["tiles"] for transfer in tile["transfers"]),
        "sequence": sequence,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "float_pooled_roc_auc", "int8_pooled_roc_auc", "layers", "tiles", "eval_seconds")}))


if __name__ == "__main__":
    main()
