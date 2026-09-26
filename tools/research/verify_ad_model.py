#!/usr/bin/env python3
"""AD source-float quality and real-input conversion checks; no INT8/FPGA claim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import tf_keras

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "compiler"))
from canonicalize import canonicalize


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(labels, scores):
    positive = scores[labels == 1][:, None]
    negative = scores[labels == 0][None, :]
    if not positive.size or not negative.size:
        raise ValueError("AUC needs both classes")
    return float(np.mean((positive > negative) + 0.5 * (positive == negative)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    if sha(a.data) != manifest["splits"]["accuracy"]["npz_sha256"]:
        raise ValueError("accuracy archive hash mismatch")
    if sha(a.source) != "651a62de1b7facd68d1269373992a2d09ffd28e6a816935d4834fcc558bde512":
        raise ValueError("source model hash mismatch")
    model = tf_keras.models.load_model(a.source, compile=False)
    original = onnx.load(a.onnx)
    canonical = canonicalize(original)
    runtimes = [ReferenceEvaluator(original), ReferenceEvaluator(canonical)]
    with np.load(a.data, allow_pickle=False) as data:
        features = data[manifest["input_name"]][:, 0, :]
        labels = data["recording_labels"]
        recording_index = data["recording_index"]
    errors = np.empty(len(features), np.float64)
    for start in range(0, len(features), 512):
        x = features[start:start + 512]
        expected = model(x, training=False).numpy()
        errors[start:start + len(x)] = np.mean(np.square(x - expected), axis=1)
    counts = np.bincount(recording_index)
    if len(labels) != 248 or not np.all(counts == 196):
        raise ValueError("unexpected recording/frame counts")
    scores = np.bincount(recording_index, weights=errors) / counts
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite reconstruction error")
    probes = []
    for index in np.linspace(0, len(features) - 1, 32, dtype=int):
        x = features[index:index + 1]
        expected = model(x, training=False).numpy()
        errors_by_stage = []
        for graph, runtime in zip((original, canonical), runtimes):
            actual = runtime.run(None, {graph.graph.input[0].name: x})[0]
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
            errors_by_stage.append(float(np.max(np.abs(actual - expected))))
        probes.append({"frame_index": int(index), "input_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                       "onnx_max_abs_error": errors_by_stage[0], "canonical_max_abs_error": errors_by_stage[1]})
    records = manifest["splits"]["accuracy"]["records"]
    ids = np.asarray([r["path"].split("_id_")[1][:2] for r in records])
    per_id = {i: auc(labels[ids == i], scores[ids == i]) for i in sorted(set(ids))}
    result = {"status": "passed", "source_sha256": sha(a.source), "onnx_sha256": sha(a.onnx),
              "canonical_sha256": hashlib.sha256(canonical.SerializeToString()).hexdigest(),
              "data_sha256": sha(a.data), "manifest_sha256": sha(a.manifest),
              "script_sha256": sha(Path(__file__)), "runtime": "tf_keras " + tf_keras.__version__,
              "scope": "research-reconstructed AD float model; official binary parity unverified; no INT8/FPGA quality claim",
              "recordings": len(labels), "frames": len(features), "atol": 1e-5, "rtol": 1e-4,
              "real_input_parity_probes": probes, "pooled_roc_auc": auc(labels, scores),
              "per_machine_id_roc_auc": per_id, "macro_machine_id_roc_auc": float(np.mean(list(per_id.values()))),
              "records": [{"path": r["path"], "label": int(label), "score": float(score)}
                          for r, label, score in zip(records, labels, scores)]}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "recordings", "frames", "pooled_roc_auc", "macro_machine_id_roc_auc")}))


if __name__ == "__main__":
    main()
