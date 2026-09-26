#!/usr/bin/env python3
"""Materialize AD vectors using pinned upstream functions and librosa 0.6.0.

The compatibility aliases below restore removed API names, not DSP algorithms.
This is a research reconstruction, without parity against official .bin inputs.
"""
import argparse
import ast
import hashlib
import json
import logging
from pathlib import Path
import sys
import types
import wave
import numpy as np

from fetch_ad_data import selection

COMMON_SHA = "061f809bb231f41549acbbfec6573e056e456c9f8644c309c13026481e768d72"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def upstream_preprocess(path):
    import joblib
    import numba
    joblib.Memory.cachedir = property(lambda self: self.location)
    sys.modules["numba.decorators"] = types.SimpleNamespace(jit=numba.jit)
    np.float, np.complex, np.int = float, complex, int
    import librosa
    if librosa.__version__ != "0.6.0" or sha(path.read_bytes()) != COMMON_SHA:
        raise ValueError("preprocessing version/hash mismatch")
    scope = {"numpy": np, "librosa": librosa, "sys": sys,
             "logger": logging.getLogger("ad-preprocessing")}
    tree = ast.parse(path.read_text())
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ("file_load", "file_to_vector_array")]
    if len(funcs) != 2:
        raise ValueError("missing upstream functions")
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(path), "exec"), scope)
    return scope["file_to_vector_array"]


def windows_from_histogram(raw):
    hist = np.frombuffer(raw, dtype="<f4")
    if hist.size != 200 * 128 or not np.isfinite(hist).all():
        raise ValueError("expected finite 200-frame, 128-bin log-mel histogram")
    hist = hist.reshape(200, 128)
    return np.stack([hist[i:i + 5].reshape(640) for i in range(196)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-name", default="input_1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.data / "raw-manifest.json"
    raw = json.loads(raw_path.read_text())
    expected = selection(args.upstream)
    if [dict((k, r[k]) for k in e) for r, e in zip(raw["records"], expected)] != expected:
        raise ValueError("raw manifest selection differs from frozen policy")
    if len(raw["records"]) != len(expected):
        raise ValueError("raw manifest recording count mismatch")
    preprocess = upstream_preprocess(args.upstream / "benchmark/training/anomaly_detection/common.py")
    splits = {}
    feature_hashes = {}
    for split in ("calibration", "accuracy"):
        records, features, sample_ids, labels, clip_indices = [], [], [], [], []
        for clip_index, record in enumerate(r for r in raw["records"] if r["split"] == split):
            path = args.data / record["path"]
            if sha(path.read_bytes()) != record["raw_sha256"]:
                raise ValueError(f"raw WAV hash mismatch: {path}")
            with wave.open(str(path), "rb") as wav:
                if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
                    raise ValueError("expected mono 16kHz PCM16 WAV")
            vectors = preprocess(str(path), n_mels=128, frames=5, n_fft=1024,
                                 hop_length=512, power=2.0, save_bin=True).astype("<f4")
            histogram = path.with_name(path.stem + "_hist_librosa.bin").read_bytes()
            framed = windows_from_histogram(histogram)
            if vectors.shape != (196, 640) or not np.array_equal(vectors, framed):
                raise ValueError("upstream vector/bin-window disagreement")
            features.append(vectors[:, None, :])
            sample_ids.extend(f"ad/{record['path']}#frame={i}" for i in range(196))
            labels.extend([record["label"]] * 196)
            clip_indices.extend([clip_index] * 196)
            records.append({**record, "frames": 196, "feature_sha256": sha(vectors.tobytes()),
                            "histogram_sha256": sha(histogram)})
            if (clip_index + 1) % 20 == 0:
                print(f"AD {split}: {clip_index + 1} recordings", flush=True)
        features = np.concatenate(features)
        feature_hashes[split] = {sha(x.tobytes()) for x in features}
        target = args.output / f"ad.{split}.npz"
        np.savez_compressed(target, **{args.input_name: features, "sample_ids": np.asarray(sample_ids),
                            "labels": np.asarray(labels), "recording_index": np.asarray(clip_indices),
                            "recording_labels": np.asarray([r["label"] for r in records])})
        splits[split] = {"count": len(features), "recording_count": len(records),
                         "npz_sha256": sha(target.read_bytes()),
                         "sample_ids_sha256": sha(("\n".join(sample_ids) + "\n").encode()),
                         "feature_content_sha256": sha(features.tobytes()), "records": records}
    if feature_hashes["calibration"] & feature_hashes["accuracy"]:
        raise ValueError("calibration/evaluation feature overlap")
    result = {"schema": 1, "workload": "ad", "input_name": args.input_name,
              "raw_manifest_sha256": sha(raw_path.read_bytes()),
              "preparation_script_sha256": sha(Path(__file__).read_bytes()),
              "upstream_preprocessing_sha256": COMMON_SHA,
              "calibration_policy": raw["calibration_policy"],
              "preprocessing": "librosa 0.6.0; mono16kHz; n_fft1024; hop512; 128 mels; power2; 10log10(mel+float64 epsilon); central frames[50:250]; five-frame windows; float32",
              "splits": splits, "calibration_accuracy_disjoint": True,
              "official_preprocessed_binary_parity_verified": False,
              "score": "mean squared reconstruction error over 640 features and all 196 frames of each recording; larger means more anomalous",
              "quality_gate": "AD INT8/FPGA ROC-AUC >=0.85 is a later-phase test, not a Phase0 result"}
    (args.output / "ad.data.json").write_text(json.dumps(result, indent=2) + "\n")
    print("AD feature provenance complete", flush=True)


if __name__ == "__main__":
    main()
