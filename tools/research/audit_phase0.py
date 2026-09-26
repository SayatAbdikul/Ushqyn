#!/usr/bin/env python3
"""Audit the frozen G0 contract and data/build evidence; does not run the board."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(upstream, primary_data, ad_data):
    base = ROOT / "benchmarks/manifests"
    counts = {"kws": (96, 4890), "vww": (11, 10961), "ad": (21952, 48608)}
    workloads = {}
    for name, expected_counts in counts.items():
        spec = read(base / f"{name}.json")
        require(spec["revision"] == "4addd0fa08d216e20637637874e084895f289da4", "upstream revision changed")
        for entry in spec["files"] + [spec["evaluator_rules"]]:
            require(digest(upstream / entry["path"]) == entry["sha256"], f"upstream hash: {entry['path']}")
        parity = read(base / spec["conversion_parity"]["report"])
        if name == "vww":
            require(parity["status"] == "passed-synthetic-float-artifact-parity", "VWW artifact parity")
            source = parity["keras_source_parity"]
            require(source["status"] == "passed" and source["samples"] == 16 and
                    (source["atol"], source["rtol"]) == (1e-5, 1e-4), "VWW original-H5 parity")
        else:
            require(parity["status"] == "passed-synthetic-source-framework-parity", f"source parity: {name}")
        require(len(parity["samples"]) == 16 and (parity["atol"], parity["rtol"]) == (1e-5, 1e-4), "parity coverage/tolerance changed")
        inventory = read(base / spec["canonical_inventory"])
        require(inventory["source_onnx_sha256"] == parity["onnx_sha256"], "inventory conversion mismatch")
        require(inventory["unknown_required_operators"] == [], f"unknown required operator: {name}")
        data = read(base / spec["dataset"]["split_manifest"])
        require(data["calibration_accuracy_disjoint"] is True, "split disjointness missing")
        cal, accuracy = [data["splits"][split] for split in ("calibration", "accuracy")]
        require((cal["count"], accuracy["count"]) == expected_counts, f"wrong sample count: {name}")
        for key in ("raw_sha256", "feature_sha256"):
            left, right = [{r[key] for r in split["records"]} for split in (cal, accuracy)]
            require(not left & right, f"calibration/accuracy {key} overlap: {name}")
            require(all(len(v) == 64 and set(v) <= set("0123456789abcdef") for v in left | right), "invalid content hash")
        require(cal["npz_sha256"] == spec["calibration"]["npz_sha256"], "calibration NPZ reference mismatch")
        require(accuracy["npz_sha256"] == spec["dataset"]["local_split_sha256"], "accuracy NPZ reference mismatch")
        local = (ad_data / "features") if name == "ad" else primary_data
        require(digest(local / f"{name}.accuracy.npz") == accuracy["npz_sha256"], f"local accuracy hash: {name}")
        if name == "ad":
            require(digest(local / "ad.calibration.npz") == cal["npz_sha256"], "local AD calibration hash")
            raw = read(base / "ad.raw-data.json")
            require(digest(base / "ad.raw-data.json") == data["raw_manifest_sha256"], "AD raw provenance link")
            require(digest(ROOT / "tools/research/prepare_ad_data.py") == data["preparation_script_sha256"], "AD preprocessing script changed")
            require(digest(ROOT / "tools/research/fetch_ad_data.py") == raw["script_sha256"], "AD retrieval script changed")
            require(len(raw["records"]) == 360, "AD raw count")
            require((cal["recording_count"], accuracy["recording_count"]) == (112, 248), "AD recording count")
            for record in raw["records"]:
                require(digest(ad_data / record["path"]) == record["raw_sha256"], f"AD WAV changed: {record['path']}")
            real = read(ROOT / "docs/research/evidence/phase0/ad-real-input-parity.json")
            require(real["status"] == "passed" and len(real["real_input_parity_probes"]) == 32, "real AD parity missing")
            require(real["data_sha256"] == accuracy["npz_sha256"] and
                    real["manifest_sha256"] == digest(base / "ad.data.json"), "real AD parity data mismatch")
            require(real["onnx_sha256"] == inventory["source_onnx_sha256"] and
                    real["canonical_sha256"] == inventory["canonical_onnx_sha256"], "real AD model mismatch")
            require(real["script_sha256"] == digest(ROOT / "tools/research/verify_ad_model.py"), "AD verification script changed")
        workloads[name] = {"source_files_verified": len(spec["files"]) + 1,
                           "manifest_sha256": digest(base / f"{name}.json"),
                           "data_manifest_sha256": digest(base / spec["dataset"]["split_manifest"]),
                           "canonical_inventory_sha256": digest(base / spec["canonical_inventory"]),
                           "calibration_vectors": cal["count"], "accuracy_vectors": accuracy["count"],
                           "macs_per_vector": inventory["total_macs"]}
    physical = ROOT / "docs/research/evidence/physical"
    board = read(physical / "summary.json")
    uart = read(physical / "uart-readback.json")
    require(uart["status"] == "measured-pass" and uart["bytes"] == 1280, "minimal UART evidence")
    require(digest(ROOT / "hardware/releases/physical/uart_loopback.fs") == board["uart_bitstream_sha256"], "UART release hash")
    artifacts = ["docs/research/novelty_matrix.md", "docs/research/PRIOR_WORK_AUDIT.md",
                 "docs/research/LAB_MEASUREMENT_PLAN.md", "docs/research/PHASE_0_CLOSURE.md",
                 "docs/research/evidence/phase0/prior-code.json", "docs/research/evidence/baseline.json",
                 "docs/research/evidence/physical/summary.json", "docs/research/evidence/physical/uart-readback.json",
                 "tools/research/ad-preprocessing.lock.txt", "tools/research/audit_phase0.py"]
    return {"status": "passed", "date": "2026-09-26", "g0_contract_complete": True,
            "scope": "claim/data/provenance/lab-plan freeze; manual claim decisions in linked review; no SOTA or power certification",
            "workloads": workloads, "artifact_sha256": {p: digest(ROOT / p) for p in artifacts},
            "primary_calibration_payloads_recomputed_this_audit": False,
            "primary_calibration_provenance": "frozen Phase1 per-record hashes and calibration reports; local accuracy payload hashes reverified",
            "measured_energy_available": False, "second_fpga_available": False,
            "pcb_revision": None, "official_mlperf_submission_certified": False,
            "later_gates": ["B02: AD INT8 and physical evaluation", "B03: matched tuned B1/B2/B3",
                            "E02: obtain instrument and measure energy", "E03: second-target validation",
                            "E04: independent reproduction and final prior-work refresh"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=ROOT / "work/upstream")
    parser.add_argument("--primary-data", type=Path, default=ROOT / "work/quality")
    parser.add_argument("--ad-data", type=Path, default=ROOT / "work/phase0-ad")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/research/evidence/phase0/closure.json")
    args = parser.parse_args()
    try:
        result = audit(args.upstream, args.primary_data, args.ad_data)
    except (ValueError, OSError, KeyError) as error:
        result = {"status": "failed", "g0_contract_complete": False, "error": str(error)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
