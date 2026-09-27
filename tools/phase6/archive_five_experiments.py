#!/usr/bin/env python3
"""Archive byte-exact evidence for the five Phase 6 follow-on experiments.

This refuses incomplete physical screens. The archive is a deterministic set
of gzip files (mtime zero), a matched board comparison, and a SHA-256 manifest.
It deliberately excludes bitstreams and generated executables, but verifies
their hashes against the original evidence before archiving.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/research/evidence/phase6/five-experiments-v1"
MAX_BYTES = 2_000_000

ARTIFACTS = {
    "methods": [
        "tools/phase6/archive_five_experiments.py",
        "tools/phase6/channel_compaction.py",
        "tools/phase6/screen_channel_compaction.py",
        "tools/phase6/pool_timing.py",
        "tools/phase6/screen_pool27.py",
        "tools/phase6/cross_layer_screen.py",
        "tools/phase6/cross_layer_compaction_screen.py",
        "tools/phase6/joint_selection_screen.py",
        "work/phase6/narrow-accum-v1/experiment.py",
        "work/phase6/narrow-accum-v1/proof.py",
        "work/phase6/narrow-accum-v1/check_schedule.py",
    ],
    "channel_compaction": [
        "work/phase6/channel-compaction-v1/report.json",
        "work/phase6/channel-compaction-v1/fused/report.json",
        "work/phase6/channel-compaction-protect29-v1/report.json",
        "work/phase6/channel-compaction-protect29-v1/fused/report.json",
    ],
    "pool_timing": [
        "work/phase6/pool-timing-v1/engine.sv",
        "work/phase6/pool-timing-v1/identity.json",
        "work/phase6/pool-timing-v1/host27.sv",
        "work/phase6/pool-timing-v1/pll27.v",
        "work/phase6/pool-timing-v1/build27.tcl",
        "work/phase6/pool-timing-v1/comparison.json",
        "work/phase6/pool-timing-v1/engine/report.json",
        "work/phase6/pool-timing-v1/engine/results.xml",
        "work/phase6/pool-timing-v1/edges/report.json",
        "work/phase6/pool-timing-v1/edges/results.xml",
        "work/phase6/pool-timing-v1/native/report.json",
        "work/phase6/pool-timing-v1/compact-native/report.json",
        "work/phase6/pool-timing-v1/integration/report.json",
        "work/phase6/pool-timing-v1/integration/results.xml",
        "work/phase6/pool-timing-v1/uart27/report.json",
        "work/phase6/pool-timing-v1/uart27/results.xml",
        "work/phase6/pool-timing-v1/route24/report.json",
        "work/phase6/pool-timing-v1/route27/report.json",
        "work/phase6/pool-timing-v1/route27/phase6_uart_burst/impl/pnr/phase6_uart_burst.rpt.txt",
        "work/phase6/pool-timing-v1/route27/phase6_uart_burst/impl/pnr/phase6_uart_burst_tr_content.html",
        "work/phase6/experiments-v1/fused-activation-v1/route24/report.json",
    ],
    "narrow_accumulator": [
        "work/phase6/narrow-accum-v1/engine.sv",
        "work/phase6/narrow-accum-v1/identity.json",
        "work/phase6/narrow-accum-v1/proof.json",
        "work/phase6/narrow-accum-v1/schedule-proof.json",
        "work/phase6/narrow-accum-v1/fixtures.json",
        "work/phase6/narrow-accum-v1/comparison.json",
        "work/phase6/narrow-accum-v1/native/report.json",
        "work/phase6/narrow-accum-v1/integration/report.json",
        "work/phase6/narrow-accum-v1/integration/results.xml",
        "work/phase6/narrow-accum-v1/route24/report.json",
        "work/phase6/narrow-accum-v1/route24/phase6_uart_burst/impl/pnr/phase6_uart_burst.rpt.txt",
        "work/phase6/narrow-accum-v1/route24/phase6_uart_burst/impl/pnr/phase6_uart_burst_tr_content.html",
    ],
    "cross_layer_fusion_screen": [
        "work/phase6/cross-layer-screen-v1/report.json",
        "work/phase6/cross-layer-compaction-v1/report.json",
    ],
    "joint_selection": [
        "work/phase6/joint-selection-v2/plan.json",
        "work/phase6/joint-selection-v2/report.json",
        *[
            f"work/phase6/joint-selection-v2/runs/{model}-{cell}-s{seed}.json"
            for model in ("kws", "vww")
            for cell in ("00", "01", "10", "11")
            for seed in (0, 6063)
        ],
    ],
}

SCREENS = {
    "grouped24": "work/phase6/fused-activation-uart256-24-screen-v1",
    "compacted24": "work/phase6/channel-compaction-v1/physical-screen-v1",
    "grouped27": "work/phase6/pool-timing-v1/physical-grouped27-v2",
    "compacted27": "work/phase6/pool-timing-v1/physical-compacted27-v1",
}
for _folder in SCREENS.values():
    ARTIFACTS.setdefault("physical", []).extend(
        f"{_folder}/{name}"
        for name in ("plan.json", "report.json", "records.jsonl", "seal.json", "program.log")
    )

REQUIRED_STATUS = {
    "work/phase6/channel-compaction-v1/report.json": "passed-replay",
    "work/phase6/channel-compaction-v1/fused/report.json": "passed",
    "work/phase6/channel-compaction-protect29-v1/report.json": "passed-replay",
    "work/phase6/channel-compaction-protect29-v1/fused/report.json": "passed",
    "work/phase6/pool-timing-v1/engine/report.json": "passed",
    "work/phase6/pool-timing-v1/edges/report.json": "passed",
    "work/phase6/pool-timing-v1/native/report.json": "passed",
    "work/phase6/pool-timing-v1/compact-native/report.json": "passed",
    "work/phase6/pool-timing-v1/integration/report.json": "passed",
    "work/phase6/pool-timing-v1/uart27/report.json": "passed",
    "work/phase6/pool-timing-v1/route24/report.json": "passed-route",
    "work/phase6/pool-timing-v1/route27/report.json": "passed-route",
    "work/phase6/experiments-v1/fused-activation-v1/route24/report.json": "passed-route",
    "work/phase6/narrow-accum-v1/comparison.json": "passed-no-go",
    "work/phase6/narrow-accum-v1/native/report.json": "passed",
    "work/phase6/narrow-accum-v1/integration/report.json": "passed",
    "work/phase6/narrow-accum-v1/route24/report.json": "passed-route",
    "work/phase6/cross-layer-screen-v1/report.json": "screened-boardless",
    "work/phase6/cross-layer-compaction-v1/report.json": "passed-software-screen",
    "work/phase6/joint-selection-v2/report.json": "passed",
    **{f"{folder}/report.json": "passed-short-screen" for folder in SCREENS.values()},
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read(relative: str) -> bytes:
    path = ROOT / relative
    if not path.is_file():
        raise ValueError(f"missing evidence: {relative}")
    return path.read_bytes()


def read_json(relative: str) -> dict:
    return json.loads(read(relative))


def expect_hash(relative: str, expected: str) -> None:
    actual = digest(read(relative))
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {relative}: {actual} != {expected}")


def check_sources(relative: str, document: dict) -> None:
    for key in ("sources", "source_sha256"):
        sources = document.get(key)
        if isinstance(sources, dict):
            for path, expected in sources.items():
                expect_hash(path, expected)
    if relative in (
        "work/phase6/channel-compaction-v1/report.json",
        "work/phase6/channel-compaction-protect29-v1/report.json",
    ):
        expect_hash("tools/phase6/channel_compaction.py", document["source_sha256"])
    if relative == "work/phase6/joint-selection-v2/report.json":
        expect_hash("work/phase6/joint-selection-v2/plan.json", document["plan_sha256"])
    if "results_sha256" in document:
        expect_hash(str(Path(relative).parent / "results.xml"), document["results_sha256"])


def check_route(relative: str) -> dict:
    report = read_json(relative)
    if report["status"] != "passed-route":
        raise ValueError(f"route did not pass: {relative}")
    check_sources(relative, report)
    if report["routed_core_fmax_mhz"] < report["core_clock_mhz"]:
        raise ValueError(f"route Fmax below operating clock: {relative}")
    if report["setup_violated_endpoints"] or report["hold_violated_endpoints"]:
        raise ValueError(f"route timing violation: {relative}")
    expect_hash(report["bitstream"], report["bitstream_sha256"])
    prefix = str(Path(report["bitstream"]).with_suffix(""))
    expect_hash(prefix + ".rpt.txt", report["route_sha256"])
    expect_hash(prefix + "_tr_content.html", report["timing_sha256"])
    return report


def check_candidate_identities() -> None:
    pool = read_json("work/phase6/pool-timing-v1/identity.json")
    expect_hash("work/phase6/pool-timing-v1/engine.sv", pool["engine_sha256"])
    expect_hash(pool["parent"], pool["parent_sha256"])
    narrow = read_json("work/phase6/narrow-accum-v1/identity.json")
    expect_hash("work/phase6/narrow-accum-v1/engine.sv", narrow["engine_sha256"])
    expect_hash("work/phase6/narrow-accum-v1/proof.json", narrow["proof_sha256"])
    expect_hash("work/phase6/narrow-accum-v1/fixtures.json", narrow["fixture_manifest_sha256"])
    expect_hash(pool["parent"], narrow["source_sha256"])
    for base, identity in (("work/phase6/pool-timing-v1/route27/report.json", pool),
                           ("work/phase6/narrow-accum-v1/route24/report.json", narrow)):
        route = read_json(base)
        if route["engine_sha256"] != identity["engine_sha256"]:
            raise ValueError(f"route/engine identity mismatch: {base}")


def check_joint_selection() -> None:
    base = "work/phase6/joint-selection-v2"
    plan = read_json(f"{base}/plan.json")
    report = read_json(f"{base}/report.json")
    expect_hash(plan["executable"], plan["executable_sha256"])
    for cell, fixture in plan["fixtures"].items():
        directory = fixture["directory"]
        for file, expected in fixture["files"].items():
            expect_hash(f"{directory}/{file}", expected)
        if fixture["files"]["input.bin"] != fixture["input_sha256"] or \
           fixture["files"]["output.bin"] != fixture["output_sha256"]:
            raise ValueError(f"joint fixture identity mismatch: {cell}")
    if len(report["runs"]) != 16:
        raise ValueError("joint selection does not have 16 runs")
    for model in ("kws", "vww"):
        for cell in ("00", "01", "10", "11"):
            for seed in (0, 6063):
                row = report["runs"][f"{model}:{cell}:{seed}"]
                path = f"{base}/runs/{model}-{cell}-s{seed}.json"
                expect_hash(path, row["report_sha256"])
                native = read_json(path)
                if native.get("status") != "passed" or native.get("stall_seed") != seed or \
                   native.get("elapsed_cycles") != row["elapsed_cycles"]:
                    raise ValueError(f"joint run mismatch: {path}")


def signed_records(raw: bytes) -> list[dict]:
    records = []
    for line in raw.splitlines():
        item = json.loads(line)
        signature = item.pop("record_sha256", None)
        canonical = json.dumps(item, sort_keys=True, separators=(",", ":")).encode()
        if signature != digest(canonical):
            raise ValueError("invalid physical record signature")
        records.append(item)
    return records


def check_screen(label: str, folder: str, expected_image: str) -> dict:
    plan_path = f"{folder}/plan.json"
    report_path = f"{folder}/report.json"
    records_path = f"{folder}/records.jsonl"
    seal_path = f"{folder}/seal.json"
    plan = read_json(plan_path)
    report = read_json(report_path)
    seal = read_json(seal_path)
    records_raw = read(records_path)
    if report.get("status") != "passed-short-screen":
        raise ValueError(f"physical screen incomplete: {label}: {report.get('status')}")
    if report.get("planned") != 10 or report.get("completed") != 10 or plan.get("planned") != 10:
        raise ValueError(f"physical screen is not 10/10: {label}")
    for field, path in (("plan_sha256", plan_path), ("report_sha256", report_path),
                        ("records_sha256", records_path)):
        expect_hash(path, seal[field])
    if report["plan_sha256"] != seal["plan_sha256"] or report["records_sha256"] != seal["records_sha256"]:
        raise ValueError(f"physical screen report/seal mismatch: {label}")
    expect_hash(f"{folder}/program.log", report["program_log_sha256"])
    image = plan["image"]
    if image["sha256"] != expected_image:
        raise ValueError(f"physical screen image mismatch: {label}")
    expect_hash(image["file"], expected_image)
    records = signed_records(records_raw)
    if len(records) != 10:
        raise ValueError(f"physical record count mismatch: {label}")
    if any(r["bitstream_sha256"] != expected_image or
           r["output_hex"] != r["expected_hex"] or
           r["input_readback_verified"] is not True for r in records):
        raise ValueError(f"physical record identity/exactness mismatch: {label}")
    clock_hz = image["clock_hz"]
    medians = {}
    timed_identity = {}
    timed_rows = {}
    for model in ("kws", "vww"):
        model_rows = [r for r in records if r["model"] == model]
        if sorted(r["kind"] for r in model_rows) != ["stress", "timed", "timed", "timed", "warmup"]:
            raise ValueError(f"physical model case mix mismatch: {label}/{model}")
        timed = sorted((r for r in model_rows if r["kind"] == "timed"), key=lambda r: r["repeat"])
        if len({r["repeat"] for r in timed}) != 3:
            raise ValueError(f"timed repeats mismatch: {label}/{model}")
        identity = {(r["input_sha256"], r["output_hex"]) for r in timed}
        if len(identity) != 1:
            raise ValueError(f"timed fixture mismatch within screen: {label}/{model}")
        timed_identity[model] = next(iter(identity))
        for row in timed:
            expected_ms = 1000 * row["elapsed_cycles"] / clock_hz
            if not math.isclose(row["device_latency_ms"], expected_ms, rel_tol=0, abs_tol=1e-8):
                raise ValueError(f"cycle/clock/latency mismatch: {label}/{model}")
        cycles = statistics.median(r["elapsed_cycles"] for r in timed)
        ms = statistics.median(r["device_latency_ms"] for r in timed)
        summary = report["summary"][model]
        summary_ms = summary.get("median_ms", summary.get("median_latency_ms",
                          summary.get("median_device_latency_ms")))
        if cycles != summary["median_cycles"] or not math.isclose(ms, summary_ms, rel_tol=0, abs_tol=1e-8):
            raise ValueError(f"reported median mismatch: {label}/{model}")
        medians[model] = {"cycles": cycles, "device_ms": ms, "inferences_per_second": 1000 / ms}
        timed_rows[model] = [
            {"repeat": r["repeat"], "input_sha256": r["input_sha256"],
             "output_hex": r["output_hex"], "cycles": r["elapsed_cycles"]}
            for r in timed
        ]
    return {
        "status": report["status"], "clock_hz": clock_hz, "bitstream_sha256": expected_image,
        "exact_records": len(records), "medians": medians,
        "timed_identity": timed_identity, "timed_rows": timed_rows,
    }


def comparison(screens: dict) -> dict:
    reference = screens["grouped24"]["timed_identity"]
    if any(row["timed_identity"] != reference for row in screens.values()):
        raise ValueError("physical timed inputs or outputs differ across screens")
    pairs = {
        "compaction_at_24_mhz": ("grouped24", "compacted24"),
        "compaction_at_27_mhz": ("grouped27", "compacted27"),
        "pool_clock_grouped": ("grouped24", "grouped27"),
        "pool_clock_compacted": ("compacted24", "compacted27"),
        "joint_vs_original": ("grouped24", "compacted27"),
    }
    rows = {}
    for key, (before, after) in pairs.items():
        metrics = {}
        for model in ("kws", "vww"):
            old = screens[before]["medians"][model]
            new = screens[after]["medians"][model]
            metrics[model] = {
                "cycle_speedup": old["cycles"] / new["cycles"],
                "latency_speedup": old["device_ms"] / new["device_ms"],
                "latency_saved_ms": old["device_ms"] - new["device_ms"],
            }
        rows[key] = {
            "before": before, "after": after, "models": metrics,
            "geometric_mean_latency_speedup": math.sqrt(
                metrics["kws"]["latency_speedup"] * metrics["vww"]["latency_speedup"]),
        }
    return {
        "schema": 1, "scope": "matched pinned-input 10/10 exact short screens; FPGA device latency only",
        "screens": screens, "pairs": rows,
        "limits": ["No full accuracy qualification", "No power/energy measurement",
                   "No long-duration stability qualification", "No SOTA or architectural novelty claim"],
    }


def archive(raw_by_path: dict[str, bytes], comparison_doc: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    entries = {}
    for relative, raw in sorted(raw_by_path.items()):
        if len(raw) > MAX_BYTES or Path(relative).suffix in (".fs", ".bin"):
            raise ValueError(f"artifact too large or binary: {relative}")
        target = OUT / "artifacts" / f"{relative}.gz"
        target.parent.mkdir(parents=True, exist_ok=True)
        compressed = gzip.compress(raw, compresslevel=6, mtime=0)
        if not target.exists() or target.read_bytes() != compressed:
            target.write_bytes(compressed)
        entries[relative] = {"sha256": digest(raw), "bytes": len(raw),
                             "archive": str(target.relative_to(ROOT)),
                             "archive_sha256": digest(compressed)}
    comparison_raw = (json.dumps(comparison_doc, sort_keys=True, indent=2) + "\n").encode()
    comparison_path = OUT / "matched-comparison.json"
    comparison_path.write_bytes(comparison_raw)
    manifest = {
        "schema": 1, "status": "complete-short-screen-archive",
        "scope": "Five Phase 6 follow-on experiments; 24 and 27 MHz KWS/VWW short board screens",
        "groups": ARTIFACTS,
        "artifacts": entries,
        "matched_comparison": {"path": str(comparison_path.relative_to(ROOT)),
                               "sha256": digest(comparison_raw)},
        "bitstreams_archived": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def build(run: bool) -> dict:
    paths = [path for group in ARTIFACTS.values() for path in group]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate archive artifact")
    raw_by_path = {path: read(path) for path in paths}
    for path, expected in REQUIRED_STATUS.items():
        report = json.loads(raw_by_path[path])
        if report.get("status") != expected:
            raise ValueError(f"report not complete: {path}: {report.get('status')} != {expected}")
        check_sources(path, report)
    for path in (
        "work/phase6/pool-timing-v1/route24/report.json",
        "work/phase6/pool-timing-v1/route27/report.json",
        "work/phase6/experiments-v1/fused-activation-v1/route24/report.json",
        "work/phase6/narrow-accum-v1/route24/report.json",
    ):
        check_route(path)
    check_candidate_identities()
    check_joint_selection()
    old_image = read_json("work/phase6/experiments-v1/fused-activation-v1/route24/report.json")["bitstream_sha256"]
    new_image = read_json("work/phase6/pool-timing-v1/route27/report.json")["bitstream_sha256"]
    screens = {
        name: check_screen(name, folder, old_image if name.endswith("24") else new_image)
        for name, folder in SCREENS.items()
    }
    comparison_doc = comparison(screens)
    if not run:
        return {"status": "preflight-passed", "files": len(paths),
                "pairs": {name: row["geometric_mean_latency_speedup"]
                          for name, row in comparison_doc["pairs"].items()}}
    manifest = archive(raw_by_path, comparison_doc)
    return {"status": manifest["status"], "files": len(paths),
            "manifest": str((OUT / "manifest.json").relative_to(ROOT)),
            "pairs": {name: row["geometric_mean_latency_speedup"]
                      for name, row in comparison_doc["pairs"].items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="write the archive after validation")
    args = parser.parse_args()
    print(json.dumps(build(args.run), sort_keys=True))
