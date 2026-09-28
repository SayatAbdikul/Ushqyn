#!/usr/bin/env python3
"""Seal the three non-streaming Phase 6 novelty screens as deterministic evidence.

The archive contains source, proof, native reports, route timing, and the
previously measured board baseline. Generated fixture binaries and bitstreams
are hash-pinned, but deliberately excluded from the Git archive.
"""

import argparse
import gzip
import hashlib
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/research/evidence/phase6/novelty-nonstream-v1"
BASE_ROUTE = "work/phase6/pool-timing-v1/route27"
CAND_ROUTE = "work/phase6/novelty-contracts-v1/route27"
BOARD = "work/phase6/pair7-fusion-board-v1/physical-short-v1"
CONTRACT = "work/phase6/novelty-contracts-v1"
CONSTANT = "work/phase6/novelty-constants-v1"
ZERO = "work/phase6/novelty-zero-block-v1"
MAX_ARCHIVED_BYTES = 2_000_000

METHODS = [
    "tools/phase6/archive_novelty_nonstream.py",
    "tools/phase6/novelty_zero_block.py",
    "tools/phase6/novelty_zero_block_profile.py",
    "tools/phase6/novelty_zero_bypass.py",
    "tools/phase6/novelty_contract.py",
    "tools/phase6/novelty_contract_edges.py",
    "tools/phase6/novelty_constants.py",
    "tools/phase6/novelty_constants_threepair.py",
    "tools/phase6/novelty_constants_decision.py",
    "tools/phase6/strip_fusion_vww.py",
    "tools/phase6/strip_fusion_pair11.py",
    "tools/phase6/strip_fusion_pair7.py",
    "test/phase6/native.cpp",
    "work/phase6/novelty-zero-block-v1/native-profile/native_profile.cpp",
    "docs/research/PHASE_6_ZERO_BLOCK_SCREEN_V1.md",
    "docs/research/PHASE_6_NONSTREAM_NOVELTY_V1.md",
]
EVIDENCE = [
    f"{ZERO}/report.json",
    f"{ZERO}/native-profile/report.json",
    *[f"{ZERO}/native-profile/{model}-s{seed}.json"
      for model in ("kws", "vww") for seed in (0, 6063)],
    "work/phase6/novelty-zero-bypass-v1/report.json",
    f"{CONTRACT}/report.json",
    f"{CONTRACT}/edges/report.json",
    f"{CONTRACT}/edges/results.xml",
    f"{CONTRACT}/decision.json",
    f"{CONTRACT}/engine.sv",
    *sorted(str(p.relative_to(ROOT)) for p in (ROOT / CONTRACT / "native").glob("*.json")),
    f"{CONSTANT}/report.json",
    f"{CONSTANT}/certificate.json",
    f"{CONSTANT}/fused/report.json",
    f"{CONSTANT}/threepair/report.json",
    f"{CONSTANT}/decision.json",
    *sorted(str(p.relative_to(ROOT)) for p in (ROOT / CONSTANT / "fused").glob("*.json")
            if p.name != "report.json"),
    *sorted(str(p.relative_to(ROOT)) for p in (ROOT / CONSTANT / "threepair").glob("*.json")
            if p.name != "report.json"),
    *sorted(str(p.relative_to(ROOT)) for p in (ROOT / CONSTANT / "threepair/fixtures").glob("*/schedule.json")),
]
ROUTES = [
    "work/phase6/pool-timing-v1/engine.sv",
    "work/phase6/pool-timing-v1/build27.tcl",
    f"{BASE_ROUTE}/report.json",
    f"{BASE_ROUTE}/phase6_uart_burst/phase6_uart_burst.gprj",
    f"{BASE_ROUTE}/phase6_uart_burst/impl/pnr/phase6_uart_burst.rpt.txt",
    f"{BASE_ROUTE}/phase6_uart_burst/impl/pnr/phase6_uart_burst_tr_content.html",
    f"{CAND_ROUTE}/report.json",
    f"{CAND_ROUTE}/phase6_uart_burst/phase6_uart_burst.gprj",
    f"{CAND_ROUTE}/phase6_uart_burst/impl/pnr/phase6_uart_burst.rpt.txt",
    f"{CAND_ROUTE}/phase6_uart_burst/impl/pnr/phase6_uart_burst_tr_content.html",
]
PHYSICAL = [f"{BOARD}/{name}" for name in (
    "plan.json", "report.json", "records.jsonl", "seal.json", "program.log",
    "pair7-comparison.json", "pair7-seal.json")]
STATUS = {
    f"{ZERO}/report.json": "passed-opportunity-screen",
    f"{ZERO}/native-profile/report.json": "passed",
    "work/phase6/novelty-zero-bypass-v1/report.json":
        "no_go_for_existing_cache_path_side_metadata_untested",
    f"{CONTRACT}/report.json": "passed-native",
    f"{CONTRACT}/edges/report.json": "passed",
    f"{CONTRACT}/decision.json": "no-go",
    f"{CAND_ROUTE}/report.json": "passed-route",
    f"{BASE_ROUTE}/report.json": "passed-route",
    f"{CONSTANT}/report.json": "passed-replay",
    f"{CONSTANT}/fused/report.json": "passed",
    f"{CONSTANT}/threepair/report.json": "passed-native",
    f"{CONSTANT}/decision.json": "no-go-board-expansion",
    f"{BOARD}/report.json": "passed-short-screen",
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read(path: str) -> bytes:
    file = ROOT / path
    if not file.is_file():
        raise ValueError(f"missing evidence: {path}")
    return file.read_bytes()


def compressed(raw: bytes) -> bytes:
    return gzip.compress(raw, compresslevel=6, mtime=0)


def project_sources(raw: bytes, original_root: str) -> list[str]:
    node = ET.fromstring(raw)
    paths = []
    for item in node.findall(".//File"):
        absolute = item.attrib["path"]
        prefix = original_root + "/"
        if not absolute.startswith(prefix):
            raise ValueError(f"route project references an external source: {absolute}")
        relative = absolute[len(prefix):]
        if ".." in Path(relative).parts or not relative:
            raise ValueError(f"unsafe route source: {relative}")
        paths.append(relative)
    if len(paths) != 18 or len(set(paths)) != len(paths):
        raise ValueError(f"unexpected route source count: {len(paths)}")
    return paths


def artifact_paths() -> dict[str, list[str]]:
    projects = [read(f"{route}/phase6_uart_burst/phase6_uart_burst.gprj")
                for route in (BASE_ROUTE, CAND_ROUTE)]
    source_paths = sorted(set().union(*(project_sources(raw, str(ROOT)) for raw in projects)))
    groups = {"methods": METHODS, "novelty_evidence": EVIDENCE,
              "routes": ROUTES, "physical_baseline": PHYSICAL,
              "route_sources": [p for p in source_paths if p not in METHODS + EVIDENCE + ROUTES + PHYSICAL]}
    flat = [p for paths in groups.values() for p in paths]
    if len(flat) != len(set(flat)):
        raise ValueError("duplicate artifact path")
    return groups


def require_hash(raw: dict[str, bytes], path: str, expected: str) -> None:
    if path not in raw or sha(raw[path]) != expected:
        raise ValueError(f"SHA-256 identity mismatch: {path}")


def document(raw: dict[str, bytes], path: str) -> dict:
    return json.loads(raw[path])


def source_audit(raw: dict[str, bytes], original_root: str) -> dict:
    projects = {}
    for label, route in (("baseline", BASE_ROUTE), ("candidate", CAND_ROUTE)):
        project = f"{route}/phase6_uart_burst/phase6_uart_burst.gprj"
        paths = project_sources(raw[project], original_root)
        projects[label] = {p: sha(raw[p]) for p in paths}
        report = document(raw, f"{route}/report.json")
        engine = "work/phase6/pool-timing-v1/engine.sv" if label == "baseline" else f"{CONTRACT}/engine.sv"
        require_hash(raw, engine, report["engine_sha256"])
        for key, filename in (("route_sha256", "phase6_uart_burst.rpt.txt"),
                              ("timing_sha256", "phase6_uart_burst_tr_content.html")):
            require_hash(raw, f"{route}/phase6_uart_burst/impl/pnr/{filename}", report[key])
        if report["routed_core_fmax_mhz"] < 27 or report["setup_violated_endpoints"] or \
                report["hold_violated_endpoints"]:
            raise ValueError(f"route did not meet 27 MHz: {route}")
    baseline, candidate = projects["baseline"], projects["candidate"]
    baseline_engine = "work/phase6/pool-timing-v1/engine.sv"
    candidate_engine = f"{CONTRACT}/engine.sv"
    if set(baseline) - {baseline_engine} != set(candidate) - {candidate_engine}:
        raise ValueError("route project source sets differ beyond engine")
    changed = [p for p in baseline if p != baseline_engine and baseline[p] != candidate[p]]
    if changed:
        raise ValueError(f"uncontrolled changed route inputs: {changed}")
    decision = document(raw, f"{CONTRACT}/decision.json")
    report = document(raw, f"{CONTRACT}/report.json")
    if decision["source_audit"]["baseline_engine_sha256"] != baseline[baseline_engine] or \
            decision["source_audit"]["candidate_engine_sha256"] != candidate[candidate_engine] or \
            report["source_sha256"] != baseline[baseline_engine] or \
            report["engine_sha256"] != candidate[candidate_engine]:
        raise ValueError("contract report does not identify routed engine pair")
    for path, expected in decision["source_audit"]["frozen_inputs"].items():
        require_hash(raw, path, expected)
    if decision["source_audit"]["route_bitstream_sha256"] != \
            document(raw, f"{CAND_ROUTE}/report.json")["bitstream_sha256"]:
        raise ValueError("contract decision bitstream mismatch")
    return {"project_source_count": 18, "changed_sources": {baseline_engine: candidate_engine},
            "baseline_project_sha256": sha(raw[f"{BASE_ROUTE}/phase6_uart_burst/phase6_uart_burst.gprj"]),
            "candidate_project_sha256": sha(raw[f"{CAND_ROUTE}/phase6_uart_burst/phase6_uart_burst.gprj"]),
            "baseline_engine_sha256": baseline[baseline_engine],
            "candidate_engine_sha256": candidate[candidate_engine]}


def physical_audit(raw: dict[str, bytes]) -> dict:
    plan = document(raw, f"{BOARD}/plan.json")
    report = document(raw, f"{BOARD}/report.json")
    seal = document(raw, f"{BOARD}/seal.json")
    for label in ("plan", "report", "records"):
        require_hash(raw, f"{BOARD}/{label}.json" if label != "records" else
                     f"{BOARD}/records.jsonl", seal[f"{label}_sha256"])
    require_hash(raw, f"{BOARD}/program.log", report["program_log_sha256"])
    if plan["image"]["sha256"] != document(raw, f"{BASE_ROUTE}/report.json")["bitstream_sha256"]:
        raise ValueError("physical image does not match frozen baseline route")
    if plan["planned"] != 10 or report["planned"] != 10 or report["completed"] != 10:
        raise ValueError("physical baseline is not a complete 10-case screen")
    rows = []
    for line in raw[f"{BOARD}/records.jsonl"].splitlines():
        row = json.loads(line)
        signature = row.pop("record_sha256")
        if signature != sha(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()):
            raise ValueError("invalid physical record signature")
        if row["output_hex"] != row["expected_hex"] or not row["input_readback_verified"] or \
                row["bitstream_sha256"] != plan["image"]["sha256"]:
            raise ValueError("physical record failed exactness or identity")
        rows.append(row)
    if len(rows) != 10:
        raise ValueError("physical record count mismatch")
    medians = {}
    for model in ("kws", "vww"):
        timed = [r["elapsed_cycles"] for r in rows if r["model"] == model and r["kind"] == "timed"]
        if len(timed) != 3 or statistics.median(timed) != report["summary"][model]["median_cycles"]:
            raise ValueError(f"physical median mismatch: {model}")
        medians[model] = int(statistics.median(timed))
    return {"bitstream_sha256": plan["image"]["sha256"],
            "clock_hz": plan["image"]["clock_hz"], "exact_records": 10,
            "median_cycles": medians}


def validate(raw: dict[str, bytes], original_root: str) -> dict:
    for path, expected in STATUS.items():
        if document(raw, path).get("status") != expected:
            raise ValueError(f"incomplete evidence: {path}")
    zero = document(raw, f"{ZERO}/report.json")
    require_hash(raw, "tools/phase6/novelty_zero_block.py", zero["source_sha256"])
    bypass = document(raw, "work/phase6/novelty-zero-bypass-v1/report.json")
    for path, expected in bypass["source_hashes"].items():
        require_hash(raw, path, expected)
    if bypass["rtl_bypass_implemented"] or bypass["native_candidate_simulated"] or \
            bypass["physical_candidate_tested"]:
        raise ValueError("zero-block status claims exceed measured scope")
    profile = document(raw, f"{ZERO}/native-profile/report.json")
    require_hash(raw, "test/phase6/native.cpp", profile["hashes"]["harness"])
    require_hash(raw, f"{ZERO}/native-profile/native_profile.cpp", profile["hashes"]["instrumented_harness"])
    edge = document(raw, f"{CONTRACT}/edges/report.json")
    require_hash(raw, f"{CONTRACT}/engine.sv", edge["engine_sha256"])
    require_hash(raw, "tools/phase6/novelty_contract_edges.py", edge["test_sha256"])
    require_hash(raw, f"{CONTRACT}/edges/results.xml", edge["results_sha256"])
    contract = document(raw, f"{CONTRACT}/report.json")
    if len(contract["native_results"]) != 10 or any(
            r["status"] != "passed" or r["cycle_delta"] != 0 for r in contract["native_results"]):
        raise ValueError("contract exact native campaign incomplete")
    constants = document(raw, f"{CONSTANT}/decision.json")
    for path, expected in constants["source_sha256"].items():
        require_hash(raw, path, expected)
    for path, expected in constants["evidence_sha256"].items():
        if path in raw:
            require_hash(raw, path, expected)
    cert = f"{CONSTANT}/certificate.json"
    for report in (f"{CONSTANT}/report.json", f"{CONSTANT}/threepair/report.json"):
        if document(raw, report)["certificate_sha256"] != sha(raw[cert]):
            raise ValueError("constant certificate identity mismatch")
    if document(raw, f"{CONSTANT}/threepair/report.json")["source_report_sha256"] != \
            sha(raw[f"{CONSTANT}/fused/report.json"]):
        raise ValueError("three-pair/fused report mismatch")
    return {"route": source_audit(raw, original_root), "physical": physical_audit(raw)}


def excluded_hashes(raw: dict[str, bytes]) -> dict[str, str]:
    external = {}
    for route in (BASE_ROUTE, CAND_ROUTE):
        path = f"{route}/phase6_uart_burst/impl/pnr/phase6_uart_burst.fs"
        external[path] = document(raw, f"{route}/report.json")["bitstream_sha256"]
    constants = document(raw, f"{CONSTANT}/decision.json")
    for path, expected in constants["evidence_sha256"].items():
        if path not in raw:
            external[path] = expected
    contract = document(raw, f"{CONTRACT}/report.json")
    for item in contract["fixtures"]:
        name = item["name"]
        external[f"{CONTRACT}/fixtures/{name}/payload.bin"] = item["certified_payload_sha256"]
    profile = document(raw, f"{ZERO}/native-profile/report.json")
    paths = {"kws": "work/phase6/channel-compaction-v1/fused/fixtures",
             "vww": "work/phase6/strip-fusion-pair7-v1/full/fixtures"}
    for model, entry in profile["models"].items():
        base = paths[model] + "/" + entry["fixture"]
        for file, expected in entry["fixture_sha256"].items():
            external[f"{base}/{file}"] = expected
    for path, expected in external.items():
        if sha(read(path)) != expected:
            raise ValueError(f"excluded artifact SHA-256 mismatch: {path}")
    return dict(sorted(external.items()))


def build() -> dict:
    groups = artifact_paths()
    raw = {p: read(p) for paths in groups.values() for p in paths}
    checks = validate(raw, str(ROOT))
    external = excluded_hashes(raw)
    OUT.mkdir(parents=True, exist_ok=True)
    entries = {}
    for path, data in sorted(raw.items()):
        if len(data) > MAX_ARCHIVED_BYTES or Path(path).suffix in (".fs", ".bin"):
            raise ValueError(f"oversized or binary archive candidate: {path}")
        target = OUT / "artifacts" / f"{path}.gz"
        target.parent.mkdir(parents=True, exist_ok=True)
        gz = compressed(data)
        if not target.exists() or target.read_bytes() != gz:
            target.write_bytes(gz)
        entries[path] = {"bytes": len(data), "sha256": sha(data),
                         "archive": str(target.relative_to(ROOT)),
                         "archive_sha256": sha(gz)}
    manifest = {"schema": 1, "status": "sealed-nonstream-screen-archive",
                "scope": "Phase 6 experiments 1, 2, 4; experiment 3 excluded by user",
                "original_root": str(ROOT), "groups": groups, "artifacts": entries,
                "excluded_generated_files": external, "checks": checks,
                "limitations": ["No physical candidate speedup measured",
                                "No zero-block RTL candidate or routed result",
                                "Only the prior selected baseline has physical board evidence"]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def verify(check_workspace: bool) -> dict:
    manifest = json.loads((OUT / "manifest.json").read_text())
    if manifest["schema"] != 1 or manifest["status"] != "sealed-nonstream-screen-archive":
        raise ValueError("bad archive manifest status")
    raw = {}
    expected_paths = {p for paths in manifest["groups"].values() for p in paths}
    if expected_paths != set(manifest["artifacts"]):
        raise ValueError("manifest group/artifact mismatch")
    for path, item in manifest["artifacts"].items():
        gz = (ROOT / item["archive"]).read_bytes()
        if sha(gz) != item["archive_sha256"] or gz[4:8] != b"\0\0\0\0":
            raise ValueError(f"gzip identity or mtime mismatch: {path}")
        data = gzip.decompress(gz)
        if gz != compressed(data) or len(data) != item["bytes"] or sha(data) != item["sha256"]:
            raise ValueError(f"deterministic payload mismatch: {path}")
        raw[path] = data
        if check_workspace and read(path) != data:
            raise ValueError(f"workspace/archive mismatch: {path}")
    checks = validate(raw, manifest["original_root"])
    if checks != manifest["checks"]:
        raise ValueError("archive semantic checks mismatch")
    if check_workspace:
        for path, expected in manifest["excluded_generated_files"].items():
            if sha(read(path)) != expected:
                raise ValueError(f"excluded workspace file mismatch: {path}")
    return {"status": "verified", "archived_files": len(raw),
            "hash_pinned_generated_files": len(manifest["excluded_generated_files"]),
            "manifest": str((OUT / "manifest.json").relative_to(ROOT)),
            "physical_exact_records": checks["physical"]["exact_records"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true", help="build and verify the evidence archive")
    group.add_argument("--verify", action="store_true", help="verify archived bytes and claims")
    parser.add_argument("--verify-workspace", action="store_true",
                        help="also compare source and excluded generated files with workspace")
    args = parser.parse_args()
    if args.run:
        build()
    print(json.dumps(verify(args.verify_workspace), sort_keys=True))
