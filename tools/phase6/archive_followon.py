#!/usr/bin/env python3
"""Pin follow-on Phase 6 evidence without promoting short screens to qualification.

This script is boardless and deterministic. Re-run it after each completed board
screen; running or failed screens are recorded as such, never as measurements.
"""

import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/research/evidence/phase6/followon"
EXPERIMENTS = ROOT / "work/phase6/experiments-v1"

ENGINES = (
    "all-exact", "pw-hit-v2", "pw-hit-v3", "dw-reuse-v3", "rq-fast-v3",
    "rq-fast-22p5-v1", "vector-lut-v1", "combined-next-v1",
    "speculative-v1", "combined-spec-v1", "combined-spec-20-v1",
    "combined-spec-22p5-v1", "combined-spec-scalar-v1",
    "combined-spec-scalar-uart-v1",
)
SCREENS = {
    "constant-filter": ("work/phase6/constant-screen-v1", "all-exact"),
    "sibling-retention": ("work/phase6/sibling-screen-v1", "all-exact"),
    "constant-plus-sibling": ("work/phase6/constant-sibling-screen-v1", "all-exact"),
    "depthwise-word-reuse": ("work/phase6/dw-reuse-screen-v1", "dw-reuse-v3"),
    "fast-requant-22p5": ("work/phase6/rq-fast-22p5-screen-v1", "rq-fast-22p5-v1"),
    "vector-lut": ("work/phase6/vector-lut-screen-v1", "vector-lut-v1"),
    "combined-next": ("work/phase6/combined-next-screen-v1", "combined-next-v1"),
    "combined-spec-vector-20p25": ("work/phase6/combined-spec-20-screen-v1", "combined-spec-20-v1"),
    "combined-spec-scalar": ("work/phase6/combined-spec-scalar-screen-v1", "combined-spec-scalar-v1"),
    "combined-spec-scalar-uart": ("work/phase6/combined-spec-scalar-uart-screen-v1", "combined-spec-scalar-uart-v1"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def relative(path):
    return str(path.relative_to(ROOT))


def require_sha(path, expected):
    if not path.is_file() or sha(path) != expected:
        raise ValueError(f"missing or changed evidence: {path}")


def save_artifact(path, artifacts):
    """Store byte-exact evidence with deterministic gzip metadata."""
    key = relative(path)
    raw = path.read_bytes()
    raw_sha = hashlib.sha256(raw).hexdigest()
    target = OUT / "artifacts" / (key + ".gz")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or gzip.decompress(target.read_bytes()) != raw:
        target.write_bytes(gzip.compress(raw, compresslevel=6, mtime=0))
    artifacts[key] = dict(archive=relative(target), sha256=raw_sha,
                          archive_sha256=sha(target))
    return artifacts[key]


def engine_summary(label, artifacts):
    root = EXPERIMENTS / label
    engine = root / "engine.sv"
    if not engine.exists():
        return dict(status="pending")
    engine_sha = sha(engine)
    row = dict(status="source-present", engine=relative(engine), engine_sha256=engine_sha)
    save_artifact(engine, artifacts)
    for filename in ("identity.json", "engine/report.json", "edges/report.json",
                     "native/report.json", "integration/report.json", "route/report.json"):
        path = root / filename
        if path.exists():
            row[filename.replace("/", "_").replace(".json", "_sha256")] = sha(path)
            save_artifact(path, artifacts)
    for stage in ("engine", "edges", "integration"):
        report_path = root / stage / "report.json"
        xml = root / stage / "results.xml"
        if report_path.exists() and xml.exists():
            require_sha(xml, read(report_path)["results_sha256"])
            save_artifact(xml, artifacts)
        elif xml.exists():
            save_artifact(xml, artifacts)
            if stage == "engine" and ET.parse(xml).findall(".//failure"):
                row["status"] = "failed-cocotb"
    identity_path = root / "identity.json"
    if identity_path.exists():
        identity = read(identity_path)
        if identity.get("engine_sha256") != engine_sha:
            raise ValueError(f"engine identity changed: {label}")
        row["parameters"] = identity.get("parameters", {})
    native_path = root / "native/report.json"
    if native_path.exists():
        native = read(native_path)
        row["native"] = dict(status=native.get("status"), cases=len(native.get("results", [])))
    route_path = root / "route/report.json"
    if route_path.exists():
        route = read(route_path)
        if route.get("engine_sha256") != engine_sha:
            raise ValueError(f"route engine changed: {label}")
        for source, expected in route.get("sources", {}).items():
            source_path = ROOT / source
            require_sha(source_path, expected)
            save_artifact(source_path, artifacts)
        row["route"] = {k: route.get(k) for k in (
            "status", "core_clock_mhz", "routed_core_fmax_mhz",
            "setup_violated_endpoints", "hold_violated_endpoints", "bitstream_sha256")}
        row["route"]["resources"] = route.get("resources")
        image_path = ROOT / route["bitstream"] if route.get("bitstream") else None
        pnr = image_path.parent if image_path else root / "route/phase6_tiled_host/impl/pnr"
        prefix = image_path.stem if image_path else "phase6_tiled_host"
        for name, field in ((prefix + ".rpt.txt", "route_sha256"),
                            (prefix + "_tr_content.html", "timing_sha256")):
            evidence = pnr / name
            if evidence.exists() and route.get(field):
                require_sha(evidence, route[field])
                save_artifact(evidence, artifacts)
        if route.get("status") == "passed-route":
            bitstream = ROOT / route["bitstream"]
            require_sha(bitstream, route["bitstream_sha256"])
            row["status"] = "passed-route"
        else:
            row["status"] = route.get("status", "route-incomplete")
    return row


def signed_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        signature = row.pop("record_sha256")
        raw = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(raw).hexdigest() != signature:
            raise ValueError(f"invalid signed physical record: {path}")
        rows.append(row)
    return rows


def summarize(rows, variant):
    if len(rows) != 10:
        raise ValueError("short screen must have ten physical inferences")
    result = {}
    for model in ("kws", "vww"):
        group = [r for r in rows if r["variant"] == variant and r["model"] == model]
        if sorted(r["kind"] for r in group) != ["stress", "timed", "timed", "timed", "warmup"]:
            raise ValueError(f"incomplete {model} screen")
        timed = [r for r in group if r["kind"] == "timed"]
        if sorted(r["repeat"] for r in timed) != [1, 2, 3]:
            raise ValueError("timing repeats changed")
        if any(r["output_hex"] != r["expected_hex"] or r.get("input_readback_verified") is False
               for r in group):
            raise ValueError("physical logits or upload readback failed")
        cycles = [r["elapsed_cycles"] for r in timed]
        median = statistics.median(cycles)
        result[model] = dict(median_cycles=median,
                             median_ms=statistics.median(r["device_latency_ms"] for r in timed),
                             range_over_median=(max(cycles)-min(cycles))/median)
    return result


def archive_runner(plan, artifacts):
    pinned = plan.get("runner_sha256")
    if not pinned:
        return
    candidates = (
        ROOT / "tools/phase6/screen_schedule.py",
        ROOT / "tools/phase6/screen_engine_schedule.py",
        EXPERIMENTS / "runner-snapshots" / (pinned + ".py"),
        ROOT / "tools/phase6/quick_experiments.py",
    )
    runner = next((p for p in candidates if p.exists() and sha(p) == pinned), None)
    if runner is None:
        raise ValueError("historical board runner source unavailable")
    save_artifact(runner, artifacts)


def screen_summary(name, directory, engine_label, artifacts):
    path = ROOT / directory / "report.json"
    if not path.exists():
        return dict(status="pending", image=engine_label)
    report = read(path)
    if report.get("status") != "passed-short-screen":
        return dict(status=report.get("status", "unknown"), image=engine_label,
                    completed=report.get("completed", 0), planned=report.get("planned"))
    root = path.parent
    plan_path, records_path = root / "plan.json", root / "records.jsonl"
    require_sha(plan_path, report["plan_sha256"])
    require_sha(records_path, report["records_sha256"])
    plan = read(plan_path)
    archive_runner(plan, artifacts)
    for path_key, sha_key in (("manifest", "manifest_sha256"),
                              ("native_report", "native_report_sha256")):
        source = Path(plan[path_key])
        require_sha(source, plan[sha_key])
        save_artifact(source, artifacts)
    if report.get("physical_board") is not True or report["completed"] != report["planned"] != 10:
        raise ValueError(f"incomplete board report: {name}")
    if report["output_mismatches"] != 0:
        raise ValueError(f"board output mismatch: {name}")
    variant = report["variant"]
    rows = signed_rows(records_path)
    derived = summarize(rows, variant)
    if derived != report["summary"]:
        raise ValueError(f"board median changed: {name}")
    image = plan["image"]
    route = read(EXPERIMENTS / engine_label / "route/report.json")
    if image["sha256"] != route["bitstream_sha256"] or route["status"] != "passed-route":
        raise ValueError(f"screened image differs from route: {name}")
    if (len(report["programming"]) != 1
            or report["programming"][0]["bitstream_sha256"] != image["sha256"]):
        raise ValueError(f"programmed image differs from route: {name}")
    if any(r["bitstream_sha256"] != image["sha256"] for r in rows):
        raise ValueError(f"row image differs from route: {name}")
    require_sha(ROOT / image["file"], image["sha256"])
    for p in (path, plan_path, records_path, root / "program.log"):
        save_artifact(p, artifacts)
    if sha(root / "program.log") != report["programming"][0]["log_sha256"]:
        raise ValueError(f"programming log changed: {name}")
    # Archive only images actually programmed and measured. All other routes
    # remain route-only evidence in the ledger.
    save_artifact(ROOT / image["file"], artifacts)
    return dict(status="board-short-screen-passed", image=engine_label,
                clock_mhz=route["core_clock_mhz"], bitstream_sha256=image["sha256"],
                physical_inferences=10, report=relative(path), report_sha256=sha(path),
                plan_sha256=sha(plan_path), records_sha256=sha(records_path), models=derived)


def baseline_summary(artifacts):
    directory = EXPERIMENTS / "quick-chain"
    report_path = directory / "report.json"
    report = read(report_path)
    plan_path, records_path = directory / "plan.json", directory / "records.jsonl"
    if report["status"] != "passed-short-screen" or report["completed"] != 10:
        raise ValueError("selected all-exact board baseline incomplete")
    require_sha(plan_path, report["plan_sha256"])
    require_sha(records_path, report["records_sha256"])
    rows = signed_rows(records_path)
    result = summarize(rows, "all-exact")
    if result != report["summary"]["all-exact"]:
        raise ValueError("all-exact baseline median changed")
    plan = read(plan_path)
    archive_runner(plan, artifacts)
    image = plan["images"]["all-exact"]
    route = read(EXPERIMENTS / "all-exact/route/report.json")
    if image["sha256"] != route["bitstream_sha256"]:
        raise ValueError("baseline bitstream differs from route")
    require_sha(ROOT / image["file"], image["sha256"])
    for path in (report_path, plan_path, records_path):
        save_artifact(path, artifacts)
    save_artifact(ROOT / image["file"], artifacts)
    return dict(status="board-short-screen-passed", image="all-exact",
                clock_mhz=route["core_clock_mhz"], physical_inferences=10,
                report=relative(report_path), report_sha256=sha(report_path), models=result)


def transfer_summary(path, expected_image_sha, artifacts):
    if not path.exists():
        return dict(status="pending")
    physical = read(path)
    if physical.get("status") != "passed-short-transfer-screen":
        return dict(status=physical.get("status", "unknown"))
    records = physical["records"]
    if (len(records) != 4 or [r["mode"] for r in records] != ["legacy", "burst", "burst", "legacy"]
            or any(not r["readback_matched"] for r in records)
            or physical["transfer_bytes"] != 8192 or physical["baud"] != 750000):
        raise ValueError("UART physical transfer coverage incomplete")
    image = ROOT / physical["tested_image_path"]
    if (physical["expected_image_sha256"] != expected_image_sha
            or sha(image) != expected_image_sha):
        raise ValueError("UART physical image changed")
    medians = {mode: statistics.median(r["upload_seconds"] for r in records if r["mode"] == mode)
               for mode in ("legacy", "burst")}
    if medians != physical["medians"] or physical["speedup"] != medians["legacy"]/medians["burst"]:
        raise ValueError("UART physical medians changed")
    save_artifact(path, artifacts)
    save_artifact(image, artifacts)
    return dict(status="board-short-transfer-passed", physical_report=relative(path),
                physical_report_sha256=sha(path), medians_seconds=medians,
                transfer_speedup=physical["speedup"], transfer_bytes=physical["transfer_bytes"])


def burst_inference_summary(artifacts):
    attempt_root = ROOT / "work/phase6/combined-spec-scalar-uart-v1/burst-inference"
    first_path = attempt_root / "report.json"
    first = None
    if first_path.exists():
        failed = read(first_path)
        if failed["status"] != "failed":
            raise ValueError("first burst inference attempt was not a failed source-stability check")
        plan_path, records_path = attempt_root / "plan.json", attempt_root / "records.jsonl"
        require_sha(plan_path, failed["plan_sha256"])
        require_sha(records_path, failed["records_sha256"])
        signed_rows(records_path)
        if "check_burst_inference.py" not in failed.get("failure", ""):
            raise ValueError("first burst inference failure cause changed")
        for path in (first_path, plan_path, records_path):
            save_artifact(path, artifacts)
        first = dict(status="failed-source-changed", completed=failed["completed"],
                     report=relative(first_path), report_sha256=sha(first_path),
                     records_sha256=sha(records_path))
    root = ROOT / "work/phase6/combined-spec-scalar-uart-v1/burst-inference-v2"
    report_path = root / "report.json"
    if not report_path.exists():
        return dict(status="pending", excluded_prior_attempt=first)
    report = read(report_path)
    if report.get("status") != "passed-exact-burst-inference":
        return dict(status=report.get("status", "unknown"),
                    completed=report.get("completed", 0), planned=report.get("planned"),
                    excluded_prior_attempt=first)
    plan_path, records_path = root / "plan.json", root / "records.jsonl"
    require_sha(plan_path, report["plan_sha256"])
    require_sha(records_path, report["records_sha256"])
    plan = read(plan_path)
    route_path = EXPERIMENTS / "combined-spec-scalar-uart-v1/route/report.json"
    route = read(route_path)
    if (report.get("physical_board") is not True or report["completed"] != report["planned"] != 2
            or report["output_mismatches"] != 0 or report["programming"] != []
            or plan["image_label"] != "combined-spec-scalar-uart-v1"
            or plan["image"]["sha256"] != report["bitstream_sha256"]
            or report["bitstream_sha256"] != route["bitstream_sha256"]
            or plan["route_report_sha256"] != sha(route_path)
            or plan["burst_bytes"] != 256 or plan["baud"] != 750000):
        raise ValueError("integrated burst inference identity or coverage incomplete")
    for key in ("manifest", "native_report", "integration_report", "prior_screen_report",
                "prior_screen_plan", "prior_program_log"):
        require_sha(Path(plan[key]), plan[key + "_sha256"])
    for source, expected in plan["dependency_sha256"].items():
        source_path = ROOT / source
        require_sha(source_path, expected)
        save_artifact(source_path, artifacts)
    prior = read(Path(plan["prior_screen_report"]))
    if (prior["status"] != "passed-short-screen" or prior["completed"] != 10
            or prior["programming"][-1]["bitstream_sha256"] != route["bitstream_sha256"]):
        raise ValueError("burst inference prior programmed screen mismatch")
    rows = signed_rows(records_path)
    if [r["model"] for r in rows] != ["kws", "vww"]:
        raise ValueError("burst inference model coverage incomplete")
    checked = {}
    for row in rows:
        model = row["model"]
        name = plan["fixture_names"][model]
        fixture = Path(plan["fixture_root"]) / name
        if row["fixture"] != name or row["kind"] != "pinned-exact":
            raise ValueError("burst inference fixture mismatch")
        files = plan["fixtures"][name]["files"]
        for filename in ("commands.bin", "payload.bin", "input.bin", "output.bin"):
            require_sha(fixture / filename, files[filename])
        payload = (fixture / "payload.bin").read_bytes()
        data = (fixture / "input.bin").read_bytes()
        expected = (fixture / "output.bin").read_bytes()
        if (row["bitstream_sha256"] != route["bitstream_sha256"]
                or row["output_hex"] != row["expected_hex"]
                or bytes.fromhex(row["output_hex"]) != expected
                or row["expected_sha256"] != hashlib.sha256(expected).hexdigest()
                or row["actual_sha256"] != row["expected_sha256"]
                or row["payload_sha256"] != hashlib.sha256(payload).hexdigest()
                or row["input_sha256"] != hashlib.sha256(data).hexdigest()
                or row["command_sha256"] != sha(fixture / "commands.bin")
                or row["payload_readback_verified"] is not True
                or row["command_readback_verified"] is not True
                or row.get("input_readback_verified") is not True
                or abs(row["device_latency_ms"] - 1000*row["elapsed_cycles"]/plan["clock_hz"]) > 1e-9):
            raise ValueError("burst inference exactness or readback failed")
        for field, content in (("payload_burst", payload), ("input_burst", data)):
            burst = row[field]
            if (burst["offset"] != 0 or burst["bytes"] != len(content)
                    or burst["frames"] != (len(content) + 255) // 256
                    or burst["sha256"] != hashlib.sha256(content).hexdigest()):
                raise ValueError("burst inference frame audit mismatch")
        checked[model] = dict(output_sha256=row["actual_sha256"],
                              elapsed_cycles=row["elapsed_cycles"],
                              payload_frames=row["payload_burst"]["frames"],
                              input_frames=row["input_burst"]["frames"])
    for path in (plan_path, report_path, records_path):
        save_artifact(path, artifacts)
    return dict(status="board-exact-burst-inference-passed", physical_inferences=2,
                report=relative(report_path), report_sha256=sha(report_path),
                plan_sha256=sha(plan_path), records_sha256=sha(records_path),
                bitstream_sha256=route["bitstream_sha256"], models=checked,
                excluded_prior_attempt=first)


def uart_summary(artifacts):
    summary_path = ROOT / "docs/research/evidence/phase6/uart-burst/summary.json"
    route = read(summary_path)
    save_artifact(summary_path, artifacts)
    row = dict(status="route-only", route=route["route"],
               source_sha256={k: route[k] for k in (
                   "command_sha256", "bridge_sha256", "top_sha256", "build_tcl_sha256")},
               tests=route["tests"])
    for field, source in (
        ("command_sha256", "rtl/phase6/uart_burst_command.sv"),
        ("bridge_sha256", "rtl/phase6/uart_burst_bridge.sv"),
        ("top_sha256", "hardware/phase6/uart_burst_tiled_host.sv"),
        ("build_tcl_sha256", "hardware/phase6/build_uart_burst.tcl"),
    ):
        require_sha(ROOT / source, route[field])
    for test in route["tests"].values():
        if "xml" in test:
            xml = ROOT / test["xml"]
            require_sha(xml, test["xml_sha256"])
            save_artifact(xml, artifacts)
    image = ROOT / route["route"]["bitstream"]
    require_sha(image, route["route"]["bitstream_sha256"])
    save_artifact(image, artifacts)
    for path, key in ((image.with_suffix(".rpt.txt"), "report_sha256"),
                      (image.with_name(image.stem + "_tr_content.html"), "timing_sha256")):
        require_sha(path, route["route"][key])
        save_artifact(path, artifacts)
    row["isolated_transfer"] = transfer_summary(
        ROOT / "work/phase6/uart-burst/physical-transfer.json",
        route["route"]["bitstream_sha256"], artifacts)
    row["status"] = row["isolated_transfer"]["status"]
    integrated = EXPERIMENTS / "combined-spec-scalar-uart-v1/route/report.json"
    if integrated.exists():
        combined_route = read(integrated)
        if combined_route["status"] == "passed-route":
            row["integrated_route"] = {k: combined_route[k] for k in (
                "status", "core_clock_mhz", "routed_core_fmax_mhz",
                "setup_violated_endpoints", "hold_violated_endpoints", "resources",
                "bitstream_sha256")}
            row["integrated_transfer"] = transfer_summary(
                ROOT / "work/phase6/combined-spec-scalar-uart-v1/physical-transfer.json",
                combined_route["bitstream_sha256"], artifacts)
            row["integrated_burst_inference"] = burst_inference_summary(artifacts)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    engines = {label: engine_summary(label, artifacts) for label in ENGINES}
    baseline = baseline_summary(artifacts)
    screens = {name: screen_summary(name, directory, label, artifacts)
               for name, (directory, label) in SCREENS.items()}
    for row in screens.values():
        if row["status"] != "board-short-screen-passed":
            continue
        for model in ("kws", "vww"):
            row["models"][model]["inferences_per_second"] = 1000 / row["models"][model]["median_ms"]
            row["models"][model]["speedup_vs_selected_all_exact"] = (
                baseline["models"][model]["median_ms"] / row["models"][model]["median_ms"])
        row["geometric_mean_speedup_vs_selected_all_exact"] = math.sqrt(math.prod(
            row["models"][model]["speedup_vs_selected_all_exact"] for model in ("kws", "vww")))
    special = {}
    for name, path in (
        ("constant_filter", ROOT / "docs/research/evidence/phase6/constant-filter-v1.json"),
        ("padded_stride", ROOT / "work/phase6/padded-stride-v1/decision.json"),
        ("dual_lane", ROOT / "docs/research/evidence/phase6/dual-lane-gate-v1.json"),
        ("geometry_area", ROOT / "docs/research/evidence/phase6/geometry-area-v1.json"),
    ):
        if path.exists():
            special[name] = dict(report=relative(path), report_sha256=sha(path), data=read(path))
            save_artifact(path, artifacts)
    if "constant_filter" in special:
        current = sha(ROOT / "tools/phase6/constant_filter.py")
        special["constant_filter"]["current_source_sha256"] = current
        special["constant_filter"]["historical_source_matches_current"] = (
            special["constant_filter"]["data"]["source_sha256"] == current)
    for p in (
        "work/phase6/constant-filter-v1/fixtures.json",
        "work/phase6/constant-filter-v1/native-all-exact/report.json",
        "work/phase6/constant-sibling-v1/fixtures.json",
        "work/phase6/constant-sibling-v1/native-all-exact/report.json",
        "work/phase6/padded-stride-v1/fixtures.json",
        "work/phase6/padded-stride-v1/native-padded-stride/report.json",
        "work/phase6/padded-stride-gate/analysis.json",
        "work/phase6/geometry-area-v1/route/report.json",
    ):
        path = ROOT / p
        if path.exists():
            save_artifact(path, artifacts)
    source_paths = [ROOT / p for p in (
        "tools/phase6/archive_followon.py", "tools/phase6/constant_filter.py",
        "tools/phase6/chain_resident.py", "tools/phase6/experiments.py",
        "tools/phase6/combined_next.py", "tools/phase6/combined_spec.py",
        "tools/phase6/combined_spec_scalar.py", "tools/phase6/screen_schedule.py",
        "tools/phase6/screen_engine_schedule.py", "tools/phase6/padded_stride_decision.py",
        "tools/phase6/dual_lane_gate.py", "tools/phase6/uart_burst.py",
        "tools/phase6/measure_uart_burst.py", "compiler/scheduler/resident_verify.py",
        "tools/phase6/combined_scalar_uart.py", "hardware/phase6/build_uart_burst.tcl",
        "tools/phase6/check_burst_inference.py",
        "hardware/phase6/uart_burst_tiled_host.sv", "rtl/phase6/uart_burst_command.sv",
        "rtl/phase6/uart_burst_bridge.sv", "rtl/phase6/geometry_engine.sv",
        "rtl/phase6/padded_stride_engine.sv",
    )]
    sources = {relative(path): save_artifact(path, artifacts)["sha256"]
               for path in source_paths if path.exists()}
    result = dict(schema=1, scope="Phase 6 follow-on short screens, native RTL, route and rejected cost gates; no full accuracy or energy qualification",
                  selected_all_exact_baseline=baseline, engines=engines, board_screens=screens,
                  uart_burst=uart_summary(artifacts), boardless_gates=special,
                  acceptance_gate=dict(exact_outputs=True, passing_route=True,
                                       minimum_geometric_mean_throughput_speedup=1.03,
                                       maximum_unexplained_single_model_regression_fraction=0.01),
                  source_sha256=sources,
                  artifacts=artifacts)
    output = OUT / "summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(output=relative(output), board_screens=sum(
        r["status"] == "board-short-screen-passed" for r in screens.values()),
        artifacts=len(artifacts)), sort_keys=True))


if __name__ == "__main__":
    main()
