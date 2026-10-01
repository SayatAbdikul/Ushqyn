#!/usr/bin/env python3
"""Route the exact full-model-tested paired-pointwise control at 27 MHz."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
import xml.etree.ElementTree as ET

from paired_pw_native import ROOT, REFERENCE, PARENT_SHA, sha, save


BASE = ROOT / "work/phase6/paired-pw-rtl-v3"
ENGINE = BASE / "engine.sv"
POOL = ROOT / "work/phase6/pool-timing-v1"
GOWIN = Path("/Applications/GowinIDE.app/Contents/Resources/Gowin_EDA/IDE")
ROUTE = BASE / "route27"


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def prepare() -> dict:
    native = json.loads((BASE / "native/report.json").read_text())
    edges = json.loads((BASE / "edges/report.json").read_text())
    audit_path = ROOT / "work/phase6/paired-pw-audit-v3/report.json"
    audit = json.loads(audit_path.read_text()) if BASE.name.endswith("v3") else None
    if native["status"] != "passed" or len(native["results"]) != 8:
        raise ValueError("full paired KWS/VWW native matrix required")
    if edges["status"] != "passed" or len(edges["cases"]) != 3:
        raise ValueError("paired RTL edge suite required")
    if audit is not None and (audit["status"] != "passed" or
            audit["engines"]["paired_v3"]["engine_sha256"] != sha(ENGINE)):
        raise ValueError("directed paired-overflow audit required")
    if native["engine_sha256"] != sha(ENGINE) or edges["engine_sha256"] != sha(ENGINE):
        raise ValueError("route engine differs from proved RTL")
    if native["parent_sha256"] != PARENT_SHA or edges["parent_sha256"] != PARENT_SHA:
        raise ValueError("selected parent identity changed")
    for path, digest in native["source_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"native source changed: {path}")
    parent = json.loads((REFERENCE / "work/phase6/engine-candidate-rtl-v2/route27/report.json").read_text())
    if parent["status"] != "passed-route" or parent["engine_sha256"] != PARENT_SHA:
        raise ValueError("selected 27 MHz parent route unavailable")
    old_sources = dict(parent["active_gprj_source_coverage"]["source_sha256"])
    old_sources.pop("work/phase6/engine-candidate-rtl-v2/engine.sv")
    route_sources = dict(old_sources, **{rel(ENGINE): sha(ENGINE)})
    if len(route_sources) != 18:
        raise ValueError("physical source count changed")
    for name, digest in route_sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"physical source changed: {name}")
    recipe = POOL / "build27.tcl"
    text = recipe.read_text()
    if f"set root {{{ROOT}}}" not in text:
        raise ValueError("27 MHz route recipe uses another checkout")
    script = BASE / "build27.tcl"
    if script.exists() and script.read_text() != text:
        raise ValueError("route recipe changed")
    script.write_text(text)
    ROUTE.mkdir(parents=True, exist_ok=True)
    inputs = {"status": "prepared", "engine": rel(ENGINE), "engine_sha256": sha(ENGINE),
              "selected_parent_engine_sha256": PARENT_SHA,
              "selected_parent_route_report_sha256": sha(REFERENCE / "work/phase6/engine-candidate-rtl-v2/route27/report.json"),
              "native_report_sha256": sha(BASE / "native/report.json"),
              "edges_report_sha256": sha(BASE / "edges/report.json"),
              "directed_audit_report_sha256": sha(audit_path) if audit is not None else None,
              "route_recipe": rel(script), "route_recipe_sha256": sha(script),
              "expected_project_sources_sha256": route_sources,
              "tool_executable": str(GOWIN / "bin/gw_sh"),
              "tool_executable_sha256": sha(GOWIN / "bin/gw_sh"),
              "core_clock_mhz": 27, "physical_board": False}
    path = ROUTE / "inputs.json"
    if path.exists() and json.loads(path.read_text()) != inputs:
        raise ValueError("immutable route inputs changed")
    save(path, inputs)
    return inputs


def route() -> dict:
    inputs = prepare()
    report_path = ROUTE / "report.json"
    if report_path.exists():
        raise ValueError("route attempt already recorded")
    env = dict(os.environ, DYLD_FRAMEWORK_PATH=str(GOWIN / "lib"),
               DYLD_LIBRARY_PATH=str(GOWIN / "lib"), PHASE6_ENGINE=str(ENGINE))
    start = time.monotonic()
    with (ROUTE / "route.log").open("w") as log:
        process = subprocess.run([str(GOWIN / "bin/gw_sh"), str(BASE / "build27.tcl")],
                                 cwd=ROUTE, env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
    report = dict(inputs, status="failed-build", process_returncode=process.returncode,
                  build_seconds=time.monotonic()-start,
                  build_log=rel(ROUTE / "route.log"),
                  build_log_sha256=sha(ROUTE / "route.log"))
    project = ROUTE / "phase6_uart_burst/phase6_uart_burst.gprj"
    if project.exists():
        active = []
        for node in ET.parse(project).findall(".//File"):
            if node.attrib.get("enable") == "1":
                path = Path(node.attrib["path"])
                active.append(rel(path if path.is_absolute() else project.parent / path))
        if len(active) != 18 or len(set(active)) != 18 or set(active) != set(inputs["expected_project_sources_sha256"]):
            raise ValueError("active Gowin source set differs from proved set")
        report["active_gprj_source_coverage"] = active
    if process.returncode:
        save(report_path, report)
        return report
    pnr = project.parent / "impl/pnr"
    routed_path = pnr / "phase6_uart_burst.rpt.txt"
    timing_path = pnr / "phase6_uart_burst_tr_content.html"
    bitstream = pnr / "phase6_uart_burst.fs"
    routed, timing = routed_path.read_text(), timing_path.read_text()
    resources = {}
    for key in ("Logic", "Register", "BSRAM", "DSP", "CLS"):
        match = re.search(rf"^\s*{key}\s*\|\s*([\d.]+)/([\d.]+)", routed, re.M)
        if not match:
            raise ValueError(f"missing routed resource {key}")
        resources[key] = {"used": float(match[1]), "available": float(match[2])}
    fmax = re.search(r"<td>27\.000\(MHz\)</td>\s*<td[^>]*>([\d.]+)\(MHz\)</td>", timing)
    if not fmax:
        raise ValueError("core timing result missing")
    report.update(resources=resources, routed_core_fmax_mhz=float(fmax[1]),
                  bitstream=rel(bitstream), bitstream_sha256=sha(bitstream),
                  route_report=rel(routed_path), route_report_sha256=sha(routed_path),
                  timing_report=rel(timing_path), timing_report_sha256=sha(timing_path))
    for kind in ("Setup", "Hold"):
        match = re.search(rf"Numbers of {kind} Violated Endpoints</td>\s*<td[^>]*>(\d+)</td>", timing)
        if not match:
            raise ValueError(f"{kind} endpoint count missing")
        report[kind.lower() + "_violated_endpoints"] = int(match[1])
    report["status"] = "passed-route" if (
        report["routed_core_fmax_mhz"] >= 27 and
        report["setup_violated_endpoints"] == report["hold_violated_endpoints"] == 0 and
        all(r["used"] <= r["available"] for r in resources.values())) else "failed-timing"
    save(report_path, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "route"))
    parser.add_argument("--variant", choices=("v2", "v3"), default="v3")
    parser.add_argument("--route-dir", default="route27")
    args = parser.parse_args()
    BASE = ROOT / f"work/phase6/paired-pw-rtl-{args.variant}"
    ENGINE = BASE / "engine.sv"
    ROUTE = BASE / args.route_dir
    result = prepare() if args.stage == "prepare" else route()
    print(json.dumps({key: result.get(key) for key in
                      ("status", "build_seconds", "routed_core_fmax_mhz", "resources")},
                     indent=2))
