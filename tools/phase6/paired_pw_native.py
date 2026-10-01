#!/usr/bin/env python3
"""Bounded exact native screen of the ordinary paired-pointwise RTL control.

The selected co-issue image and its frozen fixtures live in the parent
worktree. This script never materializes model tensors in Python: it verifies
small fixture files by streaming hashes, builds with two compiler jobs, and
runs one fixture at a time in the fixed-size C++ native harness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "work/phase6/paired-pw-rtl-v1"
ENGINE = BASE / "engine.sv"
REFERENCE = Path("/Users/sayat/.codex/worktrees/21f0/tinyML_accelerator")
PARENT_SHA = "9f934dcc27ae891e4c7b8b84bcc4b65e82d903492c1867572784fbe1cd9484ee"
RTL = ["target_pkg.sv", "requantizer.sv"]
HOST = ["scratchpad.sv", "tile_dma.sv", "tiled_core.sv", "command.sv",
        "tile_sequencer.sv", "tiled_host_bridge.sv"]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sources(engine_only: bool = False) -> list[Path]:
    result = [ROOT / "rtl/v2" / name for name in RTL] + [ENGINE]
    if not engine_only:
        result += [ROOT / "rtl/v2" / name for name in HOST]
    return result


def reference_report() -> dict:
    source = REFERENCE / "work/phase6/engine-candidate-rtl-v2"
    if sha(source / "engine.sv") != PARENT_SHA:
        raise ValueError("selected co-issue parent changed")
    report = json.loads((source / "native/report.json").read_text())
    if report["status"] != "passed" or report["engine_sha256"] != PARENT_SHA:
        raise ValueError("selected co-issue native proof incomplete")
    for name, digest in report["source_sha256"].items():
        if name.startswith("work/phase6/engine-candidate-rtl-v2/"):
            continue
        location = REFERENCE / name if name.startswith("tools/phase6/hypothesis_engine_") else ROOT / name
        if sha(location) != digest:
            raise ValueError(f"shared native source changed: {name}")
    executable = source / "native/Vv2_tiled_host_bridge"
    if sha(executable) != report["executable_sha256"]:
        raise ValueError("selected co-issue executable changed")
    return report


def edges() -> dict:
    from cocotb.runner import get_runner

    reference_report()
    build = BASE / "edges"
    runner = get_runner("verilator")
    runner.build(verilog_sources=sources(True), hdl_toplevel="v2_engine",
                 build_dir=build, build_args=["--timing", "-Wno-fatal"],
                 timescale=("1ns", "1ps"))
    test_names = ["test_cache", "test_fused_activation", "test_pool_timing"]
    paths = [str(ROOT / name) for name in ("compiler", "test/phase2", "test/phase6")]
    sys.path[:0] = paths
    runner.test(hdl_toplevel="v2_engine", test_module=test_names,
                test_dir=ROOT / "test/phase6", build_dir=build,
                results_xml=str(build / "results.xml"),
                extra_env={"PYTHONPATH": os.pathsep.join(paths + sys.path)})
    cases = ET.parse(build / "results.xml").findall(".//testcase")
    result = {"status": "passed" if len(cases) == 3 and all(
        c.find("failure") is None and c.find("error") is None for c in cases) else "failed",
        "cases": [c.get("name") for c in cases],
        "parent_sha256": PARENT_SHA, "engine_sha256": sha(ENGINE),
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources(True)},
        "results_sha256": sha(build / "results.xml")}
    save(build / "report.json", result)
    if result["status"] != "passed":
        raise ValueError("paired pointwise focused RTL cases failed")
    return result


def build_native() -> Path:
    reference_report()
    build = BASE / "native"
    build.mkdir(parents=True, exist_ok=True)
    harness = ROOT / "test/phase6/native.cpp"
    command = ["verilator", "--cc", "--exe", "--build", "-j", "2",
               "--public-flat-rw", "-Wno-fatal", "--top-module",
               "v2_tiled_host_bridge", "--Mdir", str(build),
               *map(str, sources()), str(harness)]
    with (build / "build.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       check=True)
    return build / "Vv2_tiled_host_bridge"


def matrix(limit: int | None = None) -> dict:
    parent = reference_report()
    executable = build_native()
    input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources() +
                    [ROOT / "test/phase6/native.cpp"]}
    report = {"status": "running", "scope": "native full model, fixed/stalled abstract external RAM",
              "physical_board": False, "parent_sha256": PARENT_SHA,
              "engine_sha256": sha(ENGINE), "executable_sha256": sha(executable),
              "source_sha256": input_hashes,
              "reference_report_sha256": sha(REFERENCE / "work/phase6/engine-candidate-rtl-v2/native/report.json"),
              "results": []}
    destination = BASE / "native/report.json"
    save(destination, report)
    fixtures = {}
    for row in parent["results"]:
        key = (row["model"], row["sample"], row["stall_seed"])
        if key in fixtures:
            raise ValueError("duplicate selected fixture")
        fixtures[key] = row
    selected = sorted(fixtures)
    if limit is not None:
        selected = selected[:limit]
    for model, sample, seed in selected:
        row = fixtures[(model, sample, seed)]
        expected = row["candidate"]
        directory = REFERENCE / expected["fixture_directory"]
        for name, digest in expected["fixture_files"].items():
            if sha(directory / name) != digest:
                raise ValueError(f"fixture changed: {directory / name}")
        output = BASE / "native" / f"{model}-{sample}-s{seed}.json"
        started = time.monotonic()
        subprocess.run([str(executable), str(directory), str(seed), str(output)],
                       cwd=ROOT, check=True)
        result = json.loads(output.read_text())
        if result["status"] != "passed" or result["tensor_checks"] != expected["tensor_checks"]:
            raise ValueError(f"native result incomplete: {model}/{sample}/{seed}")
        result.update(model=model, sample=sample, fixture=str(directory),
                      fixture_files_sha256=expected["fixture_files"],
                      simulation_seconds=time.monotonic() - started,
                      reference_elapsed_cycles=expected["elapsed_cycles"],
                      reference_engine_cycles=expected["engine_cycles"],
                      saved_elapsed_cycles=expected["elapsed_cycles"] - result["elapsed_cycles"],
                      saved_engine_cycles=expected["engine_cycles"] - result["engine_cycles"])
        report["results"].append(result)
        save(destination, report)
        print(model, sample, seed, "device delta", result["saved_elapsed_cycles"],
              "engine delta", result["saved_engine_cycles"], flush=True)
    report["status"] = "passed" if len(report["results"]) == len(fixtures) else "partial"
    for name, digest in input_hashes.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"native source changed during run: {name}")
    save(destination, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("edges", "native"))
    parser.add_argument("--variant", choices=("v1", "v2", "v3"), default="v3")
    parser.add_argument("--reference-root", type=Path, default=REFERENCE)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    REFERENCE = args.reference_root.resolve()
    BASE = ROOT / f"work/phase6/paired-pw-rtl-{args.variant}"
    ENGINE = BASE / "engine.sv"
    if args.stage == "edges":
        print(json.dumps(edges(), indent=2))
    else:
        matrix(args.limit)
