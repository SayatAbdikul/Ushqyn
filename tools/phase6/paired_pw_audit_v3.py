#!/usr/bin/env python3
"""Bounded paired-pointwise overflow and odd-tail check on three engine images.

The selected co-issue engine is the numerical reference. This standalone
screen uses one 32 KiB scratchpad model and does not allocate model tensors.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path("/Users/sayat/.codex/worktrees/21f0/tinyML_accelerator")
OUT = ROOT / "work/phase6/paired-pw-audit-v3"
EXPECTED_PARENT = "9f934dcc27ae891e4c7b8b84bcc4b65e82d903492c1867572784fbe1cd9484ee"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    from cocotb.runner import get_runner

    engines = {
        "selected_coissue": REFERENCE / "work/phase6/engine-candidate-rtl-v2/engine.sv",
        "paired_v2": ROOT / "work/phase6/paired-pw-rtl-v2/engine.sv",
        "paired_v3": ROOT / "work/phase6/paired-pw-rtl-v3/engine.sv",
    }
    if digest(engines["selected_coissue"]) != EXPECTED_PARENT:
        raise ValueError("selected co-issue reference changed")
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "scope": "directed engine-level paired arithmetic and odd-tail correctness", "engines": {}}
    report_path = OUT / "report.json"
    for name, engine in engines.items():
        build = OUT / name
        sources = [ROOT / "rtl/v2/target_pkg.sv", ROOT / "rtl/v2/requantizer.sv", engine]
        runner = get_runner("verilator")
        runner.build(
            verilog_sources=sources,
            hdl_toplevel="v2_engine",
            build_dir=build,
            build_args=["--timing", "-Wno-fatal", "-j", "2"],
            timescale=("1ns", "1ps"),
        )
        paths = [str(ROOT / "compiler"), str(ROOT / "tools/phase6")]
        sys.path[:0] = paths
        runner.test(
            hdl_toplevel="v2_engine",
            test_module="paired_pw_audit_test",
            test_dir=ROOT / "tools/phase6",
            build_dir=build,
            results_xml=str(build / "results.xml"),
            extra_env={"PYTHONPATH": os.pathsep.join(paths + sys.path)},
        )
        cases = ET.parse(build / "results.xml").findall(".//testcase")
        item = {
            "status": "passed" if len(cases) == 1 and all(
                case.find("failure") is None and case.find("error") is None for case in cases
            ) else "failed",
            "engine": str(engine),
            "engine_sha256": digest(engine),
            "source_sha256": {str(p): digest(p) for p in sources},
            "test_sha256": digest(ROOT / "tools/phase6/paired_pw_audit_test.py"),
            "results_sha256": digest(build / "results.xml"),
            "cases": [case.get("name") for case in cases],
        }
        report["engines"][name] = item
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if item["status"] != "passed":
            raise ValueError(f"{name} directed test failed")
    report["status"] = "passed"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
