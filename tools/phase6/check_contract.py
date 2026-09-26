#!/usr/bin/env python3
"""Archive boardless S01 examples; never opens UART or touches FPGA fixtures."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler"))
from scheduler.examples import disjoint_dma, quantized_chain
from scheduler.verify import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "work/phase6/contract.json")
    args = parser.parse_args()
    examples = {}
    for name, factory in (("quantized_chain", quantized_chain), ("disjoint_dma", disjoint_dma)):
        problem, certificate = factory()
        examples[name] = {"problem": asdict(problem), "certificate": asdict(certificate),
                          "verification": verify(problem, certificate)}
    files = ["compiler/scheduler/contract.py", "compiler/scheduler/verify.py",
             "compiler/scheduler/examples.py", "compiler/test_scheduler_contract.py",
             "tools/phase6/check_contract.py"]
    report = {"status": "passed", "scope": "synthetic fixed-task contract tests only",
              "phase6_gate_complete": False, "hardware_executable": False,
              "cycle_costs": "hand-chosen synthetic ticks; no latency prediction claim",
              "examples": examples,
              "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"passed: {len(examples)} boardless examples; G6 remains open")


if __name__ == "__main__":
    main()
