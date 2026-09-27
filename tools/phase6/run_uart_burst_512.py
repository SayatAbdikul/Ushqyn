#!/usr/bin/env python3
"""Build and test the isolated UART burst parser in RTL simulation."""

import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from cocotb.runner import get_runner

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "work/phase6/uart-burst-512/sim"


def main():
    sources = [ROOT / "rtl/v2/target_pkg.sv",
               ROOT / "rtl/phase6/uart_burst_512_command.sv"]
    runner = get_runner("verilator")
    runner.build(verilog_sources=sources, hdl_toplevel="phase6_uart_burst_512_command",
                 build_dir=BUILD, build_args=["--timing", "-Wno-fatal", "-GTILED_MEM_MAP=1",
                                                "-GTIMEOUT_CYCLES=400"],
                 timescale=("1ns", "1ps"))
    python_path = os.pathsep.join([str(ROOT / "test/phase6"), *sys.path])
    runner.test(hdl_toplevel="phase6_uart_burst_512_command",
                test_module="test_uart_burst_512", test_dir=ROOT / "test/phase6",
                build_dir=BUILD, results_xml=str(BUILD / "results.xml"),
                extra_env={"PYTHONPATH": python_path})
    tests = ET.parse(BUILD / "results.xml").findall(".//testcase")
    if len(tests) != 1 or any(case.find("failure") is not None or
                               case.find("error") is not None for case in tests):
        raise SystemExit("UART burst RTL test failed")


if __name__ == "__main__":
    main()
