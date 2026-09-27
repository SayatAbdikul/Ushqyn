#!/usr/bin/env python3
"""Run the existing tiled host integration regression on the burst bridge."""

import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from cocotb.runner import get_runner

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "work/phase6/uart-burst-512/integration"
ENGINE = ROOT / "work/phase6/experiments-v1/all-exact/engine.sv"
SOURCES = [ROOT / "rtl/v2/target_pkg.sv", ROOT / "rtl/v2/requantizer.sv",
           ENGINE, ROOT / "rtl/v2/scratchpad.sv", ROOT / "rtl/v2/tile_dma.sv",
           ROOT / "rtl/v2/tiled_core.sv",
           ROOT / "rtl/phase6/uart_burst_512_command.sv",
           ROOT / "rtl/v2/tile_sequencer.sv",
           ROOT / "rtl/phase6/uart_burst_512_bridge.sv"]


def main():
    runner = get_runner("verilator")
    runner.build(verilog_sources=SOURCES, hdl_toplevel="phase6_uart_burst_512_bridge",
                 build_dir=BUILD, build_args=["--timing", "-Wno-fatal"],
                 timescale=("1ns", "1ps"))
    sys.path[:0] = [str(ROOT / "compiler"), str(ROOT / "test/phase4"),
                    str(ROOT / "test/phase6")]
    runner.test(hdl_toplevel="phase6_uart_burst_512_bridge",
                test_module=["test_tiled_host", "test_uart_burst_512_bridge"],
                test_dir=ROOT / "test/phase6",
                build_dir=BUILD, results_xml=str(BUILD / "results.xml"),
                extra_env={"PYTHONPATH": os.pathsep.join(sys.path),
                           "PHASE4_TILED_HOST_FIXTURE": ""})
    tests = ET.parse(BUILD / "results.xml").findall(".//testcase")
    if len(tests) != 2 or any(case.find("failure") is not None or
                        case.find("error") is not None for case in tests):
        raise SystemExit("UART burst tiled-host integration failed")


if __name__ == "__main__":
    main()
