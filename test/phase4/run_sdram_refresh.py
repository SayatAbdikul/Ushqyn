#!/usr/bin/env python3
"""Run the standalone refresh scheduler with a short simulated period."""

import xml.etree.ElementTree as ET
from pathlib import Path

from cocotb.runner import get_runner

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT/'work/phase4/sim_sdram_refresh'
runner = get_runner('verilator')
runner.build(verilog_sources=[ROOT/'rtl/v2/sdram_refresh.sv'],
             hdl_toplevel='v2_sdram_refresh', build_dir=BUILD,
             parameters={'REFRESH_PERIOD_CYCLES': 8},
             build_args=['--timing', '-Wno-fatal'], always=True,
             timescale=('1ns', '1ps'))
runner.test(hdl_toplevel='v2_sdram_refresh', test_module='test_sdram_refresh',
            test_dir=Path(__file__).parent, build_dir=BUILD,
            results_xml=str(BUILD/'results.xml'))
cases = ET.parse(BUILD/'results.xml').findall('.//testcase')
if not cases or any(c.find('failure') is not None or c.find('error') is not None for c in cases):
    raise SystemExit('Phase 4 refresh scheduler regression failed')
