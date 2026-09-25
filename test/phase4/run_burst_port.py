#!/usr/bin/env python3
"""Run adapter logic against a test-only HS command timing contract."""
from pathlib import Path
import xml.etree.ElementTree as ET
from cocotb.runner import get_runner

ROOT=Path(__file__).resolve().parents[2]
BUILD=ROOT/'work/phase4/sim_burst_port'
runner=get_runner('verilator')
runner.build(verilog_sources=[ROOT/p for p in (
    'rtl/v2/hs_sdram_burst_port.sv','rtl/v2/sdram_refresh.sv',
    'test/phase4/sdram_hs_contract_model.sv')],hdl_toplevel='v2_hs_sdram_burst_port',
    build_dir=BUILD,build_args=['--timing','-Wno-fatal','--assert'],
    parameters={'CLOCK_HZ':20250000},always=True,timescale=('1ns','1ps'))
runner.test(hdl_toplevel='v2_hs_sdram_burst_port',test_module='test_burst_port',
    test_dir=Path(__file__).parent,build_dir=BUILD,results_xml=str(BUILD/'results.xml'))
cases=ET.parse(BUILD/'results.xml').findall('.//testcase')
if not cases or any(c.find('failure') is not None or c.find('error') is not None for c in cases):
    raise SystemExit('burst adapter contract test failed')
