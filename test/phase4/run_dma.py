#!/usr/bin/env python3
"""Run the abstract-port Phase 4 tile DMA RTL regression."""
import xml.etree.ElementTree as ET
from pathlib import Path
from cocotb.runner import get_runner

ROOT=Path(__file__).resolve().parents[2]
BUILD=ROOT/'work/phase4/sim_dma'
runner=get_runner('verilator')
runner.build(verilog_sources=[ROOT/'rtl/v2/tile_dma.sv'],hdl_toplevel='v2_tile_dma',
             build_dir=BUILD,build_args=['--timing','-Wno-fatal'],always=True,
             timescale=('1ns','1ps'))
runner.test(hdl_toplevel='v2_tile_dma',test_module='test_tile_dma',
            test_dir=Path(__file__).parent,build_dir=BUILD,
            results_xml=str(BUILD/'results.xml'))
cases=ET.parse(BUILD/'results.xml').findall('.//testcase')
if not cases or any(c.find('failure') is not None or c.find('error') is not None for c in cases):
    raise SystemExit('phase4 tile DMA RTL test failed')
