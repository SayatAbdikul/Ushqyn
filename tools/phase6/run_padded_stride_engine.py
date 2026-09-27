#!/usr/bin/env python3
"""Compile and run the isolated physical channel stride engine checks."""
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from cocotb.runner import get_runner

ROOT=Path(__file__).resolve().parents[2]
build=ROOT/'work/phase6/padded-stride-engine'
sources=[ROOT/'rtl/v2/target_pkg.sv',ROOT/'rtl/v2/requantizer.sv',
         ROOT/'rtl/phase6/padded_stride_engine.sv']
runner=get_runner('verilator')
runner.build(verilog_sources=sources,hdl_toplevel='v2_engine',build_dir=build,
             build_args=['--timing','-Wno-fatal'],timescale=('1ns','1ps'))
paths=[str(ROOT/'compiler'),str(ROOT/'test/phase2'),str(ROOT/'test/phase6'),*sys.path]
sys.path[:]=paths
runner.test(hdl_toplevel='v2_engine',
            test_module=['test_engine','test_cache','test_writeback','test_padded_stride_engine'],
            test_dir=ROOT/'test/phase6',build_dir=build,
            results_xml=str(build/'results.xml'),
            extra_env={'PYTHONPATH':os.pathsep.join(paths)})
cases=ET.parse(build/'results.xml').findall('.//testcase')
if len(cases)!=5 or any(c.find('failure') is not None or c.find('error') is not None for c in cases):
    raise SystemExit('physical plane RTL simulation failed')
print('5 legacy and padded stride RTL checks passed')
