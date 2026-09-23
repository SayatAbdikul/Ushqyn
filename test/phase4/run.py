#!/usr/bin/env python3
"""Run Phase 4 kernel RTL tests against the active board source manifest."""
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from cocotb.runner import get_runner

ROOT=Path(__file__).resolve().parents[2]
TARGET=json.loads((ROOT/'hardware/targets/tang_nano_20k_v2.json').read_text())
BUILD=ROOT/'work/phase4/sim_kernels'
sources=[ROOT/name for name in TARGET['sources']]
runner=get_runner('verilator')
runner.build(verilog_sources=sources,hdl_toplevel='v2_engine',build_dir=BUILD,
             build_args=['--timing','-Wno-fatal'],always=True,timescale=('1ns','1ps'))
sys.path.insert(0,str(ROOT/'compiler'))
runner.test(hdl_toplevel='v2_engine',test_module='test_kernels',
            test_dir=Path(__file__).parent,build_dir=BUILD,
            results_xml=str(BUILD/'results.xml'),
            extra_env={'PYTHONPATH':os.pathsep.join(sys.path),'REPO_ROOT':str(ROOT)})
cases=ET.parse(BUILD/'results.xml').findall('.//testcase')
if not cases or any(case.find('failure') is not None or case.find('error') is not None for case in cases):
    raise SystemExit('phase4 RTL kernel test failed')
