#!/usr/bin/env python3
"""Run mixed-model board-system RTL integration with Cocotb."""
import json
import os
import sys
from pathlib import Path
from cocotb.runner import get_runner

ROOT=Path(__file__).resolve().parents[2]
target=json.loads((ROOT/'hardware/targets/tang_nano_20k_v2.json').read_text())
build=ROOT/'work/phase3/sim_switch'
runner=get_runner('verilator')
runner.build(verilog_sources=[ROOT/s for s in target['sources']],hdl_toplevel='v2_system',
             build_dir=build,build_args=['--timing','-Wno-fatal'],parameters={'TIMEOUT_CYCLES':100},
             always=True,timescale=('1ns','1ps'))
sys.path[:0]=[str(ROOT/'test/phase2'),str(ROOT/'compiler'),str(ROOT/'tools/phase2')]
runner.test(hdl_toplevel='v2_system',test_module='test_switch',test_dir=Path(__file__).parent,
            build_dir=build,results_xml=str(build/'results.xml'),
            extra_env={'PYTHONPATH':os.pathsep.join(sys.path),'REPO_ROOT':str(ROOT)})
