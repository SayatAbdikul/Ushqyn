#!/usr/bin/env python3
"""Isolated candidate integration simulation; no Phase 5 file or device access."""
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from cocotb.runner import get_runner

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT/'work/phase6/sim_candidates'
sources = [ROOT/'rtl/v2'/name for name in ('target_pkg.sv', 'requantizer.sv', 'scratchpad.sv', 'engine.sv',
            'tile_dma.sv', 'tiled_core.sv', 'command.sv', 'tile_sequencer.sv', 'tiled_host_bridge.sv')]
runner = get_runner('verilator')
runner.build(verilog_sources=sources, hdl_toplevel='v2_tiled_host_bridge', build_dir=BUILD,
             build_args=['--timing', '-Wno-fatal'], timescale=('1ns', '1ps'))
sys.path.insert(0, str(ROOT/'compiler'))
runner.test(hdl_toplevel='v2_tiled_host_bridge', test_module='test_candidates', test_dir=ROOT/'test/phase6',
            build_dir=BUILD, results_xml=str(BUILD/'results.xml'),
            extra_env={'PYTHONPATH': os.pathsep.join(sys.path)})
cases = ET.parse(BUILD/'results.xml').findall('.//testcase')
if not cases or any(c.find('failure') is not None or c.find('error') is not None for c in cases):
    raise SystemExit('Phase 6 candidate integration simulation failed')
