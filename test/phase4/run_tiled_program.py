#!/usr/bin/env python3
"""Run packed tile-image arithmetic through the engine/DMA/SRAM RTL."""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from cocotb.runner import get_runner

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fixture', type=Path)
args = parser.parse_args()
BUILD = ROOT/'work/phase4/sim_tiled_program'
sources = [ROOT/name for name in (
    'rtl/v2/target_pkg.sv', 'rtl/v2/requantizer.sv',
    'rtl/v2/scratchpad.sv', 'rtl/v2/engine.sv',
    'rtl/v2/tile_dma.sv', 'rtl/v2/tiled_core.sv')]
runner = get_runner('verilator')
runner.build(verilog_sources=sources, hdl_toplevel='v2_tiled_core',
             build_dir=BUILD, build_args=['--timing', '-Wno-fatal'],
             always=True, timescale=('1ns', '1ps'))
sys.path.insert(0, str(ROOT/'compiler'))
runner.test(hdl_toplevel='v2_tiled_core', test_module='test_tiled_program',
            test_dir=Path(__file__).parent, build_dir=BUILD,
            results_xml=str(BUILD/'results.xml'),
            extra_env={'PYTHONPATH': os.pathsep.join(sys.path),
                       'PHASE4_TILED_FIXTURE': str(args.fixture.resolve())
                       if args.fixture else ''})
cases = ET.parse(BUILD/'results.xml').findall('.//testcase')
if not cases or any(c.find('failure') is not None or c.find('error') is not None for c in cases):
    raise SystemExit('Phase 4 packed tiled program RTL test failed')
