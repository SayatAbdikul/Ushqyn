#!/usr/bin/env python3
"""Random-stall engine regressions for isolated Phase 6 variants."""
import argparse
import os
import sys
import xml.etree.ElementTree as ET
from cocotb.runner import get_runner
from variants import ROOT, VARIANTS, sources, check_frozen

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--variant', choices=VARIANTS, default='c256p1')
args = parser.parse_args()
check_frozen()
build = ROOT/f'work/phase6/optimization/engine-{args.variant}'
runner = get_runner('verilator')
runner.build(verilog_sources=sources(args.variant, build, True), hdl_toplevel='v2_engine',
             build_dir=build, build_args=['--timing', '-Wno-fatal'], timescale=('1ns', '1ps'))
paths = [str(ROOT/'compiler'), str(ROOT/'test/phase2'), str(ROOT/'test/phase6'), *sys.path]
sys.path[:] = paths
runner.test(hdl_toplevel='v2_engine', test_module=['test_engine','test_cache'],
            test_dir=ROOT/'test/phase6', build_dir=build, results_xml=str(build/'results.xml'),
            extra_env={'PYTHONPATH': os.pathsep.join(paths)})
cases = ET.parse(build/'results.xml').findall('.//testcase')
if len(cases) != 2 or any(c.find('failure') is not None or c.find('error') is not None for c in cases):
    raise SystemExit('Phase 6 engine regression failed')
check_frozen()
