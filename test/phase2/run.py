#!/usr/bin/env python3
import argparse,json,os,sys
from pathlib import Path
from cocotb.runner import get_runner
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('suite',choices=['requantizer','engine','system','board']);a=p.parse_args()
top={'requantizer':'v2_requantizer','engine':'v2_engine','system':'v2_system','board':'tang_nano_v2'}[a.suite]
target=json.loads((ROOT/'hardware/targets/tang_nano_20k_v2.json').read_text())
sources=[ROOT/s for s in target['sources']]
r=get_runner('verilator');build=ROOT/'work/phase2'/f'sim_{a.suite}'
params={'TIMEOUT_CYCLES':100} if a.suite=='system' else ({'CLOCK_HZ':100,'BAUD':10} if a.suite=='board' else {})
r.build(verilog_sources=sources,hdl_toplevel=top,build_dir=build,build_args=['--timing','-Wno-fatal'],parameters=params,always=True,timescale=('1ns','1ps'))
sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,str(ROOT/'compiler'));sys.path.insert(0,str(ROOT/'tools/phase2'))
r.test(hdl_toplevel=top,test_module='test_'+a.suite,test_dir=Path(__file__).parent,build_dir=build,results_xml=str(build/'results.xml'),extra_env={'PYTHONPATH':os.pathsep.join(sys.path),'REPO_ROOT':str(ROOT)})
