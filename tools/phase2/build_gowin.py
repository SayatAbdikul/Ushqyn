#!/usr/bin/env python3
"""Prepare an isolated Gowin build consuming exactly the pinned source manifest."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'compiler'))
from hardware_v2 import TARGET
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
lines=[f'create_project -name tinyml_v2 -dir [pwd] -pn {TARGET["device"]} -device_version {TARGET["device_revision"]}']
lines += ['add_file {'+str(ROOT/s)+'}' for s in TARGET['sources']]
lines += ['add_file {'+str(ROOT/'hardware/tang_nano_v2.cst')+'}','add_file {'+str(ROOT/'hardware/tang_nano_v2.sdc')+'}', 'set_option -top_module tang_nano_v2','set_option -verilog_std sysv2017','set_option -output_base_name tinyml_v2','run all','exit']
(a.directory/'build.tcl').write_text('\n'.join(lines)+'\n')
print(a.directory/'build.tcl')
