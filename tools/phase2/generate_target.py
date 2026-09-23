#!/usr/bin/env python3
"""Emit/check one manifest for the compiler, RTL, simulator and Gowin."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'compiler'))
from hardware_v2 import TARGET,TARGET_PATH

def generated():
    keys={'target_id':'TARGET_ID','numerics':'NUMERICS','descriptor_version':'DESC_VERSION','protocol_version':'PROTOCOL_VERSION','clock_hz':'CLOCK_HZ','baud':'BAUD','address_bits':'ADDR_BITS','memory_bytes':'MEM_BYTES','lanes':'LANES','descriptor_bytes':'DESC_BYTES','max_transfer':'MAX_TRANSFER','timeout_cycles':'RX_TIMEOUT','max_descriptors':'MAX_DESCRIPTORS','watchdog_cycles':'WATCHDOG'}
    return '// Generated from hardware/targets/tang_nano_20k_v2.json; do not edit.\npackage target_pkg;\n'+''.join(f'    localparam integer {v} = {TARGET[k]};\n' for k,v in keys.items())+''.join(f'    localparam logic [7:0] OP_{k} = {v};\n' for k,v in TARGET['opcodes'].items())+'endpackage\n'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--manifest',type=Path);a=p.parse_args()
    out=ROOT/'rtl/v2/target_pkg.sv';data=generated()
    if a.check:
        if not out.exists() or out.read_text()!=data:raise SystemExit('v2 target package is stale')
    else:out.write_text(data)
    if a.manifest:
        sources={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in TARGET['sources']}
        a.manifest.write_text(json.dumps(dict(target=TARGET,manifest_sha256=hashlib.sha256(TARGET_PATH.read_bytes()).hexdigest(),sources_sha256=sources),indent=2)+'\n')
