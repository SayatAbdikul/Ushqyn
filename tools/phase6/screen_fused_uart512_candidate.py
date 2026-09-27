#!/usr/bin/env python3
"""Source-verified fused/512-byte UART preflight and optional short board screen.

The existing audited512 screening logic is reused with explicit isolated
engine, fixture and evidence roots. --run is the only board access path.
"""
import argparse
import json
from pathlib import Path

import screen_uart512_candidate as screen
from combined_fused_uart512 import BASE, CORE, preflight
from variants import ROOT, sha


def prepare(route_report):
    preflight()
    screen.BASE=BASE;screen.CORE=CORE;screen.FIXTURES=CORE
    plan=screen.prepare(route_report)
    plan['image']=BASE.name
    plan['dependencies'].update({str(path.relative_to(ROOT)):sha(path) for path in (
        Path(__file__),ROOT/'tools/phase6/combined_fused_uart512.py')})
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route-report',required=True,type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--port',default='/dev/cu.usbserial-20250303171')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args();plan=prepare(args.route_report)
    if args.run:
        if args.output is None:parser.error('--run requires a new output directory')
        screen.run(plan,args.output.resolve(),args.port)
    else:
        print(json.dumps(dict(status='preflight-passed',image=plan['image'],
            planned=plan['planned'],bitstream_sha256=plan['bitstream_sha256'])),flush=True)
