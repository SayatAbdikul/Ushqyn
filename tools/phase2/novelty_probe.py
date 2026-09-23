#!/usr/bin/env python3
"""R05: exact service-bound counterexample and honest overlap/pivot decision.

This tests a stated bandwidth model, not a reproduction of DeFiNES or COSMA.
The hardware memory accepts one 64-bit request per cycle. Eight FC lanes need
one input and one weight word per reduction group when neither is cached.
"""
import json
from pathlib import Path
import argparse

def service_trace(groups,ports):
    pending=[(g,operand) for g in range(groups) for operand in ('activation','weight')]
    cycles=[]
    while pending:cycles.append(pending[:ports]);del pending[:ports]
    return cycles
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    examples=[]
    for count in (8,12,65,784):
        groups=(count+7)//8;ideal=service_trace(groups,2);legal=service_trace(groups,1)
        assert len(legal)==2*groups and len(ideal)==groups
        assert all(len(c)<=1 for c in legal)
        examples.append(dict(reduction_length=count,groups=groups,unconstrained_two_port_cycles=len(ideal),physical_one_port_cycles=len(legal),trace=legal))
    result=dict(scope='exact request-service bound, not measured layer latency',examples=examples,baseline_overlap='DeFiNES already models physical memory ports; a correctly configured baseline also enforces this bound',decision='reject port-awareness alone as novelty; keep engineering contribution provisional until a stronger mechanism beats tuned legal baselines',rtl_validation='test/phase2/test_engine.py randomizes ready and response latency, enforces held requests and verifies read/write counters; test_system uses the exact single-port synchronous SRAM hierarchy')
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n')
