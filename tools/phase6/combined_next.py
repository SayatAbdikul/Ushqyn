#!/usr/bin/env python3
"""Immutable combined exact engine screen; board programming is separate.

The candidate enables the selected all-exact features plus DW word reuse,
vector activation writes and the exact fast requantizer. The geometry area
patch is applied mechanically, with its descriptor guards verified first.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import run_writeback as runner
from chain_resident import native as chain_native
from geometry_variant import apply_geometry_narrowing
from run_screening import save_json
from variants import ROOT, check_frozen, sha

BASE = ROOT/'work/phase6/experiments-v1/combined-next-v1'
ENGINE = BASE/'engine.sv'
TEMPLATE = ROOT/'rtl/phase6/experimental_engine.sv'
PARAMETERS = dict(PW_REGISTERED_WEIGHT=1, PW_FETCH_AHEAD=1,
    SCALAR_WORD_REUSE=1, PW_WORD_CACHE=1, BALANCED_SUM=1,
    ACC_WORK_BITS=33, SCALAR_LUT=1, PARAM_FAST=0, SPATIAL_DW=1,
    VALID_IN_BRAM=1, PW_HIT_BYPASS=0, DW_WORD_REUSE=1,
    RQ_FAST=1, VECTOR_LUT=1)


def prepare():
    check_frozen()
    BASE.mkdir(parents=True, exist_ok=True)
    identity = BASE/'identity.json'
    if identity.exists():
        record = json.loads(identity.read_text())
        if (record['label'] != BASE.name or record['parameters'] != PARAMETERS
                or record['engine_sha256'] != sha(ENGINE)):
            raise ValueError('immutable combined candidate identity changed')
        return record
    text = TEMPLATE.read_text()
    for key, value in PARAMETERS.items():
        kind, default = ('integer', 36) if key == 'ACC_WORK_BITS' else ('bit', 0)
        old = f'parameter {kind} {key} = {default}'
        if text.count(old) != 1:
            raise ValueError(f'missing or changed parameter default: {key}')
        text = text.replace(old, f'parameter {kind} {key} = {value}', 1)
    text = apply_geometry_narrowing(text)
    if ENGINE.exists():
        if ENGINE.read_text() != text:
            raise ValueError('existing engine differs; choose a new label')
    else:
        ENGINE.write_text(text)
    record = dict(label=BASE.name, parameters=PARAMETERS,
        template=str(TEMPLATE.relative_to(ROOT)), template_sha256=sha(TEMPLATE),
        geometry_patcher_sha256=sha(ROOT/'tools/phase6/geometry_variant.py'),
        generator_sha256=sha(Path(__file__)), engine_sha256=sha(ENGINE),
        scope='new immutable combined RTL image; no board programming')
    save_json(identity, record)
    return record


def _configure_runner():
    runner.BASE = BASE
    runner.ENGINE = ENGINE


def _cocotb(stage, top, module):
    from cocotb.runner import get_runner
    build = BASE/stage
    simulation = get_runner('verilator')
    simulation.build(verilog_sources=runner.sources(True), hdl_toplevel=top,
        build_dir=build, build_args=['--timing', '-Wno-fatal'],
        timescale=('1ns', '1ps'))
    paths=[str(ROOT/p) for p in ('compiler', 'test/phase6')]+sys.path
    sys.path[:0]=paths
    simulation.test(hdl_toplevel=top, test_module=module,
        test_dir=ROOT/'test/phase6', build_dir=build,
        results_xml=str(build/'results.xml'),
        extra_env={'PYTHONPATH': os.pathsep.join(paths)})
    cases=ET.parse(build/'results.xml').findall('.//testcase')
    if len(cases)!=1 or any(c.find('failure') is not None or c.find('error') is not None for c in cases):
        raise ValueError(f'{stage} Cocotb regression failed')
    source={str(p.relative_to(ROOT)):sha(p) for p in runner.sources(True)+[ROOT/f'test/phase6/{module}.py']}
    result=dict(status='passed', physical_board=False, tests=len(cases),
        sources=source, results_sha256=sha(build/'results.xml'))
    save_json(build/'report.json', result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare','engine','edges','vector-edges','requant-proof',
                                         'native','chain-native','route'))
    args=parser.parse_args()
    record=prepare()
    _configure_runner()
    if args.stage=='prepare': result=record
    elif args.stage=='engine':
        runner.engine();result=json.loads((BASE/'engine/report.json').read_text())
    elif args.stage in ('edges','vector-edges'):result=_cocotb(args.stage,'v2_engine','test_vector_lut_edges')
    elif args.stage=='requant-proof':result=_cocotb('requant-proof','v2_requantizer_fast','test_fast_requant')
    elif args.stage=='native':
        runner.native();result=json.loads((BASE/'native/report.json').read_text())
    elif args.stage=='chain-native':
        chain_native(BASE.name)
        result=json.loads((BASE.parent/'chain'/f'native-{BASE.name}'/'report.json').read_text())
    else:
        runner.route();result=json.loads((BASE/'route/report.json').read_text())
    check_frozen()
    print(json.dumps({'stage':args.stage,'status':result.get('status','prepared'),
                      'engine_sha256':record['engine_sha256']},sort_keys=True),flush=True)


if __name__=='__main__':main()
