#!/usr/bin/env python3
"""Versioned, short optimization experiments; no board access in these stages."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import run_writeback as runner
from variants import ROOT, check_frozen, sha
from run_screening import save_json

CONFIGS = {
    'control': {},
    'weight': {'PW_REGISTERED_WEIGHT': 1},
    'prefetch': {'PW_REGISTERED_WEIGHT': 1, 'PW_FETCH_AHEAD': 1},
    'scalar': {'PW_REGISTERED_WEIGHT': 1, 'PW_FETCH_AHEAD': 1, 'SCALAR_WORD_REUSE': 1},
    'cache': {'PW_REGISTERED_WEIGHT': 1, 'PW_FETCH_AHEAD': 1, 'SCALAR_WORD_REUSE': 1, 'PW_WORD_CACHE': 1},
    'compact': {'PW_REGISTERED_WEIGHT': 1, 'PW_FETCH_AHEAD': 1, 'SCALAR_WORD_REUSE': 1,
                'BALANCED_SUM': 1, 'ACC_WORK_BITS': 33},
    'compact-cache': {'PW_REGISTERED_WEIGHT': 1, 'PW_FETCH_AHEAD': 1, 'SCALAR_WORD_REUSE': 1,
                     'PW_WORD_CACHE': 1, 'BALANCED_SUM': 1, 'ACC_WORK_BITS': 33},
    'lut': {'PW_REGISTERED_WEIGHT': 1, 'PW_FETCH_AHEAD': 1, 'SCALAR_WORD_REUSE': 1, 'SCALAR_LUT': 1},
    'compact-lut': {'PW_REGISTERED_WEIGHT': 1, 'PW_FETCH_AHEAD': 1, 'SCALAR_WORD_REUSE': 1,
                    'BALANCED_SUM': 1, 'ACC_WORK_BITS': 33, 'SCALAR_LUT': 1},
    'params': {'PW_REGISTERED_WEIGHT': 1, 'PW_FETCH_AHEAD': 1, 'SCALAR_WORD_REUSE': 1, 'PARAM_FAST': 1},
    'combined': {'PW_REGISTERED_WEIGHT': 1, 'PW_FETCH_AHEAD': 1, 'SCALAR_WORD_REUSE': 1,
                 'PW_WORD_CACHE': 1, 'BALANCED_SUM': 1, 'ACC_WORK_BITS': 33, 'SCALAR_LUT': 1},
    'depthwise': {'PW_REGISTERED_WEIGHT': 1, 'PW_FETCH_AHEAD': 1, 'SCALAR_WORD_REUSE': 1,
                 'BALANCED_SUM': 1, 'ACC_WORK_BITS': 33, 'SPATIAL_DW': 1},
    'combined-valid': {'PW_REGISTERED_WEIGHT': 1, 'PW_FETCH_AHEAD': 1, 'SCALAR_WORD_REUSE': 1,
                 'PW_WORD_CACHE': 1, 'BALANCED_SUM': 1, 'ACC_WORK_BITS': 33, 'SCALAR_LUT': 1, 'VALID_IN_BRAM': 1},
    'all-exact': {'PW_REGISTERED_WEIGHT': 1, 'PW_FETCH_AHEAD': 1, 'SCALAR_WORD_REUSE': 1,
                 'PW_WORD_CACHE': 1, 'BALANCED_SUM': 1, 'ACC_WORK_BITS': 33, 'SCALAR_LUT': 1,
                 'VALID_IN_BRAM': 1, 'SPATIAL_DW': 1},
}
CONFIGS.update({
    'pw-hit-v2': dict(CONFIGS['all-exact'], PW_HIT_BYPASS=1),
    'dw-reuse-v2': dict(CONFIGS['all-exact'], DW_WORD_REUSE=1),
    'fetch-reuse-v2': dict(CONFIGS['all-exact'], PW_HIT_BYPASS=1, DW_WORD_REUSE=1),
    'rq-fast-v2': dict(CONFIGS['all-exact'], RQ_FAST=1),
    'combined-v2': dict(CONFIGS['all-exact'], PW_HIT_BYPASS=1, DW_WORD_REUSE=1, RQ_FAST=1),
    'pw-hit-v3': dict(CONFIGS['all-exact'], PW_HIT_BYPASS=1),
    'dw-reuse-v3': dict(CONFIGS['all-exact'], DW_WORD_REUSE=1),
    'rq-fast-v3': dict(CONFIGS['all-exact'], RQ_FAST=1),
    'combined-v3': dict(CONFIGS['all-exact'], PW_HIT_BYPASS=1, DW_WORD_REUSE=1, RQ_FAST=1),
    'vector-lut-v1': dict(CONFIGS['all-exact'], VECTOR_LUT=1),
})
BASE = ROOT / 'work/phase6/experiments-v1'


def prepare(name, label=None):
    check_frozen()
    directory = BASE / (label or name)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return prepare_locked(name, label, directory)


def prepare_locked(name, label, directory):
    record = directory / 'identity.json'
    engine = directory / 'engine.sv'
    if record.exists():
        previous = json.loads(record.read_text())
        if (previous['name'] != name or previous['parameters'] != CONFIGS[name]
                or previous['engine_sha256'] != sha(engine)):
            raise ValueError('candidate identity changed; use a new --label')
        return directory, engine
    source = ROOT / 'rtl/phase6/experimental_engine.sv'
    text = source.read_text()
    for key, value in CONFIGS[name].items():
        kind, default = ('integer', 36) if key == 'ACC_WORK_BITS' else ('bit', 0)
        old = f'parameter {kind} {key} = {default}'
        if text.count(old) != 1: raise ValueError('missing parameter: ' + key)
        text = text.replace(old, f'parameter {kind} {key} = {value}')
    directory.mkdir(parents=True, exist_ok=True)
    if engine.exists() and engine.read_text() != text:
        raise ValueError('candidate source changed; select a new --label to preserve evidence')
    if not engine.exists(): engine.write_text(text)
    identity = {'name': name, 'label': label or name, 'parameters': CONFIGS[name],
                'engine_sha256': sha(engine), 'template_sha256': sha(source),
                'template': str(source.relative_to(ROOT)), 'runner_sha256': sha(Path(__file__))}
    if record.exists():
        previous = json.loads(record.read_text())
        if previous['engine_sha256'] != identity['engine_sha256']:
            raise ValueError('candidate identity changed')
    else: save_json(record, identity)
    return directory, engine


def edges(directory, engine):
    from cocotb.runner import get_runner
    build = directory / 'edges'
    simulation = get_runner('verilator')
    simulation.build(verilog_sources=runner.sources(True), hdl_toplevel='v2_engine',
        build_dir=build, build_args=['--timing', '-Wno-fatal'], timescale=('1ns', '1ps'))
    paths = [str(ROOT / p) for p in ('compiler', 'test/phase6')] + sys.path
    sys.path[:] = paths
    edge_module = 'test_vector_lut_edges' if CONFIGS[json.loads((directory / 'identity.json').read_text())['name']].get('VECTOR_LUT') else 'test_experiment_edges'
    simulation.test(hdl_toplevel='v2_engine', test_module=edge_module,
        test_dir=ROOT / 'test/phase6', build_dir=build, results_xml=str(build / 'results.xml'),
        extra_env={'PYTHONPATH': os.pathsep.join(paths)})
    cases = ET.parse(build / 'results.xml').findall('.//testcase')
    if len(cases) != 1 or any(c.find('failure') is not None or c.find('error') is not None for c in cases):
        raise ValueError('experimental edge regression failed')
    save_json(build / 'report.json', {'status': 'passed', 'tests': len(cases),
        'sources': {str(p.relative_to(ROOT)): sha(p) for p in runner.sources(True) +
                    [ROOT / f'test/phase6/{edge_module}.py']},
        'results_sha256': sha(build / 'results.xml')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'engine', 'native', 'route', 'edges'))
    parser.add_argument('variant', choices=CONFIGS)
    parser.add_argument('--label')
    args = parser.parse_args()
    directory, engine = prepare(args.variant, args.label)
    runner.BASE = directory; runner.ENGINE = engine
    if args.stage == 'edges': edges(directory, engine)
    elif args.stage != 'prepare': getattr(runner, args.stage)()
    check_frozen()
