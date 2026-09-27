#!/usr/bin/env python3
"""Combine the verified SIMD output stream and registered tail mask.

The engine is an immutable generated snapshot.  Every stage is boardless;
physical screening uses screen_engine_schedule.py after routing passes.
"""

import argparse
import json

import combined_scalar_uart as integrated
import combined_spec as engine_runner
import output_pipeline as stream
import tail_mask as mask
from run_screening import save_json
from variants import ROOT, check_frozen, sha


BASE = ROOT / 'work/phase6/experiments-v1/stream-mask-v1'
ENGINE = BASE / 'engine.sv'
PARENT = ROOT / 'work/phase6/experiments-v1/output-pipeline-v1/engine.sv'


def prepare():
    check_frozen()
    source = PARENT.read_text()
    if source.count(mask.DECL) != 1 or source.count('(col+k<count)') != 2:
        raise ValueError('stream candidate mask patch sites changed')
    result = source.replace(mask.DECL, mask.MASK_DECL, 1)
    result = result.replace('(col+k<count)?products[k]',
                            'mac_lane_mask[k]?products[k]', 1)
    result = result.replace('if (col+k<count) begin',
                            'if (mac_lane_mask[k]) begin', 1)
    BASE.mkdir(parents=True, exist_ok=True)
    if ENGINE.exists() and ENGINE.read_text() != result:
        raise ValueError('immutable stream-mask source changed')
    if not ENGINE.exists():
        ENGINE.write_text(result)
    identity = dict(label=BASE.name, parent=str(PARENT.relative_to(ROOT)),
                    parent_sha256=sha(PARENT), engine_sha256=sha(ENGINE),
                    parameters=json.loads((PARENT.parent / 'identity.json').read_text())['parameters'],
                    change='one-result-per-cycle output stream plus registered MAC tail validity',
                    physical_board=False)
    path = BASE / 'identity.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError('immutable stream-mask identity changed')
    if not path.exists():
        save_json(path, identity)
    engine_runner.BASE = BASE
    engine_runner.ENGINE = ENGINE
    engine_runner.runner.BASE = BASE
    engine_runner.runner.ENGINE = ENGINE
    integrated.BASE = BASE
    integrated.ENGINE = ENGINE
    stream.BASE = BASE
    stream.ENGINE = ENGINE
    mask.BASE = BASE
    mask.ENGINE = ENGINE
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'engine', 'stream-edges',
                                          'native', 'integration', 'route22', 'route24',
                                          'route27'))
    stage = parser.parse_args().stage
    identity = prepare()
    if stage == 'engine':
        engine_runner.runner.engine()
    elif stage == 'stream-edges':
        engine_runner.cocotb('stream-edges', 'test_output_pipeline')
    elif stage == 'native':
        stream.native()
    elif stage == 'integration':
        integrated.integration()
    elif stage == 'route22':
        integrated.route_22_5()
    elif stage == 'route24':
        mask.route_24()
    elif stage == 'route27':
        mask.route_27()
    check_frozen()
    print(json.dumps(dict(stage=stage, engine_sha256=identity['engine_sha256'])), flush=True)


if __name__ == '__main__':
    main()
