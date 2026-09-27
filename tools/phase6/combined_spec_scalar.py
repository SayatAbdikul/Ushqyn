#!/usr/bin/env python3
"""Immutable no-vector fallback for combined-spec-v1 at 22.5 MHz."""
import argparse
import json

import combined_spec as parent
from chain_resident import native as chain_native
from run_screening import save_json
from variants import ROOT, check_frozen, sha


BASE=ROOT/'work/phase6/experiments-v1/combined-spec-scalar-v1'
ENGINE=BASE/'engine.sv'
PARENT=ROOT/'work/phase6/experiments-v1/combined-spec-v1/engine.sv'
OLD='parameter bit VECTOR_LUT = 1'
NEW='parameter bit VECTOR_LUT = 0'


def prepare():
    check_frozen()
    source=PARENT.read_text()
    if source.count(OLD)!=1 or NEW in source:
        raise ValueError('immutable VECTOR_LUT patch site changed')
    source=source.replace(OLD,NEW,1)
    BASE.mkdir(parents=True,exist_ok=True)
    if ENGINE.exists() and ENGINE.read_text()!=source:
        raise ValueError('immutable scalar fallback differs; choose a new label')
    if not ENGINE.exists():ENGINE.write_text(source)
    parameters=json.loads((PARENT.parent/'identity.json').read_text())['parameters'].copy()
    parameters['VECTOR_LUT']=0
    record=dict(label=BASE.name,parent=str(PARENT.relative_to(ROOT)),
        parent_sha256=sha(PARENT),engine_sha256=sha(ENGINE),parameters=parameters,
        change='only VECTOR_LUT default 1 to 0; geometry, DW reuse, fast RQ and speculative prefetch preserved',
        physical_board=False)
    identity=BASE/'identity.json'
    if identity.exists():
        if json.loads(identity.read_text())!=record:
            raise ValueError('immutable scalar fallback identity changed')
    else:save_json(identity,record)
    parent.BASE=BASE;parent.ENGINE=ENGINE
    parent.runner.BASE=BASE;parent.runner.ENGINE=ENGINE
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','engine','edges','geometry-rejection',
        'speculative-safety','native','chain-native','route22'))
    args=parser.parse_args();identity=prepare()
    if args.stage=='prepare':result=identity
    elif args.stage=='engine':
        parent.runner.engine();result=json.loads((BASE/'engine/report.json').read_text())
    elif args.stage=='edges':result=parent.cocotb('edges','test_experiment_edges')
    elif args.stage=='geometry-rejection':result=parent.cocotb('geometry-rejection','test_geometry_rejection')
    elif args.stage=='speculative-safety':result=parent.cocotb('speculative-safety','test_speculative_prefetch')
    elif args.stage=='native':
        parent.runner.native();result=json.loads((BASE/'native/report.json').read_text())
    elif args.stage=='chain-native':
        chain_native(BASE.name)
        result=json.loads((BASE.parent/'chain'/f'native-{BASE.name}'/'report.json').read_text())
    else:result=parent.route_22_5()
    check_frozen()
    print(json.dumps(dict(stage=args.stage,status=result.get('status','prepared'),
        engine_sha256=identity['engine_sha256']),sort_keys=True),flush=True)


if __name__=='__main__':main()
