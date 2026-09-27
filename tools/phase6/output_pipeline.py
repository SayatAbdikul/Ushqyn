#!/usr/bin/env python3
"""Isolated exact SIMD output stream; no command programs the FPGA."""
import argparse
import json
import subprocess
import time

import combined_scalar_uart as integrated
import combined_spec as engine_runner
from run_screening import save_json, verify_files
from variants import ROOT, check_frozen, sha

BASE = ROOT/'work/phase6/experiments-v1/output-pipeline-v1'
ENGINE = BASE/'engine.sv'
SOURCE = ROOT/'rtl/phase6/output_pipeline_engine.sv'
PARENT = ROOT/'work/phase6/experiments-v1/combined-spec-scalar-uart-v1/engine.sv'


def prepare():
    check_frozen()
    BASE.mkdir(parents=True, exist_ok=True)
    source = SOURCE.read_text()
    if ENGINE.exists() and ENGINE.read_text() != source:
        raise ValueError('immutable output stream snapshot changed; choose a new label')
    if not ENGINE.exists():
        ENGINE.write_text(source)
    identity = dict(label=BASE.name, parent=str(PARENT.relative_to(ROOT)),
        parent_sha256=sha(PARENT), engine_sha256=sha(ENGINE),
        source=str(SOURCE.relative_to(ROOT)), source_sha256=sha(SOURCE),
        parameters=json.loads((PARENT.parent/'identity.json').read_text())['parameters'],
        change='one result per cycle SIMD requantizer drain with elastic backpressure',
        physical_board=False)
    path = BASE/'identity.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError('immutable output stream identity changed')
    if not path.exists():
        save_json(path, identity)
    engine_runner.BASE = BASE
    engine_runner.ENGINE = ENGINE
    engine_runner.runner.BASE = BASE
    engine_runner.runner.ENGINE = ENGINE
    integrated.BASE = BASE
    integrated.ENGINE = ENGINE
    integrated.PARENT = PARENT
    return identity


def native():
    build = BASE/'native'
    build.mkdir(exist_ok=True)
    sources = engine_runner.runner.sources()
    harness = ROOT/'test/phase6/native.cpp'
    with (build/'build.log').open('w') as log:
        subprocess.run(['verilator', '--cc', '--exe', '--build', '-j', '2',
            '--public-flat-rw', '-Wno-fatal', '--top-module', 'v2_tiled_host_bridge',
            '--Mdir', str(build), *map(str, sources), str(harness)], cwd=ROOT,
            stdout=log, stderr=subprocess.STDOUT, check=True)
    exe = build/'Vv2_tiled_host_bridge'
    fixtures_root = ROOT/'work/phase6/constant-sibling-v1'
    manifest = json.loads((fixtures_root/'fixtures.json').read_text())
    assert manifest['status'] == 'passed-replay'
    report = dict(status='running', physical_board=False, label=BASE.name,
        executable_sha256=sha(exe), sources={str(p.relative_to(ROOT)): sha(p)
            for p in sources+[harness]},
        fixture_manifest_sha256=sha(fixtures_root/'fixtures.json'), results=[])
    save_json(build/'report.json', report)
    for fixture in manifest['fixtures']:
        directory = fixtures_root/'fixtures'/fixture['name']
        verify_files(directory, fixture['files'])
        for seed in (0, 6063):
            output = build/f'{fixture["name"]}-s{seed}.json'
            started = time.monotonic()
            subprocess.run([str(exe), str(directory), str(seed), str(output)], check=True)
            row = json.loads(output.read_text())
            assert row['status'] == 'passed'
            row.update(fixture=fixture['name'], fixture_files=fixture['files'],
                       simulation_seconds=time.monotonic()-started)
            report['results'].append(row)
            save_json(build/'report.json', report)
    assert len(report['results']) == 12
    report['status'] = 'passed'
    save_json(build/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'engine', 'stream-edges',
        'native', 'integration', 'route22'))
    args = parser.parse_args()
    identity = prepare()
    if args.stage == 'prepare':
        result = identity
    elif args.stage == 'engine':
        engine_runner.runner.engine()
        result = json.loads((BASE/'engine/report.json').read_text())
    elif args.stage == 'stream-edges':
        result = engine_runner.cocotb('stream-edges', 'test_output_pipeline')
    elif args.stage == 'native':
        result = native()
    elif args.stage == 'integration':
        result = integrated.integration()
    else:
        route_report = BASE/'route/report.json'
        if route_report.exists():
            old = json.loads(route_report.read_text())
            if old['status'] == 'failed-build' and 'bitstream' not in old:
                # Preserve a tool-startup failure before retrying outside the
                # macOS sandbox; successful route reports remain immutable.
                attempt = time.time_ns()
                route_report.rename(BASE/f'route/failed-build-{attempt}.json')
                log = BASE/'route/build.log'
                if log.exists():
                    log.rename(BASE/f'route/failed-build-{attempt}.log')
        result = integrated.route_22_5()
    check_frozen()
    print(json.dumps(dict(stage=args.stage, status=result.get('status', 'prepared'),
        engine_sha256=identity['engine_sha256']), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
