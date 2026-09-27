#!/usr/bin/env python3
"""Verify an exact compiler schedule with an already built engine executable."""

import argparse
import json
from pathlib import Path
import subprocess
import time

from run_screening import save_json, verify_files
from variants import ROOT, check_frozen, sha


def verify(candidate, fixtures_root, output):
    check_frozen()
    candidate = Path(candidate).resolve()
    fixtures_root = Path(fixtures_root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('preserve prior selected-schedule evidence')
    original = json.loads((candidate/'native/report.json').read_text())
    if original.get('status') != 'passed' or original.get('label') != candidate.name:
        raise ValueError('candidate native baseline missing')
    verify_files(ROOT, original['sources'])
    executable = candidate/'native/Vv2_tiled_host_bridge'
    if sha(executable) != original['executable_sha256']:
        raise ValueError('candidate native executable changed')
    manifest_path = fixtures_root/'fixtures.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'passed-replay' or len(manifest.get('fixtures', [])) != 6:
        raise ValueError('expected six exact grouped fixtures')
    output.mkdir(parents=True)
    report = dict(status='running', label=candidate.name, physical_board=False,
                  executable_sha256=sha(executable), sources=original['sources'],
                  fixture_manifest_sha256=sha(manifest_path), results=[])
    save_json(output/'report.json', report)
    for fixture in manifest['fixtures']:
        name = fixture.get('name', fixture.get('label'))
        directory = fixtures_root/'fixtures'/name
        verify_files(directory, fixture['files'])
        for seed in (0, 6063):
            result = output/f'{name}-s{seed}.json'
            started = time.monotonic()
            subprocess.run([str(executable), str(directory), str(seed), str(result)], check=True)
            row = json.loads(result.read_text())
            if row.get('status') != 'passed':
                raise ValueError(f'native mismatch: {name} seed {seed}')
            row.update(fixture=name, fixture_files=fixture['files'],
                       simulation_seconds=time.monotonic()-started)
            report['results'].append(row)
            save_json(output/'report.json', report)
    report['status'] = 'passed'
    save_json(output/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--fixtures-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.candidate, args.fixtures_root, args.output)
    print(json.dumps(dict(status=result['status'], completed=len(result['results'])),
                     sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
