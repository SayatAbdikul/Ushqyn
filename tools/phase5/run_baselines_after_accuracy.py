#!/usr/bin/env python3
"""Queue reproducible physical B1/B2 probes after both accuracy campaigns."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / 'hardware/releases/phase5/dual_resident_750k.fs'
RUNNER = ROOT / 'tools/phase5/run_baseline_probe.py'
LOG = ROOT / 'work/phase5/baseline-probe-sequence.log'
sys.path.insert(0, str(ROOT / 'tools/phase5'))
from audit import inspect_dual_release, inspect_full_campaign
from schedule import create_plan


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=64)
    parser.add_argument('--stall-minutes', type=int, default=15)
    args = parser.parse_args()
    if args.jobs < 1 or args.stall_minutes < 1:
        parser.error('positive jobs and stall interval required')
    release = inspect_dual_release(ROOT)
    if release is None:
        raise ValueError('dual-resident release evidence is absent')
    plan = create_plan()
    reports = [ROOT / f'work/phase5/dual-{name}-accuracy-full.json'
               for name in ('kws', 'vww')]
    switch_report = ROOT / 'work/phase5/dual-switch-full-v2.json'
    print('waiting for both complete accuracy campaigns', flush=True)
    while True:
        states = [read(path) for path in reports]
        if any(state and state.get('status') == 'failed' for state in states):
            raise RuntimeError('a primary accuracy campaign failed')
        switch = read(switch_report)
        if switch and switch.get('status') == 'failed':
            raise RuntimeError('switch campaign failed before accuracy')
        if all(state and state.get('status') == 'passed' for state in states):
            if not all(inspect_full_campaign(ROOT, f'{model}-accuracy', release, plan)
                       for model in ('kws', 'vww')):
                raise ValueError('complete accuracy records fail independent audit')
            break
        active = next((path for path, state in zip(reports, states)
                       if state and state.get('status') == 'running'), None)
        if active:
            rows = active.with_suffix('.jsonl')
            latest = max(active.stat().st_mtime,
                         rows.stat().st_mtime if rows.is_file() else 0)
            if time.time() - latest > args.stall_minutes * 60:
                raise TimeoutError(f'accuracy campaign stalled: {active}')
        elif switch and switch.get('status') == 'passed':
            latest = max([switch_report.stat().st_mtime] +
                         [path.stat().st_mtime for path in reports if path.is_file()])
            if time.time() - latest > args.stall_minutes * 60:
                raise TimeoutError('accuracy sequence did not start its next campaign')
        time.sleep(30)
    summaries = {}
    for model in ('kws', 'vww'):
        summaries[model] = {}
        for policy in ('B1', 'B2'):
            output = ROOT / f'work/phase5/baseline-probe-{model}-{policy.lower()}.json'
            old = read(output)
            if old is None:
                command = [sys.executable, str(RUNNER), '--model', model,
                           '--policy', policy, '--bitstream', str(RELEASE),
                           '--report', str(output), '--jobs', str(args.jobs)]
                print(f'starting {model} {policy} physical probe', flush=True)
                with LOG.open('a') as stream:
                    subprocess.run(command, cwd=ROOT, stdout=stream,
                                   stderr=subprocess.STDOUT, check=True)
                old = read(output)
            if (old is None or old.get('status') != 'passed-probe'
                    or old.get('jobs_completed') != args.jobs
                    or old.get('output_mismatches') != 0):
                raise ValueError(f'{model} {policy} physical probe did not pass')
            summaries[model][policy] = {
                'jobs': old['jobs_completed'],
                'median_cycles': old['median_cycles'],
                'median_wall_seconds': old['median_wall_seconds'],
                'correct': old['correct'],
                'records_sha256': old['records_sha256'],
            }
            print(f'{model} {policy} passed: {old["median_cycles"]} median cycles',
                  flush=True)
    summary = {'schema': 1, 'scope': 'fixed-seed probes; not complete tuned B03',
               'bitstream_sha256': release['bitstream_sha256'], 'results': summaries}
    (ROOT / 'work/phase5/baseline-probes.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
