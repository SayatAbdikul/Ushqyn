#!/usr/bin/env python3
"""Run full primary accuracy sets after a successful physical switch campaign.

Keep this separate from the already-running consecutive switch test. It waits
for that process to close the UART before starting the two accuracy campaigns.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'tools/phase5/run_dual_board.py'
RELEASE = ROOT / 'hardware/releases/phase5/dual_resident_750k.fs'
SWITCH = ROOT / 'work/phase5/dual-switch-full-v2.json'
LOG = ROOT / 'work/phase5/campaign-sequence.log'
sys.path.insert(0, str(ROOT / 'tools/phase5'))
from audit import inspect_dual_release, inspect_full_campaign
from schedule import create_plan


def report(path):
    return json.loads(path.read_text()) if path.is_file() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--switch-report', type=Path, default=SWITCH)
    parser.add_argument('--bitstream', type=Path, default=RELEASE)
    parser.add_argument('--stall-minutes', type=int, default=15)
    args = parser.parse_args()
    if args.stall_minutes < 1:
        parser.error('--stall-minutes must be positive')
    release = inspect_dual_release(ROOT)
    plan = create_plan()
    if release is None:
        raise ValueError('dual-resident release evidence is absent')
    print('waiting for strict 10,000-job switch report', flush=True)
    while True:
        state = report(args.switch_report)
        if state is not None and state.get('status') == 'passed':
            if (state['jobs_completed'] != 10000
                    or state['output_mismatches'] != 0
                    or state['bitstream_sha256'] != hashlib.sha256(
                        args.bitstream.read_bytes()).hexdigest()):
                raise ValueError('switch campaign does not match release')
            if not inspect_full_campaign(ROOT, 'switch', release, plan):
                raise ValueError('switch campaign records fail independent audit')
            break
        if state is not None and state.get('status') == 'failed':
            raise RuntimeError(f'switch campaign failed: {state.get("failure")}')
        rows = args.switch_report.with_suffix('.jsonl')
        last = rows.stat().st_mtime if rows.exists() else args.switch_report.stat().st_mtime \
            if args.switch_report.exists() else time.time()
        if time.time() - last > args.stall_minutes * 60:
            raise TimeoutError('switch campaign made no progress')
        time.sleep(30)
    for campaign in ('kws-accuracy', 'vww-accuracy'):
        output = ROOT / f'work/phase5/dual-{campaign}-full.json'
        old = report(output)
        if old is not None:
            if not inspect_full_campaign(ROOT, campaign, release, plan):
                raise FileExistsError(f'non-passing campaign report already exists: {output}')
            print(f'{campaign} already passed; skipping', flush=True)
            continue
        command = [sys.executable, str(SCRIPT), '--campaign', campaign,
                   '--bitstream', str(args.bitstream), '--report', str(output)]
        print(f'starting {campaign}', flush=True)
        with LOG.open('a') as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        result = report(output)
        if (result is None or result.get('status') != 'passed'
                or result.get('output_mismatches') != 0
                or not inspect_full_campaign(ROOT, campaign, release, plan)):
            raise ValueError(f'{campaign} did not pass')
        print(f'{campaign} passed: {result["jobs_completed"]} jobs', flush=True)
    print('both complete primary accuracy sets passed', flush=True)


if __name__ == '__main__':
    main()
