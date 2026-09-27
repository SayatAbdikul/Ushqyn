#!/usr/bin/env python3
"""Reproduce the bounded padded-stride decision from signed native evidence."""
import json
import math
from pathlib import Path

from variants import ROOT, check_frozen, sha
from run_screening import save_json, verify_files

BASE=ROOT/'work/phase6/padded-stride-v1'
REFERENCE=ROOT/'work/phase6/constant-sibling-v1'
OUTPUT=BASE/'decision.json'


def run():
    check_frozen()
    selected_path=REFERENCE/'native-all-exact/report.json'
    candidate_path=BASE/'native-padded-stride/report.json'
    selected=json.loads(selected_path.read_text())
    candidate=json.loads(candidate_path.read_text())
    manifest_path=BASE/'fixtures.json'
    manifest=json.loads(manifest_path.read_text())
    if (selected['status'],candidate['status'],manifest['status']) != (
            'passed','passed','passed-replay'):
        raise ValueError('selected/native/replay evidence missing')
    if candidate['fixture_manifest_sha256']!=sha(manifest_path):
        raise ValueError('fixture manifest identity mismatch')
    verify_files(ROOT,candidate['source_hashes'])
    for fixture in manifest['fixtures']:
        verify_files(BASE/'fixtures'/fixture['name'],fixture['files'])
    if len(candidate['results'])!=12 or any(row['status']!='passed' for row in candidate['results']):
        raise ValueError('all twelve exact native screens required')
    timing=candidate['timing']
    geometric_mean={state:math.sqrt(timing['kws'][state]['speedup']*
                                    timing['vww'][state]['speedup'])
                    for state in ('fixed','stalled')}
    candidate_commands={};selected_commands={}
    candidate_payload={};selected_payload={}
    for name in ('kws','vww'):
        current=BASE/'fixtures'/f'{name}-pinned-constant-sibling-padded-timed'
        previous=REFERENCE/'fixtures'/f'{name}-pinned-constant-sibling-timed'
        candidate_commands[name]=len((current/'commands.bin').read_bytes())//16
        selected_commands[name]=len((previous/'commands.bin').read_bytes())//16
        candidate_payload[name]=len((current/'payload.bin').read_bytes())
        selected_payload[name]=len((previous/'payload.bin').read_bytes())
    threshold=1.03
    decision='no-go-for-physical-expansion' if max(geometric_mean.values())<threshold else 'route-gate'
    report=dict(status='passed',decision=decision,physical_board=False,
        selected_schedule='constant-filter plus sibling-input retention; all-exact selected RTL',
        candidate='in-place channel-plane PACK plus padded pointwise input stride',
        candidate_policy='four KWS pointwise stages; first two VWW pointwise stages; deep VWW fallback',
        exact_native_cases=len(candidate['results']),
        fixed_and_stalled_native_cycles=timing,
        model_geometric_mean_speedup=geometric_mean,
        expansion_threshold_speedup=threshold,
        reason=('Both models improve, but the geometric mean improves less than the declared '
                '3% optimization-expansion trigger. Extra PACK commands and descriptor ABI/RTL '
                'complexity do not justify physical-board screening now.'),
        selected_command_count=selected_commands,candidate_command_count=candidate_commands,
        selected_payload_bytes=selected_payload,candidate_payload_bytes=candidate_payload,
        gate_analysis_sha256=sha(ROOT/'work/phase6/padded-stride-gate/analysis.json'),
        selected_native_report_sha256=sha(selected_path),
        candidate_native_report_sha256=sha(candidate_path),
        candidate_fixture_manifest_sha256=sha(manifest_path),
        limitation='Native RAM simulation is not routed timing, physical board latency, or power.')
    route_log=ROOT/'work/phase6/padded-stride-engine/route/build.log'
    if route_log.exists():
        report['route_attempt']=dict(status='tool-environment-failed-before-synthesis',
            build_log_sha256=sha(route_log),
            note='Gowin gw_sh exited after macOS pasteboard/hiservices errors; no fit or timing report.')
    save_json(OUTPUT,report)
    check_frozen()
    print(json.dumps({k:report[k] for k in ('status','decision',
        'model_geometric_mean_speedup')},sort_keys=True))
    return report


if __name__=='__main__':run()
