#!/usr/bin/env python3
"""Seal the bounded final-constant folding decision against three-pair VWW."""
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'compiler'),str(ROOT/'tools/phase6')]
from channel_compaction import command_stats
from hardware_v2 import Descriptor
from variants import sha

BASE=ROOT/'work/phase6/novelty-constants-v1'
THREE=BASE/'threepair'
PRIOR=ROOT/'work/phase6/strip-fusion-pair7-v1/full'


def stage_geometry(schedule,layer):
    stages=[s for s in schedule['stages'] if s['layer']==layer]
    return [dict(reduction_count=Descriptor.decode(bytes.fromhex(s['descriptor_hex'])).count,
                 output_channels=Descriptor.decode(bytes.fromhex(s['descriptor_hex'])).output_c,
                 output_bytes=Descriptor.decode(bytes.fromhex(s['descriptor_hex'])).outputs)
            for s in stages]


def main():
    prep=BASE/'report.json';certpath=BASE/'certificate.json'
    native=BASE/'fused/report.json';threepath=THREE/'report.json'
    priorpath=PRIOR/'report.json'
    a=json.loads(prep.read_text());c=json.loads(certpath.read_text())
    b=json.loads(native.read_text());r=json.loads(threepath.read_text())
    prior=json.loads(priorpath.read_text())
    if (a['status']!='passed-replay' or b['status']!='passed' or
            r['status']!='passed-native' or prior['status']!='passed-native' or
            a['source_sha256']!=sha(ROOT/'tools/phase6/novelty_constants.py') or
            a['certificate_sha256']!=sha(certpath) or
            b['prepared_report_sha256']!=sha(prep) or
            r['source_report_sha256']!=sha(native) or
            r['certificate_sha256']!=sha(certpath) or
            r['baseline_report_sha256']!=sha(priorpath) or
            r['pooled_native_executable_sha256']!=prior['selected_native_executable_sha256']):
        raise ValueError('exact experiment or matched RTL source identity changed')
    for name,digest in r['source_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError(f'changed source: {name}')
    if (len(c['constant_codes'])!=234 or
            not all(row['pool_code']==c['classifier_input_zero_point']
                    for row in c['constant_codes']) or
            c['gemm_bias_delta']!=[0,0]):
        raise ValueError('classifier constant proof changed')
    matched={}
    for seed in (0,6063):
        row=r['matched_native_comparison'][str(seed)]
        if row['saved_cycles']<=0:
            raise ValueError('final constant fold did not save cycles')
        matched[str(seed)]=dict(**row,
            vww_throughput_speedup=row['baseline_cycles']/row['candidate_cycles'],
            dual_workload_geomean_speedup=math.sqrt(row['baseline_cycles']/row['candidate_cycles']))
    before=PRIOR/'fixtures/vww-pinned-compacted-strip3-7-11-timed'
    after=THREE/'fixtures/vww-pinned-known-compacted-strip3-7-11-timed'
    old_schedule=json.loads((before/'schedule.json').read_text())
    new_schedule=json.loads((after/'schedule.json').read_text())
    geometry={str(layer):dict(baseline=stage_geometry(old_schedule,layer),
                              candidate=stage_geometry(new_schedule,layer))
              for layer in (53,55,57)}
    result=dict(schema=1,status='no-go-board-expansion',physical_board=False,
        mechanism='exact final VWW constant-channel propagation through Relu, AveragePool, Reshape and Gemm',
        board_expansion=False,
        reason='incremental dual-workload geometric-mean native speedup is below the 1.03x expansion gate on both deterministic memory-stall seeds',
        gate_dual_workload_geomean_speedup=1.03,
        matched_pooled_native_executable_sha256=r['pooled_native_executable_sha256'],
        matched_native=matched,
        proof=dict(constant_channels=c['constant_channels'],
                   nonzero_classifier_weights_removed=c['nonzero_classifier_weights_removed'],
                   gemm_bias_delta=c['gemm_bias_delta'],
                   all_constant_pool_codes_equal_classifier_zero_point=True,
                   raw_partial_sum_upper_bound=c['raw_partial_sum_upper_bound'],
                   centered_partial_sum_upper_bound=c['centered_partial_sum_upper_bound'],
                   accumulator_limit=c['accumulator_limit']),
        schedule=dict(program_bytes_before=(before/'commands.bin').stat().st_size,
                      program_bytes_after=(after/'commands.bin').stat().st_size,
                      payload_bytes_before=(before/'payload.bin').stat().st_size,
                      payload_bytes_after=(after/'payload.bin').stat().st_size,
                      command_stats_before=command_stats((before/'commands.bin').read_bytes()),
                      command_stats_after=command_stats((after/'commands.bin').read_bytes()),
                      changed_geometry=geometry),
        exact_tests=dict(
            independent_reference='all 58 VWW layers matched for pinned and seeded stress before/after symbolic fold and channel compaction',
            command_replay='three prepared fixtures; all local strip candidates independently replayed for pinned and stress',
            native='three composed full-model fixtures, each passed stall seeds 0 and 6063; pinned/stress check fixtures verified 30 intermediate tensors per seed'),
        reproduction_commands=[
            '.venv/bin/python3 tools/phase6/novelty_constants.py prepare',
            '.venv/bin/python3 tools/phase6/novelty_constants.py native',
            '.venv/bin/python3 tools/phase6/novelty_constants_threepair.py',
            '.venv/bin/python3 tools/phase6/novelty_constants_decision.py'],
        evidence_sha256={str(path.relative_to(ROOT)):sha(path)
            for path in (prep,certpath,native,threepath,priorpath,
                         before/'commands.bin',before/'payload.bin',before/'schedule.json',
                         after/'commands.bin',after/'payload.bin',after/'schedule.json')},
        source_sha256={str(path.relative_to(ROOT)):sha(path) for path in (
            Path(__file__),ROOT/'tools/phase6/novelty_constants.py',
            ROOT/'tools/phase6/novelty_constants_threepair.py')})
    target=BASE/'decision.json'
    target.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],matched_native=result['matched_native'],
                          decision=str(target.relative_to(ROOT))),sort_keys=True))


if __name__=='__main__':main()
