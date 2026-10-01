#!/usr/bin/env python3
"""Summarize source-bound engine observations; never turn occupancy into speedup."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'work/phase6/engine-profile-v1/final/report.json'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())

def run(report_path=DEFAULT):
    report_path=Path(report_path).resolve();report=read(report_path)
    if report['status']!='passed':raise ValueError('profile incomplete')
    for name,digest in report['identity']['input_files'].items():
        if sha(ROOT/name)!=digest:raise ValueError('profile input changed: '+name)
    summary=dict(status='passed',physical_board=False,profile_report=dict(path=str(report_path.relative_to(ROOT)),sha256=sha(report_path)),
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__)},models={},
        interpretation='Observed pipeline-state occupancy, not removable-stall bounds or predicted attainable speedups.',
        category_scope='Disjoint state classification, except copy and HALT assign their entire descriptor to the operator. Requantization includes fused activation and writeback.',
        event_scope={
            'req_blocked_nonprefetch':'engine mem_req && !mem_ready outside PW_MAC; additional request backpressure, not ordinary fetch stages',
            'read_pending':'read-wait states with !mem_rvalid; ordinary one-cycle SRAM delivery remains visible in wait-state occupancy',
            'scalar_hit_miss':'WIN_PREP in-bounds scalar gathers only; broadcast reuse, padding and pointwise are excluded',
            'pw_cross_reads':'accepted second-word reads from PW_X2_REQ or hit-plus-cross prefetch in PW_MAC',
            'pw_first_word_reads':'accepted first-word requests in PW_X_REQ or cache-missing PW_MAC prefetch',
            'pw_hit':'cache delivery in PW_X_REQ only; prefetch-hit decisions are separate and no global hit-rate is claimed',
            'q_output_pending':'Q_WAIT/Q_STREAM cycles without valid requantized/fused output, including pipeline fill',
            'issued_mac_lane_utilization':'sum of RTL issued useful lane counts divided by eight times arithmetic-issue states; not algorithmic FLOPs or overall hardware utilization'})
    for model,item in report['models'].items():
        rows=[]
        for entry in item['profiles']:
            p=ROOT/entry['path']
            if sha(p)!=entry['sha256']:raise ValueError('profile changed')
            r=read(p);states=r['state_cycles'];signals=r['signal_counts'];total=r['engine_cycles'];elapsed=r['sequencer_cycles']
            assert total==sum(states.values())==sum(r['categories'].values())==sum(f['engine_cycles'] for f in r['family_totals'].values())
            pw=r['family_totals'].get('pointwise_conv',{});ps=pw.get('states',{})
            cross=ps.get('PW_X2_REQ',0)+ps.get('PW_X2_WAIT',0)
            first=ps.get('PW_X_REQ',0)+ps.get('PW_X_WAIT',0)
            arithmetic=r['categories'].get('arithmetic_issue',0)
            rows.append(dict(stall_seed=r['stall_seed'],elapsed_cycles=elapsed,engine_cycles=total,
                descriptor_count=len(r['descriptors']),
                family_engine_fraction={k:v['engine_cycles']/total for k,v in r['family_totals'].items()},
                family_device_fraction={k:v['engine_cycles']/elapsed for k,v in r['family_totals'].items()},
                pointwise_second_word_state_cycles=cross,pointwise_second_word_engine_fraction=cross/total,
                pointwise_second_word_device_fraction=cross/elapsed,
                pointwise_first_word_state_cycles=first,pointwise_first_word_engine_fraction=first/total,
                pointwise_first_word_wait_cycles=ps.get('PW_X_WAIT',0),
                pointwise_first_word_wait_device_fraction=ps.get('PW_X_WAIT',0)/elapsed,
                pointwise_mac_issue_cycles=ps.get('PW_MAC',0),
                extra_request_backpressure_cycles=signals['req_blocked_nonprefetch'],
                extra_read_response_pending_cycles=signals['read_pending'],
                requant_input_backpressure_cycles=signals['q_input_blocked'],
                requant_output_writebackpressure_cycles=signals['q_write_blocked'],
                requant_pipeline_pending_cycles=signals['q_output_pending'],
                issued_mac_lane_utilization=signals['useful_macs']/(8*arithmetic),
                issued_mac_lanes_per_device_cycle=signals['useful_macs']/elapsed,
                scalar_cache_hit_events=signals['scalar_hit'],scalar_cache_miss_events=signals['scalar_miss'],
                scalar_cache_hit_fraction=signals['scalar_hit']/(signals['scalar_hit']+signals['scalar_miss']),
                source_profile_sha256=entry['sha256']))
        summary['models'][model]=rows
    output=report_path.parent/'summary.json';output.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    lines=['# Engine profile: unchanged strongest native programs','',
        'The instrumented harness reuses the original compiled RTL model and RAM stimulus. All four runs preserve exact final outputs, elapsed/engine/DMA/overlap counters, tensor-check counts and total simulated cycles. No board test was run.','',
        'Seed 0 state occupancy as a percentage of engine-busy cycles:','',
        '| Category | KWS | VWW |','| --- | ---: | ---: |']
    categories=sorted(set().union(*(set(report['models'][m]['profiles'][0]['categories']) for m in ('kws','vww'))))
    for category in categories:
        values=[100*report['models'][m]['profiles'][0]['category_engine_fraction'].get(category,0) for m in ('kws','vww')]
        lines.append(f'| {category} | {values[0]:.2f}% | {values[1]:.2f}% |')
    a=summary['models']['kws'][0];b=summary['models']['vww'][0]
    lines+=['',
        f"KWS: pointwise convolutions occupy {100*a['family_engine_fraction']['pointwise_conv']:.2f}% of engine time. Their second-word request/wait states occupy {a['pointwise_second_word_state_cycles']:,} cycles, {100*a['pointwise_second_word_device_fraction']:.2f}% of complete device latency. The four repeated pointwise descriptors have 25×5 spatial planes (125 bytes); that geometry explains why successive channels lose word alignment. This association identifies an experiment, not guaranteed removable cost.",
        '',f"VWW: pointwise convolutions occupy {100*b['family_engine_fraction']['pointwise_conv']:.2f}% of engine time. They spend {b['pointwise_first_word_wait_cycles']:,} cycles waiting for first-word delivery ({100*b['pointwise_first_word_wait_device_fraction']:.2f}% of device latency), versus {b['pointwise_mac_issue_cycles']:,} MAC-issue cycles. Several large aligned-plane descriptors have no pointwise cache delivery hits and one fetch-wait cycle per MAC issue.",
        '',f"Additional request backpressure is only {a['extra_request_backpressure_cycles']} KWS / {b['extra_request_backpressure_cycles']} VWW cycles at seed 0, with no extra read-response-pending or requantization write-backpressure cycles. Fetch-stage and output-pipeline occupancy still consume time; calling all of them arbitration stalls would be incorrect.",
        '', 'Requantization, activation lookup and writeback share states and can overlap. The profile measures their combined occupancy; it does not isolate each unit’s latency or power. Cache initialization, parameters, weight accesses and control are reported separately. The selected programs execute no COPY descriptors.',
        '', 'The per-descriptor files preserve raw 64-byte descriptor contents, command indices, PCs, source layer contracts where present, VWW component ranges, every engine state count, and scoped cache/handshake events. Seed 6063 repeats the same accounting with the original randomized external RAM model.',
        '', 'All percentages describe the fixed selected programs. A different kernel or schedule can change the denominator, overlap and work performed; no occupancy number here is a general lower bound or predicted speedup.',
        '', 'Reproduce with `python3 tools/phase6/profile_engine.py --output <fresh-work-directory>`, then `python3 tools/phase6/profile_engine_summary.py --report <fresh-work-directory>/report.json`. The first tool requires the pinned generated model archive/runtime objects and a compatible C++ compiler.']
    (report_path.parent/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'passed','summary':str(output.relative_to(ROOT)),'models':summary['models']},indent=2))
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,default=DEFAULT)
    args=parser.parse_args();run(args.report)
