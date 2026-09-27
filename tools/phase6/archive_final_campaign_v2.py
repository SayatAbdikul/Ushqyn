#!/usr/bin/env python3
"""Archive the small Phase 6 follow-up evidence without copying bitstreams.

Running/failed physical screens are recorded as pending evidence. Device
medians and speedups appear only after signed 10/10 records and the matching
route are complete. The archive is deterministic and safe to regenerate.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/research/evidence/phase6/final-campaign-v2'
MAX_ARCHIVE_BYTES=1_000_000

GROUPS={
    'campaign_ledger': [
        'tools/phase6/archive_final_campaign_v2.py',
        'docs/research/PHASE_6_FINAL_CAMPAIGN_V2.md'],
    'channel_grouping': [
        'tools/phase6/followup_graph.py',
        'docs/research/PHASE_6_CHANNEL_GROUPING.md',
        'work/phase6/followup_graph/fixtures.json',
        'work/phase6/followup_graph/native-combined-spec-scalar-v1/report.json',
        'work/phase6/followup_graph_fold/fixtures.json',
        'work/phase6/followup_graph_fold/native-combined-spec-scalar-v1/report.json'],
    'tail_prefetch': [
        'tools/phase6/prefetch_tail.py',
        'tools/phase6/prefetch_grouped.py',
        'docs/research/PHASE_6_TAIL_PREFETCH.md',
        'work/phase6/followup_graph_tail_prefetch/fixtures.json',
        'work/phase6/followup_graph_tail_prefetch/native-combined-spec-scalar-v1/report.json',
        'work/phase6/followup_graph_tail_prefetch/native-stream-mask-v1/report.json'],
    'output_pipeline': [
        'tools/phase6/output_pipeline.py',
        'work/phase6/experiments-v1/output-pipeline-v1/engine.sv',
        'work/phase6/experiments-v1/output-pipeline-v1/identity.json',
        'work/phase6/experiments-v1/output-pipeline-v1/engine/report.json',
        'work/phase6/experiments-v1/output-pipeline-v1/native/report.json',
        'work/phase6/experiments-v1/output-pipeline-v1/integration/report.json',
        'work/phase6/experiments-v1/output-pipeline-v1/route/report.json'],
    'stream_mask': [
        'tools/phase6/stream_mask.py',
        'work/phase6/experiments-v1/stream-mask-v1/engine.sv',
        'work/phase6/experiments-v1/stream-mask-v1/identity.json',
        'work/phase6/experiments-v1/stream-mask-v1/engine/report.json',
        'work/phase6/experiments-v1/stream-mask-v1/native/report.json',
        'work/phase6/experiments-v1/stream-mask-v1/integration/report.json',
        'work/phase6/experiments-v1/stream-mask-v1/route24/report.json',
        'work/phase6/experiments-v1/stream-mask-v1/route27/report.json'],
    'tail_mask': [
        'tools/phase6/tail_mask.py',
        'work/phase6/experiments-v2/tail-mask-v2/engine.sv',
        'work/phase6/experiments-v2/tail-mask-v2/identity.json',
        'work/phase6/experiments-v2/tail-mask-v2/native/report.json',
        'work/phase6/experiments-v2/tail-mask-v2/integration/report.json',
        'work/phase6/experiments-v2/tail-mask-v2/route24/report.json'],
    'fused_activation': [
        'tools/phase6/fused_activation.py',
        'work/phase6/experiments-v1/fused-activation-v1/engine.sv',
        'work/phase6/experiments-v1/fused-activation-v1/identity.json',
        'work/phase6/experiments-v1/fused-activation-v1/fixtures.json',
        'work/phase6/experiments-v1/fused-activation-v1/engine/report.json',
        'work/phase6/experiments-v1/fused-activation-v1/edges/report.json',
        'work/phase6/experiments-v1/fused-activation-v1/integration/report.json',
        'work/phase6/experiments-v1/fused-activation-v1/native/report.json',
        'work/phase6/experiments-v1/fused-activation-v1/comparison.json',
        'work/phase6/experiments-v1/fused-activation-v1/route24-inputs.json',
        'work/phase6/experiments-v1/output-fusion-prototype-v1/compiler-report.json',
        'work/phase6/experiments-v1/output-fusion-prototype-v1/rtl/report.json'],
    'fused_clock_feasibility': [
        'tools/phase6/fused_clock_feasibility.py',
        'docs/research/PHASE_6_FUSED_CLOCK_FEASIBILITY.md',
        'work/phase6/experiments-v1/fused-clock-25p5-feasibility-v1/report.json'],
    'producer_native_padding_no_go': [
        'tools/phase6/profile_padded_cost.py',
        'docs/research/PHASE_6_PRODUCER_NATIVE_PADDING_COST.md',
        'docs/research/evidence/phase6/producer-native-padding-cost-v1.json'],
    'early_exit_no_go': [
        'tools/phase6/early_exit_gate.py',
        'docs/research/PHASE_6_EARLY_EXIT_GATE.md',
        'work/phase6/early-exit-gate-v1/report.json'],
    'uart512': [
        'tools/phase6/uart_burst_512.py',
        'tools/phase6/combined_stream_uart512.py',
        'rtl/phase6/uart_burst_512_command.sv',
        'rtl/phase6/uart_burst_512_bridge.sv',
        'hardware/phase6/uart_burst_512_tiled_host.sv',
        'hardware/phase6/build_uart_burst_512.tcl',
        'test/phase6/test_uart_burst_512.py',
        'test/phase6/test_uart_burst_512_bridge.py',
        'test/phase6/test_uart_burst_512_host.py',
        'docs/research/PHASE_6_UART_512_SCREEN.md',
        'work/phase6/uart-burst-512/sim/results.xml',
        'work/phase6/uart-burst-512/integration/results.xml',
        'work/phase6/experiments-v1/stream-mask-uart512-v1/engine.sv',
        'work/phase6/experiments-v1/stream-mask-uart512-v1/identity.json',
        'work/phase6/experiments-v1/stream-mask-uart512-v1/integration/report.json',
        'work/phase6/experiments-v1/stream-mask-uart512-v1/native-grouped/report.json',
        'work/phase6/experiments-v1/stream-mask-uart512-v1/native-grouped-tail-prefetch/report.json',
        'work/phase6/experiments-v1/stream-mask-uart512-v1/route-inputs-24p0.json',
        'work/phase6/experiments-v1/stream-mask-uart512-v1/route-inputs-27p0.json',
        'work/phase6/fused-uart512-paired-transfer-v1/plan.json',
        'work/phase6/fused-uart512-paired-transfer-v1/seal.json'],
    'fused_uart512': [
        'tools/phase6/combined_fused_uart512.py',
        'tools/phase6/screen_fused_uart512_candidate.py',
        'work/phase6/experiments-v1/fused-activation-uart512-v1/engine.sv',
        'work/phase6/experiments-v1/fused-activation-uart512-v1/identity.json',
        'work/phase6/experiments-v1/fused-activation-uart512-v1/preflight-plan.json',
        'work/phase6/experiments-v1/fused-activation-uart512-v1/integration/report.json',
        'work/phase6/experiments-v1/fused-activation-uart512-v1/native-grouped-tail-prefetch/report.json',
        'work/phase6/experiments-v1/fused-activation-uart512-v1/route-inputs-24p0.json'],
    'final_fused_uart256': [
        'tools/phase6/screen_fused_uart256_candidate.py',
        'work/phase6/fused-activation-uart256-24-screen-v1/plan.json',
        'work/phase6/fused-activation-uart256-24-screen-v1/seal.json'],
}

ROUTES={
    'output_pipeline_22p5':'work/phase6/experiments-v1/output-pipeline-v1/route/report.json',
    'stream_mask_24':'work/phase6/experiments-v1/stream-mask-v1/route24/report.json',
    'stream_mask_27':'work/phase6/experiments-v1/stream-mask-v1/route27/report.json',
    'tail_mask_24':'work/phase6/experiments-v2/tail-mask-v2/route24/report.json',
    'fused_activation_24':'work/phase6/experiments-v1/fused-activation-v1/route24/report.json',
    'uart512_stream_24':'work/phase6/experiments-v1/stream-mask-uart512-v1/route24/report.json',
    'uart512_stream_27':'work/phase6/experiments-v1/stream-mask-uart512-v1/route27/report.json',
    'fused_uart512_24':'work/phase6/experiments-v1/fused-activation-uart512-v1/route24p0/report.json',
}

DEFAULT_SCREENS={
    'selected_prior':'work/phase6/combined-spec-scalar-uart-screen-v1',
    'stream_mask_grouped_prefetch_24':'work/phase6/stream-mask-grouped-prefetch-24-screen-v1',
    'fused_activation_24':'work/phase6/fused-activation-24-screen-v1',
    'fused_uart512_24':'work/phase6/fused-activation-uart512-24-screen-v1',
    'fused_uart256_24':'work/phase6/fused-activation-uart256-24-screen-v1',
}
SCREEN_ROUTE={
    'stream_mask_grouped_prefetch_24':'stream_mask_24',
    'fused_activation_24':'fused_activation_24',
    'fused_uart512_24':'fused_uart512_24',
    'fused_uart256_24':'fused_activation_24',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def get(path):
    return (ROOT/path).read_bytes()


def archive(path,artifacts):
    raw=get(path)
    if len(raw)>MAX_ARCHIVE_BYTES or Path(path).suffix in ('.fs','.bin'):
        raise ValueError(f'large/binary artifact excluded: {path}')
    target=OUT/'artifacts'/f'{path}.gz'
    target.parent.mkdir(parents=True,exist_ok=True)
    compressed=gzip.compress(raw,compresslevel=6,mtime=0)
    if not target.exists() or target.read_bytes()!=compressed:
        target.write_bytes(compressed)
    artifacts[path]=dict(sha256=sha(raw),bytes=len(raw),
        archive=str(target.relative_to(ROOT)),archive_sha256=sha(compressed))
    return raw


def json_artifact(path,artifacts):
    return json.loads(archive(path,artifacts))


def verify_report_sources(report):
    for path,expected in report.get('sources',{}).items():
        if not (ROOT/path).is_file() or sha(get(path))!=expected:
            raise ValueError(f'source changed since report: {path}')


def signed_rows(raw):
    rows=[]
    for line in raw.decode().splitlines():
        item=json.loads(line)
        signature=item.pop('record_sha256')
        canonical=json.dumps(item,sort_keys=True,separators=(',',':')).encode()
        if sha(canonical)!=signature:
            raise ValueError('physical record signature mismatch')
        rows.append(item)
    return rows


def physical(name,folder,artifacts,route):
    path=f'{folder}/report.json'
    if not (ROOT/path).is_file():
        return dict(status='pending',report=None,device_medians=None)
    report=json_artifact(path,artifacts)
    row=dict(status=report['status'],report=path,
             report_sha256=artifacts[path]['sha256'],
             completed=report.get('completed'),planned=report.get('planned'),
             device_medians=None)
    # Partial evidence must never become a latency or speedup claim.
    if report['status']!='passed-short-screen':
        return row
    records_path=f'{folder}/records.jsonl'
    records_raw=archive(records_path,artifacts)
    seal_path=f'{folder}/seal.json'
    if (ROOT/seal_path).is_file():
        seal=json_artifact(seal_path,artifacts)
        plan_path=f'{folder}/plan.json'
        archive(plan_path,artifacts)
        if (seal['plan_sha256']!=sha(get(plan_path)) or
                seal['report_sha256']!=sha(get(path)) or
                seal['records_sha256']!=sha(records_raw)):
            raise ValueError(f'physical seal mismatch: {name}')
    if (report.get('records_sha256')!=sha(records_raw) or
            report.get('completed')!=report.get('planned') or
            report.get('output_mismatches',0)!=0):
        raise ValueError(f'incomplete physical result: {name}')
    rows=signed_rows(records_raw)
    if len(rows)!=report['completed']:
        raise ValueError(f'physical record count differs: {name}')
    if name!='selected_prior':
        if route is None or route.get('status')!='passed-route':
            raise ValueError(f'completed physical screen lacks matching passed route: {name}')
        expected=route['bitstream_sha256']
        if any(item['bitstream_sha256']!=expected for item in rows):
            raise ValueError('physical rows mismatch routed bitstream')
        if any(item['bitstream_sha256']!=expected for item in report.get('programming',[])):
            raise ValueError('physical programming mismatch routed bitstream')
    if any(item['output_hex']!=item['expected_hex'] for item in rows):
        raise ValueError('physical output mismatch')
    medians={}
    pinned_io={}
    for model in ('kws','vww'):
        selected=[item for item in rows if item['model']==model and item['kind']=='timed']
        if len(selected)!=3:
            raise ValueError(f'missing three timed {model} rows')
        identities={(item['input_sha256'],item['output_hex']) for item in selected}
        if len(identities)!=1:
            raise ValueError(f'timed input/output identities differ: {name}/{model}')
        input_sha256,output_hex=identities.pop()
        pinned_io[model]=dict(input_sha256=input_sha256,output_hex=output_hex)
        median_cycles=statistics.median(item['elapsed_cycles'] for item in selected)
        median_ms=statistics.median(item['device_latency_ms'] for item in selected)
        reported=report['summary'][model]
        if median_cycles!=reported['median_cycles'] or not math.isclose(
                median_ms,reported.get('median_ms',reported.get('median_latency_ms',
                                      reported.get('median_device_latency_ms'))),
                rel_tol=0,abs_tol=1e-9):
            raise ValueError(f'median changed: {name}/{model}')
        medians[model]=dict(cycles=median_cycles,ms=median_ms,
                            clock_mhz=median_cycles/median_ms/1000)
    row.update(records=records_path,records_sha256=sha(records_raw),
               device_medians=medians,pinned_io=pinned_io,exact_records=len(rows))
    return row


def paired_transfer(artifacts,route):
    folder='work/phase6/fused-uart512-paired-transfer-v1'
    plan_path=f'{folder}/plan.json'
    report_path=f'{folder}/report.json'
    records_path=f'{folder}/records.jsonl'
    seal_path=f'{folder}/seal.json'
    if not (ROOT/report_path).is_file():
        return dict(status='pending',report=None)
    plan=json_artifact(plan_path,artifacts)
    report=json_artifact(report_path,artifacts)
    row=dict(status=report['status'],report=report_path,
             report_sha256=artifacts[report_path]['sha256'])
    if report['status']!='passed-paired-transfer':
        return row
    seal=json_artifact(seal_path,artifacts)
    records_raw=archive(records_path,artifacts)
    if (route is None or route['status']!='passed-route' or
            plan['bitstream_sha256']!=route['bitstream_sha256'] or
            sha(get(plan_path))!=seal['plan_sha256'] or
            sha(get(report_path))!=seal['report_sha256'] or
            sha(records_raw)!=report['records_sha256'] or
            sha(records_raw)!=seal['records_sha256']):
        raise ValueError('paired transfer source/route seal mismatch')
    rows=signed_rows(records_raw)
    if len(rows)!=6 or report['completed']!=6 or report['planned']!=6:
        raise ValueError('paired transfer row count mismatch')
    medians={}
    for chunk in (256,512):
        selected=[item for item in rows if item['chunk_bytes']==chunk]
        if (len(selected)!=3 or
                {item['pair'] for item in selected}!={1,2,3} or
                any(item['frame_count']*chunk!=plan['input_bytes'] or
                    item['input_sha256']!=plan['input_sha256'] or
                    item['bitstream_sha256']!=route['bitstream_sha256'] or
                    not item['precondition_verified'] or
                    not item['readback_verified'] for item in selected)):
            raise ValueError(f'paired transfer invalid {chunk}-byte rows')
        median=statistics.median(item['upload_seconds'] for item in selected)
        if not math.isclose(median,report['summary']['median_upload_seconds'][str(chunk)],
                            rel_tol=0,abs_tol=1e-12):
            raise ValueError(f'paired transfer median mismatch: {chunk}')
        medians[str(chunk)]=median
    ratio=medians['256']/medians['512']
    if not math.isclose(ratio,report['summary']['speedup_512_over_256'],
                        rel_tol=0,abs_tol=1e-12):
        raise ValueError('paired transfer speedup mismatch')
    row.update(records=records_path,records_sha256=sha(records_raw),
               route='fused_uart512_24',input_bytes=plan['input_bytes'],
               exact_records=len(rows),median_upload_seconds=medians,
               speedup_512_over_256=ratio)
    return row


def build(extra_screens=None):
    OUT.mkdir(parents=True,exist_ok=True)
    artifacts={}
    candidates={}
    for name,paths in GROUPS.items():
        row={}
        for path in paths:
            if not (ROOT/path).is_file():
                row[path]=dict(status='pending')
                continue
            raw=archive(path,artifacts)
            record=dict(status='archived',sha256=sha(raw),bytes=len(raw))
            if path.endswith('.json'):
                content=json.loads(raw)
                if 'status' in content: record['result_status']=content['status']
                verify_report_sources(content)
                if path.endswith('/identity.json') and 'engine_sha256' in content:
                    engine_path=str(Path(path).parent/'engine.sv')
                    if (ROOT/engine_path).is_file() and sha(get(engine_path))!=content['engine_sha256']:
                        raise ValueError(f'candidate engine identity changed: {name}')
            row[path]=record
        candidates[name]=row
    routes={}
    route_reports={}
    for name,path in ROUTES.items():
        if not (ROOT/path).is_file():
            routes[name]=dict(status='pending',report=path)
            continue
        report=json_artifact(path,artifacts)
        verify_report_sources(report)
        route_reports[name]=report
        row=dict(status=report['status'],report=path,
                 report_sha256=artifacts[path]['sha256'],
                 clock_mhz=report.get('core_clock_mhz'),
                 routed_fmax_mhz=report.get('routed_core_fmax_mhz'),
                 bitstream_sha256=report.get('bitstream_sha256'),
                 resources=report.get('resources'))
        if report['status']=='passed-route':
            if (row['routed_fmax_mhz'] is None or
                    row['routed_fmax_mhz']<row['clock_mhz'] or
                    report.get('setup_violated_endpoints') or
                    report.get('hold_violated_endpoints')):
                raise ValueError(f'route gate invalid: {name}')
            bitstream=ROOT/report['bitstream']
            if bitstream.exists() and sha(bitstream.read_bytes())!=report['bitstream_sha256']:
                raise ValueError(f'routed bitstream changed: {name}')
            row['bitstream_archived']=False
        routes[name]=row
    screens=dict(DEFAULT_SCREENS)
    if extra_screens:
        for name,folder in extra_screens.items():
            if name in screens: raise ValueError(f'duplicate screen {name}')
            screens[name]=folder
    physical_rows={}
    for name,folder in screens.items():
        route=route_reports.get(SCREEN_ROUTE.get(name))
        if route is None and name!='selected_prior':
            report_path=ROOT/folder/'report.json'
            if report_path.is_file():
                candidate=json.loads(report_path.read_text())
                if candidate.get('status')=='passed-short-screen':
                    hashes={item['bitstream_sha256'] for item in candidate.get('programming',[])}
                    matches=[r for r in route_reports.values() if r.get('status')=='passed-route'
                             and r.get('bitstream_sha256') in hashes]
                    if len(hashes)==1 and len(matches)==1:
                        route=matches[0]
        physical_rows[name]=physical(name,folder,artifacts,route)
    baseline=physical_rows['selected_prior']['device_medians']
    if baseline is None:
        raise ValueError('selected prior physical baseline missing')
    for name,row in physical_rows.items():
        if name=='selected_prior' or row['device_medians'] is None: continue
        if row['pinned_io']!=physical_rows['selected_prior']['pinned_io']:
            raise ValueError(f'pinned input/output identities differ from baseline: {name}')
        speedups={model:baseline[model]['ms']/row['device_medians'][model]['ms']
                  for model in ('kws','vww')}
        decomposition={}
        for model in ('kws','vww'):
            cycle_gain=baseline[model]['cycles']/row['device_medians'][model]['cycles']
            clock_gain=(row['device_medians'][model]['clock_mhz']/
                        baseline[model]['clock_mhz'])
            if not math.isclose(cycle_gain*clock_gain,speedups[model],
                                rel_tol=1e-12):
                raise ValueError(f'cycle/clock decomposition mismatch: {name}/{model}')
            decomposition[model]=dict(cycle_speedup=cycle_gain,
                                      clock_speedup=clock_gain)
        row['speedup_vs_selected_prior']=speedups
        row['latency_decomposition_vs_selected_prior']=decomposition
        row['geometric_mean_speedup']=math.sqrt(speedups['kws']*speedups['vww'])
    summary=dict(schema=1,status='boardless-evidence-archived',
        scope='short exact KWS/VWW screens; no full accuracy, energy or SOTA claim',
        gates=dict(native='pinned/stress full-model exact with fixed and stalled RAM',
            route='all resources fit, no setup/hold violations, routed Fmax >= operating clock',
            board='10/10 exact short-screen records with signed identities and three timed runs per model',
            expand='>=1.03 geometric-mean physical device throughput versus selected prior image',
            host='UART transfer gains reported separately from FPGA device FPS'),
        candidates=candidates,routes=routes,physical=physical_rows,
        paired_transfer=paired_transfer(artifacts,route_reports.get('fused_uart512_24')),
        artifacts=artifacts)
    if any(row['device_medians'] is not None for name,row in physical_rows.items()
           if name!='selected_prior'):
        summary['status']='short-physical-evidence-archived'
    selected=physical_rows['fused_uart256_24']
    if (selected['device_medians'] is not None and
            summary['paired_transfer']['status']=='passed-paired-transfer'):
        summary['selected_image']=dict(status='selected-after-short-screen',
            screen='fused_uart256_24',route='fused_activation_24',
            bitstream_sha256=route_reports['fused_activation_24']['bitstream_sha256'],
            core_clock_mhz=24,uart_burst_bytes=256,
            basis='10/10 exact physical screen; paired 512-byte upload no-go',
            scope='device latency and paired fresh-image upload only')
    (OUT/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen',action='append',default=[],metavar='NAME=WORK_FOLDER',
                        help='additional named physical screen under the repository')
    args=parser.parse_args()
    screens={}
    for item in args.screen:
        if '=' not in item: raise SystemExit('--screen requires NAME=WORK_FOLDER')
        name,folder=item.split('=',1)
        if not name or not folder: raise SystemExit('empty screen name or path')
        screens[name]=folder
    summary=build(screens)
    print(json.dumps(dict(status=summary['status'],candidates=len(summary['candidates']),
        routes={name:row['status'] for name,row in summary['routes'].items()},
        physical={name:row['status'] for name,row in summary['physical'].items()}),
        sort_keys=True))
