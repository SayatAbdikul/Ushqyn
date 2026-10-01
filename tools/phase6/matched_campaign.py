#!/usr/bin/env python3
"""Matched policy comparison on one immutable 27 MHz Tang Nano 20K image.

Preparation is boardless. Execution programs volatile SRAM once and runs short,
order-balanced measurements. Historical build paths are mapped explicitly to
the current checkout without changing any historical file or its digest.
"""
import argparse
import fcntl
import json
import math
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'compiler'), str(ROOT/'tools/phase6')]
from variants import check_frozen, sha

BASE = ROOT/'work/phase6/matched-baselines-v1'
POOL = ROOT/'work/phase6/pool-timing-v1'
REFERENCE = ROOT/'work/phase6/pair7-fusion-board-v1/physical-short-v1'
REFERENCE_FIXTURES = ROOT/'work/phase6/strip-fusion-pair7-v1/full/fixtures'
IMAGE_SHA = '3e943639382e4efea62d4d2072c016fa40eacc7a97ef6bad7f5bfdc69d8dafce'
NATIVE_SHA = 'b97fbf944827a5345670f3603e7f10368a897e6bcb19d16cc56905d8e06d16ec'
REQUIRED = ('commands.bin', 'payload.bin', 'input.bin', 'output.bin',
            'checks.txt', 'schedule.json')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temporary.replace(path)


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def local(name):
    path = (ROOT/name).resolve()
    path.relative_to(ROOT)
    return path


def verify(directory, files):
    directory = Path(directory).resolve()
    for name, digest in files.items():
        path = (directory/name).resolve()
        path.relative_to(directory)
        if sha(path) != digest:
            raise ValueError(f'evidence changed: {path}')


def hardware_contract():
    """Verify the existing physical seal, RTL, native binary and routed image."""
    frozen = check_frozen()
    plan, report, seal = (read(REFERENCE/name) for name in
                          ('plan.json', 'report.json', 'seal.json'))
    for name in ('plan', 'report', 'records'):
        filename = name+('.jsonl' if name=='records' else '.json')
        if sha(REFERENCE/filename) != seal[name+'_sha256']:
            raise ValueError(f'historical physical {name} seal changed')
    if (report['status'] != 'passed-short-screen' or report['completed'] != 10 or
            report['physical_board'] is not True or
            report['plan_sha256'] != sha(REFERENCE/'plan.json') or
            report['records_sha256'] != sha(REFERENCE/'records.jsonl')):
        raise ValueError('selected image has no sealed physical reference')
    route = read(POOL/'route27/report.json')
    native = read(POOL/'native/report.json')
    coverage = plan['pool27_source_coverage']
    verify(ROOT, coverage['source_sha256'])
    verify(ROOT, route['sources'])
    verify(ROOT, native['sources'])
    verify(ROOT, {coverage['gowin_project']:coverage['gowin_project_sha256']})
    verify(ROOT, {route['bitstream']:IMAGE_SHA})
    pnr=POOL/'route27/phase6_uart_burst/impl/pnr'
    raw_route={relative(pnr/'phase6_uart_burst.rpt.txt'):route['route_sha256'],
               relative(pnr/'phase6_uart_burst_tr_content.html'):route['timing_sha256']}
    verify(ROOT,raw_route)
    if (local(plan['image']['file']) != local(route['bitstream']) or
            sha(local(plan['image']['file'])) != IMAGE_SHA or
            plan['image']['sha256'] != IMAGE_SHA or
            route['bitstream_sha256'] != IMAGE_SHA or
            plan['image']['clock_hz'] != 27_000_000 or
            route['status'] != 'passed-route' or route['core_clock_mhz'] != 27 or
            route['routed_core_fmax_mhz'] < 27 or
            route['setup_violated_endpoints'] or route['hold_violated_endpoints'] or
            route['uart_divider'] != 36 or plan['baud'] != 750_000 or
            plan['protocol_burst_bytes'] != 256 or
            native['status'] != 'passed' or native['executable_sha256'] != NATIVE_SHA or
            sha(POOL/'native/Vv2_tiled_host_bridge') != NATIVE_SHA or
            any(r['used'] > r['available'] for r in route['resources'].values())):
        raise ValueError('hardware, clock, resources or native image mismatch')
    for name, digest in plan['image']['evidence'].items():
        path = POOL/('route27/report.json' if name=='route' else name+'/report.json')
        if sha(path) != digest:
            raise ValueError(f'selected hardware evidence changed: {name}')

    # Historical XML contains absolute paths. Derive its original root from
    # sealed fixture metadata, then require exact coverage after relocation.
    suffix = Path('work/phase6/strip-fusion-pair7-v1/full/fixtures')
    original = Path(plan['fixture_root'])
    if tuple(original.parts[-len(suffix.parts):]) != suffix.parts:
        raise ValueError('unexpected historical checkout identity')
    original_root = original.parents[len(suffix.parts)-1]
    selected = set()
    for node in ET.parse(local(coverage['gowin_project'])).findall('.//File'):
        if node.attrib.get('enable') != '1':
            continue
        path = Path(node.attrib['path'])
        name = str(path.relative_to(original_root)) if path.is_absolute() else str(path)
        local(name)
        selected.add(name)
    expected = set(coverage['source_sha256'])-{'work/phase6/pool-timing-v1/build27.tcl'}
    if selected != expected or len(selected) != 18:
        raise ValueError('relocated Gowin source set differs from sealed source set')
    sources = dict(coverage['source_sha256'], **native['sources'])
    sources.update(raw_route)
    for path in (POOL/'route27/report.json', POOL/'native/report.json',
                 POOL/'native/Vv2_tiled_host_bridge',
                 local(coverage['gowin_project']), local(route['bitstream']),
                 *(REFERENCE/name for name in ('plan.json','report.json','records.jsonl','seal.json'))):
        sources[relative(path)] = sha(path)
    return dict(image=plan['image'], resources=route['resources'], clock_hz=27_000_000,
        baud=750_000, burst_bytes=256, lanes=8, sram_bytes=32768,
        sdram_bytes=8*1024*1024, command_bytes=32768, frozen_files=frozen,
        native_sha256=NATIVE_SHA, source_sha256=sources,
        original_checkout=str(original_root), current_checkout=str(ROOT),
        timing_boundary='sequencer start to HALT; excludes input/model UART upload and output readback',
        arithmetic='unchanged INT8 operands, INT32 accumulators and exact per-layer requantization',
        historical_plan_sha256=sha(REFERENCE/'plan.json'))


def fixture(directory, files, model, sample, native, replay, *, historical=False):
    """Require exact paired inputs/logits and native evidence before board use."""
    directory = local(directory)
    if any(name not in files for name in REQUIRED):
        raise ValueError('incomplete fixture hashes')
    verify(directory, files)
    schedule = read(directory/'schedule.json')
    if (schedule['program_sha256'] != files['commands.bin'] or
            schedule['image_sha256'] != files['payload.bin'] or
            (directory/'commands.bin').stat().st_size != 16*schedule['command_count'] or
            (directory/'commands.bin').stat().st_size > 32768 or
            (directory/'payload.bin').stat().st_size > 8*1024*1024 or
            (directory/'input.bin').stat().st_size != (490 if model=='kws' else 27648) or
            (directory/'output.bin').stat().st_size != schedule['final_output']['bytes'] or
            replay.get('status') != 'passed' or not native or
            any(row.get('status') != 'passed' or row.get('tensor_checks',0)<1
                for row in native)):
        raise ValueError(f'{model}/{sample}: invalid ABI, replay, capacity or native certificate')
    for row in native:
        if (row.get('physical_board') is not False or
                row.get('stall_seed',row.get('seed')) not in (0,6063) or
                any(not isinstance(row.get(key),int) or row[key] <= 0
                    for key in ('elapsed_cycles','engine_cycles','dma_cycles')) or
                not isinstance(row.get('overlap_cycles'),int) or
                not 0<=row['overlap_cycles']<=min(row['engine_cycles'],row['dma_cycles']) or
                row['engine_cycles']+row['dma_cycles']-row['overlap_cycles']>row['elapsed_cycles']):
            raise ValueError(f'{model}/{sample}: malformed native timing certificate')
        if not historical and (row.get('executable_sha256') != NATIVE_SHA or
                row.get('fixture_files') != files):
            raise ValueError(f'{model}/{sample}: native result is not bound to these exact fixture files')
    if sample=='pinned' and (schedule.get('snapshot_regions') or schedule.get('snapshots_enabled')):
        raise ValueError('timed fixture includes intermediate snapshot overhead')
    checks=[]
    for line in (directory/'checks.txt').read_text().splitlines():
        address, name = line.split()
        if name not in files or int(address)<0:
            raise ValueError('unhashed or invalid native tensor check')
        path=(directory/name).resolve()
        path.relative_to(directory)
        if int(address)+path.stat().st_size>8*1024*1024:
            raise ValueError('tensor check outside SDRAM')
        checks.append((int(address),name))
    if (schedule['final_output']['ext'],'output.bin') not in checks:
        raise ValueError('native checks omit final output at declared address')
    if any(row['tensor_checks']!=len(checks) for row in native):
        raise ValueError('native tensor check count differs from checks.txt')
    reference = read(REFERENCE/'plan.json')
    ref_files = reference['fixtures'][reference['fixture_names'][model][sample]]['files']
    for name in ('input.bin','output.bin'):
        if files[name] != ref_files[name]:
            raise ValueError(f'{model}/{sample}: unmatched {name}')
    return dict(directory=relative(directory), files=files, model=model, sample=sample,
                native=native, replay=replay)


def reference_policy():
    old = read(REFERENCE/'plan.json')
    result = {}
    for model in ('kws','vww'):
        result[model] = {}
        for sample in ('pinned','stress'):
            label = old['fixture_names'][model][sample]
            row = old['fixtures'][label]
            result[model][sample] = fixture(relative(REFERENCE_FIXTURES/label),
                row['files'],model,sample,row['native'],row['verification'],historical=True)
    return result


def current_policy(path):
    report=read(path)
    if (report['status']!='passed' or report['native_sha256']!=NATIVE_SHA or
            report['hardware_sha256']!=IMAGE_SHA):
        raise ValueError('common-optimized current schedule lacks exact native coverage')
    verify(ROOT,report['sources'])
    result={}
    for model in ('kws','vww'):
        result[model]={}
        for sample in ('pinned','stress'):
            row=report['models'][model]['fixtures'][sample]
            if not row['command_identity_unchanged'] or not row['payload_identity_unchanged']:
                raise ValueError('current schedule differs from its prior replay certificate')
            result[model][sample]=fixture(row['directory'],row['files'],model,sample,
                                         row['native'],row['replay'])
    return result,report


def b1b2_policies(path):
    report = read(path)
    if report['status'] != 'passed' or report['native']['executable_sha256'] != NATIVE_SHA:
        raise ValueError('B1/B2 tuning is incomplete or used a different native engine')
    verify(ROOT, report['compiler_sources'])
    policies = {}
    dependencies = dict(report['compiler_sources'])
    for model in ('kws','vww'):
        model_report=report['models'][model]
        verify(ROOT,model_report['provenance']['sources'])
        dependencies.update(model_report['provenance']['sources'])
        for policy in ('B1','B2'):
            frontier=model_report.get('frontier',{}).get(policy,[model_report['selected'][policy]])
            if not 1<=len(frontier)<=3:
                raise ValueError('board frontier exceeds declared search budget')
            for rank,row in enumerate(frontier):
                label=policy if rank==0 else f'{policy}-alt{rank}'
                candidate = model_report['candidates'][row['candidate']]
                if (candidate['status'] != 'passed' or
                        {n.get('stall_seed',n.get('seed')) for n in
                         (row['native'],row['stalled_timed'])} != {0,6063}):
                    raise ValueError('baseline timed fixture lacks two native seeds')
                stress = next(v for v in row['validation'] if v['sample']=='stress')
                policies.setdefault(label,{})[model] = dict(
                    pinned=fixture(row['directory'],row['files'],model,'pinned',
                        [row['native'],row['stalled_timed']],candidate['replay']),
                    stress=fixture(stress['directory'],stress['files'],model,'stress',
                        [stress['native']],stress['replay']))
    return policies, dependencies


def prepare(b1b2, b3=None, repeats=3, current=BASE/'current-v4/report.json'):
    if repeats < 1:
        raise ValueError('at least one timed repeat required')
    hardware = hardware_contract()
    policies, dependencies = b1b2_policies(b1b2)
    policies['current'],current_report=current_policy(current)
    dependencies = dict(dependencies)
    dependencies[relative(b1b2)] = sha(b1b2)
    dependencies.update(current_report['sources'])
    dependencies[relative(current)]=sha(current)
    # Independently reconstruct the shared typed graph using the pinned
    # compiler sources; equality includes every weight and quantizer byte.
    from matched_b1b2_generic import matched_model
    from matched_current import graph_identity
    graphs={}
    for model in ('kws','vww'):
        graph=matched_model(model)[2]
        digest=graph_identity(graph)
        if digest!=current_report['models'][model]['graph_sha256']:
            raise ValueError(f'{model}: baseline/current optimized typed graphs differ')
        graphs[model]=digest
    if b3 is not None:
        report = read(b3)
        if (report['status'] != 'passed' or report['native_sha256'] != NATIVE_SHA or
                not report.get('coverage')):
            raise ValueError('B3 adaptation is incomplete or not pinned to the same engine')
        verify(ROOT,report['compiler_sources'])
        for model in ('kws','vww'):
            if report['models'][model]['graph_sha256']!=graphs[model]:
                raise ValueError(f'{model}: B3 used a different common optimized graph')
            pins=report['models'][model]['provenance']['sources']
            verify(ROOT,pins);dependencies.update(pins)
            frontier=report.get('frontier',{}).get(model,[report['selected'][model]])
            if not 1<=len(frontier)<=3:
                raise ValueError('B3 board frontier exceeds declared search budget')
            for rank,selection in enumerate(frontier):
                label='B3-adaptation' if rank==0 else f'B3-adaptation-alt{rank}'
                policies.setdefault(label,{})[model]={}
                for sample in ('pinned','stress'):
                    row = selection[sample]
                    if {n.get('stall_seed',n.get('seed')) for n in row['native']}!={0,6063}:
                        raise ValueError('B3 fixture lacks both exact native stall seeds')
                    policies[label][model][sample] = fixture(row['directory'],
                        row['files'],model,sample,row['native'],row['replay'])
        dependencies.update(report['compiler_sources'])
        dependencies[relative(b3)] = sha(b3)
    for name in ('matched_campaign.py','run_screening.py','run_priority.py',
                 'screen_engine_schedule.py','uart_burst.py','variants.py'):
        path = ROOT/'tools/phase6'/name
        dependencies[relative(path)] = sha(path)
    for path in (ROOT/'tools/phase4/tiled_host.py',ROOT/'tools/phase2/host.py',
                 ROOT/'work/phase6/phase5-frozen-files.json'):
        dependencies[relative(path)]=sha(path)
    groups={}
    for model in ('kws','vww'):
        seen={};groups[model]={}
        for policy,models in policies.items():
            if model not in models:continue
            identity=[]
            for sample in ('pinned','stress'):
                row=models[model][sample]
                # Metadata-only schedule labels can differ. Execution bytes,
                # all oracle checks and readout addresses must be identical.
                identity.append(tuple(sorted((n,d) for n,d in row['files'].items()
                                             if n!='schedule.json')))
                final=read(local(row['directory'])/'schedule.json')['final_output']
                identity.append(tuple(sorted(final.items())))
            identity=tuple(identity)
            representative=seen.setdefault(identity,policy)
            groups[model].setdefault(representative,[]).append(policy)
    return dict(schema=1,status='prepared',physical_board=False,hardware=hardware,
        policies=policies,source_sha256=dependencies,repeats=repeats,common_graph_sha256=graphs,
        measurement_groups=groups,
        planned=sum(len(rows) for rows in groups.values())*(1+2*repeats),
        order='stress by rotated policy order; timed rounds alternate forward/reverse policy order; warmup immediately after every reload before each timing',
        scope='short matched correctness and latency screen; not accuracy, endurance or energy validation',
        b3_coverage=read(b3)['coverage'] if b3 else None)


def audit_plan(plan):
    verify(ROOT,plan['hardware']['source_sha256'])
    verify(ROOT,plan['source_sha256'])
    for policy in plan['policies'].values():
        for model in policy.values():
            for row in model.values():
                verify(local(row['directory']),row['files'])


def run(plan, output, port):
    import serial
    import run_priority as priority
    import run_screening as screening
    from screen_engine_schedule import execute_verified_sample
    from uart_burst import BurstTiledClient
    screening.require_board_free(port)
    audit_plan(plan)
    output = Path(output)
    if output.exists():
        raise FileExistsError('choose a fresh output directory to preserve evidence')
    lock = (ROOT/'work/phase6/physical-board.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # A task moved into a worktree may coexist with an older compiler/queue.
    # Coordinate with its existing board lock without writing that checkout.
    historical_lock=None
    old_path=Path(plan['hardware']['original_checkout'])/'work/phase6/physical-board.lock'
    try:
        if old_path.exists() and old_path.resolve()!=Path(lock.name).resolve():
            historical_lock=old_path.open('r')
            fcntl.flock(historical_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:
        if historical_lock is not None:historical_lock.close()
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        raise
    output.mkdir(parents=True)
    save(output/'plan.json',plan)
    report = dict(schema=1,status='running',physical_board=True,completed=0,
        planned=plan['planned'],plan_sha256=sha(output/'plan.json'),
        started_at=screening.timestamp(),records=[])
    def update():
        report['updated_at']=screening.timestamp()
        save(output/'report.json',report)
    def interrupt(signum, frame):
        raise KeyboardInterrupt('matched campaign interrupted; completed records preserved')
    old_handlers={sig:signal.signal(sig,interrupt) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        update()
        with (output/'program.log').open('x') as log:
            subprocess.run(['openFPGALoader','-b','tangnano20k','--ftdi-serial',
                '2025030317','--freq','2500000','-m','-v',
                str(local(plan['hardware']['image']['file']))],
                stdout=log,stderr=subprocess.STDOUT,timeout=90,check=True)
        report['program_log_sha256']=sha(output/'program.log')
        time.sleep(1)
        with serial.Serial(port,plan['hardware']['baud'],timeout=5,write_timeout=5) as uart:
            uart.reset_input_buffer()
            client=BurstTiledClient(uart)
            client.capabilities()
            with (output/'records.jsonl').open('x') as stream:
                for mi,model in enumerate(('kws','vww')):
                    names=list(plan['measurement_groups'][model])
                    names=names[mi:]+names[:mi]
                    stages=[('stress',0)]
                    stages += [('pinned',r) for r in range(plan['repeats'])]
                    for si,(sample,repeat) in enumerate(stages):
                        order=names if si%2==0 else list(reversed(names))
                        for policy in order:
                            row=plan['policies'][policy][model][sample]
                            directory=local(row['directory'])
                            report['current']=f'{model}/{policy}/{sample}/{repeat}/load'
                            update()
                            schedule,loaded=screening.load_fixture(client,directory,row)
                            for kind in (('stress',) if sample=='stress' else ('warmup','timed')):
                                report['current']=f'{model}/{policy}/{kind}/{repeat}'
                                update()
                                record=execute_verified_sample(client,schedule,
                                    (directory/'input.bin').read_bytes(),
                                    (directory/'output.bin').read_bytes(),30)
                                record.update(policy=policy,model=model,kind=kind,repeat=repeat,
                                    equivalent_policies=plan['measurement_groups'][model][policy],
                                    at=screening.timestamp(),load_seconds=loaded if kind!='timed' else 0,
                                    input_sha256=row['files']['input.bin'],
                                    bitstream_sha256=IMAGE_SHA,
                                    device_latency_ms=record['elapsed_cycles']/27_000,
                                    order_index=len(report['records']))
                                priority.append_row(stream,record)
                                report['records'].append(record)
                                report['completed']=len(report['records'])
                                update()
                                print(report['current'],record['elapsed_cycles'],flush=True)
        audit_plan(plan)
        if (priority.read_rows(output/'records.jsonl') != report['records'] or
                report['completed'] != report['planned']):
            raise ValueError('physical record count or seal mismatch')
        summary={}
        for policy in plan['policies']:
            summary[policy]={}
            for model in plan['policies'][policy]:
                representative=next(p for p,members in plan['measurement_groups'][model].items()
                                    if policy in members)
                rows=[r for r in report['records'] if r['policy']==representative and
                      r['model']==model and r['kind']=='timed']
                if len(rows)!=plan['repeats']:
                    raise ValueError('incomplete timing group')
                cycles=[r['elapsed_cycles'] for r in rows]
                median=statistics.median(cycles)
                summary[policy][model]=dict(median_cycles=median,timed_cycles=cycles,
                    measured_policy=representative,shared_identical_execution=representative!=policy,
                    median_device_latency_ms=median/27_000,
                    device_inferences_per_second=27_000_000/median,
                    range_over_median=(max(cycles)-min(cycles))/median)
        comparisons={}
        for policy in summary:
            if policy=='current':continue
            speedups={m:summary[policy][m]['median_cycles']/
                      summary['current'][m]['median_cycles'] for m in summary[policy]}
            comparisons[policy]=dict(current_latency_speedup=speedups,
                geometric_mean_speedup=math.prod(speedups.values())**(1/len(speedups)))
        best={}
        for family in ('B1','B2','B3-adaptation'):
            options=[p for p in summary if p==family or p.startswith(family+'-alt')]
            if not options:continue
            winners={model:min((p for p in options if model in summary[p]),
                        key=lambda p:(summary[p][model]['median_cycles'],p))
                     for model in ('kws','vww')}
            ratios={m:summary[p][m]['median_cycles']/summary['current'][m]['median_cycles']
                    for m,p in winners.items()}
            best[family]=dict(selected_policies=winners,current_latency_speedup=ratios,
                geometric_mean_speedup=math.prod(ratios.values())**0.5)
        report.update(status='passed-matched-short-screen',summary=summary,
            comparisons=comparisons,best_matched_baselines=best,
            records_sha256=sha(output/'records.jsonl'),
            finished_at=screening.timestamp())
        update()
        save(output/'seal.json',{name+'_sha256':sha(output/filename) for name,filename in
            [('plan','plan.json'),('report','report.json'),('records','records.jsonl')]})
        return report
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                      failure=repr(error))
        if (output/'records.jsonl').exists():
            report['records_sha256']=sha(output/'records.jsonl')
        update()
        raise
    finally:
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
        if historical_lock is not None:
            fcntl.flock(historical_lock,fcntl.LOCK_UN)
            historical_lock.close()
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--b1b2',type=Path,default=BASE/'b1b2-final-v1/report.json')
    parser.add_argument('--b3',type=Path)
    parser.add_argument('--current',type=Path,default=BASE/'current-v4/report.json')
    parser.add_argument('--output',type=Path,default=BASE/'physical-v1')
    parser.add_argument('--port',default='/dev/cu.usbserial-20250303171')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--hardware-only',action='store_true')
    args=parser.parse_args()
    if args.hardware_only:
        print(json.dumps(hardware_contract(),indent=2,sort_keys=True))
        return
    plan=prepare(args.b1b2,args.b3,args.repeats,args.current)
    save(BASE/'campaign-plan.json',plan)
    if args.run:
        result=run(plan,args.output,args.port)
        print(json.dumps(result['comparisons'],indent=2,sort_keys=True))
    else:
        print(json.dumps(dict(status='preflight-passed',planned=plan['planned'],
            policies=list(plan['policies']),bitstream_sha256=IMAGE_SHA),sort_keys=True))


if __name__=='__main__':main()
