#!/usr/bin/env python3
"""Short physical RTL screen on unchanged strongest B3 schedules.

One stress, one warmup and three timed inferences per model. No endurance,
automatic retry, flash programming or unverified substitute timing evidence.
"""
import argparse
import fcntl
import gzip
import json
import math
from pathlib import Path
import signal
import statistics
import subprocess
import time
import xml.etree.ElementTree as ET

import matched_campaign as common

ROOT = common.ROOT
BASE = ROOT/'work/phase6/engine-candidate-rtl-v2'
REFERENCE = ROOT/'work/phase6/generic-b3-board-v2/physical'
FINAL = ROOT/'work/phase6/matched-baselines-v1/b3-final/report.json'


def prepare(route_path):
    common.check_frozen()
    pins = {}
    def pin(path):
        path = Path(path).resolve()
        pins[common.relative(path)] = common.sha(path)
        return common.read(path)
    historical = pin(REFERENCE/'report.json')
    original_plan = pin(REFERENCE/'plan.json')
    seal = pin(REFERENCE/'seal.json')
    for key, filename in [('plan','plan.json'),('report','report.json'),('records','records.jsonl')]:
        path = REFERENCE/filename
        if common.sha(path) != seal[key+'_sha256']:
            raise ValueError('historical physical seal changed')
        pins[common.relative(path)] = common.sha(path)
    if (historical['status'] != 'passed-matched-short-screen' or
            historical['physical_board'] is not True or historical['completed'] != 21 or
            historical['plan_sha256'] != common.sha(REFERENCE/'plan.json')):
        raise ValueError('historical physical evidence incomplete')
    hardware=original_plan['hardware']
    if (hardware['clock_hz']!=27_000_000 or hardware['baud']!=750_000
            or hardware['lanes']!=8 or hardware['sram_bytes']!=32768
            or hardware['image']['sha256']!=common.IMAGE_SHA
            or historical['records_sha256']!=seal['records_sha256']):
        raise ValueError('historical hardware/counter contract differs')
    for name in ('work/phase6/pool-timing-v1/route27/report.json',
            'work/phase6/pool-timing-v1/route27/phase6_uart_burst/impl/pnr/phase6_uart_burst.rpt.txt',
            'work/phase6/pool-timing-v1/route27/phase6_uart_burst/impl/pnr/phase6_uart_burst_tr_content.html'):
        expected=hardware['source_sha256'][name]
        common.verify(ROOT,{name:expected});pins[name]=expected
    from run_priority import read_rows
    if read_rows(REFERENCE/'records.jsonl') != historical['records']:
        raise ValueError('historical signed records differ')
    final = pin(FINAL)
    native = pin(BASE/'native/report.json')
    edges = pin(BASE/'edges/report.json')
    identity = pin(BASE/'identity.json')
    route = pin(route_path)
    engine_key = common.relative(BASE/'engine.sv')
    engine_hash = common.sha(BASE/'engine.sv')
    if (final['status'] != 'passed' or native['status'] != 'passed' or edges['status'] != 'passed'
            or identity['engine_sha256'] != engine_hash or native['engine_sha256'] != engine_hash
            or edges['source_sha256'][engine_key] != engine_hash or route['engine_sha256'] != engine_hash
            or native['final_baseline']['sha256'] != common.sha(FINAL)
            or native['baseline_executable_sha256'] != common.NATIVE_SHA):
        raise ValueError('candidate RTL/native identity invalid')
    if (route['status'] != 'passed-route' or route['core_clock_mhz'] != 27
            or route['routed_core_fmax_mhz'] < 27 or route['setup_violated_endpoints']
            or route['hold_violated_endpoints'] or route['uart_divider'] != 36
            or any(row['used'] > row['available'] for row in route['resources'].values())):
        raise ValueError('candidate has no passing resource-feasible 27 MHz route')
    coverage = route.get('active_gprj_source_coverage')
    if (not coverage or coverage['active_count'] != 18 or len(coverage['source_sha256']) != 18
            or coverage['source_sha256'].get(engine_key) != engine_hash
            or coverage['source_sha256'] != route['expected_gprj_sources']
            or route['uart_baud'] != 750_000 or route['uart_burst_bytes'] != 256):
        raise ValueError('candidate active project/host coverage differs')
    common.verify(ROOT, coverage['source_sha256'])
    pins.update(coverage['source_sha256'])
    for path, digest in ((coverage['gowin_project'],coverage['gowin_project_sha256']),
            (route['inputs'],route['inputs_sha256']),
            (route['build_log'],route['build_log_sha256']),
            (route['route_report'],route['route_sha256']),
            (route['timing_report'],route['timing_sha256'])):
        common.verify(ROOT,{path:digest}); pins[path]=digest
    for report in (native, edges, route):
        source_pins = report.get('source_sha256', report.get('sources'))
        if not source_pins:
            raise ValueError('candidate evidence lacks source pins')
        common.verify(ROOT, source_pins)
        pins.update(source_pins)
    if pins.get(engine_key) != engine_hash:
        raise ValueError('route source pins omit selected engine')
    for path, expected in [(BASE/'native/Vv2_tiled_host_bridge',native['executable_sha256']),
            (BASE/'edges/results.xml',edges['results_sha256']),
            (common.local(route['bitstream']),route['bitstream_sha256'])]:
        if common.sha(path) != expected: raise ValueError('candidate artifact digest mismatch')
        pins[common.relative(path)] = expected
    cases = ET.parse(BASE/'edges/results.xml').findall('.//testcase')
    if len(cases) != edges['tests'] or any(c.find('failure') is not None or c.find('error') is not None for c in cases):
        raise ValueError('edge test results invalid')
    # The route producer supplies raw timing/resource files and project inputs.
    for path, digest in route.get('artifact_sha256', {}).items():
        common.verify(ROOT, {path:digest}); pins[path] = digest
    fixtures = {}
    for model in ('kws','vww'):
        fixtures[model] = {}
        for sample in ('pinned','stress'):
            fixture = original_plan['policies']['B3-adaptation'][model][sample]
            selected = final['selected'][model][sample]
            common.verify(common.local(fixture['directory']), fixture['files'])
            common.verify(common.local(selected['directory']), selected['files'])
            # The historical VWW adapter changes only schedule metadata. Every
            # execution byte and oracle must be the original B3-selected bytes.
            if ({k:v for k,v in fixture['files'].items() if k != 'schedule.json'} !=
                    {k:v for k,v in selected['files'].items() if k != 'schedule.json'}):
                raise ValueError('historical fixture changes execution/oracle bytes')
            before = common.read(common.local(selected['directory'])/'schedule.json')
            after = common.read(common.local(fixture['directory'])/'schedule.json')
            before['command_count'] = len((common.local(selected['directory'])/'commands.bin').read_bytes())//16
            before.setdefault('snapshot_regions', {})
            if before != after: raise ValueError('unexplained historical schedule metadata adapter')
            candidates = [row['candidate'] for row in native['results'] if row['model']==model and row['sample']==sample]
            if {row['stall_seed'] for row in candidates} != {0,6063} or len(candidates) != 2:
                raise ValueError('candidate lacks both exact native seeds')
            for row in candidates:
                if (row['status'] != 'passed' or row['fixture_files'] != selected['files']
                        or row['executable_sha256'] != native['executable_sha256']):
                    raise ValueError('candidate native certificate unbound')
            for row in (fixture, selected):
                pins.update({common.relative(common.local(row['directory'])/name):digest for name,digest in row['files'].items()})
            fixtures[model][sample] = dict(directory=fixture['directory'],files=fixture['files'],
                candidate_native=candidates,selected_directory=selected['directory'],
                metadata_adapter='historical command_count/snapshot_regions metadata only; execution/oracle bytes identical')
    for filename in ('physical_engine_candidate.py','matched_campaign.py','run_screening.py',
            'run_priority.py','screen_engine_schedule.py','uart_burst.py','variants.py'):
        path = ROOT/'tools/phase6'/filename; pins[common.relative(path)] = common.sha(path)
    for filename in ('tools/phase4/tiled_host.py','tools/phase2/host.py'):
        pins[filename] = common.sha(ROOT/filename)
    path=ROOT/'work/phase6/phase5-frozen-files.json';pins[common.relative(path)]=common.sha(path)
    return dict(status='preflight-passed',physical_board=False,planned=10,clock_hz=27_000_000,
        baud=750_000,bitstream=route['bitstream'],bitstream_sha256=route['bitstream_sha256'],
        route=route,fixtures=fixtures,source_sha256=pins,
        original_checkout=original_plan['hardware']['original_checkout'],
        baseline=historical['summary']['B3-adaptation'],baseline_resources=original_plan['hardware']['resources'],
        scope='same B3 programs; one stress + one warmup + three pinned timings per model',
        timing_boundary=original_plan['hardware']['timing_boundary'],
        limitations=['Short separate-session comparison with sealed historical physical baseline; not an interleaved A/B campaign.',
            'Pinned/stress exactness only, not full-dataset accuracy, endurance or measured power.'])


def audit(plan):
    common.verify(ROOT,plan['source_sha256'])


def archive(output):
    output = Path(output).resolve()
    plan = common.read(output/'plan.json'); report = common.read(output/'report.json')
    if report['status'] != 'passed-short-screen': raise ValueError('only successful evidence can be archived')
    audit(plan)
    target = output/'archive'
    target.mkdir(exist_ok=False)
    entries = dict(plan['source_sha256'])
    for filename in ('plan.json','report.json','records.jsonl','seal.json','program.log'):
        path=output/filename; entries[common.relative(path)] = common.sha(path)
    manifest = dict(schema=1,status='sealed',scope=plan['scope'],files={})
    for name, digest in sorted(entries.items()):
        raw=common.local(name).read_bytes()
        if common.sha(common.local(name)) != digest: raise ValueError('archive source changed')
        compressed=gzip.compress(raw,mtime=0)
        destination=target/'artifacts'/Path(name+'.gz')
        destination.parent.mkdir(parents=True,exist_ok=True); destination.write_bytes(compressed)
        manifest['files'][name]=dict(sha256=digest,bytes=len(raw),gzip_sha256=common.sha(destination),
            archive_path=str(destination.relative_to(target)))
        if gzip.decompress(destination.read_bytes()) != raw: raise ValueError('archive roundtrip failed')
    common.save(target/'manifest.json',manifest)
    for name,row in manifest['files'].items():
        path=target/row['archive_path']
        if common.sha(path)!=row['gzip_sha256']: raise ValueError('archive compressed hash failed')
        import hashlib
        if hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest()!=row['sha256']:
            raise ValueError('archive raw hash failed')
    verification=dict(status='passed',manifest_sha256=common.sha(target/'manifest.json'),
        files=len(manifest['files']),uncompressed_bytes=sum(r['bytes'] for r in manifest['files'].values()),
        compressed_bytes=sum((target/r['archive_path']).stat().st_size for r in manifest['files'].values()),
        limitations='Portable evidence archive, not an EDA/compiler toolchain distribution or full reproducible environment image.')
    common.save(target/'verification.json',verification)
    return verification


def run(plan, output, port):
    import serial
    import run_screening as screening
    import run_priority as priority
    from screen_engine_schedule import execute_verified_sample
    from uart_burst import BurstTiledClient
    output=Path(output).resolve()
    audit(plan);screening.require_board_free(port)
    if output.exists():raise FileExistsError('choose a fresh physical evidence directory')
    locks=[]
    old_handlers={}
    try:
        paths=[(ROOT/'work/phase6/physical-board.lock','a'),
            (Path(plan['original_checkout'])/'work/phase6/physical-board.lock','r')]
        for path,mode in paths:
            if mode=='r' and not path.exists():continue
            lock=path.open(mode);locks.append(lock);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        screening.require_board_free(port)
        output.mkdir(parents=True);common.save(output/'plan.json',plan)
        report=dict(status='running',physical_board=True,planned=10,completed=0,records=[],
            started_at=screening.timestamp(),plan_sha256=common.sha(output/'plan.json'))
        def save():
            report['updated_at']=screening.timestamp();common.save(output/'report.json',report)
        def interrupt(signum,frame):raise KeyboardInterrupt('completed signed evidence preserved')
        old_handlers={sig:signal.signal(sig,interrupt) for sig in (signal.SIGINT,signal.SIGTERM)}
        save()
        try:
            with (output/'program.log').open('x') as log:
                subprocess.run(['openFPGALoader','-b','tangnano20k','--ftdi-serial','2025030317',
                    '--freq','2500000','-m','-v',str(common.local(plan['bitstream']))],
                    stdout=log,stderr=subprocess.STDOUT,timeout=90,check=True)
            report['program_log_sha256']=common.sha(output/'program.log');save();time.sleep(1)
            with serial.Serial(port,plan['baud'],timeout=5,write_timeout=5) as uart, (output/'records.jsonl').open('x') as stream:
                uart.reset_input_buffer();client=BurstTiledClient(uart);client.capabilities()
                for model in ('kws','vww'):
                    for sample in ('stress','pinned'):
                        fixture=plan['fixtures'][model][sample];directory=common.local(fixture['directory'])
                        report['current']=f'{model}/{sample}/load';save()
                        schedule,loaded=screening.load_fixture(client,directory,fixture)
                        for i,kind in enumerate(('stress',) if sample=='stress' else ('warmup','timed','timed','timed')):
                            report['current']=f'{model}/{kind}/{i}';save()
                            row=execute_verified_sample(client,schedule,(directory/'input.bin').read_bytes(),
                                (directory/'output.bin').read_bytes(),30)
                            checks=[]
                            if sample=='stress':
                                for line in (directory/'checks.txt').read_text().splitlines():
                                    address,name=line.split();wanted=(directory/name).read_bytes()
                                    actual=client.read_external(int(address),len(wanted))
                                    if actual!=wanted:raise screening.OutputMismatch(f'{model}/{name}: stress tensor mismatch')
                                    checks.append(dict(file=name,bytes=len(actual),sha256=priority.digest(actual)))
                            row.update(model=model,sample=sample,kind=kind,repeat=max(0,i-1),
                                fixture=fixture['directory'],input_sha256=fixture['files']['input.bin'],
                                bitstream_sha256=plan['bitstream_sha256'],at=screening.timestamp(),
                                device_latency_ms=row['elapsed_cycles']/27_000,load_seconds=loaded if i==0 else 0,
                                stress_tensor_checks=checks)
                            priority.append_row(stream,row);report['records'].append(row)
                            report['completed']=len(report['records']);save()
                            print(report['current'],row['elapsed_cycles'],flush=True)
            audit(plan)
            if priority.read_rows(output/'records.jsonl')!=report['records'] or report['completed']!=10:
                raise ValueError('signed record completeness mismatch')
            summary={}
            for model in ('kws','vww'):
                rows=[r for r in report['records'] if r['model']==model and r['kind']=='timed']
                cycles=[r['elapsed_cycles'] for r in rows]
                if len(cycles)!=3:raise ValueError('missing timing records')
                median=statistics.median(cycles);baseline=plan['baseline'][model]['median_cycles']
                summary[model]=dict(timed_cycles=cycles,median_cycles=median,median_device_latency_ms=median/27_000,
                    device_inferences_per_second=27_000_000/median,baseline_median_cycles=baseline,
                    speedup=baseline/median,latency_reduction_fraction=1-median/baseline,
                    range_over_median=(max(cycles)-min(cycles))/median,
                    median_engine_cycles=statistics.median(r['engine_cycles'] for r in rows),
                    median_dma_cycles=statistics.median(r['dma_cycles'] for r in rows),
                    median_overlap_cycles=statistics.median(r['overlap_cycles'] for r in rows))
            report.update(status='passed-short-screen',summary=summary,
                geometric_mean_speedup=math.prod(r['speedup'] for r in summary.values())**0.5,
                resources=plan['route']['resources'],routed_core_fmax_mhz=plan['route']['routed_core_fmax_mhz'],
                records_sha256=common.sha(output/'records.jsonl'),finished_at=screening.timestamp())
            save();common.save(output/'seal.json',{k+'_sha256':common.sha(output/f) for k,f in
                [('plan','plan.json'),('report','report.json'),('records','records.jsonl')]})
            return report
        except BaseException as error:
            report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',failure=repr(error))
            if (output/'records.jsonl').exists():report['records_sha256']=common.sha(output/'records.jsonl')
            save();raise
    finally:
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
        for lock in reversed(locks):fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route-report',type=Path,default=BASE/'route27/report.json')
    parser.add_argument('--output',type=Path,default=BASE/'physical/screen-v1')
    parser.add_argument('--port',default='/dev/cu.usbserial-20250303171')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--archive-only',action='store_true')
    args=parser.parse_args()
    if args.archive_only:print(json.dumps(archive(args.output),indent=2))
    else:
        plan=prepare(args.route_report)
        if args.run:
            report=run(plan,args.output,args.port)
            print(json.dumps(dict(summary=report['summary'],archive=archive(args.output)),indent=2))
        else:print(json.dumps(dict(status=plan['status'],planned=plan['planned'],resources=plan['route']['resources']),indent=2))
