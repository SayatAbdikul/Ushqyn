#!/usr/bin/env python3
"""Short physical screen for a separately verified engine and compiler schedule.

The selected routed image is immutable. Each model runs one stress input,
one pinned warmup and three pinned timings. --run is the only hardware path.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np
import serial

import quick_experiments as quick
import run_priority as priority
import run_screening as screening
from variants import ROOT, check_frozen
from tiled_host import TiledClient
from host import STATUS, decode_status


IMAGE_LABEL = 'all-exact'
IMAGE_ROOT = ROOT / 'work/phase6/experiments-v1/all-exact'


def _fixture_names(records):
    names = {}
    for model in ('kws', 'vww'):
        options = {}
        for sample, ending in (('pinned', '-timed'), ('stress', '-check')):
            matches = [name for name, row in records.items()
                       if row['model'] == model and name.startswith(f'{model}-{sample}-')
                       and name.endswith(ending)]
            if len(matches) != 1:
                raise ValueError(f'{model} {sample} requires one {ending} fixture')
            options[sample] = matches[0]
        names[model] = options
    return names


def _selected_image():
    """Validate the immutable selected bitstream without pinning mutable compiler sources."""
    engine = json.loads((IMAGE_ROOT/'engine/report.json').read_text())
    native = json.loads((IMAGE_ROOT/'native/report.json').read_text())
    edge = json.loads((IMAGE_ROOT/'edges/report.json').read_text())
    route = json.loads((IMAGE_ROOT/'route/report.json').read_text())
    identity = json.loads((IMAGE_ROOT/'identity.json').read_text())
    source_key = next(k for k in engine['sources'] if k.endswith('/engine.sv'))
    engine_file = ROOT/source_key
    engine_sha = screening.sha(engine_file)
    if (engine['status'] != 'passed' or native['status'] != 'passed' or edge['status'] != 'passed'
            or engine_sha != engine['sources'][source_key]
            or screening.sha(IMAGE_ROOT/'engine.sv') != engine_sha
            or native['sources'].get(source_key) != engine_sha
            or edge['sources'].get(source_key) != engine_sha
            or route['engine_sha256'] != engine_sha):
        raise ValueError('selected engine or RTL simulation identity mismatch')
    engine_xml = IMAGE_ROOT/'engine/results.xml'
    cases = ET.parse(engine_xml).findall('.//testcase')
    if (screening.sha(engine_xml) != engine['results_sha256'] or len(cases) != 3
            or any(c.find('failure') is not None or c.find('error') is not None for c in cases)):
        raise ValueError('selected engine RTL tests failed')
    if identity['parameters'].get('SCALAR_LUT'):
        edge_xml = IMAGE_ROOT/'edges/results.xml'
        edge_cases = ET.parse(edge_xml).findall('.//testcase')
        if (screening.sha(edge_xml) != edge['results_sha256'] or len(edge_cases) != 1
                or any(c.find('failure') is not None or c.find('error') is not None for c in edge_cases)):
            raise ValueError('selected activation edge tests failed')
    if (route['status'] != 'passed-route' or route['setup_violated_endpoints']
            or route['hold_violated_endpoints'] or route['routed_core_fmax_mhz'] < route['core_clock_mhz']
            or any(r['used'] > r['available'] for r in route['resources'].values())):
        raise ValueError('selected route is not eligible')
    if route.get('sources'):
        screening.verify_files(ROOT,route['sources'])
    screening.verify_files(ROOT,{route['bitstream']:route['bitstream_sha256']})
    return dict(file=route['bitstream'],sha256=route['bitstream_sha256'],
        clock_hz=round(route['core_clock_mhz']*1_000_000),
        evidence={s:screening.sha(IMAGE_ROOT/s/'report.json') for s in ('engine','native','edges','route')})


def prepare(fixtures_root, manifest_path=None, native_path=None, variant=None):
    """Verify all identities and RTL coverage without touching JTAG or UART."""
    check_frozen()
    fixtures_root = Path(fixtures_root).resolve()
    manifest_path = Path(manifest_path).resolve() if manifest_path else fixtures_root/'fixtures.json'
    native_path = Path(native_path).resolve() if native_path else fixtures_root/'native-all-exact/report.json'
    fixture_directory = fixtures_root/'fixtures'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'passed-replay' or not manifest.get('fixtures'):
        raise ValueError('independent fixture replay missing')
    records = {}
    for row in manifest['fixtures']:
        name = row['name']
        if (name in records or Path(name).name != name or row.get('model') not in ('kws', 'vww')
                or row.get('verification', {}).get('status') != 'passed'):
            raise ValueError('invalid or duplicate fixture record')
        for required in ('commands.bin', 'payload.bin', 'schedule.json', 'input.bin', 'output.bin', 'checks.txt'):
            if required not in row['files']:
                raise ValueError(f'{name} missing {required}')
        directory = fixture_directory/name
        screening.verify_files(directory, row['files'])
        schedule = json.loads((directory/'schedule.json').read_text())
        if (screening.sha(directory/'commands.bin') != schedule['program_sha256']
                or screening.sha(directory/'payload.bin') != schedule['image_sha256']
                or (directory/'commands.bin').stat().st_size != 16*schedule['command_count']):
            raise ValueError(f'{name} schedule identity mismatch')
        records[name] = row
    names = _fixture_names(records)
    image = _selected_image()
    route = json.loads((IMAGE_ROOT/'route/report.json').read_text())
    native = json.loads(native_path.read_text())
    if (native.get('status') != 'passed' or native.get('label') != route.get('source_label',IMAGE_LABEL)
            or native.get('fixture_manifest_sha256') != screening.sha(manifest_path)
            or route['engine_sha256'] not in [v for k, v in native['sources'].items() if k.endswith('engine.sv')]
            or screening.sha(IMAGE_ROOT/'native/Vv2_tiled_host_bridge') != native['executable_sha256']):
        raise ValueError('native full-model RTL evidence or engine identity mismatch')
    for name, row in records.items():
        for seed in (0, 6063):
            matches = [r for r in native['results'] if r['fixture'] == name and r['stall_seed'] == seed]
            if (len(matches) != 1 or matches[0]['status'] != 'passed'
                    or matches[0]['fixture_files'] != row['files'] or matches[0]['tensor_checks'] < 1):
                raise ValueError(f'{name} missing full-model RTL coverage at seed {seed}')
    paused = ROOT/'work/phase6/priority-v1'
    if json.loads((paused/'report.json').read_text())['status'] != 'interrupted':
        raise ValueError('full accuracy state changed')
    variant = variant or fixtures_root.name
    if not variant or '/' in variant or '\\' in variant:
        raise ValueError('invalid variant label')
    return dict(schema=1, variant=variant, image=image, image_label=IMAGE_LABEL,
        fixture_root=str(fixture_directory), manifest=str(manifest_path),
        manifest_sha256=screening.sha(manifest_path), native_report=str(native_path),
        native_report_sha256=screening.sha(native_path), fixture_names=names,
        fixtures=records, planned=10, repeats=3, physical_board=True,
        runner_sha256=screening.sha(Path(__file__)),
        dependency_sha256={str(p.relative_to(ROOT)):screening.sha(p) for p in (
            ROOT/'tools/phase6/quick_experiments.py', ROOT/'tools/phase6/run_screening.py',
            ROOT/'tools/phase6/run_priority.py', ROOT/'tools/phase4/tiled_host.py',
            ROOT/'tools/phase2/host.py')},
        preserved_accuracy={p.name:screening.sha(p) for p in paused.iterdir()
                            if p.suffix in ('.json','.jsonl')},
        scope='one stress, one warmup, three timed runs per model; final logits only on FPGA')


def execute_verified_sample(client, schedule, data, expected, timeout=30):
    """Execute one inference with input, status, counters and logits verified."""
    client.write(0x41001c, b'\0\0')
    started = time.monotonic()
    client.write_external(0, data)
    uploaded = time.monotonic()
    if client.read_external(0, len(data)) != data:
        raise AssertionError('input SDRAM readback mismatch')
    readback = time.monotonic()
    client.write(0x410000, b'\x01')
    deadline = time.monotonic()+timeout
    while True:
        status = decode_status(client.exchange(STATUS))
        if status['error'] or status['protocol_errors']:
            raise AssertionError(f'FPGA/protocol failure: {status}')
        if not status['busy']:
            break
        if time.monotonic()>deadline:
            raise TimeoutError('ambiguous FPGA launch; not retried')
    row = screening.check_counters(client.read(0x410000,32),schedule['command_count'])
    output = schedule['final_output']
    actual = client.read_external(output['ext'],output['bytes'])
    if actual != expected:
        raise screening.OutputMismatch(f'FPGA logits differ: actual={actual.hex()} expected={expected.hex()}')
    row.update(input_upload_seconds=uploaded-started,
        input_readback_seconds=readback-uploaded,
        input_readback_verified=True,
        wall_seconds=time.monotonic()-started,
        output_hex=actual.hex(),expected_hex=expected.hex(),
        prediction=int(np.argmax(np.frombuffer(actual,np.int8))))
    return row


def main():
    global IMAGE_LABEL, IMAGE_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-label', required=True,
                        help='routed engine candidate under work/phase6/experiments-v1/')
    parser.add_argument('--fixtures-root', type=Path, required=True,
                        help='directory containing fixtures.json and fixtures/')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--native-report', type=Path)
    parser.add_argument('--variant')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--port', default='/dev/cu.usbserial-20250303171')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if Path(args.image_label).name != args.image_label or args.image_label in ('.','..'):
        raise ValueError('invalid engine label')
    IMAGE_LABEL = args.image_label
    IMAGE_ROOT = ROOT / 'work/phase6/experiments-v1' / IMAGE_LABEL
    plan = prepare(args.fixtures_root, args.manifest, args.native_report, args.variant)
    if not args.run:
        print(json.dumps(dict(status='preflight-passed', variant=plan['variant'],
            planned=plan['planned'], manifest_sha256=plan['manifest_sha256'],
            native_report_sha256=plan['native_report_sha256']),sort_keys=True))
        return
    if args.output is None:
        parser.error('--run requires a fresh --output directory')
    screening.require_board_free(args.port)
    if args.output.exists():
        raise FileExistsError('preserve prior evidence; choose a fresh output directory')
    lock = (ROOT/'work/phase6/physical-board.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    args.output.mkdir(parents=True)
    screening.save_json(args.output/'plan.json',plan)
    report = dict(schema=1,status='running',started_at=screening.timestamp(),pid=os.getpid(),
        physical_board=True,variant=plan['variant'],planned=10,completed=0,output_mismatches=0,
        plan_sha256=screening.sha(args.output/'plan.json'),programming=[])
    def save():
        report['updated_at']=screening.timestamp()
        screening.save_json(args.output/'report.json',report)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('interrupted; completed signed records preserved')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    rows=[];save()
    try:
        with (args.output/'records.jsonl').open('x') as stream:
            image=plan['image']
            screening.verify_files(ROOT,{image['file']:image['sha256']})
            report['current']={'operation':'programming'};save()
            log=args.output/'program.log'
            with log.open('x') as output:
                subprocess.run(['openFPGALoader','-b','tangnano20k','--ftdi-serial','2025030317',
                    '--freq','2500000','-m','-v',str(ROOT/image['file'])],
                    stdout=output,stderr=subprocess.STDOUT,timeout=90,check=True)
            report['programming'].append(dict(at=screening.timestamp(),bitstream_sha256=image['sha256'],
                                              log_sha256=screening.sha(log)))
            save();time.sleep(1)
            with serial.Serial(args.port,750000,timeout=5,write_timeout=5) as uart:
                uart.reset_input_buffer();client=TiledClient(uart);client.capabilities()
                for model in ('kws','vww'):
                    stress_name=plan['fixture_names'][model]['stress']
                    stress=Path(plan['fixture_root'])/stress_name
                    report['current']={'model':model,'operation':'load-and-verify-stress'};save()
                    stress_schedule,stress_loaded=screening.load_fixture(client,stress,plan['fixtures'][stress_name])
                    report['current']['operation']='stress';save()
                    data=(stress/'input.bin').read_bytes()
                    expected=(stress/'output.bin').read_bytes()
                    row=execute_verified_sample(client,stress_schedule,data,expected,30)
                    row['device_latency_ms']=row['elapsed_cycles']/image['clock_hz']*1000
                    row.update(variant=plan['variant'],model=model,repeat=0,kind='stress',
                        fixture=stress_name,input_sha256=priority.digest(data),
                        bitstream_sha256=image['sha256'],at=screening.timestamp(),
                        load_seconds=stress_loaded)
                    priority.append_row(stream,row);rows.append(row)
                    report['completed']=len(rows);save()

                    name=plan['fixture_names'][model]['pinned']
                    directory=Path(plan['fixture_root'])/name
                    report['current']={'model':model,'operation':'load-and-verify-pinned'};save()
                    schedule,loaded=screening.load_fixture(client,directory,plan['fixtures'][name])
                    report['current']['operation']='warmup-and-timed';save()
                    for case in range(1,5):
                        data=(directory/'input.bin').read_bytes()
                        expected=(directory/'output.bin').read_bytes()
                        row=execute_verified_sample(client,schedule,data,expected,30)
                        row['device_latency_ms']=row['elapsed_cycles']/image['clock_hz']*1000
                        row.update(variant=plan['variant'],model=model,repeat=max(0,case-1),
                            kind='warmup' if case==1 else 'timed',
                            fixture=name, input_sha256=priority.digest(data),
                            bitstream_sha256=image['sha256'],at=screening.timestamp(),
                            load_seconds=loaded if case==1 else 0)
                        priority.append_row(stream,row);rows.append(row)
                        report['completed']=len(rows);save()
                    print(f'{screening.timestamp()} {report["completed"]}/10 {plan["variant"]} {model}',flush=True)
        stored_rows=priority.read_rows(args.output/'records.jsonl')
        if stored_rows != rows:
            raise ValueError('signed record round-trip mismatch')
        summary=quick.summarize(stored_rows,[plan['variant']])[plan['variant']]
        screening.verify_files(ROOT/'work/phase6/priority-v1',plan['preserved_accuracy'])
        screening.verify_files(ROOT,plan['dependency_sha256'])
        screening.verify_files(ROOT,{image['file']:image['sha256']})
        if (screening.sha(Path(__file__)) != plan['runner_sha256']
                or screening.sha(Path(plan['manifest'])) != plan['manifest_sha256']
                or screening.sha(Path(plan['native_report'])) != plan['native_report_sha256']):
            raise ValueError('screen evidence source identity changed during run')
        report.update(status='passed-short-screen',summary=summary,
                      finished_at=screening.timestamp(),
                      records_sha256=screening.sha(args.output/'records.jsonl'))
        save();print(json.dumps(summary,sort_keys=True),flush=True)
    except BaseException as error:
        report['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
        report['failure']=repr(error)
        if isinstance(error,screening.OutputMismatch):report['output_mismatches']+=1
        if (args.output/'records.jsonl').exists():
            report['records_sha256']=screening.sha(args.output/'records.jsonl')
        save();raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
