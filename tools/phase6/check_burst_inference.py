#!/usr/bin/env python3
"""Check one pinned KWS and VWW inference on an already programmed burst image.

Without --run, this performs a read-only evidence preflight. With --run, it
acquires the Phase 6 board lock, uses 256-byte UART burst writes for fixture
payloads and inputs, checks SDRAM/command readback and exact logits, and
records signed results. It never programs the FPGA or retries a launch.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time
import xml.etree.ElementTree as ET

import serial

import run_priority as priority
import run_screening as screening
import screen_engine_schedule as screen
from uart_burst import BURST_BYTES, BurstTiledClient
from variants import ROOT, check_frozen
from host import STATUS, decode_status


LABEL = 'combined-spec-scalar-uart-v1'
IMAGE_ROOT = ROOT/'work/phase6/experiments-v1'/LABEL
DEFAULT_FIXTURES = ROOT/'work/phase6/constant-sibling-v1'
DEFAULT_NATIVE = DEFAULT_FIXTURES/f'native-{LABEL}/report.json'
DEFAULT_PRIOR = ROOT/'work/phase6/combined-spec-scalar-uart-screen-v1'


class AuditedBurstClient(BurstTiledClient):
    """Record successful external writes; the superclass emits burst frames."""

    def __init__(self, uart):
        super().__init__(uart)
        self.burst_writes = []

    def write_external(self, offset, data):
        started = time.monotonic()
        super().write_external(offset, data)
        self.burst_writes.append(dict(offset=offset, bytes=len(data),
            frames=(len(data)+BURST_BYTES-1)//BURST_BYTES,
            sha256=priority.digest(data), seconds=time.monotonic()-started))


def _verified_xml(path, report, expected_cases):
    cases=ET.parse(path).findall('.//testcase')
    if (screening.sha(path)!=report['results_sha256'] or len(cases)!=expected_cases
            or any(case.find('failure') is not None or case.find('error') is not None
                   for case in cases)):
        raise ValueError(f'RTL regression failed or changed: {path}')


def prepare(fixtures_root=DEFAULT_FIXTURES, native_report=DEFAULT_NATIVE,
            prior_screen=DEFAULT_PRIOR):
    """Verify all file and selected-board-image identities without touching UART."""
    check_frozen()
    fixtures_root=Path(fixtures_root).resolve()
    native_report=Path(native_report).resolve()
    prior_screen=Path(prior_screen).resolve()
    screen.IMAGE_LABEL=LABEL
    screen.IMAGE_ROOT=IMAGE_ROOT
    baseline=screen.prepare(fixtures_root,native_path=native_report,variant='burst-pinned-exact')
    route=json.loads((IMAGE_ROOT/'route/report.json').read_text())
    if (route.get('bridge')!='uart-burst' or route['core_clock_mhz']!=22.5
            or baseline['image']['sha256']!=route['bitstream_sha256']):
        raise ValueError('selected route is not the 22.5 MHz UART burst image')
    integration_path=IMAGE_ROOT/'integration/report.json'
    integration=json.loads(integration_path.read_text())
    if integration.get('status')!='passed' or integration.get('tests')!=2:
        raise ValueError('integrated UART RTL tests did not pass')
    screening.verify_files(ROOT,integration['sources'])
    _verified_xml(IMAGE_ROOT/'integration/results.xml',integration,2)

    prior_report_path=prior_screen/'report.json'
    prior_plan_path=prior_screen/'plan.json'
    prior_log_path=prior_screen/'program.log'
    prior=json.loads(prior_report_path.read_text())
    earlier=json.loads(prior_plan_path.read_text())
    if (prior.get('status')!='passed-short-screen' or prior.get('completed')!=10
            or prior['plan_sha256']!=screening.sha(prior_plan_path)
            or prior['records_sha256']!=screening.sha(prior_screen/'records.jsonl')
            or earlier.get('image_label')!=LABEL
            or earlier['image']['sha256']!=route['bitstream_sha256']
            or not prior.get('programming')
            or prior['programming'][-1]['bitstream_sha256']!=route['bitstream_sha256']
            or prior['programming'][-1]['log_sha256']!=screening.sha(prior_log_path)):
        raise ValueError('completed matching board programming screen is required')
    priority.read_rows(prior_screen/'records.jsonl')
    chosen={model:baseline['fixture_names'][model]['pinned'] for model in ('kws','vww')}
    fixtures={name:baseline['fixtures'][name] for name in chosen.values()}
    dependency_paths=(Path(__file__).resolve(),ROOT/'tools/phase6/uart_burst.py',
        ROOT/'tools/phase6/run_screening.py',ROOT/'tools/phase6/screen_engine_schedule.py',
        ROOT/'tools/phase6/run_priority.py',ROOT/'tools/phase4/tiled_host.py',
        ROOT/'tools/phase2/host.py')
    return dict(schema=1,status='preflight-passed',physical_board=True,
        image_label=LABEL,image=baseline['image'],clock_hz=baseline['image']['clock_hz'],
        fixture_root=baseline['fixture_root'],fixtures=fixtures,fixture_names=chosen,
        manifest=baseline['manifest'],manifest_sha256=baseline['manifest_sha256'],
        native_report=baseline['native_report'],native_report_sha256=baseline['native_report_sha256'],
        integration_report=str(integration_path),integration_report_sha256=screening.sha(integration_path),
        prior_screen_report=str(prior_report_path),prior_screen_report_sha256=screening.sha(prior_report_path),
        prior_screen_plan=str(prior_plan_path),prior_screen_plan_sha256=screening.sha(prior_plan_path),
        prior_program_log=str(prior_log_path),prior_program_log_sha256=screening.sha(prior_log_path),
        route_report_sha256=screening.sha(IMAGE_ROOT/'route/report.json'),
        dependency_sha256={str(p.relative_to(ROOT)):screening.sha(p) for p in dependency_paths},
        planned=2,burst_bytes=BURST_BYTES,baud=750000,
        scope='one exact pinned inference per model on the already programmed UART burst image; no programming',
        live_image_identity='attested by the completed matching programming log and read-only burst feature probe; the FPGA does not expose a bitstream digest')


def _verify_plan(plan):
    screening.verify_files(ROOT,plan['dependency_sha256'])
    screening.verify_files(ROOT,{plan['image']['file']:plan['image']['sha256']})
    for model in ('kws','vww'):
        name=plan['fixture_names'][model]
        screening.verify_files(Path(plan['fixture_root'])/name,plan['fixtures'][name]['files'])
    for key in ('manifest','native_report','integration_report','prior_screen_report',
                'prior_screen_plan','prior_program_log'):
        if screening.sha(plan[key])!=plan[key+'_sha256']:
            raise ValueError(f'evidence changed during UART burst inference: {key}')
    if screening.sha(IMAGE_ROOT/'route/report.json')!=plan['route_report_sha256']:
        raise ValueError('routed image evidence changed during UART burst inference')
    check_frozen()


def run(plan,port,output):
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError('preserve prior burst inference evidence; choose a fresh output')
    lock_path=ROOT/'work/phase6/physical-board.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        screening.require_board_free(port)
        _verify_plan(plan)
        output.mkdir(parents=True)
        screening.save_json(output/'plan.json',plan)
        report=dict(schema=1,status='running',physical_board=True,
            started_at=screening.timestamp(),planned=2,completed=0,output_mismatches=0,
            plan_sha256=screening.sha(output/'plan.json'),
            bitstream_sha256=plan['image']['sha256'],programming=[])
        def save():
            report['updated_at']=screening.timestamp()
            screening.save_json(output/'report.json',report)
        def interrupted(signum,frame):
            raise KeyboardInterrupt('interrupted; completed signed records preserved')
        signal.signal(signal.SIGTERM,interrupted)
        signal.signal(signal.SIGINT,interrupted)
        save()
        try:
            with (output/'records.jsonl').open('x') as stream:
                with serial.Serial(port,plan['baud'],timeout=5,write_timeout=5) as uart:
                    uart.reset_input_buffer()
                    client=AuditedBurstClient(uart)
                    client.capabilities()  # read-only legacy target and burst feature probe
                    status=decode_status(client.exchange(STATUS))
                    if status['busy'] or status['error'] or status['protocol_errors']:
                        raise ValueError('FPGA is not idle and error-free')
                    for model in ('kws','vww'):
                        name=plan['fixture_names'][model]
                        directory=Path(plan['fixture_root'])/name
                        report['current']={'model':model,'operation':'load-pinned-burst'};save()
                        before=len(client.burst_writes)
                        schedule,load_seconds=screening.load_fixture(client,directory,plan['fixtures'][name])
                        payload_calls=client.burst_writes[before:]
                        payload=(directory/'payload.bin').read_bytes()
                        if (len(payload_calls)!=1 or payload_calls[0]['offset']!=0
                                or payload_calls[0]['sha256']!=priority.digest(payload)
                                or payload_calls[0]['frames']!=(len(payload)+BURST_BYTES-1)//BURST_BYTES):
                            raise AssertionError('fixture payload did not use verified burst writes')
                        report['current']['operation']='exact-output-inference';save()
                        data=(directory/'input.bin').read_bytes()
                        expected=(directory/'output.bin').read_bytes()
                        before=len(client.burst_writes)
                        row=screen.execute_verified_sample(client,schedule,data,expected,30)
                        input_calls=client.burst_writes[before:]
                        if (len(input_calls)!=1 or input_calls[0]['offset']!=0
                                or input_calls[0]['sha256']!=priority.digest(data)
                                or input_calls[0]['frames']!=(len(data)+BURST_BYTES-1)//BURST_BYTES):
                            raise AssertionError('inference input did not use verified burst writes')
                        row['device_latency_ms']=row['elapsed_cycles']/plan['clock_hz']*1000
                        row.update(model=model,fixture=name,kind='pinned-exact',
                            bitstream_sha256=plan['image']['sha256'],at=screening.timestamp(),
                            command_sha256=screening.sha(directory/'commands.bin'),
                            payload_sha256=priority.digest(payload),input_sha256=priority.digest(data),
                            expected_sha256=priority.digest(expected),
                            actual_sha256=priority.digest(bytes.fromhex(row['output_hex'])),
                            payload_burst=payload_calls[0],input_burst=input_calls[0],
                            payload_readback_verified=True,command_readback_verified=True,
                            load_seconds=load_seconds)
                        priority.append_row(stream,row)
                        report['completed']+=1;save()
                        print(f'{screening.timestamp()} {report["completed"]}/2 {model}',flush=True)
            rows=priority.read_rows(output/'records.jsonl')
            if ([r['model'] for r in rows]!=['kws','vww'] or
                    any(r['output_hex']!=r['expected_hex'] or
                        r['actual_sha256']!=r['expected_sha256'] for r in rows)):
                raise ValueError('signed inference evidence mismatch')
            _verify_plan(plan)
            if screening.sha(output/'plan.json')!=report['plan_sha256']:
                raise ValueError('burst inference plan changed during run')
            report.update(status='passed-exact-burst-inference',finished_at=screening.timestamp(),
                records_sha256=screening.sha(output/'records.jsonl'))
            report.pop('current',None);save()
        except BaseException as error:
            report['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
            report['failure']=repr(error)
            if isinstance(error,screening.OutputMismatch):report['output_mismatches']+=1
            if (output/'records.jsonl').exists():
                report['records_sha256']=screening.sha(output/'records.jsonl')
            save()
            raise
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures-root',type=Path,default=DEFAULT_FIXTURES)
    parser.add_argument('--native-report',type=Path,default=DEFAULT_NATIVE)
    parser.add_argument('--prior-screen',type=Path,default=DEFAULT_PRIOR)
    parser.add_argument('--port',default='/dev/cu.usbserial-20250303171')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    plan=prepare(args.fixtures_root,args.native_report,args.prior_screen)
    if not args.run:
        print(json.dumps(dict(status=plan['status'],image_label=LABEL,
            bitstream_sha256=plan['image']['sha256'],fixtures=plan['fixture_names'],
            planned=plan['planned']),sort_keys=True))
        return
    if args.output is None:parser.error('--run requires a fresh --output directory')
    result=run(plan,args.port,args.output)
    print(json.dumps(dict(status=result['status'],completed=result['completed'],
        records_sha256=result['records_sha256']),sort_keys=True))


if __name__=='__main__':main()
