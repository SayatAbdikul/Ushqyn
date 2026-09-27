#!/usr/bin/env python3
"""Preflight and optional ten-inference board screen of compacted schedules.

The FPGA image is the existing routed fused-activation UART256 24 MHz image.
Only immutable command and payload bytes differ. Without --run this script
reads evidence files and never opens a board device.
"""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import statistics
import subprocess
import time

import serial

import run_priority as priority
import run_screening as screening
import screen_engine_schedule as physical
import screen_fused_uart256_candidate as selected
from uart_burst import BURST_BYTES, BurstTiledClient
from variants import ROOT, check_frozen, sha


BASE=ROOT/'work/phase6/channel-compaction-v1'
SOURCE=BASE/'fused'
SELECTED=ROOT/'work/phase6/experiments-v1/fused-activation-v1'


def prepare():
    check_frozen()
    old=selected.prepare()  # Full preflight of the unchanged routed RTL image.
    report=json.loads((SOURCE/'report.json').read_text())
    graph=json.loads((BASE/'report.json').read_text())
    native=json.loads((SELECTED/'native/report.json').read_text())
    if (report['status']!='passed' or graph['status']!='passed-replay' or
            graph['source_sha256']!=sha(ROOT/'tools/phase6/channel_compaction.py') or
            report['source_report_sha256']!=sha(BASE/'report.json') or
            report['executable_sha256']!=native['executable_sha256'] or
            sha(SELECTED/'native/Vv2_tiled_host_bridge')!=report['executable_sha256'] or
            old['image']['sha256']!=
                '03e57521885f1a9d6b4b4cef7e374122ba948028ed379d963c97f794efc9f195' or
            old['image']['clock_hz']!=24_000_000 or BURST_BYTES!=256):
        raise ValueError('compaction evidence or selected routed image changed')
    if len(report['results'])!=6:
        raise ValueError('expected six compacted fused fixtures')
    records={}
    for row in report['results']:
        name=row['label'];folder=SOURCE/'fixtures'/name
        if (Path(name).name!=name or name in records or
                row['verification']['status']!='passed' or
                row['lifetimes']['status']!='passed' or
                [(v['seed'],v['status']) for v in row['native']]!=
                    [(0,'passed'),(6063,'passed')]):
            raise ValueError(f'{name}: missing exact replay/native coverage')
        screening.verify_files(folder,row['files'])
        schedule=json.loads((folder/'schedule.json').read_text())
        commands=folder/'commands.bin';payload=folder/'payload.bin'
        if (sha(commands)!=schedule['program_sha256'] or
                sha(payload)!=schedule['image_sha256'] or
                commands.stat().st_size!=16*schedule['command_count'] or
                commands.stat().st_size>32768 or payload.stat().st_size>8*1024*1024 or
                schedule['final_output']['bytes']!=(folder/'output.bin').stat().st_size):
            raise ValueError(f'{name}: invalid command/payload capacity or identity')
        if row['model']=='kws' and (folder/'input.bin').stat().st_size!=490:
            raise ValueError('unexpected public KWS input shape')
        if row['model']=='vww' and (folder/'input.bin').stat().st_size!=27_648:
            raise ValueError('unexpected public VWW input shape')
        records[name]=row
    names=physical._fixture_names(records)
    for model in ('kws','vww'):
        new=SOURCE/'fixtures'/names[model]['pinned']/'output.bin'
        old_fixture=SELECTED/'fixtures'/old['fixture_names'][model]['pinned']/'output.bin'
        if new.read_bytes()!=old_fixture.read_bytes():
            raise ValueError(f'{model}: compacted pinned logits changed')
    plan=dict(old,variant='channel-compaction-fused-uart256-v1',
        fixture_root=str((SOURCE/'fixtures').resolve()),
        fixture_names=names,fixtures=records,
        compacted_report_sha256=sha(SOURCE/'report.json'),
        compacted_graph_sha256=sha(BASE/'report.json'),
        compacted_source_sha256=sha(ROOT/'tools/phase6/channel_compaction.py'),
        selected_image_report_sha256=sha(SELECTED/'route24/report.json'),
        physical_runner_sha256=sha(Path(__file__)),
        scope='unchanged selected FPGA image; one stress, one pinned warmup and three pinned timings per model')
    return plan


def run(plan,output,port):
    screening.require_board_free(port)
    if output.exists():
        raise FileExistsError('preserve previous evidence; choose a fresh output')
    lock=(ROOT/'work/phase6/physical-board.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    output.mkdir(parents=True)
    screening.save_json(output/'plan.json',plan)
    report=dict(schema=1,status='running',physical_board=True,
        variant=plan['variant'],planned=10,completed=0,
        started_at=screening.timestamp(),plan_sha256=sha(output/'plan.json'))
    def save():
        report['updated_at']=screening.timestamp()
        screening.save_json(output/'report.json',report)
    def interrupted(signum,frame):
        raise KeyboardInterrupt('compacted short screen interrupted; signed records preserved')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    rows=[];loads={};save()
    try:
        image=plan['image']
        screening.verify_files(ROOT,{image['file']:image['sha256']})
        report['current']='programming';save()
        with (output/'program.log').open('x') as log:
            subprocess.run(['openFPGALoader','-b','tangnano20k',
                '--ftdi-serial','2025030317','--freq','2500000','-m','-v',
                str(ROOT/image['file'])],stdout=log,stderr=subprocess.STDOUT,
                timeout=90,check=True)
        report['program_log_sha256']=sha(output/'program.log')
        time.sleep(1)
        with serial.Serial(port,plan['baud'],timeout=5,write_timeout=5) as uart:
            uart.reset_input_buffer()
            client=BurstTiledClient(uart)
            client.capabilities()
            with (output/'records.jsonl').open('x') as stream:
                for model in ('kws','vww'):
                    for sample in ('stress','pinned'):
                        name=plan['fixture_names'][model][sample]
                        folder=Path(plan['fixture_root'])/name
                        report['current']=f'{model}-{sample}-load';save()
                        schedule,loaded=screening.load_fixture(client,folder,plan['fixtures'][name])
                        loads[name]=loaded
                        for case in range(1 if sample=='stress' else 4):
                            report['current']=f'{model}-{sample}-{case}';save()
                            data=(folder/'input.bin').read_bytes()
                            expected=(folder/'output.bin').read_bytes()
                            row=physical.execute_verified_sample(client,schedule,data,expected,30)
                            row.update(variant=plan['variant'],model=model,fixture=name,
                                kind=('stress' if sample=='stress' else 'warmup' if case==0 else 'timed'),
                                repeat=case if sample=='stress' else max(0,case-1),
                                device_latency_ms=row['elapsed_cycles']/image['clock_hz']*1000,
                                input_sha256=priority.digest(data),
                                bitstream_sha256=image['sha256'],burst_bytes=BURST_BYTES,
                                load_seconds=loaded if case==0 else 0,at=screening.timestamp())
                            priority.append_row(stream,row);rows.append(row)
                            report['completed']=len(rows);save()
                    print(f'{screening.timestamp()} {model}: {len(rows)}/10 exact',flush=True)
        if priority.read_rows(output/'records.jsonl')!=rows or len(rows)!=10:
            raise ValueError('signed board record count or contents changed')
        summary={}
        for model in ('kws','vww'):
            timed=[r for r in rows if r['model']==model and r['kind']=='timed']
            if len(timed)!=3:raise ValueError(f'{model}: missing timed repeats')
            cycles=statistics.median(r['elapsed_cycles'] for r in timed)
            summary[model]=dict(median_cycles=cycles,
                median_device_latency_ms=cycles/image['clock_hz']*1000,
                device_inferences_per_second=image['clock_hz']/cycles,
                timed_cycles=[r['elapsed_cycles'] for r in timed],
                pinned_fixture_load_seconds=loads[plan['fixture_names'][model]['pinned']])
        screening.verify_files(ROOT,plan['dependency_sha256'])
        screening.verify_files(ROOT,plan['physical_dependencies'])
        screening.verify_files(ROOT,plan['source_coverage']['source_sha256'])
        screening.verify_files(ROOT,{image['file']:image['sha256']})
        for row in plan['fixtures'].values():
            screening.verify_files(Path(plan['fixture_root'])/row['label'],row['files'])
        if (sha(Path(__file__))!=plan['physical_runner_sha256'] or
                sha(ROOT/'tools/phase6/channel_compaction.py')!=plan['compacted_source_sha256'] or
                sha(SOURCE/'report.json')!=plan['compacted_report_sha256'] or
                sha(BASE/'report.json')!=plan['compacted_graph_sha256'] or
                sha(SELECTED/'route24/report.json')!=plan['selected_image_report_sha256']):
            raise ValueError('compaction or route evidence changed during physical run')
        report.update(status='passed-short-screen',summary=summary,
            records_sha256=sha(output/'records.jsonl'),finished_at=screening.timestamp())
        save()
        screening.save_json(output/'seal.json',dict(report_sha256=sha(output/'report.json'),
            plan_sha256=sha(output/'plan.json'),records_sha256=sha(output/'records.jsonl'),
            runner_sha256=plan['physical_runner_sha256']))
        print(json.dumps(summary,sort_keys=True),flush=True)
    except BaseException as error:
        report['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
        report['failure']=repr(error)
        if (output/'records.jsonl').exists():
            report['records_sha256']=sha(output/'records.jsonl')
        save();raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--port',default='/dev/cu.usbserial-20250303171')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    plan=prepare()
    if args.run:
        if args.output is None:parser.error('--run requires a fresh --output directory')
        run(plan,args.output.resolve(),args.port)
    else:
        print(json.dumps(dict(status='preflight-passed',planned=plan['planned'],
            bitstream_sha256=plan['image']['sha256'],
            compacted_report_sha256=plan['compacted_report_sha256'],
            fixture_names=plan['fixture_names']),sort_keys=True),flush=True)


if __name__=='__main__':main()
