#!/usr/bin/env python3
"""Strict boardless preflight and optional VWW x-tile physical comparison.

The full-width and 16+8-column programs use exactly the same frozen 27 MHz
image, model, arithmetic and device-cycle boundary. This is an executable
restricted DeFiNES-style point, not a complete DeFiNES reproduction.
"""
import argparse
import fcntl
import json
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode=True

import serial

import matched_baselines_board as shared
import matched_defines_x_diagnostics as diagnostics
import run_priority as priority
import run_screening as screening
import screen_engine_schedule as physical
import screen_pair7_fusion as selected
from uart_burst import BURST_BYTES,BurstTiledClient
from variants import ROOT,sha

BASE=ROOT/'work/phase6/matched-baselines-v1'
TIMED=BASE/'b3-x-tiles-v2'
DIAGNOSTIC=BASE/'b3-x-diagnostics-v1'
VARIANTS=('full_width','x_tiles')
require=shared.require


def checked_timed(raw,reference):
    fixture=TIMED/'fixture'
    files=raw['fixture_files_sha256']
    for name,digest in files.items():
        require(sha(fixture/name)==digest,f'timed x-tile fixture changed: {name}')
    require(raw['segment_replay']['status']=='passed' and
            raw['source_full_model_replay']['status']=='passed' and
            raw['splice']['command_count']==1471 and
            raw['segment']['max_copy_chain_descriptors']==256,
            'full-model compositional replay or ABI capacity evidence changed')
    native=[]
    for seed,row in zip((0,6063),raw['native']):
        report=TIMED/f'native-s{seed}.json'
        require(shared.json_file(report)==row and
                row['status']=='passed' and row['stall_seed']==seed and row['tensor_checks']==1,
                'timed native result changed')
        native.append(dict(row,report=str(report.relative_to(ROOT)),report_sha256=sha(report)))
    item=dict(directory=str(fixture.relative_to(ROOT)),files=files,
              replay=dict(status='passed',method='source full-model replay plus exact segment DMA/COPY replay and byte-identical splice'),
              native=native)
    return shared.validate_fixture(item,'vww','pinned_timed',reference)


def checked_diagnostic(raw,sample,reference):
    fixture=raw['directory']
    require(raw['replay']['status']=='passed' and
            raw['replay']['source']['status']=='passed' and
            raw['replay']['segment']['status']=='passed' and
            raw['splice']['command_count']==1483,
            f'{sample}: diagnostic replay or command count changed')
    item=shared.validate_fixture(raw,'vww',sample,reference)
    folder=ROOT/fixture
    checks=(folder/'checks.txt').read_text().splitlines()
    require(len(checks)==8 and
            all((folder/f'layer12-y{y}-x{x}.bin').stat().st_size==32*8*width
                for y in (0,8,16) for x,width in ((0,16),(16,8))) and
            (folder/'layer14-full.bin').stat().st_size==18432,
            f'{sample}: intermediate tile coverage incomplete')
    require(all(row['tensor_checks']==8 for row in raw['native']),
            f'{sample}: native tile checks incomplete')
    return item


def prepare(seed=20260928):
    """Read-only: validate source, fixtures, exactness and shared physical ABI."""
    reference=selected.prepare()
    require(reference['image']['sha256']==shared.IMAGE_SHA and
            reference['image']['clock_hz']==shared.CLOCK_HZ and
            BURST_BYTES==256 and sha(shared.NATIVE)==shared.NATIVE_SHA,
            'selected 27 MHz image/native/UART changed')
    timed_report=shared.json_file(TIMED/'report.json')
    diag_report=shared.json_file(DIAGNOSTIC/'report.json')
    require(timed_report.get('schema')==diag_report.get('schema')==1 and
            timed_report.get('status')==diag_report.get('status')=='passed-native' and
            timed_report['native_executable_sha256']==shared.NATIVE_SHA and
            diag_report['selected_native_executable_sha256']==shared.NATIVE_SHA and
            diag_report['timed_report_sha256']==sha(TIMED/'report.json') and
            timed_report['physical_board'] is False and diag_report['physical_board'] is False,
            'x-tile native or parent evidence changed')
    sources=shared.checked_hash_tree(diag_report['source_sha256'],'x-tile compiler sources')
    sources.update(shared.checked_hash_tree(diag_report['model_source_sha256'],'x-tile model sources'))
    require(timed_report['model_source_sha256']==diag_report['model_source_sha256'],
            'timed/diagnostic model identities differ')
    require(sha(ROOT/'tools/phase6/matched_defines_x_tiles.py')==timed_report['source_sha256'],
            'timed x-tile compiler source changed')
    for name,digest in timed_report['frozen_source_files_sha256'].items():
        require(sha(ROOT/timed_report['frozen_source_fixture']/name)==digest,
                f'frozen full-width source fixture changed: {name}')
    ref=shared.reference_policy(reference)['models']['vww']['fixtures']
    candidate={
        'pinned_timed':checked_timed(timed_report,ref['pinned_timed']),
        'pinned_check':checked_diagnostic(diag_report['fixtures']['pinned_check'],
                                          'pinned_check',ref['pinned_timed']),
        'stress_check':checked_diagnostic(diag_report['fixtures']['stress_check'],
                                          'stress_check',ref['stress_check']),
    }
    for name in ('input.bin','output.bin'):
        require(candidate['pinned_timed']['files'][name]==candidate['pinned_check']['files'][name],
                f'pinned x-tile {name} differs between timed/check')
    variants={
        'full_width':dict(description='selected full-width 12-row VWW layer11–14 cut',
                          fixtures={key:ref[key] for key in ('pinned_timed','stress_check')}),
        'x_tiles':dict(description='three 8-row bands, each 8x16+8x8 output tiles',
                       fixtures=candidate),
    }
    order=list(VARIANTS);random.Random(seed).shuffle(order)
    dependencies=(Path(__file__),ROOT/'tools/phase6/matched_baselines_board.py',
        ROOT/'tools/phase6/run_priority.py',ROOT/'tools/phase6/run_screening.py',
        ROOT/'tools/phase6/screen_engine_schedule.py',ROOT/'tools/phase6/uart_burst.py',
        ROOT/'tools/phase4/tiled_host.py',ROOT/'tools/phase2/host.py')
    return dict(schema=1,status='passed-preflight',physical_board=False,
        image=reference['image'],clock_hz=shared.CLOCK_HZ,baud=750000,
        burst_bytes=BURST_BYTES,seed=seed,planned=10,block_order=order,
        native_executable_sha256=shared.NATIVE_SHA,
        reference_plan_sha256=shared.canonical_sha(reference),
        manifests={str((TIMED/'report.json').relative_to(ROOT)):sha(TIMED/'report.json'),
                   str((DIAGNOSTIC/'report.json').relative_to(ROOT)):sha(DIAGNOSTIC/'report.json')},
        source_sha256=sources,
        runner_sources={str(p.resolve().relative_to(ROOT)):sha(p) for p in dependencies},
        variants=variants,
        scope='VWW x-tile versus selected full-width short physical screen',
        not_claimed=['complete DeFiNES adaptation','B03 closure','SOTA',
                     'full accuracy','endurance','energy'])


def summarize(rows):
    require(len(rows)==10,'ten board records required')
    out={}
    for variant in VARIANTS:
        group=[row for row in rows if row['variant']==variant]
        require(sorted(row['kind'] for row in group)==
                ['stress','timed','timed','timed','warmup'],
                f'{variant}: incomplete physical coverage')
        require(all(row['bitstream_sha256']==shared.IMAGE_SHA and
                    row['core_clock_hz']==shared.CLOCK_HZ and
                    row['output_hex']==row['expected_hex'] and
                    row['device_latency_ms']==row['elapsed_cycles']/shared.CLOCK_HZ*1000
                    for row in group),f'{variant}: clock, image, or logits differ')
        timed=[row for row in group if row['kind']=='timed']
        require(sorted(row['repeat'] for row in timed)==[0,1,2],
                f'{variant}: timed repetition incomplete')
        cycles=[row['elapsed_cycles'] for row in timed]
        median=statistics.median(cycles)
        out[variant]=dict(median_cycles=median,timed_cycles=cycles,
            median_device_latency_ms=median/shared.CLOCK_HZ*1000,
            frames_per_second=shared.CLOCK_HZ/median,
            spread_fraction=(max(cycles)-min(cycles))/median)
    return dict(variants=out,
        full_width_over_x_tiles_speedup=out['x_tiles']['median_cycles']/out['full_width']['median_cycles'])


def run(plan,output,port):
    screening.require_board_free(port)
    require(not output.exists(),'choose a fresh output; prior evidence is immutable')
    lock=(ROOT/'work/phase6/physical-board.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows,report,handlers=[],{},{}
    def save():
        report['updated_at']=screening.timestamp()
        screening.save_json(output/'report.json',report)
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f'signal {signum}; completed records preserved')
    try:
        require(prepare(plan['seed'])==plan,'preflight changed before board lock')
        output.mkdir(parents=True)
        screening.save_json(output/'plan.json',plan)
        report.update(schema=1,status='running',physical_board=True,planned=10,
            completed=0,started_at=screening.timestamp(),
            plan_sha256=sha(output/'plan.json'),output_mismatches=0)
        for signum in (signal.SIGINT,signal.SIGTERM):
            handlers[signum]=signal.signal(signum,interrupted)
        save()
        with (output/'program.log').open('x') as log:
            subprocess.run(['openFPGALoader','-b','tangnano20k','--ftdi-serial',
                '2025030317','--freq','2500000','-m','-v',
                str(ROOT/plan['image']['file'])],stdout=log,
                stderr=subprocess.STDOUT,timeout=90,check=True)
        report['program_log_sha256']=sha(output/'program.log')
        time.sleep(1)
        with serial.Serial(port,plan['baud'],timeout=5,write_timeout=5) as uart:
            uart.reset_input_buffer();client=BurstTiledClient(uart);client.capabilities()
            with (output/'records.jsonl').open('x') as stream:
                for variant in plan['block_order']:
                    for sample in ('stress_check','pinned_timed'):
                        fixture=plan['variants'][variant]['fixtures'][sample]
                        folder=ROOT/fixture['directory']
                        report['current']=f'{variant}/{sample}/load';save()
                        schedule,loaded=screening.load_fixture(client,folder,fixture)
                        kinds=('stress',) if sample=='stress_check' else (
                            'warmup','timed','timed','timed')
                        for index,kind in enumerate(kinds):
                            report['current']=f'{variant}/{kind}/{index}';save()
                            data=(folder/'input.bin').read_bytes()
                            expected=(folder/'output.bin').read_bytes()
                            row=physical.execute_verified_sample(client,schedule,data,expected,30)
                            row.update(variant=variant,model='vww',sample=sample,kind=kind,
                                repeat=max(0,index-1),fixture=fixture['directory'],
                                bitstream_sha256=shared.IMAGE_SHA,core_clock_hz=shared.CLOCK_HZ,
                                device_latency_ms=row['elapsed_cycles']/shared.CLOCK_HZ*1000,
                                input_sha256=fixture['files']['input.bin'],
                                commands_sha256=fixture['files']['commands.bin'],
                                payload_sha256=fixture['files']['payload.bin'],
                                burst_bytes=BURST_BYTES,load_seconds=loaded if index==0 else 0,
                                at=screening.timestamp())
                            priority.append_row(stream,row);rows.append(row)
                            report['completed']=len(rows);save()
                    print(f'{variant}: {len(rows)}/10 exact',flush=True)
        require(priority.read_rows(output/'records.jsonl')==rows,
                'signed physical records changed')
        require(prepare(plan['seed'])==plan,'preflight changed during board run')
        report.update(status='passed-x-tile-short-screen',summary=summarize(rows),
            records_sha256=sha(output/'records.jsonl'),finished_at=screening.timestamp())
        save()
        screening.save_json(output/'seal.json',dict(plan_sha256=sha(output/'plan.json'),
            report_sha256=sha(output/'report.json'),
            records_sha256=sha(output/'records.jsonl'),
            runner_sha256=sha(Path(__file__))))
        return report
    except BaseException as error:
        if output.exists() and report:
            report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                failure=repr(error))
            if isinstance(error,screening.OutputMismatch):report['output_mismatches']+=1
            if (output/'records.jsonl').exists():report['records_sha256']=sha(output/'records.jsonl')
            save()
        raise
    finally:
        for signum,handler in handlers.items():signal.signal(signum,handler)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE/'b3-x-board')
    parser.add_argument('--port',default='/dev/cu.usbserial-20250303171')
    parser.add_argument('--seed',type=int,default=20260928)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    plan=prepare(args.seed)
    if args.run:
        result=run(plan,shared.path_in_repo(args.output),args.port)
        print(json.dumps(dict(status=result['status'],summary=result['summary']),sort_keys=True))
    else:
        print(json.dumps(dict(status=plan['status'],planned=plan['planned'],
            clock_hz=plan['clock_hz'],image_sha256=plan['image']['sha256'],
            block_order=plan['block_order'],plan_sha256=shared.canonical_sha(plan)),sort_keys=True))


if __name__=='__main__':main()
