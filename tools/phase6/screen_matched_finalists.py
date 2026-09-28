#!/usr/bin/env python3
"""One-flash VWW finalist campaign from frozen strict preflight plans.

The read-only default composes the B1/B2 tuning, B3 spatial, and B3 cache
preflights. Byte-identical B4/recompute fixtures run once and carry both
labels. The optional x-tile plan adds one distinct schedule and another
byte-identical B4 control. --run uses one JTAG programming operation and one
UART session for one stress, one warmup, and three timed inferences per
distinct schedule.
"""
from __future__ import annotations

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
import run_priority as priority
import run_screening as screening
import screen_engine_schedule as physical
import screen_b1b2_vww_tuning as b1b2
import screen_b3_spatial as spatial
import screen_b3_cache as cache
from uart_burst import BURST_BYTES, BurstTiledClient
from variants import ROOT, sha


BASE=ROOT/'work/phase6/matched-baselines-v1'
DEFAULT_OUTPUT=BASE/'combined-finalists-board'
ROLES=('pinned_timed','stress_check')
IDENTITY_FILES=('commands.bin','payload.bin','input.bin','output.bin')
require=shared.require


def fixture_signature(fixtures: dict) -> tuple:
    """Capture exactly the program/input/output consumed in both sample roles."""
    require(set(ROLES)<=set(fixtures), 'finalist lacks timed or stress fixture')
    result=[]
    for role in ROLES:
        row=fixtures[role]
        require(set(IDENTITY_FILES)<=set(row['files']), f'{role}: incomplete fixture identity')
        result.extend(row['files'][name] for name in IDENTITY_FILES)
        result.extend((row['final_output']['ext'],row['final_output']['bytes'],row['command_count']))
    return tuple(result)


def prepare(tuning_manifest=b1b2.MANIFEST, spatial_manifest=spatial.MANIFEST,
            cache_manifest=cache.MANIFEST, seed=20260928, x_manifest=None):
    """Compose existing strict gates without touching JTAG, UART or output files."""
    source_paths={
        'b1b2_tuning':shared.path_in_repo(tuning_manifest),
        'b3_spatial':shared.path_in_repo(spatial_manifest),
        'b3_cache':shared.path_in_repo(cache_manifest),
    }
    source_plans={
        'b1b2_tuning':b1b2.prepare(source_paths['b1b2_tuning'],seed),
        'b3_spatial':spatial.prepare(source_paths['b3_spatial'],seed),
        'b3_cache':cache.prepare(source_paths['b3_cache'],seed),
    }
    if x_manifest is not None:
        import screen_b3_x_tiles as x_tiles
        x_path=shared.path_in_repo(x_manifest)
        require(x_path==x_tiles.TIMED/'report.json',
                'x-tile preflight requires its frozen timed report')
        source_paths['b3_x_tiles']=x_path
        source_plans['b3_x_tiles']=x_tiles.prepare(seed)
    for label,plan in source_plans.items():
        require(plan['status']=='passed-preflight' and plan['clock_hz']==shared.CLOCK_HZ and
                plan['baud']==750000 and plan['burst_bytes']==BURST_BYTES and
                plan['image']['sha256']==shared.IMAGE_SHA and
                plan['native_executable_sha256']==shared.NATIVE_SHA,
                f'{label}: selected image/engine/clock/UART changed')
        if label=='b3_x_tiles':
            # The x-tile gate revalidates the selected reference engine and both
            # frozen reports, but its plan does not expose engine_sha256 itself.
            require(sha(shared.ENGINE)==source_plans['b3_spatial']['engine_sha256'] and
                    plan['manifests'][str(source_paths[label].relative_to(ROOT))]==
                    sha(source_paths[label]),
                    'x-tile reference engine or source manifest changed')
        else:
            require(plan['engine_sha256']==sha(shared.ENGINE) and
                    plan['manifest_sha256']==sha(source_paths[label]),
                    f'{label}: source manifest changed')
    reference_image=source_plans['b3_spatial']['image']
    require(all(plan['image']==reference_image for plan in source_plans.values()),
            'preflight plans disagree on bitstream identity')
    labelled={}
    labels=[
        ('b1b2_tuning','B1_selected'),('b1b2_tuning','B1_full_prefetch'),
        ('b1b2_tuning','B2_selected'),('b1b2_tuning','B2_half_prefetch'),
        ('b3_spatial','B4'),('b3_spatial','h8'),('b3_spatial','h12'),
        ('b3_cache','recompute'),('b3_cache','vertical_cache')]
    if x_manifest is not None:
        labels.extend((('b3_x_tiles','full_width'),('b3_x_tiles','x_tiles')))
    for label,name in labels:
        row=source_plans[label]['variants'][name]
        labelled[name]=dict(origin_plan=label,fixtures=row['fixtures'],
                            signature=fixture_signature(row['fixtures']))
    signatures={}
    variants={}
    aliases={}
    for name,row in labelled.items():
        signature=row['signature']
        if signature not in signatures:
            signatures[signature]=name
            variants[name]=dict(fixtures=row['fixtures'],source_labels=[name],
                                source_plans=[row['origin_plan']],
                                signature_sha256=shared.canonical_sha(signature))
        else:
            canonical=signatures[signature]
            variants[canonical]['source_labels'].append(name)
            variants[canonical]['source_plans'].append(row['origin_plan'])
        aliases[name]=signatures[signature]
    require(aliases['B4']==aliases['recompute']=='B4',
            'recompute is no longer byte-identical to selected B4 for both samples')
    if x_manifest is not None:
        require(aliases['full_width']=='B4',
                'x-tile full-width control is no longer byte-identical to selected B4')
    expected={'B1_selected','B1_full_prefetch','B2_selected',
              'B2_half_prefetch','B4','h8','h12','vertical_cache'}
    if x_manifest is not None:expected.add('x_tiles')
    require(set(variants)==expected,
            'new duplicate or missing finalist requires a new physical plan')
    require(all(fixtures['pinned_timed']['files']['input.bin']==
                variants['B4']['fixtures']['pinned_timed']['files']['input.bin'] and
                fixtures['pinned_timed']['files']['output.bin']==
                variants['B4']['fixtures']['pinned_timed']['files']['output.bin'] and
                fixtures['stress_check']['files']['input.bin']==
                variants['B4']['fixtures']['stress_check']['files']['input.bin'] and
                fixtures['stress_check']['files']['output.bin']==
                variants['B4']['fixtures']['stress_check']['files']['output.bin']
                for fixtures in (row['fixtures'] for row in variants.values())),
            'finalists no longer share pinned/stress VWW inputs and logits')
    order=list(variants)
    random.Random(seed).shuffle(order)
    dependencies=(Path(__file__),ROOT/'tools/phase6/screen_b1b2_vww_tuning.py',
        ROOT/'tools/phase6/screen_b3_spatial.py',ROOT/'tools/phase6/screen_b3_cache.py',
        ROOT/'tools/phase6/matched_baselines_board.py',
        ROOT/'tools/phase6/run_priority.py',ROOT/'tools/phase6/run_screening.py',
        ROOT/'tools/phase6/screen_engine_schedule.py',ROOT/'tools/phase6/uart_burst.py',
        ROOT/'tools/phase4/tiled_host.py',ROOT/'tools/phase2/host.py')
    if x_manifest is not None:
        dependencies+=(ROOT/'tools/phase6/screen_b3_x_tiles.py',)
    return dict(schema=1,status='passed-preflight',physical_board=False,
        source_manifests={label:dict(path=str(path.relative_to(ROOT)),sha256=sha(path))
                          for label,path in source_paths.items()},
        source_plan_sha256={label:shared.canonical_sha(plan)
                            for label,plan in source_plans.items()},
        image=reference_image,engine_sha256=sha(shared.ENGINE),
        native_executable_sha256=shared.NATIVE_SHA,
        runner_sources={str(path.relative_to(ROOT)):sha(path) for path in dependencies},
        variants=variants,aliases=aliases,block_order=order,seed=seed,
        planned=5*len(order),clock_hz=shared.CLOCK_HZ,baud=750000,
        burst_bytes=BURST_BYTES,programming_operations=1,uart_sessions=1,
        scope=f'one-flash exact VWW finalist comparison across {len(source_plans)} frozen plans',
        not_claimed=['full DeFiNES reproduction','B03 closure','G6 closure',
                     'complete-set accuracy','energy','SOTA'])


def reprepare(plan):
    """Recheck the exact manifests recorded in a signed run plan."""
    sources=plan['source_manifests']
    base={'b1b2_tuning','b3_spatial','b3_cache'}
    require(set(sources) in (base,base|{'b3_x_tiles'}),
            'combined source manifest set changed')
    return prepare(
        tuning_manifest=ROOT/sources['b1b2_tuning']['path'],
        spatial_manifest=ROOT/sources['b3_spatial']['path'],
        cache_manifest=ROOT/sources['b3_cache']['path'],
        seed=plan['seed'],
        x_manifest=ROOT/sources['b3_x_tiles']['path'] if 'b3_x_tiles' in sources else None)


def summarize(rows, plan):
    require(len(rows)==plan['planned'], 'incomplete combined finalist campaign')
    outcomes={}
    for name in plan['variants']:
        group=[row for row in rows if row['variant']==name]
        require(sorted(row['kind'] for row in group)==
                ['stress','timed','timed','timed','warmup'],
                f'{name}: unbalanced stress/warmup/timings')
        require(all(row['bitstream_sha256']==shared.IMAGE_SHA and
                    row['core_clock_hz']==shared.CLOCK_HZ and
                    row['output_hex']==row['expected_hex'] and
                    row['device_latency_ms']==row['elapsed_cycles']/shared.CLOCK_HZ*1000
                    for row in group),f'{name}: physical identity or logits changed')
        timed=[row for row in group if row['kind']=='timed']
        require(sorted(row['repeat'] for row in timed)==[0,1,2],
                f'{name}: timing repeats incomplete')
        cycles=[row['elapsed_cycles'] for row in timed]
        median=statistics.median(cycles)
        outcomes[name]=dict(median_cycles=median,timed_cycles=cycles,
            median_device_latency_ms=median/shared.CLOCK_HZ*1000,
            frames_per_second=shared.CLOCK_HZ/median,
            spread_fraction=(max(cycles)-min(cycles))/median,
            source_labels=plan['variants'][name]['source_labels'])
    ranking=sorted(outcomes,key=lambda name:(outcomes[name]['median_cycles'],name))
    b4=outcomes['B4']['median_cycles']
    families={
        'B1':('B1_selected','B1_full_prefetch'),
        'B2':('B2_selected','B2_half_prefetch'),
        'pair11_height':('B4','h8','h12'),
        'source_halo_cache':('B4','vertical_cache'),
    }
    if 'x_tiles' in outcomes:families['spatial_x']=('B4','x_tiles')
    return dict(variants=outcomes,ranking=ranking,
        fastest=ranking[0],fastest_by_family={family:min(names,key=lambda name:
            (outcomes[name]['median_cycles'],name)) for family,names in families.items()},
        speedup_vs_B4={name:b4/row['median_cycles'] for name,row in outcomes.items()},
        aliases=plan['aliases'])


def run(plan, output, port):
    screening.require_board_free(port)
    require(not output.exists(), 'choose a fresh output directory; prior records are immutable')
    lock=(ROOT/'work/phase6/physical-board.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows=[];report={};old_handlers={}
    def save():
        report['updated_at']=screening.timestamp()
        screening.save_json(output/'report.json',report)
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f'signal {signum}; completed records preserved')
    try:
        require(reprepare(plan)==plan,
                'combined evidence changed before board use')
        output.mkdir(parents=True)
        screening.save_json(output/'plan.json',plan)
        report.update(schema=1,status='running',physical_board=True,
            planned=plan['planned'],completed=0,programming_operations=0,uart_sessions=0,
            started_at=screening.timestamp(),plan_sha256=sha(output/'plan.json'),
            output_mismatches=0)
        for signum in (signal.SIGINT,signal.SIGTERM):
            old_handlers[signum]=signal.signal(signum,interrupted)
        save()
        with (output/'program.log').open('x') as log:
            subprocess.run(['openFPGALoader','-b','tangnano20k','--ftdi-serial','2025030317',
                            '--freq','2500000','-m','-v',str(ROOT/plan['image']['file'])],
                           stdout=log,stderr=subprocess.STDOUT,timeout=90,check=True)
        report['programming_operations']=1
        report['program_log_sha256']=sha(output/'program.log')
        save()
        time.sleep(1)
        with serial.Serial(port,plan['baud'],timeout=5,write_timeout=5) as uart:
            report['uart_sessions']=1;save()
            uart.reset_input_buffer()
            client=BurstTiledClient(uart)
            client.capabilities()
            with (output/'records.jsonl').open('x') as stream:
                for name in plan['block_order']:
                    for sample in ('stress_check','pinned_timed'):
                        fixture=plan['variants'][name]['fixtures'][sample]
                        folder=ROOT/fixture['directory']
                        report['current']=f'{name}/{sample}/load';save()
                        schedule,loaded=screening.load_fixture(client,folder,fixture)
                        kinds=('stress',) if sample=='stress_check' else (
                            'warmup','timed','timed','timed')
                        for index,kind in enumerate(kinds):
                            report['current']=f'{name}/{kind}/{index}';save()
                            data=(folder/'input.bin').read_bytes()
                            expected=(folder/'output.bin').read_bytes()
                            row=physical.execute_verified_sample(client,schedule,data,expected,30)
                            row.update(variant=name,source_labels=plan['variants'][name]['source_labels'],
                                model='vww',sample=sample,kind=kind,repeat=max(0,index-1),
                                fixture=fixture['directory'],bitstream_sha256=shared.IMAGE_SHA,
                                core_clock_hz=shared.CLOCK_HZ,
                                device_latency_ms=row['elapsed_cycles']/shared.CLOCK_HZ*1000,
                                input_sha256=fixture['files']['input.bin'],
                                commands_sha256=fixture['files']['commands.bin'],
                                payload_sha256=fixture['files']['payload.bin'],
                                burst_bytes=BURST_BYTES,load_seconds=loaded if index==0 else 0,
                                at=screening.timestamp())
                            priority.append_row(stream,row);rows.append(row)
                            report['completed']=len(rows);save()
                    print(f'{name}: {len(rows)}/{plan["planned"]} exact',flush=True)
        require(priority.read_rows(output/'records.jsonl')==rows,
                'signed physical records changed')
        require(reprepare(plan)==plan,
                'combined evidence changed during campaign')
        report.update(status='passed-combined-finalists',summary=summarize(rows,plan),
            records_sha256=sha(output/'records.jsonl'),finished_at=screening.timestamp())
        save()
        screening.save_json(output/'seal.json',dict(plan_sha256=sha(output/'plan.json'),
            report_sha256=sha(output/'report.json'),records_sha256=sha(output/'records.jsonl'),
            runner_sha256=sha(Path(__file__))))
        return report
    except BaseException as error:
        if output.exists() and report:
            report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                          failure=repr(error))
            if isinstance(error,screening.OutputMismatch):report['output_mismatches']+=1
            if (output/'records.jsonl').exists():
                report['records_sha256']=sha(output/'records.jsonl')
            save()
        raise
    finally:
        for signum,handler in old_handlers.items():signal.signal(signum,handler)
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tuning-manifest',type=Path,default=b1b2.MANIFEST)
    parser.add_argument('--spatial-manifest',type=Path,default=spatial.MANIFEST)
    parser.add_argument('--cache-manifest',type=Path,default=cache.MANIFEST)
    parser.add_argument('--x-manifest',type=Path,help='include frozen B3 x-tile timed report')
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--port',default='/dev/cu.usbserial-20250303171')
    parser.add_argument('--seed',type=int,default=20260928)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    plan=prepare(args.tuning_manifest,args.spatial_manifest,args.cache_manifest,args.seed,
                 args.x_manifest)
    if args.run:
        result=run(plan,shared.path_in_repo(args.output),args.port)
        print(json.dumps(dict(status=result['status'],summary=result['summary']),sort_keys=True))
    else:
        print(json.dumps(dict(status=plan['status'],planned=plan['planned'],
            unique_variants=list(plan['variants']),aliases=plan['aliases'],
            block_order=plan['block_order'],image_sha256=plan['image']['sha256'],
            plan_sha256=shared.canonical_sha(plan)),sort_keys=True))


if __name__=='__main__':main()
