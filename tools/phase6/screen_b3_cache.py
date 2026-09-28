#!/usr/bin/env python3
"""Preflight, or explicitly compare VWW halo caching with selected recompute.

Both schedules run on the same frozen 27 MHz FPGA image. Each receives one
stress input, one warmup and three pinned timings. No board access by default.
This is a restricted mechanism ablation, not a full DeFiNES or G6 evaluation.
"""

import argparse
import fcntl
import json
from pathlib import Path
import random
import signal
import statistics
import struct
import subprocess
import sys
import time

sys.dont_write_bytecode = True

import serial
import matched_baselines_board as shared
import run_priority as priority
import run_screening as screening
import screen_engine_schedule as physical
import screen_pair7_fusion as selected
from uart_burst import BURST_BYTES, BurstTiledClient
from variants import ROOT, sha


BASE = ROOT / 'work/phase6/matched-baselines-v1'
MANIFEST = BASE / 'b3/report.json'
VARIANTS = ('recompute', 'vertical_cache')
CMD = struct.Struct('<BBHIII')
require = shared.require


def checked_symbolic(raw):
    """A status string alone does not establish independent command replay."""
    replay = raw.get('replay', {})
    require(replay.get('status') == 'passed', 'symbolic command replay has not passed')
    path = shared.path_in_repo(replay['report'])
    require(sha(path) == replay['report_sha256'], 'symbolic replay report changed')
    result = shared.json_file(path)
    require(result.get('status') == 'passed' and result.get('model') == 'vww' and
            result.get('replay', {}).get('status') == 'passed' and
            shared.path_in_repo(result['fixture']) == shared.path_in_repo(raw['directory']),
            'symbolic replay model/fixture/status differs')
    for name in ('commands.bin', 'payload.bin', 'input.bin', 'output.bin', 'schedule.json'):
        require(result.get('fixture_files_sha256', {}).get(name) == raw['files'][name],
                f'symbolic replay fixture differs: {name}')
    require(result.get('oracle_output_sha256') == raw['files']['output.bin'],
            'symbolic oracle output differs')
    return dict(report=str(path.relative_to(ROOT)), report_sha256=sha(path))


def traffic(directory):
    blob = (directory / 'commands.bin').read_bytes()
    commands = [CMD.unpack_from(blob, offset) for offset in range(0, len(blob), CMD.size)]
    require(all(op != 1 or flags in (0, 1) for op, flags, *_ in commands),
            'unknown DMA direction')
    return dict(command_count=len(commands),
        external_read_bytes=sum(c[5] for c in commands if c[0] == 1 and c[1] == 1),
        external_write_bytes=sum(c[5] for c in commands if c[0] == 1 and c[1] == 0),
        engine_dispatches=sum(c[0] == 2 for c in commands))


def prepare(manifest=MANIFEST, seed=20260928):
    reference = selected.prepare()
    require(reference['image']['sha256'] == shared.IMAGE_SHA and
            reference['image']['clock_hz'] == shared.CLOCK_HZ and BURST_BYTES == 256,
            'selected route/image/clock/UART differs')
    manifest = shared.path_in_repo(manifest)
    data = shared.json_file(manifest)
    require(data.get('schema') == 1 and data.get('status') == 'passed-native',
            'B3 manifest native checks are incomplete')
    require(data.get('engine_sha256') == sha(shared.ENGINE) and
            data.get('native_executable_sha256') == shared.NATIVE_SHA == sha(shared.NATIVE),
            'cache experiment engine/native executable differs')
    source_hashes = shared.checked_hash_tree(data.get('source_sha256'), 'B3 sources')
    require('compiler/scheduler/defines_verify.py' in source_hashes,
            'independent symbolic verifier source is not pinned')
    if 'model_source_sha256' in data:
        source_hashes.update(shared.checked_hash_tree(data['model_source_sha256'], 'B3 model sources'))
    cache = data['vertical_cache_candidate']
    inputs = {'recompute': data['policies']['B3_restricted']['models']['vww']['fixtures'],
              'vertical_cache': cache['fixtures']}
    variants = {}
    for variant, raw_fixtures in inputs.items():
        require({'pinned_timed', 'stress_check'} <= set(raw_fixtures),
                f'{variant}: timed and stress fixtures required')
        fixtures = {}
        for sample, old_sample in (('pinned_timed', 'pinned'), ('stress_check', 'stress')):
            raw = raw_fixtures[sample]
            name = reference['fixture_names']['vww'][old_sample]
            baseline = reference['fixtures'][name]
            if variant == 'recompute':
                require(all(raw['files'][key] == baseline['files'][key] for key in
                            ('commands.bin', 'payload.bin', 'input.bin', 'output.bin')),
                        'recompute control differs from selected B4')
            else:
                require(raw['files']['commands.bin'] != baseline['files']['commands.bin'],
                        'cache candidate is a duplicate of recompute')
            fixtures[sample] = shared.validate_fixture(raw, 'vww', sample, baseline)
            fixtures[sample]['symbolic_evidence'] = checked_symbolic(raw)
        variants[variant] = dict(fixtures=fixtures,
            timed_traffic=traffic(ROOT / fixtures['pinned_timed']['directory']))
    order = list(VARIANTS)
    random.Random(seed).shuffle(order)
    dependencies = (Path(__file__), ROOT / 'tools/phase6/matched_baselines_board.py',
                    ROOT / 'tools/phase6/run_priority.py', ROOT / 'tools/phase6/run_screening.py',
                    ROOT / 'tools/phase6/screen_engine_schedule.py', ROOT / 'tools/phase6/uart_burst.py',
                    ROOT / 'tools/phase4/tiled_host.py', ROOT / 'tools/phase2/host.py')
    return dict(schema=1, status='passed-preflight', manifest=str(manifest.relative_to(ROOT)),
        manifest_sha256=sha(manifest), image=reference['image'],
        native_executable_sha256=shared.NATIVE_SHA, engine_sha256=sha(shared.ENGINE),
        reference_plan_sha256=shared.canonical_sha(reference), source_sha256=source_hashes,
        runner_sources={str(p.relative_to(ROOT)): sha(p) for p in dependencies},
        variants=variants, block_order=order, seed=seed, planned=10,
        clock_hz=shared.CLOCK_HZ, baud=750000, burst_bytes=BURST_BYTES,
        source_halo_read_bytes_avoided=cache.get('saved_external_bytes'),
        additional_copy_descriptors=cache.get('new_copy_descriptors'),
        scope='VWW-only exact full-width source-halo caching ablation on one fixed image',
        not_claimed=['full DeFiNES reproduction', 'B03 closure', 'G6 closure', 'full accuracy',
                     'endurance', 'energy', 'SOTA'])


def summarize(rows):
    require(len(rows) == 10, 'ten board records required')
    summary = {}
    for variant in VARIANTS:
        group = [row for row in rows if row['variant'] == variant]
        require(sorted(row['kind'] for row in group) ==
                ['stress', 'timed', 'timed', 'timed', 'warmup'], 'unbalanced cache/recompute checks')
        require(all(row['bitstream_sha256'] == shared.IMAGE_SHA and
                    row['output_hex'] == row['expected_hex'] and
                    row['device_latency_ms'] == row['elapsed_cycles'] / shared.CLOCK_HZ * 1000
                    for row in group), 'physical exactness, clock or image differs')
        timed = [row for row in group if row['kind'] == 'timed']
        require(sorted(row['repeat'] for row in timed) == [0, 1, 2], 'timing repetitions differ')
        cycles = [row['elapsed_cycles'] for row in timed]
        median = statistics.median(cycles)
        summary[variant] = dict(median_cycles=median, timed_cycles=cycles,
            median_device_latency_ms=median / shared.CLOCK_HZ * 1000,
            frames_per_second=shared.CLOCK_HZ / median,
            spread_fraction=(max(cycles) - min(cycles)) / median,
            median_host_wall_seconds=statistics.median(row['wall_seconds'] for row in timed))
    return dict(variants=summary,
        cache_throughput_speedup=summary['recompute']['median_cycles'] /
                                summary['vertical_cache']['median_cycles'],
        cache_latency_change_fraction=summary['vertical_cache']['median_cycles'] /
                                      summary['recompute']['median_cycles'] - 1)


def run(plan, output, port):
    screening.require_board_free(port)
    require(not output.exists(), 'choose a fresh output directory; existing records are immutable')
    lock = (ROOT / 'work/phase6/physical-board.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report, rows, old_handlers = {}, [], {}
    def save():
        report['updated_at'] = screening.timestamp()
        screening.save_json(output / 'report.json', report)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}; completed records preserved')
    try:
        require(prepare(ROOT / plan['manifest'], plan['seed']) == plan,
                'cache evidence changed before programming')
        output.mkdir(parents=True)
        screening.save_json(output / 'plan.json', plan)
        report.update(schema=1, status='running', physical_board=True, planned=10, completed=0,
                      started_at=screening.timestamp(), plan_sha256=sha(output / 'plan.json'),
                      output_mismatches=0)
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, interrupted)
        save()
        with (output / 'program.log').open('x') as log:
            subprocess.run(['openFPGALoader', '-b', 'tangnano20k', '--ftdi-serial', '2025030317',
                            '--freq', '2500000', '-m', '-v', str(ROOT / plan['image']['file'])],
                           stdout=log, stderr=subprocess.STDOUT, timeout=90, check=True)
        report['program_log_sha256'] = sha(output / 'program.log')
        time.sleep(1)
        with serial.Serial(port, plan['baud'], timeout=5, write_timeout=5) as uart:
            uart.reset_input_buffer()
            client = BurstTiledClient(uart)
            client.capabilities()
            with (output / 'records.jsonl').open('x') as stream:
                for variant in plan['block_order']:
                    for sample in ('stress_check', 'pinned_timed'):
                        fixture = plan['variants'][variant]['fixtures'][sample]
                        folder = ROOT / fixture['directory']
                        report['current'] = f'{variant}/{sample}/load'; save()
                        schedule, loaded = screening.load_fixture(client, folder, fixture)
                        kinds = ('stress',) if sample == 'stress_check' else ('warmup', 'timed', 'timed', 'timed')
                        for index, kind in enumerate(kinds):
                            report['current'] = f'{variant}/{kind}/{index}'; save()
                            data = (folder / 'input.bin').read_bytes()
                            expected = (folder / 'output.bin').read_bytes()
                            row = physical.execute_verified_sample(client, schedule, data, expected, 30)
                            row.update(variant=variant, model='vww', sample=sample, kind=kind,
                                repeat=max(0, index - 1), fixture=fixture['directory'],
                                bitstream_sha256=shared.IMAGE_SHA, core_clock_hz=shared.CLOCK_HZ,
                                device_latency_ms=row['elapsed_cycles'] / shared.CLOCK_HZ * 1000,
                                input_sha256=fixture['files']['input.bin'],
                                commands_sha256=fixture['files']['commands.bin'],
                                payload_sha256=fixture['files']['payload.bin'],
                                burst_bytes=BURST_BYTES, load_seconds=loaded if index == 0 else 0,
                                at=screening.timestamp())
                            priority.append_row(stream, row); rows.append(row)
                            report['completed'] = len(rows); save()
                    print(f'{variant}: {len(rows)}/10 exact', flush=True)
        require(priority.read_rows(output / 'records.jsonl') == rows, 'signed physical records changed')
        require(prepare(ROOT / plan['manifest'], plan['seed']) == plan, 'cache evidence changed during run')
        report.update(status='passed-cache-short-screen', summary=summarize(rows),
                      records_sha256=sha(output / 'records.jsonl'), finished_at=screening.timestamp())
        save()
        screening.save_json(output / 'seal.json', dict(plan_sha256=sha(output / 'plan.json'),
            report_sha256=sha(output / 'report.json'), records_sha256=sha(output / 'records.jsonl'),
            runner_sha256=sha(Path(__file__))))
        return report
    except BaseException as error:
        if output.exists() and report:
            report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                          failure=repr(error))
            if isinstance(error, screening.OutputMismatch):
                report['output_mismatches'] += 1
            if (output / 'records.jsonl').exists():
                report['records_sha256'] = sha(output / 'records.jsonl')
            save()
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--output', type=Path, default=BASE / 'b3-cache-board')
    parser.add_argument('--port', default='/dev/cu.usbserial-20250303171')
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    plan = prepare(args.manifest, args.seed)
    if args.run:
        report = run(plan, shared.path_in_repo(args.output), args.port)
        print(json.dumps(dict(status=report['status'], summary=report['summary']), sort_keys=True))
    else:
        print(json.dumps(dict(status=plan['status'], planned=plan['planned'],
            clock_hz=plan['clock_hz'], block_order=plan['block_order'],
            plan_sha256=shared.canonical_sha(plan),
            traffic={name: value['timed_traffic'] for name, value in plan['variants'].items()}), sort_keys=True))


if __name__ == '__main__':
    main()
