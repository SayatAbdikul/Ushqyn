#!/usr/bin/env python3
"""Check whether native-tuned B1/B2 VWW tile winners also win on the FPGA.

The frozen 27 MHz image, compiler report, and original board campaign are
read-only. Generation creates diagnostic stress fixtures for the two unmeasured
finalists. Preflight opens no board; --run performs a separate 20-run campaign.
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

sys.dont_write_bytecode = True

import serial

import matched_b1b2 as baseline
import matched_baselines_board as shared
import run_priority as priority
import run_screening as screening
import screen_engine_schedule as physical
from uart_burst import BURST_BYTES, BurstTiledClient
from variants import ROOT, sha


BASE = ROOT / 'work/phase6/matched-baselines-v1'
BASELINE_REPORT = BASE / 'b1b2/report.json'
EXPERIMENT = BASE / 'b1b2-vww-tuning'
MANIFEST = EXPERIMENT / 'report.json'
VARIANTS = ('B1_selected', 'B1_full_prefetch', 'B2_selected', 'B2_half_prefetch')
ALTERNATIVES = {
    'B1_full_prefetch': ('B1', False, True),
    'B2_half_prefetch': ('B2', True, True),
}
require = shared.require


def candidate(report, policy, half, prefetch):
    rows = report['policies'][policy]['models']['vww']['tuning_candidates']
    found = [row for row in rows if row.get('half') is half and
             row.get('prefetch') is prefetch]
    require(len(rows) == 4 and len(found) == 1 and found[0]['status'] == 'passed-native',
            f'{policy}: frozen four-point tuning candidate missing')
    return found[0]


def generate(output=EXPERIMENT):
    """Create only new stress fixtures and their native evidence."""
    require(not output.exists(), 'choose a fresh experiment directory')
    report = shared.json_file(BASELINE_REPORT)
    require(baseline.validate_report(report), 'frozen B1/B2 report failed validation')
    original, pinned, compacted, maps, _, _, _ = baseline.model('vww')
    value = baseline.sample_value('vww', 'stress', pinned)
    oracle = baseline.check_oracles(original, compacted, maps, value)
    manifest = dict(schema=1, status='running', physical_board=False,
        baseline_report_sha256=sha(BASELINE_REPORT),
        image_bitstream_sha256=report['image_bitstream_sha256'],
        native_executable_sha256=report['native_executable_sha256'],
        source_sha256={str(Path(__file__).resolve().relative_to(ROOT)):
                       sha(Path(__file__))},
        alternatives={})
    output.mkdir(parents=True)
    for variant, (policy, half, prefetch) in ALTERNATIVES.items():
        frozen = candidate(report, policy, half, prefetch)
        directory = output / 'fixtures' / f'{variant.lower()}-stress_check'
        stress = baseline.fused_fixture('vww', compacted, original, pinned, maps,
            sample='stress', snapshots=True, policy=policy, half=half,
            prefetch=prefetch, directory=directory, value=value, oracle=oracle)
        require(stress['seed_replay']['status'] == stress['replay']['status'] == 'passed',
                f'{variant}: stress command replay failed')
        stress['native'] = [baseline.measure(directory, seed) for seed in (0, 6063)]
        stress['selected_b4_identity'] = baseline.compare_selected_b4('vww', 'stress_check', stress)
        manifest['alternatives'][variant] = dict(policy=policy, half=half,
            prefetch=prefetch, pinned_candidate_files=frozen['files'],
            stress_check=stress)
        baseline.save(output / 'report.json', manifest)
    manifest['status'] = 'passed-native'
    baseline.save(output / 'report.json', manifest)
    prepare(output / 'report.json')
    return manifest


def prepare(manifest=MANIFEST, seed=20260928):
    """Read-only evidence gate, with no JTAG/UART access."""
    manifest = shared.path_in_repo(manifest)
    base_report = shared.json_file(BASELINE_REPORT)
    require(baseline.validate_report(base_report), 'frozen B1/B2 evidence changed')
    base_plan = shared.prepare([BASELINE_REPORT], seed)
    require(base_plan['image']['sha256'] == shared.IMAGE_SHA and
            base_plan['clock_hz'] == shared.CLOCK_HZ and BURST_BYTES == 256,
            'routed image, clock, or UART protocol changed')
    data = shared.json_file(manifest)
    require(data.get('schema') == 1 and data.get('status') == 'passed-native' and
            data.get('physical_board') is False and
            data.get('baseline_report_sha256') == sha(BASELINE_REPORT) and
            data.get('image_bitstream_sha256') == shared.IMAGE_SHA and
            data.get('native_executable_sha256') == shared.NATIVE_SHA,
            'alternative evidence or shared hardware identity changed')
    sources = shared.checked_hash_tree(data.get('source_sha256'), 'VWW tuning sources')
    require(str(Path(__file__).resolve().relative_to(ROOT)) in sources,
            'experiment generator/runner source not pinned')
    require(set(data.get('alternatives', {})) == set(ALTERNATIVES),
            'VWW finalist set changed')
    reference = base_plan['policies']['B4']['models']['vww']['fixtures']
    variants = {}
    for policy in ('B1', 'B2'):
        name = f'{policy}_selected'
        current = base_plan['policies'][policy]['models']['vww']['fixtures']
        variants[name] = dict(policy=policy, source='frozen selected winner',
            fixtures={key: current[key] for key in ('pinned_timed', 'stress_check')})
    for name, (policy, half, prefetch) in ALTERNATIVES.items():
        frozen = candidate(base_report, policy, half, prefetch)
        raw = data['alternatives'][name]
        require((raw['policy'], raw['half'], raw['prefetch']) ==
                (policy, half, prefetch) and
                raw['pinned_candidate_files'] == frozen['files'],
                f'{name}: alternative differs from frozen native candidate')
        timed = dict(frozen, native=[frozen['native'][str(x)] for x in (0, 6063)])
        stress = raw['stress_check']
        require(stress['policy'] == policy and stress['half'] is half and
                stress['prefetch'] is prefetch and stress['snapshots'] is True and
                stress['sample'] == 'stress' and
                stress['seed_replay']['status'] == 'passed',
                f'{name}: stress fixture does not use the declared candidate')
        fixtures = {
            'pinned_timed': shared.validate_fixture(timed, 'vww', 'pinned_timed',
                                                     reference['pinned_timed']),
            'stress_check': shared.validate_fixture(stress, 'vww', 'stress_check',
                                                    reference['stress_check']),
        }
        variants[name] = dict(policy=policy, source='frozen native candidate',
            fixtures=fixtures, native_worst_cycles=frozen['cycle_objective'])
    order = list(VARIANTS)
    rng = random.Random(seed)
    for _ in range(100):
        rng.shuffle(order)
        if all(variants[a]['policy'] != variants[b]['policy']
               for a, b in zip(order, order[1:])):
            break
    else:
        raise ValueError('cannot interleave B1/B2 finalist blocks')
    dependencies = (Path(__file__), ROOT / 'tools/phase6/matched_baselines_board.py',
        ROOT / 'tools/phase6/run_priority.py', ROOT / 'tools/phase6/run_screening.py',
        ROOT / 'tools/phase6/screen_engine_schedule.py', ROOT / 'tools/phase6/uart_burst.py',
        ROOT / 'tools/phase4/tiled_host.py', ROOT / 'tools/phase2/host.py')
    return dict(schema=1, status='passed-preflight', physical_board=False,
        manifest=str(manifest.relative_to(ROOT)), manifest_sha256=sha(manifest),
        baseline_report_sha256=sha(BASELINE_REPORT),
        selected_reference_plan_sha256=base_plan['selected_reference_plan_sha256'],
        image=base_plan['image'], engine_sha256=base_plan['engine_sha256'],
        native_executable_sha256=base_plan['native_executable_sha256'],
        source_sha256={**base_plan['source_sha256'], **sources},
        runner_sources={str(p.resolve().relative_to(ROOT)): sha(p) for p in dependencies},
        variants=variants, block_order=order, seed=seed, planned=20,
        clock_hz=shared.CLOCK_HZ, baud=750000, burst_bytes=BURST_BYTES,
        scope='VWW-only FPGA ranking check for four frozen B1/B2 tile finalists',
        not_claimed=['full grid physical tuning', 'full DeFiNES reproduction',
                     'accuracy', 'energy', 'SOTA'])


def summarize(rows):
    require(len(rows) == 20, 'all twenty exact board records are required')
    out = {}
    for name in VARIANTS:
        group = [row for row in rows if row['variant'] == name]
        require(sorted(row['kind'] for row in group) ==
                ['stress', 'timed', 'timed', 'timed', 'warmup'],
                f'{name}: incomplete physical coverage')
        require(all(row['bitstream_sha256'] == shared.IMAGE_SHA and
                    row['core_clock_hz'] == shared.CLOCK_HZ and
                    row['output_hex'] == row['expected_hex'] and
                    row['device_latency_ms'] == row['elapsed_cycles'] / shared.CLOCK_HZ * 1000
                    for row in group), f'{name}: image, clock, or logits changed')
        timed = [row for row in group if row['kind'] == 'timed']
        require(sorted(row['repeat'] for row in timed) == [0, 1, 2],
                f'{name}: timed repeats incomplete')
        cycles = [row['elapsed_cycles'] for row in timed]
        median = statistics.median(cycles)
        out[name] = dict(median_cycles=median, timed_cycles=cycles,
            median_device_latency_ms=median / shared.CLOCK_HZ * 1000,
            frames_per_second=shared.CLOCK_HZ / median,
            spread_fraction=(max(cycles)-min(cycles))/median)
    return dict(variants=out,
        physically_fastest={policy: min((f'{policy}_selected',
            'B1_full_prefetch' if policy == 'B1' else 'B2_half_prefetch'),
            key=lambda name: out[name]['median_cycles']) for policy in ('B1', 'B2')})


def run(plan, output, port):
    screening.require_board_free(port)
    require(not output.exists(), 'choose a fresh output directory; prior evidence is immutable')
    lock = (ROOT / 'work/phase6/physical-board.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows, report, old_handlers = [], {}, {}
    def save():
        report['updated_at'] = screening.timestamp()
        screening.save_json(output / 'report.json', report)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}; completed records preserved')
    try:
        require(prepare(ROOT / plan['manifest'], plan['seed']) == plan,
                'preflight evidence changed before board lock')
        output.mkdir(parents=True)
        screening.save_json(output / 'plan.json', plan)
        report.update(schema=1, status='running', physical_board=True, planned=20,
            completed=0, started_at=screening.timestamp(),
            plan_sha256=sha(output / 'plan.json'), output_mismatches=0)
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, interrupted)
        save()
        with (output / 'program.log').open('x') as log:
            subprocess.run(['openFPGALoader', '-b', 'tangnano20k', '--ftdi-serial',
                '2025030317', '--freq', '2500000', '-m', '-v',
                str(ROOT / plan['image']['file'])], stdout=log,
                stderr=subprocess.STDOUT, timeout=90, check=True)
        report['program_log_sha256'] = sha(output / 'program.log')
        time.sleep(1)
        with serial.Serial(port, plan['baud'], timeout=5, write_timeout=5) as uart:
            uart.reset_input_buffer()
            client = BurstTiledClient(uart)
            client.capabilities()
            with (output / 'records.jsonl').open('x') as stream:
                for name in plan['block_order']:
                    for sample in ('stress_check', 'pinned_timed'):
                        fixture = plan['variants'][name]['fixtures'][sample]
                        folder = ROOT / fixture['directory']
                        report['current'] = f'{name}/{sample}/load'; save()
                        schedule, loaded = screening.load_fixture(client, folder, fixture)
                        kinds = ('stress',) if sample == 'stress_check' else (
                            'warmup', 'timed', 'timed', 'timed')
                        for index, kind in enumerate(kinds):
                            report['current'] = f'{name}/{kind}/{index}'; save()
                            data = (folder / 'input.bin').read_bytes()
                            expected = (folder / 'output.bin').read_bytes()
                            row = physical.execute_verified_sample(
                                client, schedule, data, expected, 30)
                            row.update(variant=name, policy=plan['variants'][name]['policy'],
                                model='vww', sample=sample, kind=kind,
                                repeat=max(0, index-1), fixture=fixture['directory'],
                                bitstream_sha256=shared.IMAGE_SHA,
                                core_clock_hz=shared.CLOCK_HZ,
                                device_latency_ms=row['elapsed_cycles']/shared.CLOCK_HZ*1000,
                                input_sha256=fixture['files']['input.bin'],
                                commands_sha256=fixture['files']['commands.bin'],
                                payload_sha256=fixture['files']['payload.bin'],
                                burst_bytes=BURST_BYTES,
                                load_seconds=loaded if index == 0 else 0,
                                at=screening.timestamp())
                            priority.append_row(stream, row); rows.append(row)
                            report['completed'] = len(rows); save()
                    print(f'{name}: {len(rows)}/20 exact', flush=True)
        require(priority.read_rows(output / 'records.jsonl') == rows,
                'signed physical records changed')
        require(prepare(ROOT / plan['manifest'], plan['seed']) == plan,
                'preflight evidence changed during physical campaign')
        report.update(status='passed-vww-finalist-screen', summary=summarize(rows),
            records_sha256=sha(output / 'records.jsonl'),
            finished_at=screening.timestamp())
        save()
        screening.save_json(output / 'seal.json', dict(
            plan_sha256=sha(output / 'plan.json'),
            report_sha256=sha(output / 'report.json'),
            records_sha256=sha(output / 'records.jsonl'),
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
    parser.add_argument('--output', type=Path, default=EXPERIMENT / 'board')
    parser.add_argument('--port', default='/dev/cu.usbserial-20250303171')
    parser.add_argument('--seed', type=int, default=20260928)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--generate-stress', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.generate_stress:
        result = generate()
        print(json.dumps(dict(status=result['status'],
                              alternatives=list(result['alternatives'])), sort_keys=True))
        return
    plan = prepare(args.manifest, args.seed)
    if args.run:
        result = run(plan, shared.path_in_repo(args.output), args.port)
        print(json.dumps(dict(status=result['status'], summary=result['summary']), sort_keys=True))
    else:
        print(json.dumps(dict(status=plan['status'], planned=plan['planned'],
            clock_hz=plan['clock_hz'], image_sha256=plan['image']['sha256'],
            block_order=plan['block_order'], plan_sha256=shared.canonical_sha(plan)),
            sort_keys=True))


if __name__ == '__main__':
    main()
