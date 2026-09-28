#!/usr/bin/env python3
"""Read-only preflight, or a short B1/B2/B4 and optional restricted-B3 campaign.

Each policy/model receives one stress check, one warmup and three timings.
Native checks and the selected 27 MHz route must pass before JTAG is opened.
This is a policy screen, not full accuracy, endurance, B03 or G6 certification.
"""

import argparse
import fcntl
import json
import math
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True

import serial

import run_priority as priority
import run_screening as screening
import screen_engine_schedule as physical
import screen_pair7_fusion as selected
from uart_burst import BURST_BYTES, BurstTiledClient
from variants import ROOT, sha


BASE = ROOT / 'work/phase6/matched-baselines-v1'
DEFAULT_MANIFESTS = (BASE / 'b1b2/report.json', BASE / 'b3/report.json')
NATIVE = ROOT / 'work/phase6/pool-timing-v1/native/Vv2_tiled_host_bridge'
ENGINE = ROOT / 'work/phase6/pool-timing-v1/engine.sv'
IMAGE_SHA = '3e943639382e4efea62d4d2072c016fa40eacc7a97ef6bad7f5bfdc69d8dafce'
NATIVE_SHA = 'b97fbf944827a5345670f3603e7f10368a897e6bcb19d16cc56905d8e06d16ec'
CLOCK_HZ = 27_000_000
REQUIRED_POLICIES = ('B1', 'B2')
BASELINE_POLICIES = (*REQUIRED_POLICIES, 'B3_restricted')
POLICIES = (*BASELINE_POLICIES, 'B4')
MODELS = ('kws', 'vww')
FILES = ('commands.bin', 'payload.bin', 'input.bin', 'output.bin',
         'checks.txt', 'schedule.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path_in_repo(value):
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    require(path.is_relative_to(ROOT), f'path outside the project: {value}')
    return path


def json_file(path):
    return json.loads(path.read_text())


def canonical_sha(value):
    return priority.digest(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def checked_hash_tree(value, label):
    """Validate nested model/file provenance and return normalized file pins."""
    require(isinstance(value, dict) and value, f'{label}: missing file hashes')
    checked = {}
    for name, digest in value.items():
        if isinstance(digest, dict):
            children = checked_hash_tree(digest, f'{label}/{name}')
        else:
            require(isinstance(digest, str) and len(digest) == 64 and
                    all(c in '0123456789abcdef' for c in digest), f'{label}: invalid SHA256')
            path = path_in_repo(name)
            require(sha(path) == digest, f'{label}: source changed: {name}')
            children = {str(path.relative_to(ROOT)): digest}
        for key, child in children.items():
            require(key not in checked or checked[key] == child, f'{label}: conflicting file hashes')
            checked[key] = child
    return checked


def checked_b3_selection(record, models, reference):
    """Bind the restricted VWW argmin and explicit KWS fallback to emitted code."""
    require(record.get('independent_policy_generation') is True,
            'B3_restricted requires independently generated policy-selection evidence')
    path = path_in_repo(record['selection_report'])
    require(sha(path) == record['selection_report_sha256'], 'B3 selection report changed')
    selection = json_file(path)
    require(selection.get('status') == 'passed-native', 'B3 selection is incomplete')
    variants = selection.get('variants')
    require(isinstance(variants, dict) and variants, 'B3 candidate catalogue is missing')
    objectives = {}
    for name, candidate in variants.items():
        native = candidate.get('native', [])
        require(len(native) == 2 and
                {row.get('stall_seed', row.get('seed')) for row in native} == {0, 6063} and
                all(row.get('status') == 'passed' and row.get('elapsed_cycles', 0) > 0 for row in native),
                f'B3 selection candidate {name} lacks exact native coverage')
        objective = max(row['elapsed_cycles'] for row in native)
        require(candidate.get('worst_seed_cycles') == objective, 'B3 selection objective differs')
        objectives[name] = objective
    winner = min(objectives, key=lambda name: (objectives[name], name))
    require(selection.get('selection') == winner, 'B3 selected candidate is not the declared argmin')
    files = variants[winner]['files']
    chosen = models['vww']['fixtures']['pinned_timed']
    for name in ('commands.bin', 'payload.bin', 'input.bin', 'output.bin'):
        require(files.get(name) == chosen['files'][name], f'B3 winner fixture differs: {name}')
    require(record.get('fixed_fallback_models') == ['kws'],
            'B3 must declare its unsearched KWS fallback')
    for sample, old_sample in (('pinned_timed', 'pinned'), ('stress_check', 'stress')):
        name = reference['fixture_names']['kws'][old_sample]
        base_files = reference['fixtures'][name]['files']
        fallback = models['kws']['fixtures'][sample]['files']
        require(all(fallback[key] == base_files[key] for key in
                    ('commands.bin', 'payload.bin', 'input.bin', 'output.bin')),
                'B3 KWS fallback differs from the declared selected baseline')
    return dict(report=str(path.relative_to(ROOT)), report_sha256=sha(path),
                winner=winner, winner_worst_seed_cycles=objectives[winner],
                candidates=len(variants), fixed_fallback_models=['kws'])


def validate_fixture(raw, model, sample, reference):
    """Bind replay, file contents and native results to one selected fixture."""
    folder = path_in_repo(raw['directory'])
    files = raw['files']
    require(isinstance(files, dict) and all(name in files for name in FILES),
            f'{folder}: missing required fixture hashes')
    require(all(Path(name).name == name for name in files), 'fixture file path is not flat')
    screening.verify_files(folder, files)
    require(raw.get('replay', {}).get('status') == 'passed',
            f'{folder}: independent command replay has not passed')
    schedule = json_file(folder / 'schedule.json')
    command_bytes = (folder / 'commands.bin').stat().st_size
    payload_bytes = (folder / 'payload.bin').stat().st_size
    expected_input = 490 if model == 'kws' else 27_648
    expected_output = 12 if model == 'kws' else 2
    require(0 < command_bytes <= 32_768 and command_bytes == 16 * schedule['command_count'],
            f'{folder}: command size/count out of range')
    require(0 < payload_bytes <= 8 * 1024 * 1024, f'{folder}: SDRAM capacity exceeded')
    require(schedule['program_sha256'] == files['commands.bin'] and
            schedule['image_sha256'] == files['payload.bin'],
            f'{folder}: schedule and fixture identities differ')
    output = schedule['final_output']
    require(output['bytes'] == expected_output and
            0 <= output['ext'] <= 8 * 1024 * 1024 - output['bytes'],
            f'{folder}: final output address/shape invalid')
    require((folder / 'input.bin').stat().st_size == expected_input and
            (folder / 'output.bin').stat().st_size == expected_output,
            f'{folder}: public tensor shape changed')
    for name in ('input.bin', 'output.bin'):
        require(files[name] == reference['files'][name],
                f'{folder}: public {name} differs from the frozen reference')
    checks = []
    for line in (folder / 'checks.txt').read_text().splitlines():
        address, name = line.split()
        require(name in files, f'{folder}: unchecked tensor file {name}')
        address = int(address)
        require(0 <= address <= 8 * 1024 * 1024 - (folder / name).stat().st_size,
                f'{folder}: tensor check range invalid')
        checks.append((address, name))
    require(checks, f'{folder}: no native tensor checks')
    require(any(address == output['ext'] and files[name] == files['output.bin']
                for address, name in checks), f'{folder}: native final-output check absent')
    if sample == 'pinned_timed':
        require(schedule.get('snapshots_enabled') is False and
                not schedule.get('snapshot_regions') and len(checks) == 1,
                f'{folder}: timed fixture contains diagnostic snapshots')
    else:
        require(len(checks) > 1, f'{folder}: stress fixture lacks intermediate checks')
    native = raw.get('native', [])
    require(isinstance(native, list) and len(native) == 2,
            f'{folder}: two native memory seeds required')
    seeds = set()
    evidence = {}
    for row in native:
        seed = row.get('stall_seed', row.get('seed'))
        require(seed in (0, 6063) and seed not in seeds, f'{folder}: invalid native seed')
        seeds.add(seed)
        report = path_in_repo(row['report'])
        require(sha(report) == row['report_sha256'], f'{folder}: native report changed')
        result = json_file(report)
        require(result.get('status') == 'passed' and row.get('status') == 'passed' and
                result.get('physical_board') is False and result.get('stall_seed') == seed and
                result.get('tensor_checks') == len(checks),
                f'{folder}: incomplete native exactness coverage')
        for name in ('elapsed_cycles', 'engine_cycles', 'dma_cycles', 'overlap_cycles'):
            if name in row:
                require(row[name] == result[name], f'{folder}: native {name} differs')
        e, c, d, o = (result[name] for name in
                      ('elapsed_cycles', 'engine_cycles', 'dma_cycles', 'overlap_cycles'))
        require(min(e, c, d) > 0 and 0 <= o <= min(c, d) and c + d - o <= e,
                f'{folder}: invalid native counters')
        evidence[str(report.relative_to(ROOT))] = row['report_sha256']
    return dict(directory=str(folder.relative_to(ROOT)), files=files,
                replay=raw['replay'], native=native, native_evidence=evidence,
                command_count=schedule['command_count'], final_output=output)


def make_blocks(seed, policies=POLICIES):
    blocks = [dict(policy=policy, model=model) for policy in policies for model in MODELS]
    rng = random.Random(seed)
    for _ in range(1000):
        rng.shuffle(blocks)
        if all(left['policy'] != right['policy'] for left, right in zip(blocks, blocks[1:])):
            break
    else:
        raise ValueError('cannot construct interleaved policy order')
    return blocks


def reference_policy(reference):
    """Remeasure the already validated selected schedule in this same campaign."""
    models = {}
    for model in MODELS:
        fixtures = {}
        for sample, old_sample in (('pinned_timed', 'pinned'), ('stress_check', 'stress')):
            name = reference['fixture_names'][model][old_sample]
            prior = reference['fixtures'][name]
            native = []
            for seed in (0, 6063):
                report = (ROOT / 'work/phase6/pool-timing-v1/compact-native' /
                          f'{name}-s{seed}.json' if model == 'kws' else
                          ROOT / 'work/phase6/strip-fusion-pair7-v1/full' /
                          f'{name}-seed{seed}.json')
                raw = json_file(report)
                native.append(dict(raw, report=str(report.relative_to(ROOT)), report_sha256=sha(report)))
            raw = dict(directory=str(Path(reference['fixture_root']) / name),
                       files=prior['files'], replay=prior['verification'], native=native)
            fixtures[sample] = validate_fixture(raw, model, sample, prior)
        models[model] = dict(fixtures=fixtures)
    return dict(description='selected 27 MHz compacted graph and three VWW strip pairs', models=models)


def prepare(manifests, seed=20260928):
    """No files written and no device opened, including when a gate fails."""
    reference = selected.prepare()
    require(reference['image']['sha256'] == IMAGE_SHA and
            reference['image']['clock_hz'] == CLOCK_HZ and BURST_BYTES == 256,
            'selected image/clock/UART differs from the frozen comparison')
    require(sha(NATIVE) == NATIVE_SHA, 'selected native executable changed')
    source_hashes = {}
    manifest_hashes = {}
    policies = {}
    for manifest in manifests:
        manifest = path_in_repo(manifest)
        data = json_file(manifest)
        require(data.get('schema') == 1 and data.get('status') == 'passed-native',
                f'{manifest}: candidate generation/native checks are incomplete')
        require(data.get('engine_sha256') == sha(ENGINE) and
                data.get('native_executable_sha256') == NATIVE_SHA,
                f'{manifest}: engine/native executable differs')
        sources = checked_hash_tree(data.get('source_sha256'), str(manifest))
        if 'model_source_sha256' in data:
            sources.update(checked_hash_tree(data['model_source_sha256'],
                                            f'{manifest}/model_source_sha256'))
        for key, digest in sources.items():
            require(key not in source_hashes or source_hashes[key] == digest,
                    f'{manifest}: conflicting source versions')
            source_hashes[key] = digest
        manifest_hashes[str(manifest.relative_to(ROOT))] = sha(manifest)
        for policy, policy_record in data.get('policies', {}).items():
            # This backend supports a restricted policy catalogue. An input's
            # historical B3 label must not imply a full published reproduction.
            source_policy = policy
            if policy == 'B3':
                policy = 'B3_restricted'
            require(policy in BASELINE_POLICIES and policy not in policies, 'duplicate or unknown policy')
            require(set(policy_record['models']) == set(MODELS), f'{policy}: both models required')
            policies[policy] = dict(description=policy_record.get('description'),
                source_policy=source_policy, limitations=policy_record.get('limitations', []), models={})
            if policy == 'B3_restricted':
                require(policy_record.get('independent_policy_generation') is True,
                        'B3_restricted requires independently generated policy-selection evidence')
                policies[policy].update(fully_eligible_b3=False,
                    independent_policy_generation=True,
                    comparison_scope='restricted executable policy adaptation, not full DeFiNES reproduction')
            for model in MODELS:
                raw_fixtures = policy_record['models'][model]['fixtures']
                require({'pinned_timed', 'stress_check'} <= set(raw_fixtures) <=
                        {'pinned_timed', 'pinned_check', 'stress_check'},
                        f'{policy}/{model}: explicit timed and stress fixtures required')
                fixtures = {}
                for sample in raw_fixtures:
                    old_sample = 'stress' if sample == 'stress_check' else 'pinned'
                    name = reference['fixture_names'][model][old_sample]
                    fixtures[sample] = validate_fixture(raw_fixtures[sample], model, sample,
                                                         reference['fixtures'][name])
                policies[policy]['models'][model] = dict(fixtures=fixtures)
            if policy == 'B3_restricted':
                policies[policy]['selection_evidence'] = checked_b3_selection(
                    policy_record, policies[policy]['models'], reference)
    require(set(REQUIRED_POLICIES) <= set(policies), 'both B1/B2 policies must pass before board use')
    policies['B4'] = reference_policy(reference)
    for policy in policies:
        for model in MODELS:
            for sample, fixture in policies[policy]['models'][model]['fixtures'].items():
                reference_sample = 'stress_check' if sample == 'stress_check' else 'pinned_timed'
                base_files = policies['B4']['models'][model]['fixtures'][reference_sample]['files']
                fixture['duplicate_commands_of_B4'] = fixture['files']['commands.bin'] == base_files['commands.bin']
                fixture['duplicate_program_and_payload_of_B4'] = (
                    fixture['duplicate_commands_of_B4'] and
                    fixture['files']['payload.bin'] == base_files['payload.bin'])
    policy_order = tuple(policy for policy in POLICIES if policy in policies)
    runner_sources = (Path(__file__), ROOT / 'tools/phase6/run_priority.py',
                      ROOT / 'tools/phase6/run_screening.py',
                      ROOT / 'tools/phase6/screen_engine_schedule.py',
                      ROOT / 'tools/phase6/uart_burst.py', ROOT / 'tools/phase4/tiled_host.py',
                      ROOT / 'tools/phase2/host.py')
    return dict(schema=1, status='passed-preflight', image=reference['image'],
                native_executable=str(NATIVE.relative_to(ROOT)),
                native_executable_sha256=NATIVE_SHA, engine_sha256=sha(ENGINE),
                selected_reference_plan_sha256=canonical_sha(reference),
                manifests=manifest_hashes, source_sha256=source_hashes,
                runner_sources={str(p.relative_to(ROOT)): sha(p) for p in runner_sources},
                policies=policies, seed=seed, blocks=make_blocks(seed, policy_order),
                planned=10 * len(policies), clock_hz=CLOCK_HZ, baud=750000, burst_bytes=BURST_BYTES,
                scope='matched short policy screen: final board logits and native intermediate tensors',
                not_claimed=['full accuracy', 'endurance', 'full DeFiNES reproduction',
                             'B03 closure', 'G6 closure', 'SOTA', 'measured energy'])


def summarize(rows, policies=POLICIES):
    require(len(rows) == 10 * len(policies), 'wrong board record count for selected policies')
    summary = {}
    for policy in policies:
        summary[policy] = {}
        for model in MODELS:
            group = [row for row in rows if row['policy'] == policy and row['model'] == model]
            require(sorted(row['kind'] for row in group) ==
                    ['stress', 'timed', 'timed', 'timed', 'warmup'], 'unbalanced record coverage')
            timed = [row for row in group if row['kind'] == 'timed']
            require(sorted(row['repeat'] for row in timed) == [0, 1, 2], 'invalid timed repeats')
            require(all(row['bitstream_sha256'] == IMAGE_SHA and
                        row['output_hex'] == row['expected_hex'] and
                        row['device_latency_ms'] == row['elapsed_cycles'] / CLOCK_HZ * 1000
                        for row in group), 'physical record identity, clock or exactness changed')
            cycles = [row['elapsed_cycles'] for row in timed]
            median = statistics.median(cycles)
            summary[policy][model] = dict(median_cycles=median,
                median_device_latency_ms=median / CLOCK_HZ * 1000,
                device_inferences_per_second=CLOCK_HZ / median, timed_cycles=cycles,
                spread_fraction=(max(cycles) - min(cycles)) / median,
                median_host_wall_seconds=statistics.median(row['wall_seconds'] for row in timed),
                duplicate_commands_of_B4=all(row['duplicate_commands_of_B4'] for row in timed),
                duplicate_program_and_payload_of_B4=all(
                    row['duplicate_program_and_payload_of_B4'] for row in timed))
    comparisons = {}
    for policy in (policy for policy in policies if policy != 'B1'):
        ratios = {model: summary['B1'][model]['median_cycles'] /
                        summary[policy][model]['median_cycles'] for model in MODELS}
        comparisons[f'{policy}_versus_B1'] = dict(per_model_speedup=ratios,
                                                geomean_speedup=math.sqrt(math.prod(ratios.values())))
    for policy in (policy for policy in policies if policy != 'B4'):
        ratios = {model: summary[policy][model]['median_cycles'] /
                        summary['B4'][model]['median_cycles'] for model in MODELS}
        comparisons[f'B4_versus_{policy}'] = dict(per_model_speedup=ratios,
            geomean_speedup=math.sqrt(math.prod(ratios.values())),
            independent_performance_evidence=not any(
                summary[policy][model]['duplicate_program_and_payload_of_B4'] for model in MODELS),
            duplicate_program_models=[model for model in MODELS
                if summary[policy][model]['duplicate_program_and_payload_of_B4']])
    return dict(policies=summary, comparisons=comparisons)


def run(plan, manifests, output, port):
    screening.require_board_free(port)
    require(not output.exists(), 'choose a fresh output directory; prior evidence is immutable')
    lock = (ROOT / 'work/phase6/physical-board.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = []
    old_handlers = {}
    report = {}
    def save():
        report['updated_at'] = screening.timestamp()
        screening.save_json(output / 'report.json', report)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}; completed records retained, no automatic retry')
    try:
        require(prepare(manifests, plan['seed']) == plan, 'evidence changed before board lock')
        output.mkdir(parents=True)
        screening.save_json(output / 'plan.json', plan)
        report.update(schema=1, status='running', physical_board=True, planned=plan['planned'],
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
                for block_index, block in enumerate(plan['blocks']):
                    policy, model = block['policy'], block['model']
                    fixtures = plan['policies'][policy]['models'][model]['fixtures']
                    for sample in ('stress_check', 'pinned_timed'):
                        fixture = fixtures[sample]
                        folder = ROOT / fixture['directory']
                        report['current'] = f'{policy}/{model}/{sample}/load'; save()
                        schedule, loaded = screening.load_fixture(client, folder, fixture)
                        kinds = ('stress',) if sample == 'stress_check' else ('warmup', 'timed', 'timed', 'timed')
                        for index, kind in enumerate(kinds):
                            report['current'] = f'{policy}/{model}/{kind}/{index}'; save()
                            data, expected = (folder / 'input.bin').read_bytes(), (folder / 'output.bin').read_bytes()
                            row = physical.execute_verified_sample(client, schedule, data, expected, 30)
                            row.update(policy=policy, model=model, sample=sample, kind=kind,
                                repeat=max(0, index - 1), block=block_index,
                                fixture=fixture['directory'], bitstream_sha256=IMAGE_SHA,
                                input_sha256=fixture['files']['input.bin'],
                                commands_sha256=fixture['files']['commands.bin'],
                                payload_sha256=fixture['files']['payload.bin'],
                                duplicate_commands_of_B4=fixture['duplicate_commands_of_B4'],
                                duplicate_program_and_payload_of_B4=fixture['duplicate_program_and_payload_of_B4'],
                                device_latency_ms=row['elapsed_cycles'] / CLOCK_HZ * 1000,
                                core_clock_hz=CLOCK_HZ, burst_bytes=BURST_BYTES,
                                load_seconds=loaded if index == 0 else 0, at=screening.timestamp())
                            priority.append_row(stream, row); rows.append(row)
                            report['completed'] = len(rows); save()
                    print(f'{policy} {model}: {len(rows)}/{plan["planned"]} exact', flush=True)
        require(priority.read_rows(output / 'records.jsonl') == rows, 'signed records changed')
        require(prepare(manifests, plan['seed']) == plan, 'evidence changed during physical campaign')
        report.update(status='passed-matched-short-screen', summary=summarize(rows, tuple(plan['policies'])),
                      finished_at=screening.timestamp(), records_sha256=sha(output / 'records.jsonl'))
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
    parser.add_argument('--manifest', action='append', type=Path,
                        help='schema-1 native-passed manifest; repeat for separate policy reports')
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument('--output', type=Path, default=BASE / 'board')
    parser.add_argument('--port', default='/dev/cu.usbserial-20250303171')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    manifests = args.manifest or [path for path in DEFAULT_MANIFESTS if path.exists()]
    require(manifests, 'no baseline manifest exists yet')
    plan = prepare(manifests, args.seed)
    if args.run:
        result = run(plan, manifests, path_in_repo(args.output), args.port)
        print(json.dumps(dict(status=result['status'], completed=result['completed'],
                              summary=result['summary']), sort_keys=True))
    else:
        print(json.dumps(dict(status=plan['status'], planned=plan['planned'],
                              clock_hz=CLOCK_HZ, image_sha256=IMAGE_SHA,
                              plan_sha256=canonical_sha(plan), blocks=plan['blocks']), sort_keys=True))


if __name__ == '__main__':
    main()
