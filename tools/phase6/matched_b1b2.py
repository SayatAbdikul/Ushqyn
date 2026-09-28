#!/usr/bin/env python3
"""Build matched conventional and liveness-only Phase 6 accelerator baselines.

B1 materializes each completed layer in SDRAM. B2 keeps an exact full tensor
in SRAM when its live range fits. Both use the same compacted INT8 graph,
constant-filter lowering, tagged activation lookup, legal immutable DMA
prefetch, 27 MHz routed engine, and native timing boundary. No FPGA hardware
source is modified here.
"""
import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'compiler'), str(ROOT / 'tools/phase6')]

import chain_resident
import constant_filter
from matched_b1b2_prefetch import next_input_candidates
from channel_compaction import compact_channels, check_oracles, command_stats
from followup_graph import group_channels, trim_constant_only_loads
from fused_activation import lower_fixture, check_table_lifetimes
from prefetch_tail import analyze_fixture, commands, reorder
from run_boardless import load_model
from scheduler.fused_verify import replay_fused
from scheduler.resident import compile_resident
from scheduler.resident_verify import replay_resident
from variants import check_frozen, sha

BASE = ROOT / 'work/phase6/matched-baselines-v1/b1b2'
NATIVE = ROOT / 'work/phase6/pool-timing-v1/native/Vv2_tiled_host_bridge'
ENGINE = ROOT / 'work/phase6/pool-timing-v1/engine.sv'
ROUTE = ROOT / 'work/phase6/pool-timing-v1/route27/report.json'
IMAGE = ROOT / 'work/phase6/pool-timing-v1/route27/phase6_uart_burst/impl/pnr/phase6_uart_burst.fs'
B4 = ROOT / 'work/phase6/strip-fusion-pair7-v1/full/fixtures'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def rebuild_serial_stages(schedule):
    """Rebuild the chain command bytes after restoring required partial stores."""
    command_list = []
    contracts = {}

    def emit(op, flags=0, a=0, b=0, c=0):
        command_list.append((op, flags, 0, a, b, c))

    def dma(transfer):
        emit(1, int(transfer['direction'] == 'to_sram'),
             transfer['ext'], transfer['sram'], transfer['bytes'])
        emit(3, 2)

    for stage in schedule['stages']:
        if 'padded_pack' in stage:
            raise ValueError('partial-consumer restore does not support PACK')
        for load in stage['loads']:
            dma(load)
        contracts[str(len(command_list))] = dict(
            layer=stage['layer'], first_element=stage['first_element'])
        emit(2, 0, stage['pc'], stage['live'][0] | (stage['live'][1] << 16))
        emit(3, 3)
        if schedule['snapshots_enabled']:
            region = schedule['snapshot_regions'][stage['layer']]
            dma(dict(direction='from_sram',
                     ext=region['ext'] + stage['first_element'], **stage['output']))
        if stage['store'] is not None:
            dma(stage['store'])
    emit(0)
    packed = b''.join(struct.pack('<BBHIII', *command) for command in command_list)
    if len(packed) > 32768:
        raise ValueError('restored chain exceeds command capacity')
    schedule['run_contracts'] = contracts
    schedule['program_sha256'] = hashlib.sha256(packed).hexdigest()
    schedule['command_count'] = len(command_list)
    schedule['program_bytes'] = len(packed)
    return packed


@contextlib.contextmanager
def compiler_policy(policy, half):
    original_chain = constant_filter.compile_chain
    original_resident = chain_resident.compile_resident

    if policy == 'B1':
        def conventional(program, snapshots, **kwargs):
            if kwargs.get('padded_pointwise'):
                raise ValueError('B1 wrapper does not support padded pointwise')
            return compile_resident(program, prefer_half=half,
                                    overlap=False, snapshots=snapshots)
        constant_filter.compile_chain = conventional
    elif policy == 'B2':
        def resident_choice(program, **kwargs):
            kwargs['prefer_half'] = half
            return compile_resident(program, **kwargs)
        def liveness_chain(program, snapshots, **kwargs):
            code, payload, schedule = original_chain(program, snapshots, **kwargs)
            if half:
                _, _, materialized = compile_resident(
                    program, prefer_half=True, overlap=False, snapshots=snapshots)
                stages = schedule['stages']
                ordinary = materialized['stages']
                if len(stages) != len(ordinary):
                    raise ValueError('liveness/ordinary stage counts differ')
                restored = []
                for index, (live, plain) in enumerate(zip(stages, ordinary)):
                    if (live['layer'], live['first_element']) != (plain['layer'], plain['first_element']):
                        raise ValueError('liveness/ordinary stage coordinates differ')
                    store = plain['store']
                    if live['store'] is not None or store is None:
                        continue
                    following = next((j for j in range(index + 1, len(stages))
                                      if stages[j]['group'] != live['group']), None)
                    if following is None:
                        continue
                    group = stages[following]['group']
                    consumers = []
                    for stage in stages[following:]:
                        if stage['group'] != group:
                            break
                        consumers.append(stage)
                    first, end = store['ext'], store['ext'] + store['bytes']
                    needs_external_slice = any(
                        load['role'] == 'input' and
                        first <= load['ext'] and load['ext'] + load['bytes'] <= end
                        for stage in consumers for load in stage['loads'])
                    if needs_external_slice:
                        live['store'] = dict(store, sram=live['output']['sram'])
                        restored.append(dict(producer_layer=live['layer'],
                                             consumer_group=group,
                                             bytes=store['bytes'], ext=store['ext']))
                schedule['restored_partial_consumer_stores'] = restored
                schedule['removed_transfer_bytes'] -= sum(x['bytes'] for x in restored)
                if restored:
                    code = rebuild_serial_stages(schedule)
            return code, payload, schedule
        chain_resident.compile_resident = resident_choice
        constant_filter.compile_chain = liveness_chain
    else:
        raise ValueError('unknown baseline policy')
    try:
        yield
    finally:
        constant_filter.compile_chain = original_chain
        chain_resident.compile_resident = original_resident


def compile_seed(program, policy, *, half, snapshots):
    # The tagged epilogue lowering assumes serial seed stage chunks. Both
    # policies receive the same legal post-fusion prefetch tuning option.
    with compiler_policy(policy, half):
        code, payload, schedule = constant_filter.compile_constant_chain(
            program, snapshots, reuse_sibling_inputs=(policy == 'B2'))
    code, schedule = trim_constant_only_loads(code, schedule)
    return code, payload, schedule


def model(program_name):
    original, pinned, _, source_hashes = load_model(program_name)
    grouped, group_maps, _ = group_channels(original)
    compacted, kept, changes, aligned = compact_channels(grouped)
    maps = {name: tuple(group_maps[name][index] for index in keep)
            for name, keep in kept.items()}
    return original, pinned, compacted, maps, source_hashes, changes, aligned


def sample_value(name, sample, pinned):
    if sample == 'pinned':
        return pinned
    return np.random.default_rng(6157).integers(-128, 128, pinned.shape, dtype=np.int8)


def fixture_files(directory, code, payload, schedule, initial, oracle, program):
    files = {'commands.bin': code, 'payload.bin': payload,
             'input.bin': initial.tobytes(),
             'output.bin': oracle[program.outputs[0]].tobytes()}
    checks = [f'{schedule["final_output"]["ext"]} output.bin']
    for index, region in schedule['snapshot_regions'].items():
        files[f'layer-{index}.bin'] = oracle[program.layers[int(index)].output].tobytes()
        checks.append(f'{region["ext"]} layer-{index}.bin')
    files['checks.txt'] = ('\n'.join(checks) + '\n').encode()
    files['schedule.json'] = (json.dumps(schedule, indent=2, sort_keys=True) + '\n').encode()
    directory.mkdir(parents=True, exist_ok=True)
    for filename, data in files.items():
        (directory / filename).write_bytes(data)
    return {filename: sha(directory / filename) for filename in files}


def fused_fixture(program_name, compacted, original, pinned, maps, *, sample,
                  snapshots, policy, half, prefetch, directory,
                  value=None, oracle=None):
    if value is None:
        value = sample_value(program_name, sample, pinned)
    if oracle is None:
        oracle = check_oracles(original, compacted, maps, value)
    initial = oracle[compacted.layers[0].output]
    seed_code, seed_payload, seed_schedule = compile_seed(
        compacted, policy, half=half, snapshots=snapshots)
    seed_replay = replay_resident(
        compacted, seed_code, seed_payload, {compacted.inputs[0]: value},
        run_contracts=seed_schedule['run_contracts'],
        constant_contracts=seed_schedule.get('constant_contracts'),
        pack_contracts=seed_schedule.get('pack_contracts'),
        final_output=seed_schedule['final_output'],
        snapshot_regions=seed_schedule['snapshot_regions'], oracle=oracle)
    seed = directory.parent / (directory.name + '-seed')
    fixture_files(seed, seed_code, seed_payload, seed_schedule, initial, oracle, compacted)
    code, payload, schedule = lower_fixture(seed)
    working = directory.parent / (directory.name + '-prefetch-input')
    fixture_files(working, code, payload, schedule, initial, oracle, compacted)
    analysis = analyze_fixture(working)
    input_analysis = next_input_candidates(working)
    selected_prefetch = (analysis['candidates'] + input_analysis['candidates']) if prefetch else []
    reordered, mapping = reorder(commands(code), selected_prefetch)
    code = b''.join(struct.pack('<BBHIII', *command) for command in reordered)
    schedule['run_contracts'] = {str(mapping[int(index)]): contract
                                 for index, contract in schedule['run_contracts'].items()}
    schedule['constant_contracts'] = {str(mapping[int(index)]): contract
                                      for index, contract in schedule['constant_contracts'].items()}
    schedule['program_sha256'] = hashlib.sha256(code).hexdigest()
    schedule['tail_prefetch'] = dict(bytes=sum(x['bytes'] for x in selected_prefetch),
                                     transfers=len(selected_prefetch),
                                     eligible_bytes=analysis['immediately_legal_prefetch_bytes'],
                                     eligible_transfers=analysis['immediately_legal_count'],
                                     eligible_next_input_bytes=input_analysis['eligible_bytes'],
                                     eligible_next_input_transfers=len(input_analysis['candidates']),
                                     same_layer_adjacencies=input_analysis['same_layer_adjacencies'],
                                     rejected_input_hazards=input_analysis['rejected'])
    lifetime = check_table_lifetimes(code, payload, schedule)
    replay = replay_fused(
        compacted, code, payload, {compacted.inputs[0]: value},
        run_contracts=schedule['run_contracts'],
        constant_contracts=schedule['constant_contracts'],
        final_output=schedule['final_output'],
        snapshot_regions=schedule['snapshot_regions'], oracle=oracle)
    files = fixture_files(directory, code, payload, schedule, initial, oracle, compacted)
    return dict(directory=str(directory.relative_to(ROOT)), files=files,
                seed_replay=seed_replay, replay=replay, lifetime=lifetime,
                command_stats=command_stats(code),
                model_input_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                output_sha256=files['output.bin'],
                prefetch_bytes=sum(x['bytes'] for x in selected_prefetch),
                eligible_prefetch_bytes=analysis['immediately_legal_prefetch_bytes'],
                eligible_next_input_bytes=input_analysis['eligible_bytes'],
                eligible_next_input_transfers=len(input_analysis['candidates']),
                same_layer_adjacencies=input_analysis['same_layer_adjacencies'],
                policy=policy, half=half, prefetch=prefetch,
                snapshots=snapshots, sample=sample)


def measure(directory, seed):
    path = directory.parent / (directory.name + f'-native-s{seed}.json')
    subprocess.run([str(NATIVE), str(directory), str(seed), str(path)], check=True)
    result = json.loads(path.read_text())
    if result['status'] != 'passed':
        raise ValueError(f'native mismatch: {directory} seed {seed}')
    return dict(result, report=str(path.relative_to(ROOT)), report_sha256=sha(path))


def compare_selected_b4(name, role, entry):
    reference_role = 'timed' if role == 'pinned_timed' else 'check'
    sample = 'stress' if role == 'stress_check' else 'pinned'
    stem = (f'{name}-{sample}-compacted-fused-{reference_role}' if name == 'kws' else
            f'{name}-{sample}-compacted-strip3-7-11-{reference_role}')
    reference = B4 / stem
    if role == 'pinned_check' and not reference.is_dir():
        reference = B4 / (stem.removesuffix('-check') + '-timed')
    if not reference.is_dir():
        raise ValueError(f'selected B4 fixture missing: {reference}')
    identity = {}
    for filename in ('input.bin', 'output.bin'):
        digest = sha(reference / filename)
        if entry['files'][filename] != digest:
            raise ValueError(f'{name}/{role}: {filename} differs from selected B4')
        identity[filename] = digest
    return dict(directory=str(reference.relative_to(ROOT)), files_sha256=identity)


def validate_report(report):
    if (report['status'] != 'passed-native' or
            report['image_bitstream_sha256'] != sha(IMAGE) or
            report['engine_sha256'] != sha(ENGINE) or
            report['native_executable_sha256'] != sha(NATIVE)):
        raise ValueError('common image/native provenance changed')
    for path, digest in report['source_sha256'].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f'compiler source changed: {path}')
    for sources in report['model_source_sha256'].values():
        for path, digest in sources.items():
            if sha(ROOT / path) != digest:
                raise ValueError(f'baseline model source changed: {path}')
    for name in ('kws', 'vww'):
        entries = []
        for policy in ('B1', 'B2'):
            model_report = report['policies'][policy]['models'][name]
            if len(model_report['tuning_candidates']) != 4:
                raise ValueError('declared tuning budget not exhausted')
            for role, item in model_report['fixtures'].items():
                directory = ROOT / item['directory']
                for filename, digest in item['files'].items():
                    if sha(directory / filename) != digest:
                        raise ValueError(f'fixture changed: {directory / filename}')
                schedule = json.loads((directory / 'schedule.json').read_text())
                if (schedule['program_sha256'] != item['files']['commands.bin'] or
                        schedule['image_sha256'] != item['files']['payload.bin'] or
                        item['seed_replay']['status'] != 'passed' or
                        item['replay']['status'] != 'passed'):
                    raise ValueError(f'fixture replay/hash mismatch: {directory}')
                if len(item['native']) != 2 or [x['stall_seed'] for x in item['native']] != [0, 6063]:
                    raise ValueError(f'native seed coverage incomplete: {directory}')
                for native in item['native']:
                    source = ROOT / native['report']
                    if (sha(source) != native['report_sha256'] or
                            json.loads(source.read_text()) !=
                            {key: value for key, value in native.items()
                             if key not in ('report', 'report_sha256')} or
                            native['status'] != 'passed' or
                            native['tensor_checks'] != (1 if role == 'pinned_timed' else
                                                        12 if name == 'kws' else 30)):
                        raise ValueError(f'native evidence changed: {source}')
                reference = item['selected_b4_identity']
                for filename in ('input.bin', 'output.bin'):
                    if (sha(ROOT / reference['directory'] / filename) !=
                            reference['files_sha256'][filename] or
                            item['files'][filename] != reference['files_sha256'][filename]):
                        raise ValueError(f'selected B4 identity changed: {name}/{role}')
                entries.append((policy, role, item))
        for role in ('pinned_timed', 'pinned_check', 'stress_check'):
            a = next(item for policy, key, item in entries if policy == 'B1' and key == role)
            b = next(item for policy, key, item in entries if policy == 'B2' and key == role)
            if any(a['files'][file] != b['files'][file] for file in ('input.bin', 'output.bin')):
                raise ValueError(f'B1/B2 input/output differs: {name}/{role}')
    return True


def run(output=BASE):
    check_frozen()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    native_report = json.loads((NATIVE.parent / 'report.json').read_text())
    if native_report['status'] != 'passed' or sha(NATIVE) != native_report['executable_sha256']:
        raise ValueError('selected 27 MHz native executable changed')
    route = json.loads(ROUTE.read_text())
    report = dict(schema=1, status='running', physical_board=False,
                  scope='matched B1 layerwise and B2 liveness baselines on frozen pool27 image',
                  engine_sha256=sha(ENGINE), image_engine_sha256=sha(ENGINE),
                  image_bitstream_sha256=sha(IMAGE),
                  route_report_sha256=sha(ROUTE), native_executable_sha256=sha(NATIVE),
                  common_clock_mhz=route['core_clock_mhz'],
                  route_fmax_mhz=route['routed_core_fmax_mhz'],
                  tuning_budget={policy: [{'half': h, 'prefetch': p}
                                          for h in (False, True) for p in (False, True)]
                                 for policy in ('B1', 'B2')},
                  policies={}, model_source_sha256={},
                  selected_b4_report_sha256=sha(B4.parent / 'report.json'),
                  source_sha256={str(path.relative_to(ROOT)): sha(path) for path in (
                      Path(__file__), ROOT / 'compiler/scheduler/resident.py',
                      ROOT / 'tools/phase6/matched_b1b2_prefetch.py',
                      ROOT / 'compiler/scheduler/resident_verify.py',
                      ROOT / 'compiler/scheduler/fused_verify.py',
                      ROOT / 'compiler/phase4_compile.py',
                      ROOT / 'compiler/phase4_tiling.py',
                      ROOT / 'compiler/hardware_v2.py',
                      ROOT / 'compiler/integer_reference.py',
                      ROOT / 'compiler/static_pipeline.py',
                      ROOT / 'compiler/quantization.py',
                      ROOT / 'tools/phase6/chain_resident.py',
                      ROOT / 'tools/phase6/constant_filter.py',
                      ROOT / 'tools/phase6/channel_compaction.py',
                      ROOT / 'tools/phase6/followup_graph.py',
                      ROOT / 'tools/phase6/fused_activation.py',
                      ROOT / 'tools/phase6/prefetch_tail.py',
                      ROOT / 'tools/phase6/output_pipeline_fusion.py',
                      ROOT / 'tools/phase6/run_boardless.py',
                      ROOT / 'tools/phase6/padded_descriptor.py')})
    for policy in ('B1', 'B2'):
        report['policies'][policy] = {'models': {}}
        for name in ('kws', 'vww'):
            original, pinned, compacted, maps, hashes, changes, aligned = model(name)
            report['model_source_sha256'][name] = hashes
            pinned_oracle = check_oracles(original, compacted, maps, pinned)
            report['policies'][policy]['models'][name] = dict(
                source_hashes=hashes, compaction_changes=changes,
                alignment_retained=aligned, tuning_candidates=[])
            candidates = []
            for choice in report['tuning_budget'][policy]:
                suffix = f'half{int(choice["half"])}-prefetch{int(choice["prefetch"])}'
                directory = output / 'tuning' / f'{policy.lower()}-{name}-pinned-timed-{suffix}'
                try:
                    item = fused_fixture(name, compacted, original, pinned, maps,
                        sample='pinned', snapshots=False, policy=policy,
                        half=choice['half'], prefetch=choice['prefetch'], directory=directory,
                        value=pinned, oracle=pinned_oracle)
                    item['native'] = {str(seed): measure(directory, seed) for seed in (0, 6063)}
                    item['cycle_objective'] = max(row['elapsed_cycles'] for row in item['native'].values())
                    item['status'] = 'passed-native'
                except Exception as error:
                    item = dict(policy=policy, choice=choice, status='infeasible',
                                reason=f'{type(error).__name__}: {error}')
                candidates.append(item)
                report['policies'][policy]['models'][name]['tuning_candidates'] = candidates
                save(output / 'report.json', report)
                print(policy, name, suffix, item['status'],
                      item.get('cycle_objective', item.get('reason')), flush=True)
            feasible = [item for item in candidates if item['status'] == 'passed-native']
            if not feasible:
                raise ValueError(f'{policy}/{name}: no feasible baseline')
            winner = min(feasible, key=lambda item: (item['cycle_objective'],
                                                       item['native']['0']['elapsed_cycles']))
            report['policies'][policy]['models'][name]['selected_choice'] = dict(
                half=winner['half'], prefetch=winner['prefetch'])
            chosen = {}
            for sample, snapshots, role in (('pinned', False, 'pinned_timed'),
                                            ('pinned', True, 'pinned_check'),
                                            ('stress', True, 'stress_check')):
                value = sample_value(name, sample, pinned)
                oracle = (pinned_oracle if sample == 'pinned' else
                          check_oracles(original, compacted, maps, value))
                directory = output / 'fixtures' / f'{policy.lower()}-{name}-{role}'
                entry = fused_fixture(name, compacted, original, pinned, maps,
                    sample=sample, snapshots=snapshots, policy=policy,
                    half=winner['half'], prefetch=winner['prefetch'], directory=directory,
                    value=value, oracle=oracle)
                entry['native'] = [measure(directory, seed) for seed in (0, 6063)]
                entry['selected_b4_identity'] = compare_selected_b4(name, role, entry)
                chosen[role] = entry
                report['policies'][policy]['models'][name]['fixtures'] = chosen
                save(output / 'report.json', report)
            if len({row['output_sha256'] for row in chosen.values()}) != 1:
                # Stress input normally produces different logits. Verify only
                # pinned timed/check equality below.
                if chosen['pinned_timed']['output_sha256'] != chosen['pinned_check']['output_sha256']:
                    raise ValueError('timed/check output mismatch')
    report['status'] = 'passed-native'
    validate_report(report)
    save(output / 'report.json', report)
    check_frozen()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE)
    args = parser.parse_args()
    result = run(args.output)
    print(result['status'])
