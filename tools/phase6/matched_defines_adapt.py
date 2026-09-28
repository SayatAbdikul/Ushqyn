#!/usr/bin/env python3
"""Screen executable DeFiNES depth-first modes on the frozen 27 MHz backend.

The author implementation supplies tile/overlap geometry.  This adapter only
assigns Tang Nano costs after lowering to the existing command ABI and exact
native RTL simulation.  It deliberately labels the restricted executable
catalogue and records modes that this backend cannot yet lower.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import copy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'compiler'), str(ROOT / 'tools/phase6')]

from hardware_v2 import Descriptor
from run_boardless import load_model
from followup_graph import group_channels
from channel_compaction import compact_channels
from scheduler.defines_adapter import segment_workload
from scheduler.defines_verify import replay_defines
from integer_reference import evaluate
from output_pipeline_fusion import decode_fused
import strip_fusion_pair7 as pair7_builder
import strip_fusion_pair11 as pair11_builder

REVISION = '7097d6090dc22321e44ce91434e7cc23b065864f'
UPSTREAM = ROOT / f'work/phase6/defines-source/DeFiNES-{REVISION}'
sys.path.insert(0, str(UPSTREAM))
from classes.workload.dnn_workload import DNNWorkload
from classes.stages.DepthFirstStage import backpropagate_tilesize

OUT = ROOT / 'work/phase6/matched-baselines-v1/b3'
ENGINE = ROOT / 'work/phase6/pool-timing-v1/engine.sv'
NATIVE = ROOT / 'work/phase6/pool-timing-v1/native/Vv2_tiled_host_bridge'
CMD = struct.Struct('<BBHIII')


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def source_audit() -> dict:
    pins = json.loads((ROOT / 'docs/research/evidence/phase0/prior-code.json').read_text())
    for entry in pins['files']:
        assert file_sha(UPSTREAM / entry['path']) == entry['sha256'], entry['path']
    return {'revision': REVISION, 'prior_code_pin_sha256': file_sha(ROOT / 'docs/research/evidence/phase0/prior-code.json'),
            'source_tar_sha256': file_sha(ROOT / 'work/phase6/defines-source.tar.gz'),
            'depth_first_stage_sha256': file_sha(UPSTREAM / 'classes/stages/DepthFirstStage.py'),
            'adapter_sha256': file_sha(ROOT / 'compiler/scheduler/defines_adapter.py')}


def geometry() -> dict:
    original, _, _, _ = load_model('vww')
    grouped, _, _ = group_channels(original)
    compacted, _, _, _ = compact_channels(grouped)
    cases = []
    # The three existing-ABI full-width strip pairs cover 3-6, 7-10, 11-14.
    for first, width, height in ((3, 48, 24), (7, 24, 12), (11, 24, 12),
                                 (11, 16, 8)):
        full_width=compacted.tensors[compacted.layers[first+3].output].shape[3]
        for hcache, vcache in ((False, False), (True, False), (True, True)):
            graph = DNNWorkload(segment_workload(compacted, first, first + 4))
            _, col_out, col_in, row_out, row_in = backpropagate_tilesize(
                graph, width, height, hcache, hcache, vcache, vcache)
            first_input = graph.get_node_with_id(-1)
            terminal = graph.get_node_with_id(first + 3)
            assert (terminal.loop_dim_size['OX'], terminal.loop_dim_size['OY']) == (width, height)
            rows = row_in.get((graph.get_node_with_id(first), 'I'), ('?', 0))[1]
            cols = col_in.get((graph.get_node_with_id(first), 'I'), ('?', 0))[1]
            cases.append({'layers': [first, first + 3], 'tile': [height, width],
                          'mode': {'horizontal_cache': hcache, 'vertical_cache': vcache},
                          'lowering': ('supported full-width strip endpoint' if width == full_width
                                       else 'not lowered: contiguous aligned DMA/COPY cannot gather interior 18-wide rows from 24-wide source or scatter 16-wide output rows'),
                          'in_regime_input_tile': [first_input.loop_dim_size['OY'], first_input.loop_dim_size['OX']],
                          'input_overlap_rows': rows, 'input_overlap_columns': cols,
                          'source': 'upstream backpropagate_tilesize; border pads checked in existing exact strip oracle'})
    return {'status': 'passed', 'cases': cases}


PAIRS = (
    # first-DW RUN, first-PW RUN, next-DW RUN, source channels, first source SRAM,
    # source plane bytes, halo start row, row width, second source SRAM, second
    # plane bytes, source external plane bytes, scratch SRAM.
    dict(layer=3, dw=74, pw=84, next_dw=142, channels=8, first_sram=128,
         first_plane=1200, halo_row=23, width=48, second_sram=18560,
         second_plane=1200, external_plane=2304, scratch=28672),
    dict(layer=7, dw=226, pw=236, next_dw=342, channels=16, first_sram=128,
         first_plane=1200, halo_row=24, width=48, second_sram=9344,
         second_plane=1152, external_plane=2304, scratch=28672),
    dict(layer=11, dw=490, pw=500, next_dw=638, channels=32, first_sram=128,
         first_plane=312, halo_row=11, width=24, second_sram=10112,
         second_plane=312, external_plane=576, scratch=24576),
)


def cache_variant(source: Path, target: Path) -> dict:
    """Keep vertical halo rows in spare SRAM using legal COPY descriptors.

    This tests the available ABI, including the cost of COPY dispatches. It
    leaves all convolution, weights, INT8 arithmetic, and DDR slots unchanged.
    """
    src_code = (source / 'commands.bin').read_bytes()
    commands = [CMD.unpack_from(src_code, i) for i in range(0, len(src_code), CMD.size)]
    payload = bytearray((source / 'payload.bin').read_bytes())
    schedule = json.loads((source / 'schedule.json').read_text())
    assert len(commands) <= 2048 and schedule['command_count'] == len(commands)
    assert schedule['program_sha256'] == sha(src_code) and schedule['image_sha256'] == sha(payload)
    append_at = (len(payload) + 7) & ~7
    payload.extend(b'\0' * (append_at - len(payload)))

    def copy_commands(src: int, dst: int, size: int) -> list[tuple]:
        assert src % 8 == dst % 8 == size % 8 == 0
        descriptor = Descriptor(3, input=src, output=dst, count=size, outputs=size, next_pc=64)
        descriptor.validate()
        ptr = len(payload)
        assert ptr % 8 == 0
        payload.extend(descriptor.encode() + Descriptor(0).encode())
        return [(1, 1, 0, ptr, 0, 128), (3, 2, 0, 0, 0, 0),
                (2, 0, 0, 0, 32768 << 16, 0), (3, 3, 0, 0, 0, 0)]

    insert: dict[int, list[tuple]] = {}
    replace: dict[int, tuple] = {}
    patches = []
    for template in PAIRS:
        pair = dict(template)
        relevant = [(int(index), value) for index, value in schedule['run_contracts'].items()
                    if value.get('layer') in (pair['layer'], pair['layer'] + 2)
                    and 'strip_origin_y' in value]
        first_height = 24 if pair['layer'] == 3 else 12
        pair['dw'] = next(i for i, row in relevant if row['layer'] == pair['layer'] and row['strip_origin_y'] == 0)
        pair['pw'] = next(i for i, row in relevant if row['layer'] == pair['layer'] + 2 and row['strip_origin_y'] == 0)
        pair['next_dw'] = next(i for i, row in relevant if row['layer'] == pair['layer'] and row['strip_origin_y'] == first_height)
        layer = pair['layer']
        halo = pair['first_plane'] - pair['halo_row'] * pair['width']
        # Pair 3: 96 B/channel; pair 7: 48; pair 11: 48.
        assert halo in (48, 96) and halo % 8 == 0
        assert pair['scratch'] + halo * pair['channels'] <= 32768
        assert commands[pair['dw']][0] == commands[pair['pw']][0] == commands[pair['next_dw']][0] == 2
        assert commands[pair['dw'] + 1] == (3, 3, 0, 0, 0, 0)
        save_idx = pair['dw'] + 2
        for channel in range(pair['channels']):
            src = pair['first_sram'] + channel * pair['first_plane'] + pair['halo_row'] * pair['width']
            scratch = pair['scratch'] + channel * halo
            insert.setdefault(save_idx, []).extend(copy_commands(src, scratch, halo))
            old = (1, 1, 0, 36864 + channel * pair['external_plane'] + pair['halo_row'] * pair['width'],
                   pair['second_sram'] + channel * pair['second_plane'], pair['second_plane'])
            hits = [i for i in range(pair['pw'] + 2, pair['next_dw']) if commands[i] == old]
            assert len(hits) == 1, (layer, channel, hits)
            index = hits[0]
            assert commands[index + 1] == (3, 2, 0, 0, 0, 0)
            replace[index] = (1, 1, 0, old[3] + halo, old[4] + halo, old[5] - halo)
            insert.setdefault(index + 2, []).extend(copy_commands(scratch, old[4], halo))
            patches.append({'layer': layer, 'channel': channel, 'original_prefetch_command': index,
                            'saved_sram': src, 'scratch_sram': scratch, 'restored_sram': old[4],
                            'saved_bytes': halo, 'remaining_external_bytes': old[5] - halo})
    transformed, shift = [], {}
    for i, command in enumerate(commands):
        transformed.extend(insert.get(i, ()))
        shift[i] = len(transformed)
        transformed.append(replace.get(i, command))
    assert len(transformed) <= 2048 and len(transformed) == len(commands) + 8 * sum(p['channels'] for p in PAIRS)
    assert sum(p['saved_bytes'] for p in patches) == 3072
    code = b''.join(CMD.pack(*item) for item in transformed)
    assert len(code) <= 32768 and len(payload) <= 8 * 1024 * 1024
    altered = dict(schedule)
    altered['schema'] = 2
    altered['catalogue'] = 'DeFiNES-inspired full-width VWW depth-first, vertical source-halo SRAM cache'
    altered['program_sha256'] = sha(code)
    altered['image_sha256'] = sha(payload)
    altered['command_count'] = len(transformed)
    altered['program_bytes'] = len(code)
    for key in ('run_contracts', 'constant_contracts', 'pack_contracts'):
        altered[key] = {str(shift[int(k)]): v for k, v in schedule.get(key, {}).items()}
    altered['vertical_cache'] = {'source_code_sha256': sha(src_code), 'source_image_sha256': sha((source / 'payload.bin').read_bytes()),
                                 'copy_descriptors': len(patches) * 2,
                                 'saved_external_bytes': sum(p['saved_bytes'] for p in patches),
                                 'patches': patches,
                                 'stale_prior_splice_offsets': True}
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.name not in ('commands.bin', 'payload.bin', 'schedule.json'):
            shutil.copy2(path, target / path.name)
    (target / 'commands.bin').write_bytes(code)
    (target / 'payload.bin').write_bytes(payload)
    save(target / 'schedule.json', altered)
    return {'status': 'lowered', 'command_count': len(transformed), 'new_copy_descriptors': len(patches) * 2,
            'saved_external_bytes': 3072, 'program_sha256': sha(code), 'image_sha256': sha(payload),
            'patches': len(patches)}


def native(fixture: Path, label: str, seeds=(0, 6063)) -> list[dict]:
    results = []
    for seed in seeds:
        path = OUT / 'native' / f'{label}-s{seed}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(NATIVE), str(fixture), str(seed), str(path)], check=True, cwd=ROOT)
        record = json.loads(path.read_text())
        assert record['status'] == 'passed' and record['tensor_checks'] >= 1
        results.append({'stall_seed': seed, 'status': record['status'], 'tensor_checks': record['tensor_checks'],
                        'elapsed_cycles': record['elapsed_cycles'], 'dma_cycles': record['dma_cycles'],
                        'engine_cycles': record['engine_cycles'], 'report': str(path.relative_to(ROOT)),
                        'report_sha256': file_sha(path)})
    return results


def checked_fixture(path: Path) -> dict:
    expected = ('commands.bin', 'payload.bin', 'input.bin', 'output.bin', 'checks.txt', 'schedule.json')
    assert all((path / name).exists() for name in expected)
    schedule = json.loads((path / 'schedule.json').read_text())
    assert sha((path / 'commands.bin').read_bytes()) == schedule['program_sha256']
    assert sha((path / 'payload.bin').read_bytes()) == schedule['image_sha256']
    assert 'output.bin' in (path / 'checks.txt').read_text()
    assert (path / 'commands.bin').stat().st_size == CMD.size * schedule['command_count']
    return {'directory': str(path.relative_to(ROOT)),
            'files': {p.name: file_sha(p) for p in path.iterdir() if p.is_file()},
            'replay': {'status': 'not-run', 'scope': 'hash and command-envelope checks only; exact RTL native is separate'}}


def symbolic(fixture: Path, model: str, label: str) -> dict:
    """Replay every command against an independent full-model integer oracle."""
    fixture = (ROOT / fixture).resolve()
    original, _, _, _ = load_model(model)
    grouped, _, _ = group_channels(original)
    graph, _, _, _ = compact_channels(grouped)
    model_input = graph.inputs[0]
    tensor = graph.tensors[model_input]
    input_bytes = (fixture / 'input.bin').read_bytes()
    if graph.layers[0].op == 'Transpose':
        first = graph.layers[0]
        physical_shape = graph.tensors[first.output].shape
        input_value = np.frombuffer(input_bytes, dtype=np.int8).reshape(physical_shape).transpose(
            np.argsort(first.attributes['perm']))
    else:
        input_value = np.frombuffer(input_bytes, dtype=np.int8).reshape(tensor.shape)
    assert input_value.shape == tensor.shape
    oracle = evaluate(graph, {model_input: input_value})
    assert oracle[graph.outputs[0]].tobytes() == (fixture / 'output.bin').read_bytes(), label
    schedule = json.loads((fixture / 'schedule.json').read_text())
    code = (fixture / 'commands.bin').read_bytes()
    payload = (fixture / 'payload.bin').read_bytes()
    result = replay_defines(graph, code, payload, {model_input: input_value},
                            run_contracts=schedule['run_contracts'],
                            final_output=schedule['final_output'],
                            snapshot_regions=schedule.get('snapshot_regions'),
                            constant_contracts=schedule.get('constant_contracts'),
                            pack_contracts=schedule.get('pack_contracts'), oracle=oracle)
    assert result['status'] == 'passed'
    record = {'status': 'passed', 'model': model,
              'fixture': str(fixture.relative_to(ROOT)),
              'fixture_files_sha256': {name: file_sha(fixture / name) for name in
                                       ('commands.bin', 'payload.bin', 'input.bin', 'output.bin', 'schedule.json')},
              'oracle_output_sha256': sha(oracle[graph.outputs[0]].tobytes()),
              'replay': result}
    path = OUT / 'replay' / f'{label}.json'
    save(path, record)
    return {'status': 'passed', 'report': str(path.relative_to(ROOT)),
            'report_sha256': file_sha(path), 'engine_runs': result['engine_runs'],
            'scope': result['scope']}


def kws_endpoint_selection() -> dict:
    """Select the legal KWS LBL endpoint on the same two-seed native objective."""
    parent = ROOT / 'work/phase6/matched-baselines-v1/b1b2/report.json'
    upstream = json.loads(parent.read_text())
    assert upstream['status'] == 'passed-native'
    variants = {}
    for policy in ('B1', 'B2'):
        source = upstream['policies'][policy]['models']['kws']['fixtures']['pinned_timed']
        fixture = ROOT / source['directory']
        row = checked_fixture(fixture)
        for name, digest in source['files'].items():
            assert row['files'][name] == digest, (policy, name)
        row['replay'] = symbolic(fixture, 'kws', f'kws-endpoint-{policy.lower()}')
        row['native'] = native(fixture, f'kws-endpoint-{policy.lower()}')
        row['worst_seed_cycles'] = max(item['elapsed_cycles'] for item in row['native'])
        row['source_policy'] = policy
        variants[policy] = row
    winner = min(variants, key=lambda name: (variants[name]['worst_seed_cycles'], name))
    report = {'schema': 1, 'status': 'passed-native',
              'selection_rule': 'minimum worst native elapsed_cycles over seeds 0 and 6063; tie by policy name',
              'scope': 'KWS full-layer legal LBL endpoints inherited from B1/B2 on matched engine and backend',
              'baseline_report': str(parent.relative_to(ROOT)),
              'baseline_report_sha256': file_sha(parent),
              'selection': winner, 'winner_worst_seed_cycles': variants[winner]['worst_seed_cycles'],
              'variants': variants}
    save(OUT / 'kws-endpoint-selection.json', report)
    return report


def splice_pair7_without_pair11(source: Path, target: Path, candidate_code: bytes,
                                candidate_payload: bytes, candidate: dict) -> dict:
    """Generalize the exact pair-7 splice to a layer-11 successor without fusion."""
    old_code = (source / 'commands.bin').read_bytes()
    old_payload = (source / 'payload.bin').read_bytes()
    old_schedule = json.loads((source / 'schedule.json').read_text())
    old = [CMD.unpack_from(old_code, i) for i in range(0, len(old_code), CMD.size)]
    stages = old_schedule['stages']
    first = next(s for s in stages if s['layer'] == 7)
    following = next(s for s in stages if s['layer'] == 11)

    def command_index(load):
        command = (1, 1, 0, load['ext'], load['sram'], load['bytes'])
        matches = [i for i, entry in enumerate(old) if entry == command]
        assert len(matches) == 1, (load, matches)
        return matches[0]

    descriptor = command_index(next(t for t in first['loads'] if t['role'] == 'descriptor'))
    cut_first = command_index(next(t for t in first['loads'] if t['role'] == 'parameter'))
    cut_end = command_index(next(t for t in following['loads'] if t['role'] == 'descriptor'))
    assert cut_first < descriptor < cut_end
    assert old[cut_first-1] == (3, 2, 0, 0, 0, 0)
    assert all(not cut_first <= int(k) < cut_end or 7 <= value['layer'] <= 10
               for k, value in old_schedule['run_contracts'].items())
    activation_ext = next(t['ext'] for t in first['loads'] if t['role'] == 'input')
    assert activation_ext == 36864
    payload = bytearray(old_payload)
    appended = (len(payload) + 7) & ~7
    payload.extend(b'\0' * (appended - len(payload)))
    payload.extend(candidate_payload[36864:])
    candidate_commands = [CMD.unpack_from(candidate_code, i) for i in range(0, len(candidate_code), CMD.size)]
    transfers = {t['command']: t for t in candidate['transfers']}
    runs = {r['command']: r for r in candidate['runs']}
    snapshots = {int(k): v for k, v in old_schedule['snapshot_regions'].items()}
    # The preceding layer-wise engine still owns source SRAM. Keep its wait,
    # intermediate snapshots (when present), and final SDRAM materialization.
    # The original layer-7 parameter prefetches before this wait are redundant.
    waits = [i for i in range(cut_first, descriptor)
             if old[i] == (3, 3, 0, 0, 0, 0)]
    block, new_runs = (list(old[waits[-1]:descriptor]) if waits else []), []
    for i, original in enumerate(candidate_commands[:-1]):
        op, flags, reserved, a, b, size = original
        if op == 1:
            transfer = transfers[i]
            if transfer['role'].startswith(('source_strip', 'prefetch_source', 'output_strip')):
                a += activation_ext
            else:
                assert flags == 1 and a >= 36864
                a = appended + a - 36864
            original = (op, flags, reserved, a, b, size)
        if op == 2:
            new_runs.append((len(block), runs[i]))
        block.append(original)
        if op == 3 and flags == 3 and i-1 in runs:
            stage = runs[i-1]
            layer = 8 if stage['layer'] == 0 else 10
            if layer in snapshots:
                descriptor, _ = decode_fused(bytes.fromhex(stage['descriptor_hex']))
                for channel in range(16 if layer == 8 else 32):
                    external = snapshots[layer]['ext'] + channel*576 + stage['strip']*288
                    assert external % 8 == (descriptor.output + channel*288) % 8 == 0
                    block.extend(((1, 0, 0, external, descriptor.output + channel*288, 288),
                                  (3, 2, 0, 0, 0, 0)))
    # The old layer-9 output store follows successor parameter prefetches.
    # Candidate strips already materialize the full output and snapshots; only
    # the successor's independent parameter loads may be retained.
    for load in following['loads']:
        if load['role'] != 'parameter':
            continue
        index = command_index(load)
        if cut_first <= index < cut_end:
            assert old[index + 1] == (3, 2, 0, 0, 0, 0)
            block.extend((old[index], old[index + 1]))
    combined = old[:cut_first] + block + old[cut_end:]
    code = b''.join(CMD.pack(*item) for item in combined)
    assert len(combined) <= 2048 and len(code) <= 32768 and len(payload) <= 8*1024*1024
    delta = len(block) - (cut_end - cut_first)
    schedule = copy.deepcopy(old_schedule)
    schedule['stages'] = [s for s in stages if not 7 <= s['layer'] <= 10]
    for key in ('run_contracts', 'constant_contracts', 'pack_contracts'):
        schedule[key] = {str(i if i < cut_first else i + delta): value
                         for index, value in old_schedule.get(key, {}).items()
                         if not cut_first <= (i := int(index)) < cut_end}
    for index, stage in new_runs:
        schedule['run_contracts'][str(cut_first + index)] = {
            'layer': 7 if stage['layer'] == 0 else 9,
            'fused_activation_layer': 8 if stage['layer'] == 0 else 10,
            'fused_table_sha256': stage['table_sha256'],
            'strip_origin_y': stage['strip']*12, 'strip_height': 12}
    schedule.update(command_count=len(combined), program_bytes=len(code),
                    program_sha256=sha(code), image_sha256=sha(payload),
                    catalogue='independently selected depth-first pair7 cut with layer-wise layer11 successor',
                    pair7_splice={'source_fixture': source.name, 'cut_first': cut_first,
                                  'cut_end': cut_end, 'inserted_commands': len(block),
                                  'source_sha256': sha(old_code), 'source_payload_sha256': sha(old_payload)})
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)
    (target / 'commands.bin').write_bytes(code)
    (target / 'payload.bin').write_bytes(payload)
    save(target / 'schedule.json', schedule)
    return {'status': 'lowered', 'cut_first': cut_first, 'cut_end': cut_end,
            'command_count': len(combined), 'program_sha256': sha(code)}


def restore_preceding_store(source: Path, target: Path) -> dict:
    """Retain a preceding layer-wise producer store removed by legacy splice."""
    old = [CMD.unpack_from((source / 'commands.bin').read_bytes(), i)
           for i in range(0, (source / 'commands.bin').stat().st_size, CMD.size)]
    prior = json.loads((source / 'schedule.json').read_text())
    target_code = (target / 'commands.bin').read_bytes()
    current = [CMD.unpack_from(target_code, i) for i in range(0, len(target_code), CMD.size)]
    schedule = json.loads((target / 'schedule.json').read_text())
    cut = schedule['pair7_splice']['cut_first']
    first = next(s for s in prior['stages'] if s['layer'] == 7)
    load = next(t for t in first['loads'] if t['role'] == 'descriptor')
    descriptor = old.index((1, 1, 0, load['ext'], load['sram'], load['bytes']))
    waits = [i for i in range(cut, descriptor) if old[i] == (3, 3, 0, 0, 0, 0)]
    assert len(waits) == 1
    prefix = old[waits[0]:descriptor]
    assert any(item[0] == 1 and item[1] == 0 for item in prefix)
    current[cut:cut] = prefix
    code = b''.join(CMD.pack(*item) for item in current)
    delta = len(prefix)
    for key in ('run_contracts', 'constant_contracts', 'pack_contracts'):
        schedule[key] = {str(i if i < cut else i + delta): value
                         for index, value in schedule.get(key, {}).items()
                         for i in (int(index),)}
    schedule['pair7_splice']['preceding_producer_store_restored'] = delta
    schedule['pair7_splice']['inserted_commands'] += delta
    schedule.update(command_count=len(current), program_bytes=len(code), program_sha256=sha(code))
    (target / 'commands.bin').write_bytes(code)
    save(target / 'schedule.json', schedule)
    return {'status': 'lowered', 'inserted_commands': delta, 'program_sha256': sha(code)}


def enumerate_catalogue() -> dict:
    """Evaluate all eight legal combinations of the three fixed fusion cuts."""
    base = ROOT / 'work/phase6/channel-compaction-v1/fused/fixtures/vww-pinned-compacted-fused-timed'
    source3 = ROOT / 'work/phase6/strip-fusion-vww-v1/full/fixtures/vww-pinned-compacted-strip-timed'
    source311 = ROOT / 'work/phase6/strip-fusion-pair11-v1/full/fixtures/vww-pinned-compacted-strip3-11-timed'
    source3711 = ROOT / 'work/phase6/strip-fusion-pair7-v1/full/fixtures/vww-pinned-compacted-strip3-7-11-timed'
    block11 = pair11_builder.model_block()[1]
    code11, image11, plan11 = pair11_builder.build(block11)
    block7 = pair7_builder.model_block()[1]
    code7, image7, plan7 = pair7_builder.build(block7)
    input_value = np.frombuffer((base / 'input.bin').read_bytes(), dtype=np.int8)
    root = OUT / 'catalogue'
    roots = {'000': base, '100': source3, '101': source311, '111': source3711}
    for bits in ('001', '011'):
        src = roots['000'] if bits == '001' else roots['001']
        dest = root / f'pair-{bits}'
        if bits == '001':
            pair11_builder.splice(src, dest, code11, image11, plan11, input_value, False)
        else:
            pair7_builder.splice(src, dest, code7, image7, plan7, input_value, False)
            restore_preceding_store(src, dest)
        roots[bits] = dest
    for bits in ('010', '110'):
        src = roots['000'] if bits == '010' else roots['100']
        dest = root / f'pair-{bits}'
        splice_pair7_without_pair11(src, dest, code7, image7, plan7)
        roots[bits] = dest
    expected_input = sha((base / 'input.bin').read_bytes())
    expected_output = sha((base / 'output.bin').read_bytes())
    variants = {}
    for bits in sorted(roots):
        path = roots[bits]
        assert sha((path / 'input.bin').read_bytes()) == expected_input
        assert sha((path / 'output.bin').read_bytes()) == expected_output
        row = checked_fixture(path)
        row['cut_bits_3_7_11'] = bits
        row['replay'] = symbolic(path, 'vww', f'catalogue-{bits}')
        row['native'] = native(path, f'catalogue-{bits}')
        row['worst_seed_cycles'] = max(r['elapsed_cycles'] for r in row['native'])
        variants[bits] = row
        save(OUT / 'catalogue.partial.json', {'status': 'running', 'variants': variants})
    winner = min(variants, key=lambda bits: (variants[bits]['worst_seed_cycles'], bits))
    selected = variants[winner]
    report = {'schema': 1, 'status': 'passed-native', 'selection_rule': 'minimum worst native elapsed_cycles over seeds 0 and 6063; tie by cut bits',
              'cut_bits_order': [3, 7, 11], 'mode': 'vertical recompute; horizontal cache vacuous at full width',
              'upstream_geometry_report': str((OUT/'upstream-geometry.json').relative_to(ROOT)),
              'upstream_geometry_sha256': file_sha(OUT/'upstream-geometry.json'),
              'candidate_tile_shapes': {'3': [24,48], '7': [12,24], '11': [12,24]},
              'upstream_policy_role': 'author backpropagate_tilesize establishes candidate halo/cache geometry; local backend lowers feasible fixed cuts and exact RTL native cycles select among them',
              'selection': winner, 'winner_worst_seed_cycles': selected['worst_seed_cycles'], 'variants': variants,
              'coverage': 'all 2^3 combinations of the three legal fixed full-width four-layer VWW strip cuts; arbitrary tiles and deeper fusion absent'}
    save(OUT / 'catalogue-selection.json', report)
    return report


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = source_audit()
    shape = geometry()
    save(OUT / 'upstream-geometry.json', shape)
    kws_selection = kws_endpoint_selection()
    assert kws_selection['selection'] == 'B2'
    selected_src = ROOT / 'work/phase6/strip-fusion-pair7-v1/full/fixtures'
    candidates = {}
    for model, src_name, dst_name in (
        ('kws', 'kws-pinned-compacted-fused-timed', 'kws-pinned-timed'),
        ('kws', 'kws-pinned-compacted-fused-check', 'kws-pinned-check'),
        ('kws', 'kws-stress-compacted-fused-check', 'kws-stress-check'),
        ('vww', 'vww-pinned-compacted-strip3-7-11-timed', 'vww-pinned-timed'),
        ('vww', 'vww-pinned-compacted-strip3-7-11-check', 'vww-pinned-check'),
        ('vww', 'vww-stress-compacted-strip3-7-11-check', 'vww-stress-check')):
        source = selected_src / src_name
        if model == 'kws' and not source.exists():
            source = ROOT / 'work/phase6/channel-compaction-v1/fused/fixtures' / src_name
        dest = OUT / 'fixtures' / dst_name
        shutil.copytree(source, dest, dirs_exist_ok=True)
        row = checked_fixture(dest)
        row['source_directory'] = str(source.relative_to(ROOT))
        row['replay'] = symbolic(dest, model, dst_name)
        row['native'] = native(dest, dst_name)
        role = 'stress_check' if 'stress' in dst_name else 'pinned_check' if 'check' in dst_name else 'pinned_timed'
        candidates.setdefault(model, {})[role] = row
    for role in ('pinned_timed', 'pinned_check', 'stress_check'):
        selected_files = candidates['kws'][role]['files']
        endpoint_files = json.loads((ROOT/'work/phase6/matched-baselines-v1/b1b2/report.json').read_text())['policies'][
            kws_selection['selection']]['models']['kws']['fixtures'][role]['files']
        assert all(selected_files[name] == endpoint_files[name] for name in
                   ('commands.bin', 'payload.bin', 'input.bin', 'output.bin'))
    selection = enumerate_catalogue()
    winner = selection['selection']
    assert winner == '111'  # report outcome, not a scheduler constraint
    assert selection['variants'][winner]['files']['commands.bin'] == candidates['vww']['pinned_timed']['files']['commands.bin']
    assert selection['variants'][winner]['files']['payload.bin'] == candidates['vww']['pinned_timed']['files']['payload.bin']
    cache_source = selected_src / 'vww-pinned-compacted-strip3-7-11-timed'
    cache_dest = OUT / 'candidates' / 'vww-pinned-vertical-cache'
    cache = cache_variant(cache_source, cache_dest)
    cache['fixture'] = checked_fixture(cache_dest)
    cache['fixture']['replay'] = symbolic(cache_dest, 'vww', 'vww-pinned-vertical-cache')
    cache['native'] = native(cache_dest, 'vww-pinned-vertical-cache')
    cache['fixture']['native'] = cache['native']
    stress_source = selected_src / 'vww-stress-compacted-strip3-7-11-check'
    stress_dest = OUT / 'candidates' / 'vww-stress-vertical-cache'
    cache_stress = cache_variant(stress_source, stress_dest)
    cache_stress['fixture'] = checked_fixture(stress_dest)
    cache_stress['fixture']['replay'] = symbolic(stress_dest, 'vww', 'vww-stress-vertical-cache')
    cache_stress['native'] = native(stress_dest, 'vww-stress-vertical-cache')
    cache_stress['fixture']['native'] = cache_stress['native']
    cache['fixtures'] = {'pinned_timed': cache['fixture'], 'stress_check': cache_stress['fixture']}
    report = {'schema': 1, 'status': 'passed-native', 'physical_board': False,
              'scope': 'restricted executable DeFiNES adaptation; all tiles full-width and three fixed VWW 4-layer depth-first blocks',
              'source_sha256': {name: file_sha(ROOT/name) for name in (
                  'tools/phase6/matched_defines_adapt.py', 'tools/phase6/strip_fusion_vww.py',
                  'tools/phase6/strip_fusion_pair11.py', 'tools/phase6/strip_fusion_pair7.py',
                  'tools/phase6/channel_compaction.py', 'tools/phase6/followup_graph.py',
                  'tools/phase6/run_boardless.py', 'compiler/scheduler/defines_adapter.py',
                  'compiler/hardware_v2.py', 'compiler/integer_reference.py',
                  'compiler/scheduler/fused_verify.py', 'compiler/scheduler/defines_verify.py',
                  'compiler/static_pipeline.py', 'compiler/phase4_compile.py',
                  'compiler/canonicalize.py', 'compiler/quantization.py',
                  'compiler/phase4_tiling.py', 'compiler/scheduler/contract.py',
                  'compiler/scheduler/resident.py', 'tools/phase6/fused_activation.py',
                  'tools/phase6/output_pipeline_fusion.py',
                  'work/phase6/channel-compaction-v1/fused/report.json',
                  'work/phase6/strip-fusion-vww-v1/full/report.json',
                  'work/phase6/strip-fusion-pair11-v1/full/report.json',
                  'work/phase6/strip-fusion-pair7-v1/full/report.json',
                  'work/phase6/matched-baselines-v1/b1b2/report.json',
                  'work/phase6/defines-source.tar.gz',
                  f'work/phase6/defines-source/DeFiNES-{REVISION}/classes/workload/dnn_workload.py',
                  f'work/phase6/defines-source/DeFiNES-{REVISION}/classes/stages/DepthFirstStage.py',
                  f'work/phase6/defines-source/DeFiNES-{REVISION}/classes/hardware/architecture/memory_level.py')},
              'engine_sha256': file_sha(ENGINE), 'native_executable_sha256': file_sha(NATIVE),
              'model_source_sha256': {path: digest for model in ('kws', 'vww')
                                      for path, digest in load_model(model)[3].items()},
              'upstream': provenance, 'upstream_geometry_report': str((OUT / 'upstream-geometry.json').relative_to(ROOT)),
              'modes': {'fully_recompute': 'executable via selected strip pair schedules; source halo refetched from SDRAM',
                        'horizontal_cache_vertical_recompute': 'horizontal cache is vacuous for full-width strips',
                        'full_cache': 'tested via legal SRAM COPY descriptors; per-channel halo retained across output overwrite'},
              'unsupported': ['arbitrary x/y tile sizes', 'arbitrary fusion depths and stack cuts',
                              'on-chip horizontal overlap when tile width is below full width; the interior 8x16 layer11-14 tile fits SRAM but needs row gather/scatter support',
                              'cross-layer cache placement beyond these three fixed blocks',
                              'upstream ideal-array energy/latency model is not the Tang Nano backend'],
              'omitted_resource_feasible_point': {
                  'layers': [11, 14], 'output_tile': [8, 16],
                  'upstream_input_tile_no_cache': [10, 18],
                  'estimated_sram_bytes_with_descriptors_weights_params': 16896,
                  'estimated_horizontal_halo_bytes': 640,
                  'estimated_vertical_halo_bytes': 1152,
                  'capacity_limit_bytes': 32768,
                  'lowering_blocker': 'aligned contiguous DMA/COPY cannot row-pack 18-wide interior input from a 24-wide plane; naive six-tile output scatter uses 1536 DMA plus 1536 WAIT commands, exceeding the 2048-command store',
                  'status': 'geometry/resource feasibility only; no compiled executable or timing result'},
              'policies': {'B3_restricted': {
                  'description': 'fixed full-width DeFiNES-style depth-first cut catalogue on inherited eight-lane backend',
                  'independent_policy_generation': True,
                  'fully_eligible_b3': False,
                  'selection_report': str((OUT/'catalogue-selection.json').relative_to(ROOT)),
                  'selection_report_sha256': file_sha(OUT/'catalogue-selection.json'),
                  'fixed_fallback_models': ['kws'],
                  'kws_endpoint_selection_report': str((OUT/'kws-endpoint-selection.json').relative_to(ROOT)),
                  'kws_endpoint_selection_report_sha256': file_sha(OUT/'kws-endpoint-selection.json'),
                  'models': {model: {'fixtures': rows} for model, rows in candidates.items()}}},
              'vertical_cache_candidate': cache, 'vertical_cache_stress': cache_stress,
              'limitations': ['the selected VWW recompute command stream is byte-identical to B4 and is a valid tie only in the restricted catalogue',
                              'KWS B2 endpoint wins the restricted LBL selection and is byte-identical to B4; no separate KWS strip mode is lowerable',
                              'arbitrary tile widths/heights and deeper fuse depths could dominate this subset and remain untested']}
    save(OUT / 'report.json', report)
    print(json.dumps({'status': report['status'], 'selected_cycles': {model: [r['elapsed_cycles'] for r in rows['pinned_timed']['native']]
                      for model, rows in candidates.items()},
                      'cache_cycles': [r['elapsed_cycles'] for r in cache['native']]}, indent=2))


if __name__ == '__main__':
    main()
