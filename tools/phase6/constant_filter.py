#!/usr/bin/env python3
"""Experimental exact constant-filter lowering on the existing tiled ABI.

An entire all-zero output filter has an input-independent INT8 output, but
usually shares a tile with nonzero filters. Split that tile into contiguous
channel runs. DMA bias-derived bytes into its output SRAM for zero runs and
execute ordinary Conv descriptors for the remaining runs. All other stages
and the model's quantization boundaries retain the chain schedule semantics.

This is an isolated experiment. It neither alters the pinned model nor the
Phase 5 compiler/RTL, and it makes no physical speedup claim by itself.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'compiler'))
sys.path.insert(0, str(ROOT / 'tools/phase6'))

from hardware_v2 import Descriptor
from integer_reference import evaluate, rounded
from quantization import requantize
from chain_resident import compile_chain
from run_boardless import load_model
from scheduler.resident_verify import replay_resident
from variants import check_frozen, sha
from padded_descriptor import decode_experimental, encode_pointwise


def filter_constants(program, layer_index):
    """Return the exact constant code per channel, or None for a live filter."""
    layer = program.layers[layer_index]
    if layer.op != 'Conv' or layer.attributes.get('group', 1) != 1:
        return None
    p = layer.parameters
    weights = np.asarray(p['weight'], dtype=np.int8)
    zero = np.all(weights == 0, axis=tuple(range(1, weights.ndim)))
    output_zp = program.tensors[layer.output].quantization.zero_point
    codes = []
    for channel, is_zero in enumerate(zero):
        if not is_zero:
            codes.append(None)
            continue
        # A zero row makes the zero-point correction exactly zero. A copied
        # value must be the result of requantizing the bias, not INT8 zero.
        if int(p['corrected_bias'][channel]) != int(p['bias'][channel]):
            raise ValueError('zero filter has nonzero input correction')
        bias = int(p['bias'][channel])
        multiplier, shift = int(p['multiplier'][channel]), int(p['shift'][channel])
        value = int(requantize(np.asarray([bias], dtype=np.int64), multiplier, shift, output_zp)[0])
        independent = int(rounded(np.asarray([bias * multiplier], dtype=np.int64), shift, output_zp)[0])
        if value != independent:
            raise ValueError('constant requantization disagreement')
        codes.append(value)
    return tuple(codes)


def _runs(codes):
    """Maximal alternating zero-filter/nonzero-filter runs."""
    beginning = 0
    while beginning < len(codes):
        constant = codes[beginning] is not None
        end = beginning + 1
        while end < len(codes) and (codes[end] is not None) == constant:
            end += 1
        yield beginning, end, constant
        beginning = end


def compile_constant_chain(program, snapshots=False, *, reuse_sibling_inputs=False,
                           padded_pointwise=False):
    """Compile a normal chain and replace every eligible Conv tile exactly."""
    constants = {i: filter_constants(program, i) for i in range(len(program.layers))
                 if program.layers[i].op == 'Conv' and
                 program.layers[i].attributes.get('group', 1) == 1}
    active_channels = {i: tuple(code is None for code in row)
                       for i,row in constants.items()} if padded_pointwise else None
    original_code, raw_payload, original = compile_chain(program, snapshots,
        reuse_sibling_inputs=reuse_sibling_inputs, padded_pointwise=padded_pointwise,
        active_pointwise_channels=active_channels)
    if not any(any(code is not None for code in row) for row in constants.values()):
        return original_code, raw_payload, dict(original, constant_filter=dict(
            enabled=False, skipped_dense_macs=0, zero_channels=0, split_tiles=0,
            reason='no all-zero Conv output filters'))

    payload = bytearray(raw_payload)
    stages = copy.deepcopy(original['stages'])
    commands = []
    contracts = {}
    pack_contracts = {}
    constant_contracts = {}
    skipped_macs = 0
    constant_outputs = 0
    split_tiles = 0
    descriptor_runs = 0
    constant_runs = 0

    def emit(op, flags=0, a=0, b=0, c=0):
        commands.append((op, flags, a, b, c))

    def dma(t):
        if (t['sram'] % 8 or t['ext'] % 8 or not 0 < t['bytes'] <= 32768
                or t['sram'] + t['bytes'] > 32768 or t['ext'] + t['bytes'] > len(payload)):
            raise ValueError('invalid constant-filter DMA geometry')
        emit(1, int(t['direction'] == 'to_sram'), t['ext'], t['sram'], t['bytes'])
        emit(3, 2)

    for s in stages:
        layer_index = s['layer']
        layer = program.layers[layer_index]
        d, is_padded = decode_experimental(bytes.fromhex(s['descriptor_hex']))
        plane = math.prod(program.tensors[layer.output].shape[2:]) if layer.op == 'Conv' else 0
        selected = (constants.get(layer_index) or ())[s['first_element']//plane:
                     (s['first_element']+d.outputs)//plane] if layer.op == 'Conv' else ()
        needs_split = bool(selected) and any(code is not None for code in selected)
        if 'padded_pack' in s:
            for t in s['loads']:
                if t['role'] == 'input':
                    dma(t)
            dma(s['padded_pack']['descriptor_load'])
            pack_contracts[str(len(commands))] = dict(layer=layer_index,
                                                       first_element=s['first_element'])
            emit(2, 0, s['pc'], s['live'][0] | s['live'][1] << 16)
            emit(3, 3)
        if not needs_split:
            for t in s['loads']:
                if 'padded_pack' in s and t['role'] == 'input': continue
                dma(t)
            contracts[str(len(commands))] = dict(layer=layer_index, first_element=s['first_element'])
            emit(2, 0, s['pc'], s['live'][0] | s['live'][1] << 16)
            emit(3, 3)
        else:
            split_tiles += 1
            if d.opcode != 4 or d.outputs != len(selected)*plane or d.output_c != len(selected):
                raise ValueError('constant-filter lowering only supports ordinary Conv channel tiles')
            if d.output % 8 or d.outputs % 8:
                raise ValueError('constant-filter tile output must have aligned bounds')
            # Keep the original input, weight and parameter loads once. They
            # share an SRAM region across all split descriptors. The unused
            # whole-tile descriptor transfer can be omitted.
            for t in s['loads']:
                if t['role'] != 'descriptor' and not ('padded_pack' in s and t['role'] == 'input'):
                    dma(t)
            tile = bytearray(d.outputs)
            for channel, value in enumerate(selected):
                if value is not None:
                    tile[channel*plane:(channel+1)*plane] = bytes([value & 255])*plane
            parts = list(_runs(selected))
            # Do all DMA fills before any Conv run: aligned fills may touch a
            # neighboring live-filter byte, which the later Conv overwrites.
            for begin, end, is_constant in parts:
                if not is_constant:
                    continue
                first = begin*plane
                last = end*plane
                aligned_first = first & ~7
                aligned_last = (last+7) & ~7
                if aligned_last > d.outputs:
                    raise ValueError('constant DMA extends beyond output tile')
                ext = (len(payload)+7) & ~7
                payload.extend(bytes(ext-len(payload)))
                payload.extend(tile[aligned_first:aligned_last])
                byte_count = aligned_last-aligned_first
                key = str(len(commands))
                constant_contracts[key] = dict(layer=layer_index,
                    first_element=s['first_element']+first,
                    bytes=last-first, dma_first=aligned_first,
                    dma_bytes=byte_count, channel_first=s['first_element']//plane+begin,
                    channels=end-begin, tile_first_element=s['first_element'],
                    tile_output_sram=d.output, tile_output_bytes=d.outputs)
                dma(dict(direction='to_sram', ext=ext, sram=d.output+aligned_first, bytes=byte_count))
                constant_runs += 1
                constant_outputs += last-first
                skipped_macs += (end-begin)*plane*d.count
            for begin, end, is_constant in parts:
                if is_constant:
                    continue
                sub = copy.deepcopy(d)
                sub.output += begin*plane
                sub.weight += begin*d.row_stride
                sub.params += begin*16
                sub.output_c = end-begin
                sub.outputs = (end-begin)*plane
                sub.validate()
                ext = (len(payload)+7) & ~7
                payload.extend(bytes(ext-len(payload)))
                payload.extend((encode_pointwise(sub) if is_padded else sub.encode())+
                               Descriptor(0).encode())
                dma(dict(direction='to_sram', ext=ext, sram=s['pc'], bytes=128))
                contracts[str(len(commands))] = dict(layer=layer_index,
                    first_element=s['first_element']+begin*plane)
                emit(2, 0, s['pc'], s['live'][0] | s['live'][1] << 16)
                emit(3, 3)
                descriptor_runs += 1
            s['constant_filter_parts'] = [dict(first_channel=s['first_element']//plane+begin,
                                               channels=end-begin,
                                               constant=is_constant)
                                          for begin, end, is_constant in parts]
        if snapshots:
            region = original['snapshot_regions'][layer_index]
            dma(dict(direction='from_sram', ext=region['ext']+s['first_element'], **s['output']))
        if s['store'] is not None:
            dma(s['store'])
    emit(0)
    code = b''.join(struct.pack('<BBHIII', op, flags, 0, a, b, c)
                    for op, flags, a, b, c in commands)
    if len(code) > 32768 or len(payload) > 8*1024*1024:
        raise ValueError('constant-filter program capacity')
    schedule = dict(original, stages=stages, run_contracts=contracts,
        pack_contracts=pack_contracts,
        constant_contracts=constant_contracts,
        catalogue='experimental exact constant-filter DMA plus ordinary Conv runs',
        command_count=len(commands), program_bytes=len(code),
        program_sha256=hashlib.sha256(code).hexdigest(),
        image_sha256=hashlib.sha256(payload).hexdigest(),
        constant_filter=dict(enabled=True, split_tiles=split_tiles,
                             constant_runs=constant_runs, ordinary_conv_runs=descriptor_runs,
                             zero_channels=sum(code is not None for row in constants.values() for code in row),
                             constant_output_bytes=constant_outputs,
                             skipped_dense_macs=skipped_macs))
    return code, bytes(payload), schedule


def prepare(output, *, reuse_sibling_inputs=False, padded_pointwise=False):
    """Emit isolated native fixtures and check constants on pinned/stress data."""
    check_frozen()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name in ('kws', 'vww'):
        program, pinned, _, source_hashes = load_model(name)
        for sample, value in (('pinned', pinned),
                              ('stress', np.random.default_rng(6073).integers(-128, 128, pinned.shape, dtype=np.int8))):
            oracle = evaluate(program, {program.inputs[0]: value})
            initial = oracle[program.layers[0].output] if program.layers[0].op == 'Transpose' else value
            # Independent centered-input oracle verifies every emitted
            # constant channel for two unrelated input tensors.
            for layer_index, layer in enumerate(program.layers):
                codes = filter_constants(program, layer_index)
                if codes is None:
                    continue
                for channel, code_value in enumerate(codes):
                    if code_value is not None and not np.all(oracle[layer.output][:, channel] == code_value):
                        raise ValueError(f'{name} layer {layer_index} channel {channel}: nonconstant oracle')
            for snapshots in ((True, False) if sample == 'pinned' else (True,)):
                code, payload, schedule = compile_constant_chain(program, snapshots=snapshots,
                    reuse_sibling_inputs=reuse_sibling_inputs,
                    padded_pointwise=padded_pointwise)
                replay = replay_resident(program, code, payload, {program.inputs[0]: value},
                    run_contracts=schedule['run_contracts'],
                    constant_contracts=schedule.get('constant_contracts'),
                    pack_contracts=schedule.get('pack_contracts'),
                    final_output=schedule['final_output'],
                    snapshot_regions=schedule['snapshot_regions'], oracle=oracle)
                suffix = 'check' if snapshots else 'timed'
                technique = ('constant-sibling-padded' if padded_pointwise else
                             'constant-sibling' if reuse_sibling_inputs else 'constant')
                fixture = output / 'fixtures' / f'{name}-{sample}-{technique}-{suffix}'
                fixture.mkdir(parents=True, exist_ok=True)
                files = dict(
                    **{'commands.bin': code, 'payload.bin': payload, 'input.bin': initial.tobytes(),
                       'output.bin': oracle[program.outputs[0]].tobytes()},
                    **{f'layer-{i}.bin': oracle[layer.output].tobytes()
                       for i, layer in enumerate(program.layers) if i in schedule['snapshot_regions']})
                checks = [f'{schedule["final_output"]["ext"]} output.bin']
                checks += [f'{region["ext"]} layer-{i}.bin'
                           for i, region in schedule['snapshot_regions'].items()]
                files['checks.txt'] = ('\n'.join(checks)+'\n').encode()
                files['schedule.json'] = (json.dumps(schedule, indent=2, sort_keys=True)+'\n').encode()
                for filename, data in files.items():
                    (fixture/filename).write_bytes(data)
                records.append(dict(name=fixture.name, model=name, sample=sample,
                                    pins=source_hashes,
                                    verification=replay,
                                    constants=schedule['constant_filter'],
                                    files={key: sha(fixture/key) for key in files}))
    manifest = dict(schema=1, status='passed-replay', physical_board=False,
                    claim='exact bias-derived constants checked against independent full-model oracle; physical board pending',
                    fixtures=records)
    if reuse_sibling_inputs:
        manifest['sibling_input_retention'] = True
        manifest['claim'] = ('exact bias-derived constants plus sibling input retention '
                             'checked against independent full-model oracle; physical board pending')
    if padded_pointwise:
        manifest['padded_pointwise'] = True
        manifest['claim'] = ('experimental in-place PACK and padded pointwise channels checked '
                             'against independent full-model oracle; native RTL pending')
    (output/'fixtures.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    check_frozen()
    return manifest


def native(output, label='all-exact'):
    """Run the pinned RTL host bridge on all check/timed fixtures, no board."""
    check_frozen()
    manifest_path = output/'fixtures.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'passed-replay':
        raise ValueError('replay manifest missing')
    candidate = ROOT/'work/phase6/experiments-v1'/label
    original = json.loads((candidate/'native/report.json').read_text())
    if original['status'] != 'passed':
        raise ValueError('selected RTL native baseline not passed')
    for filename, digest in original['sources'].items():
        if sha(ROOT/filename) != digest:
            raise ValueError(f'native RTL source changed: {filename}')
    executable = candidate/'native/Vv2_tiled_host_bridge'
    if original.get('executable_sha256') and sha(executable) != original['executable_sha256']:
        raise ValueError('selected RTL native executable changed')
    target = output/f'native-{label}'
    target.mkdir(parents=True, exist_ok=True)
    report_path = target/'report.json'
    report = dict(status='running', label=label, physical_board=False,
                  executable_sha256=sha(executable), sources=original['sources'],
                  fixture_manifest_sha256=sha(manifest_path), results=[])
    for fixture in manifest['fixtures']:
        directory = output/'fixtures'/fixture['name']
        for filename, digest in fixture['files'].items():
            if sha(directory/filename) != digest:
                raise ValueError(f'fixture changed: {fixture["name"]}/{filename}')
        for seed in (0, 6063):
            result_path = target/f'{fixture["name"]}-s{seed}.json'
            start = time.monotonic()
            subprocess.run([str(executable), str(directory), str(seed), str(result_path)], check=True)
            row = json.loads(result_path.read_text())
            if row['status'] != 'passed':
                raise ValueError('constant-filter RTL mismatch')
            row.update(fixture=fixture['name'], fixture_files=fixture['files'],
                       simulation_seconds=time.monotonic()-start)
            report['results'].append(row)
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    report['status'] = 'passed'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    check_frozen()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--native', nargs='?', const='all-exact')
    parser.add_argument('--sibling-inputs', action='store_true')
    parser.add_argument('--padded-pointwise', action='store_true')
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT/('work/phase6/padded-stride-v1' if args.padded_pointwise else
                            'work/phase6/constant-sibling-v1' if args.sibling_inputs
                            else 'work/phase6/constant-filter-v1')
    if args.native:
        report = native(args.output, args.native)
        print(report['status'], len(report['results']), flush=True)
    else:
        manifest = prepare(args.output, reuse_sibling_inputs=args.sibling_inputs,
                           padded_pointwise=args.padded_pointwise)
        for row in manifest['fixtures']:
            print(row['name'], row['constants'], flush=True)
