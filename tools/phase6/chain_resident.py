#!/usr/bin/env python3
"""Experimental full-tensor SRAM retention, verified by independent replay.

Uses the existing ABI and exact arithmetic. Serialization is intentional and
its performance cost is measured, not omitted. Partial tensors fall back to
SDRAM; a failed allocation keeps the original legal stage placement.
"""
import copy
import argparse
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

from variants import ROOT, check_frozen, sha
from run_boardless import load_model
from integer_reference import evaluate
from hardware_v2 import Descriptor
from scheduler.resident import compile_resident
from scheduler.resident_verify import replay_resident
from run_screening import save_json, verify_files
from padded_descriptor import align8, encode_pack, encode_pointwise, physical_input_bytes

BASE = ROOT / 'work/phase6/experiments-v1/chain'
SIBLING_BASE = ROOT / 'work/phase6/experiments-v1/sibling-input-retention'


def _input_load(stage):
    return next((t for t in stage['loads'] if t['role'] == 'input'), None)


def _sibling_predecessors(stages):
    """Find producer/activation/producer triples consuming the same input tile.

    The retained input must have identical external bytes and tensor identity.
    Its SRAM address is then verified by the allocation and independent replay.
    """
    result = set()
    for i in range(len(stages)-2):
        first, activation, sibling = stages[i:i+3]
        first_input, sibling_input = _input_load(first), _input_load(sibling)
        if (not first['inplace'] and activation['inplace'] and not sibling['inplace']
                and first['layer'] == sibling['layer']
                and first['group'] == activation['group'] == sibling['group']
                and first['first_element'] == activation['first_element']
                and sibling['first_element'] > first['first_element']
                and first_input is not None and sibling_input is not None
                and (first_input['ext'], first_input['bytes']) ==
                    (sibling_input['ext'], sibling_input['bytes'])):
            result.add(i)
    return result


def _profitable_padded_pointwise(d, active_channels=None):
    """A structural gate: save at least 20 cross-word reads per PACK word."""
    if (d.opcode != 4 or (d.kernel_h, d.kernel_w, d.stride_h, d.stride_w) != (1, 1, 1, 1)
            or any((d.pad_top, d.pad_bottom, d.pad_left, d.pad_right))):
        return False
    logical = d.input_h*d.input_w
    if logical == align8(logical):
        return False
    crossing = sum(((ch*logical+pixel)&7)+min(8,logical-pixel)>8
                   for ch in range(d.input_c) for pixel in range(0,logical,8))
    return crossing*(d.output_c if active_channels is None else active_channels) >= 20*d.input_c*(align8(logical)//8)


def compile_chain(program, snapshots, *, reuse_sibling_inputs=False, padded_pointwise=False,
                  active_pointwise_channels=None):
    _, raw, original = compile_resident(program, prefer_half=False, overlap=False, snapshots=snapshots)
    payload = bytearray(raw)
    stages = copy.deepcopy(original['stages'])
    aliases = {}
    for layer in program.layers:
        aliases[layer.output] = aliases.get(layer.inputs[0], layer.inputs[0]) if layer.op in (
            'Identity', 'Flatten', 'Reshape', 'Transpose') else layer.output
    sibling_predecessors = _sibling_predecessors(stages) if reuse_sibling_inputs else set()
    previous = None
    retained = 0
    retained_sibling_input_bytes = 0
    pending_input = None
    shared_input = None
    for index, s in enumerate(stages):
        layer = program.layers[s['layer']]
        input_load = _input_load(s)
        can_retain = (previous is not None and
            aliases.get(layer.inputs[0], layer.inputs[0]) == aliases.get(program.layers[previous['layer']].output) and
            ((s['inplace'] and previous['first_element'] == s['first_element']) or
             (previous['first_element'] == 0 and previous['output']['bytes'] == math.prod(program.tensors[layer.inputs[0]].shape))))
        # The inplace producer/activation pair is already legal in the seed schedule.
        if s['inplace'] and not can_retain: raise ValueError('broken producer/activation adjacency')
        candidate = shared_input if not s['inplace'] else None
        shared_input = None
        reuse_input = (candidate is not None and input_load is not None and
            (s['layer'], input_load['ext'], input_load['bytes']) ==
            (candidate['layer'], candidate['ext'], candidate['bytes']))
        protected = pending_input if s['inplace'] and index-1 in sibling_predecessors else None
        pending_input = None

        def place(*, reuse, protect, padded=False):
            d = Descriptor.decode(bytes.fromhex(s['descriptor_hex']))
            input_size = (physical_input_bytes(d) if padded else
                          d.outputs if s['inplace'] else input_load['bytes'])
            occupied = [(0, 128)]
            if protect is not None:
                occupied.append((protect['sram'], protect['sram']+protect['bytes']))

            def allocate(size, fixed=None):
                padded = (size+7)//8*8
                if fixed is None:
                    fixed = 128
                    for lo, hi in sorted(occupied):
                        if fixed+padded <= lo: break
                        fixed = max(fixed, hi)
                if fixed % 8 or fixed+padded > 32768 or any(fixed < hi and lo < fixed+padded for lo, hi in occupied):
                    raise ValueError('no contiguous SRAM region')
                occupied.append((fixed, fixed+padded))
                return fixed

            fixed = previous['output']['sram'] if can_retain else candidate['sram'] if reuse else None
            d.input = allocate(input_size, fixed)
            d.output = d.input if s['inplace'] else allocate(d.outputs)
            replacements = {}
            for t in sorted((t for t in s['loads'] if t['role'] == 'parameter'), key=lambda t: -t['bytes']):
                replacements[t['sram']] = allocate(t['bytes'])
            for field in ('weight', 'params'):
                address = getattr(d, field)
                if address in replacements: setattr(d, field, replacements[address])
            d.next_pc = 64; d.validate()
            if padded: encode_pointwise(d)
            return d, replacements, occupied

        original_descriptor = Descriptor.decode(bytes.fromhex(s['descriptor_hex']))
        active_channels = None
        if active_pointwise_channels is not None and s['layer'] in active_pointwise_channels:
            plane = math.prod(program.tensors[layer.output].shape[2:])
            first = s['first_element']//plane
            active_channels = sum(active_pointwise_channels[s['layer']][first:first+original_descriptor.output_c])
        select_padded = padded_pointwise and _profitable_padded_pointwise(original_descriptor, active_channels)
        placed = False
        for attempt_padded in ((True, False) if select_padded else (False,)):
            try:
                d, replacements, occupied = place(reuse=reuse_input, protect=protected,
                                                   padded=attempt_padded)
            except ValueError:
                continue
            select_padded = attempt_padded
            placed = True
            break
        if not placed:
            select_padded = False
            if reuse_input or protected is not None:
                # A group with no room for the protected input uses the
                # original legal per-tile input load and activation layout.
                reuse_input = False
                protected = None
                try:
                    d, replacements, occupied = place(reuse=False, protect=None)
                except ValueError:
                    if s['inplace']: raise
                    if index in sibling_predecessors:
                        pending_input = dict(layer=s['layer'], ext=input_load['ext'],
                                             bytes=input_load['bytes'], sram=original_descriptor.input)
                    previous = s
                    continue
            else:
                if s['inplace']: raise
                if index in sibling_predecessors:
                    pending_input = dict(layer=s['layer'], ext=input_load['ext'],
                                         bytes=input_load['bytes'], sram=original_descriptor.input)
                previous = s
                continue
        loads = []
        descriptor_bytes = encode_pointwise(d) if select_padded else d.encode()
        ext = len(payload); payload.extend(descriptor_bytes+Descriptor(0).encode())
        loads.append(dict(direction='to_sram', ext=ext, sram=0, bytes=128, role='descriptor'))
        for t in s['loads']:
            if t['role'] == 'descriptor': continue
            t = copy.deepcopy(t)
            if t['role'] == 'input':
                if can_retain:
                    retained += t['bytes']; continue
                if reuse_input:
                    retained += t['bytes']; retained_sibling_input_bytes += t['bytes']; continue
                t['sram'] = d.input
            else: t['sram'] = replacements[t['sram']]
            loads.append(t)
        if can_retain and previous['store'] is not None:
            retained += previous['store']['bytes']; previous['store'] = None
        if s['store'] is not None: s['store']['sram'] = d.output
        s.update(pc=0, live=[0, max(hi for _, hi in occupied)], loads=loads,
                 descriptor_hex=descriptor_bytes.hex(), output=dict(sram=d.output, bytes=d.outputs))
        if select_padded:
            pack_ext = len(payload)
            payload.extend(encode_pack(d.input,d.input_h,d.input_w,d.input_c)+Descriptor(0).encode())
            s['padded_pack'] = dict(descriptor_load=dict(direction='to_sram',ext=pack_ext,
                sram=0,bytes=128,role='pack_descriptor'),input_sram=d.input,
                logical_plane=d.input_h*d.input_w,physical_plane=align8(d.input_h*d.input_w),
                channels=d.input_c)
        if s['inplace']:
            shared_input = protected
        elif index in sibling_predecessors:
            pending_input = dict(layer=s['layer'], ext=input_load['ext'],
                                 bytes=input_load['bytes'], sram=d.input)
        previous = s
    commands = []; contracts = {}; pack_contracts={}
    def emit(op, flags=0, a=0, b=0, c=0): commands.append((op, flags, a, b, c))
    def dma(t):
        emit(1, int(t['direction']=='to_sram'), t['ext'], t['sram'], t['bytes']); emit(3, 2)
    for s in stages:
        if 'padded_pack' in s:
            for t in s['loads']:
                if t['role']=='input': dma(t)
            dma(s['padded_pack']['descriptor_load'])
            pack_contracts[str(len(commands))]=dict(layer=s['layer'],
                first_element=s['first_element'])
            emit(2,0,s['pc'],s['live'][0] | s['live'][1]<<16);emit(3,3)
        for t in s['loads']: dma(t)
        contracts[str(len(commands))] = dict(layer=s['layer'], first_element=s['first_element'])
        emit(2, 0, s['pc'], s['live'][0] | s['live'][1] << 16); emit(3, 3)
        if snapshots:
            region = original['snapshot_regions'][s['layer']]
            dma(dict(direction='from_sram', ext=region['ext']+s['first_element'], **s['output']))
        if s['store'] is not None: dma(s['store'])
    emit(0)
    code = b''.join(struct.pack('<BBHIII', op, flags, 0, a, b, c) for op, flags, a, b, c in commands)
    if len(code)>32768 or len(payload)>8*1024*1024: raise ValueError('program capacity')
    schedule = dict(original, stages=stages, run_contracts=contracts,
        catalogue='experimental full-tensor chain retention; serialized DMA',
        removed_transfer_bytes=retained,
        removed_sibling_input_bytes=retained_sibling_input_bytes,
        sibling_input_retention=reuse_sibling_inputs,
        padded_pointwise=padded_pointwise, pack_contracts=pack_contracts,
        command_count=len(commands), program_bytes=len(code))
    import hashlib
    schedule['program_sha256']=hashlib.sha256(code).hexdigest()
    schedule['image_sha256']=hashlib.sha256(payload).hexdigest()
    return code, bytes(payload), schedule


def prepare(*, reuse_sibling_inputs=False):
    base = SIBLING_BASE if reuse_sibling_inputs else BASE
    check_frozen(); records=[]
    for name in ('kws', 'vww'):
        program, x, _, pins = load_model(name)
        for sample, value in (('pinned', x), ('stress', np.random.default_rng(6062).integers(-128,128,x.shape,dtype=np.int8))):
            oracle=evaluate(program,{program.inputs[0]:value})
            initial=oracle[program.layers[0].output] if program.layers[0].op=='Transpose' else value
            for snapshots in ((True, False) if sample=='pinned' else (True,)):
                stem = 'sibling' if reuse_sibling_inputs else 'chain'
                label=f'{name}-{sample}-{stem}-'+('check' if snapshots else 'timed')
                path=base/'fixtures'/label; path.mkdir(parents=True,exist_ok=True)
                commands,payload,schedule=compile_chain(program,snapshots,
                    reuse_sibling_inputs=reuse_sibling_inputs)
                verification=replay_resident(program,commands,payload,{program.inputs[0]:value},
                    run_contracts=schedule['run_contracts'],final_output=schedule['final_output'],
                    snapshot_regions=schedule['snapshot_regions'],oracle=oracle)
                files={'commands.bin':commands,'payload.bin':payload,'input.bin':initial.tobytes(),
                       'output.bin':oracle[program.outputs[0]].tobytes()}
                checks=[f'{schedule["final_output"]["ext"]} output.bin']
                for i,region in schedule['snapshot_regions'].items():
                    files[f'layer-{i}.bin']=oracle[program.layers[i].output].tobytes()
                    checks.append(f'{region["ext"]} layer-{i}.bin')
                files['checks.txt']=('\n'.join(checks)+'\n').encode()
                files['schedule.json']=(json.dumps(schedule,indent=2,sort_keys=True)+'\n').encode()
                for file,data in files.items(): (path/file).write_bytes(data)
                row=dict(name=label,model=name,pins=pins,verification=verification,
                    removed_transfer_bytes=schedule['removed_transfer_bytes'],
                    removed_sibling_input_bytes=schedule['removed_sibling_input_bytes'],
                    files={file:sha(path/file) for file in files})
                records.append(row); print(label,verification,schedule['removed_transfer_bytes'],
                    'sibling',schedule['removed_sibling_input_bytes'],flush=True)
    save_json(base/'fixtures.json',dict(status='passed-replay',fixtures=records,
        sibling_input_retention=reuse_sibling_inputs))
    check_frozen()


def native(label, *, reuse_sibling_inputs=False):
    base = SIBLING_BASE if reuse_sibling_inputs else BASE
    candidate=BASE.parent/label
    original=json.loads((candidate/'native/report.json').read_text())
    if original['status']!='passed': raise ValueError('engine full-model regression missing')
    verify_files(ROOT,original['sources'])
    executable=candidate/'native/Vv2_tiled_host_bridge'
    manifest=json.loads((base/'fixtures.json').read_text())
    out=base/f'native-{label}';out.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',physical_board=False,label=label,
                executable_sha256=sha(executable),sources=original['sources'],
                fixture_manifest_sha256=sha(base/'fixtures.json'),results=[])
    for fixture in manifest['fixtures']:
        directory=base/'fixtures'/fixture['name']; verify_files(directory,fixture['files'])
        for seed in (0,6063):
            result=out/f'{fixture["name"]}-s{seed}.json'
            subprocess.run([str(executable),str(directory),str(seed),str(result)],check=True)
            row=json.loads(result.read_text());row.update(fixture=fixture['name'],fixture_files=fixture['files'])
            if row['status']!='passed': raise ValueError('chain native mismatch')
            report['results'].append(row);save_json(out/'report.json',report)
    report['status']='passed';save_json(out/'report.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native')
    parser.add_argument('--sibling-inputs', action='store_true')
    args=parser.parse_args()
    if args.native: native(args.native, reuse_sibling_inputs=args.sibling_inputs)
    else: prepare(reuse_sibling_inputs=args.sibling_inputs)
