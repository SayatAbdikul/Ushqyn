"""Compile legal autonomous schedules with hybrid ping-pong SRAM placement."""

import copy
import hashlib
import struct

from hardware_v2 import Descriptor
from phase4_tiling import align8, EXT_BYTES

PROGRAM_BYTES = 32768


def compile_sequence(plan, image, overlap=True, snapshots=False):
    """Return command bytes, SDRAM image and a reviewable schedule manifest.

    Each tile <=16 KiB alternates banks. Larger tiles use the whole scratchpad
    and impose a barrier. Only same-layer input or immutable next-layer
    parameters are prefetched before the current output is committed.
    """
    if hashlib.sha256(image).hexdigest() != plan['parameter_image_sha256']:
        raise ValueError('parameter image hash mismatch')
    payload = bytearray(image)
    payload.extend(bytes(align8(len(payload))-len(payload)))
    tiles = []
    bank = 0
    snapshot_regions = {}
    for layer in plan['layers']:
        for original in layer['tiles']:
            tile = copy.deepcopy(original)
            base = bank * 16384 if tile['scratch_bytes'] <= 16384 else 0
            bank = 1 - bank if tile['scratch_bytes'] <= 16384 else 0
            d = Descriptor.decode(bytes.fromhex(tile['descriptor_hex']))
            for name in ('input', 'output', 'weight', 'params', 'next_pc'):
                if name in ('input', 'output', 'next_pc') or getattr(d, name):
                    setattr(d, name, getattr(d, name) + base)
            d.validate()
            tile.update(layer=layer['index'], pc=base,
                        live=[base, base+tile['scratch_bytes']], descriptor_hex=d.encode().hex())
            for r in tile['regions'].values():
                r['base'] += base
            for transfer in tile['transfers']:
                transfer['sram'] += base
            desc_ext = len(payload)
            payload.extend(d.encode() + Descriptor(0).encode())
            tile['loads'] = [dict(direction='to_sram', ext=desc_ext,
                                  sram=base, bytes=128, role='descriptor')]
            for t in tile['transfers'][:-1]:
                tile['loads'].append(dict(t, role=('input' if
                    t['sram']==d.input else 'parameter')))
            tiles.append(tile)
    cursor = align8(len(payload))
    if snapshots:
        for layer in plan['layers']:
            if layer['tiles']:
                snapshot_regions[layer['index']] = {'ext': cursor, 'bytes': layer['output_bytes']}
                cursor += align8(layer['output_bytes'])
    if cursor > EXT_BYTES:
        raise ValueError('program image and snapshots exceed SDRAM')
    commands, prefetched = [], set()

    def emit(op, flags=0, a=0, b=0, c=0):
        commands.append((op, flags, a, b, c))

    def dma(t):
        emit(1, t['direction']=='to_sram', t['ext'], t['sram'], t['bytes'])
        emit(3, 2)

    overlap_bytes = 0
    for index, tile in enumerate(tiles):
        for j, load in enumerate(tile['loads']):
            if (index, j) not in prefetched:
                dma(load)
        emit(2, 0, tile['pc'], tile['live'][0] | (tile['live'][1]<<16))
        if overlap and index+1 < len(tiles):
            following = tiles[index+1]
            disjoint = (tile['live'][1] <= following['live'][0] or
                        following['live'][1] <= tile['live'][0])
            if disjoint:
                for j, load in enumerate(following['loads']):
                    if load['role'] != 'input' or following['layer'] == tile['layer']:
                        dma(load)
                        prefetched.add((index+1, j))
                        overlap_bytes += load['bytes']
        emit(3, 3)
        dma(tile['transfers'][-1])
        if snapshots:
            layer = plan['layers'][tile['layer']]
            write = dict(tile['transfers'][-1])
            write['ext'] += snapshot_regions[tile['layer']]['ext'] - layer['output_slot']*plan['activation_slot_bytes']
            dma(write)
    emit(0)
    program = b''.join(struct.pack('<BBHIII', op, flags, 0, a, b, c)
                       for op, flags, a, b, c in commands)
    if len(program)>PROGRAM_BYTES:
        raise ValueError(f'command list exceeds {PROGRAM_BYTES} bytes: {len(program)}')
    final_layer = next(l for l in reversed(plan['layers']) if l['tiles'])
    record = {'schema': 1, 'overlap_enabled': overlap, 'snapshots_enabled': snapshots,
              'command_count': len(commands), 'program_bytes': len(program),
              'prefetch_payload_bytes': overlap_bytes,
              'program_sha256': hashlib.sha256(program).hexdigest(),
              'image_sha256': hashlib.sha256(payload).hexdigest(),
              'snapshot_regions': snapshot_regions, 'tiles': tiles,
              'final_output': {'ext': final_layer['output_slot']*plan['activation_slot_bytes'],
                               'bytes': final_layer['output_bytes']}}
    return program, bytes(payload), record
