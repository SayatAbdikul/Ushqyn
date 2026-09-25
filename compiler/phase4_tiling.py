"""Conservative, geometry-only off-chip schedule for the frozen primary models.

This plans one descriptor at a time using the v2 kernel ABI. It deliberately
does not manufacture weights, quantization parameters, an SDRAM controller, or
an executable board image. External activation slots alternate after each
compute node; a tile's input remains live until its output DMA completes.
"""

import hashlib
import json
import math
from pathlib import Path

from hardware_v2 import Descriptor, TARGET

ROOT = Path(__file__).resolve().parents[1]
EXT_BYTES = 8 * 1024 * 1024
SRAM_BYTES = TARGET['memory_bytes']


def align8(n):
    return (n + 7) & ~7


def elements(shape):
    return math.prod(shape)


def _scratch(input_bytes, output_bytes, weight_bytes, param_bytes):
    regions = {}
    end = 128  # one 64-byte descriptor and one HALT descriptor
    for name, size in (('input', input_bytes), ('output', output_bytes),
                       ('weight', weight_bytes), ('params', param_bytes)):
        if size:
            end = align8(end)
            regions[name] = {'base': end, 'bytes': size}
            end += size
    if end > SRAM_BYTES:
        return None
    return regions, end


def _kind(node, input_shape, output_shape):
    op = node['op']
    if op == 'Conv':
        group = node['attributes'].get('group', 1)
        if group == 1:
            return 'conv'
        if group == input_shape[1] == output_shape[1]:
            return 'depthwise'
        raise ValueError('unsupported group convolution')
    return {'Gemm': 'gemm', 'MaxPool': 'maxpool',
            'AveragePool': 'avgpool', 'GlobalAveragePool': 'avgpool',
            'Relu': 'relu', 'Clip': 'clip'}[op]


def plan_inventory(path):
    path = Path(path)
    return plan_graph(json.loads(path.read_text()),
                      hashlib.sha256(path.read_bytes()).hexdigest())


def plan_graph(inventory, inventory_sha256):
    """Plan a verified inventory, including one derived from a typed Program."""
    tensors = {t['name']: t for t in inventory['tensors']}
    nodes = inventory['operators']
    if not nodes:
        raise ValueError('empty inventory')
    max_activation = max(elements(t['shape']) for t in tensors.values()
                         if t['name'] in {n['inputs'][0] for n in nodes} |
                         {n['outputs'][0] for n in nodes})
    slot_bytes = align8(max_activation)
    parameter_cursor = 2 * slot_bytes
    if parameter_cursor >= EXT_BYTES:
        raise ValueError('activation slots exceed external memory')
    current_slot = 0
    layers = []
    for index, node in enumerate(nodes):
        op = node['op']
        input_shape = tensors[node['inputs'][0]]['shape']
        output_shape = tensors[node['outputs'][0]]['shape']
        input_bytes, output_bytes = elements(input_shape), elements(output_shape)
        if op in ('Reshape', 'Flatten', 'Identity'):
            if input_bytes != output_bytes:
                raise ValueError(f'node {index}: alias changes element count')
            layers.append({'index': index, 'kind': 'alias', 'tiles': []})
            continue
        if op == 'Transpose' and index == 0 and node['attributes'].get('perm') == [0, 3, 1, 2]:
            # The declared host boundary provides the transposed bytes in slot 0.
            layers.append({'index': index, 'kind': 'host_layout', 'tiles': []})
            continue
        kind = _kind(node, input_shape, output_shape)
        a = node['attributes']
        weight_shape = tensors[node['inputs'][1]]['shape'] if kind in ('conv', 'depthwise', 'gemm') else None
        if weight_shape:
            reduction = elements(weight_shape[1:])
            row_stride = align8(reduction)
            rows = weight_shape[0]
            weight_base = parameter_cursor
            parameter_cursor += rows * row_stride
            parameter_cursor = align8(parameter_cursor)
            params_base = parameter_cursor
            parameter_cursor += rows * 16
        else:
            row_stride = 0
            weight_base = None
            params_base = parameter_cursor if kind in ('relu', 'clip', 'maxpool', 'avgpool') else None
            if params_base is not None:
                parameter_cursor += 16
        parameter_cursor = align8(parameter_cursor)
        if parameter_cursor > EXT_BYTES:
            raise ValueError('parameter image exceeds external memory')
        input_base = current_slot * slot_bytes
        output_base = (1-current_slot) * slot_bytes
        tiles = []
        spatial = kind in ('conv', 'depthwise', 'maxpool', 'avgpool')
        if spatial:
            if len(input_shape) != 4 or len(output_shape) != 4 or input_shape[0] != 1 or output_shape[0] != 1:
                raise ValueError(f'node {index}: requires batch-one NCHW')
            input_plane = input_shape[2] * input_shape[3]
            output_plane = output_shape[2] * output_shape[3]
            total = output_shape[1]
        elif kind == 'gemm':
            input_plane = input_bytes
            output_plane = 1
            total = output_bytes
        else:
            input_plane = output_plane = 1
            total = output_bytes
            if input_bytes != output_bytes:
                raise ValueError(f'node {index}: elementwise size mismatch')
        first = 0
        while first < total:
            choice = None
            for count in range(total-first, 0, -1):
                end = first + count
                # The current DMA accepts aligned base addresses; a tail may
                # have non-multiple-of-eight length, but the next tile may not.
                if end < total and ((end * output_plane) % 8 or
                                    (kind in ('depthwise', 'maxpool', 'avgpool') and end * input_plane % 8)):
                    continue
                ib = input_bytes if kind in ('conv', 'gemm') else count * input_plane
                ob = count * output_plane
                wb = count * row_stride if weight_shape else 0
                pb = count * 16 if weight_shape else (16 if params_base is not None else 0)
                fit = _scratch(ib, ob, wb, pb)
                if fit is not None:
                    choice = (count, ib, ob, wb, pb, fit)
                    break
            if choice is None:
                raise ValueError(f'node {index}: no aligned tile fits 32 KiB SRAM at {first}')
            count, ib, ob, wb, pb, (regions, used) = choice
            iext = input_base + (first * input_plane if kind in ('depthwise', 'maxpool', 'avgpool', 'relu', 'clip') else 0)
            oext = output_base + first * output_plane
            if iext % 8 or oext % 8:
                raise ValueError(f'node {index}: unaligned tile base')
            kwargs = dict(opcode=TARGET['opcodes'][{'conv':'CONV','depthwise':'DWCONV',
                          'gemm':'GEMM','maxpool':'MAXPOOL','avgpool':'AVGPOOL',
                          'relu':'RELU','clip':'CLIP'}[kind]],
                          input=regions['input']['base'], output=regions['output']['base'],
                          weight=regions.get('weight', {}).get('base', 0),
                          params=regions.get('params', {}).get('base', 0),
                          count=ib if kind not in ('conv', 'depthwise') else reduction,
                          outputs=ob, row_stride=row_stride, next_pc=64)
            if spatial:
                kh, kw = (weight_shape[2:] if weight_shape else a.get('kernel_shape', input_shape[2:]))
                sh, sw = a.get('strides', [1, 1])
                pt, pl, pb_pad, pr = a.get('pads', [0, 0, 0, 0])
                kwargs.update(kernel_h=kh, kernel_w=kw, stride_h=sh, stride_w=sw,
                              pad_top=pt, pad_left=pl, pad_bottom=pb_pad, pad_right=pr,
                              input_h=input_shape[2], input_w=input_shape[3],
                              input_c=count if kind in ('depthwise', 'maxpool', 'avgpool') else input_shape[1],
                              output_c=count)
                if kind in ('maxpool', 'avgpool'):
                    kwargs['count'] = ib
            if kind == 'gemm':
                kwargs['count'] = input_bytes
            descriptor = Descriptor(**kwargs)
            descriptor.validate()
            transfers = [{'direction': 'to_sram', 'ext': iext,
                          'sram': regions['input']['base'], 'bytes': ib}]
            if wb:
                transfers.append({'direction': 'to_sram', 'ext': weight_base+first*row_stride,
                                  'sram': regions['weight']['base'], 'bytes': wb})
            if pb:
                transfers.append({'direction': 'to_sram',
                                  'ext': params_base+(first*16 if weight_shape else 0),
                                  'sram': regions['params']['base'], 'bytes': pb})
            transfers.append({'direction': 'from_sram', 'ext': oext,
                              'sram': regions['output']['base'], 'bytes': ob})
            for transfer in transfers:
                if (transfer['ext'] % 8 or transfer['sram'] % 8 or
                        transfer['bytes'] < 1 or transfer['bytes'] > SRAM_BYTES or
                        transfer['ext'] + transfer['bytes'] > EXT_BYTES or
                        transfer['sram'] + transfer['bytes'] > SRAM_BYTES):
                    raise ValueError(f'node {index}: illegal DMA transfer')
            tiles.append({'first': first, 'count': count, 'scratch_bytes': used,
                          'descriptor_hex': descriptor.encode().hex(),
                          'regions': regions, 'transfers': transfers})
            first += count
        layers.append({'index': index, 'kind': kind, 'input_slot': current_slot,
                       'output_slot': 1-current_slot, 'input_bytes': input_bytes,
                       'output_bytes': output_bytes, 'tiles': tiles})
        current_slot = 1-current_slot
    return {'schema': 1, 'inventory_sha256': inventory_sha256,
            'external_bytes': EXT_BYTES, 'sram_bytes': SRAM_BYTES,
            'activation_slot_bytes': slot_bytes, 'parameter_end': parameter_cursor,
            'final_output_slot': current_slot, 'layers': layers}
