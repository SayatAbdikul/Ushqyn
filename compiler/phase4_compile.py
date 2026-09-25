"""Materialize the Phase 4 tile plan from a calibrated integer Program.

This produces an external parameter image and per-tile SRAM descriptors. It
does not claim that the board can execute the schedule yet: the host-command
board image still needs an SDRAM/DMA sequencer.
"""

import hashlib
import json
import math
import struct

import numpy as np

from phase4_tiling import plan_graph
from quantization import multiplier_shift


def _inventory(program):
    if len(program.inputs) != 1 or program.constants:
        raise ValueError('tiled graph requires one activation input and no runtime constants')
    tensors = [{'name': name, 'shape': list(t.shape)}
               for name, t in program.tensors.items()]
    nodes = []
    previous = program.inputs[0]
    for index, layer in enumerate(program.layers):
        if len(layer.inputs) != 1 or layer.inputs[0] != previous:
            raise ValueError(f'node {index}: tiled graph must be a linear chain')
        inputs = list(layer.inputs)
        if layer.op in ('Conv', 'Gemm'):
            weight = layer.parameters['weight']
            weight_name = f'__tile_weight_{index}'
            tensors.append({'name': weight_name, 'shape': list(weight.shape)})
            inputs.append(weight_name)
        nodes.append({'op': layer.op, 'inputs': inputs,
                      'outputs': [layer.output], 'attributes': layer.attributes})
        previous = layer.output
    if program.outputs != [previous]:
        raise ValueError('tiled graph requires its final layer as the sole output')
    return {'tensors': tensors, 'operators': nodes}


def _parameter_rows(program, layer):
    p = layer.parameters
    iq = program.tensors[layer.inputs[0]].quantization
    oq = program.tensors[layer.output].quantization
    if layer.op in ('Conv', 'Gemm'):
        weights = np.asarray(p['weight'], dtype=np.int8)
        reduction = math.prod(weights.shape[1:])
        stride = (reduction + 7) & ~7
        rows = np.zeros((weights.shape[0], stride), dtype=np.int8)
        rows[:, :reduction] = weights.reshape(weights.shape[0], reduction)
        parameters = b''.join(
            struct.pack('<iiBbbbbb', int(p['corrected_bias'][c]),
                        int(p['multiplier'][c]), int(p['shift'][c]),
                        oq.zero_point, iq.zero_point, -128, 127, 0) + b'\0\0'
            for c in range(weights.shape[0]))
        return rows.tobytes(), parameters
    if layer.op in ('Relu', 'Clip', 'MaxPool', 'AveragePool', 'GlobalAveragePool'):
        ratio = iq.scale / oq.scale
        if layer.op in ('AveragePool', 'GlobalAveragePool'):
            shape = program.tensors[layer.inputs[0]].shape
            kh, kw = layer.attributes.get('kernel_shape', shape[2:])
            ratio /= kh * kw
        multiplier, shift = multiplier_shift(ratio)
        low, high = (p['clip_bounds'].tolist() if layer.op == 'Clip' else
                     (iq.zero_point, 127) if layer.op == 'Relu' else (-128, 127))
        return b'', (struct.pack('<iiBbbbbb', 0, multiplier, shift,
                                  oq.zero_point, iq.zero_point, low, high, 0)
                     + b'\0\0')
    return b'', b''


def compile_tiled(program):
    """Return (tile plan, immutable external parameter image).

    Activation slot zero is intentionally blank. The caller supplies a
    quantized input there before executing the first tile.
    """
    inventory = _inventory(program)
    digest = hashlib.sha256(json.dumps(inventory, sort_keys=True,
                                      separators=(',', ':')).encode()).hexdigest()
    plan = plan_graph(inventory, digest)
    image = bytearray(plan['parameter_end'])
    for layer, scheduled in zip(program.layers, plan['layers']):
        weights, parameters = _parameter_rows(program, layer)
        for tile in scheduled['tiles']:
            regions = tile['regions']
            transfers = {t['sram']: t for t in tile['transfers']
                         if t['direction'] == 'to_sram'}
            first, count = tile['first'], tile['count']
            if 'weight' in regions:
                stride = len(weights) // len(layer.parameters['weight'])
                data = weights[first*stride:(first+count)*stride]
                t = transfers[regions['weight']['base']]
                if len(data) != t['bytes']:
                    raise ValueError('weight tile length disagrees with DMA plan')
                image[t['ext']:t['ext']+len(data)] = data
            if 'params' in regions:
                data = (parameters[first*16:(first+count)*16]
                        if 'weight' in regions else parameters)
                t = transfers[regions['params']['base']]
                if len(data) != t['bytes']:
                    raise ValueError('parameter tile length disagrees with DMA plan')
                image[t['ext']:t['ext']+len(data)] = data
    plan['parameter_image_sha256'] = hashlib.sha256(image).hexdigest()
    plan['parameter_image_bytes'] = len(image)
    return plan, bytes(image)
