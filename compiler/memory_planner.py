"""Single-port Tang Nano SRAM allocation with explicit tensor lifetimes.

The eight byte lanes form one 64-bit request port.  A 32-KiB instance maps to
16 Gowin BSRAM blocks in the routed v2 implementation; this is an observed
implementation parameter, not a claim that arbitrary layouts cost 16 blocks.
"""
from dataclasses import dataclass
from math import prod,ceil


@dataclass(frozen=True)
class Region:
    name: str
    size: int
    first: int
    last: int
    kind: str
    data: bytes = b''


def align8(value):
    return (value + 7) & ~7


def allocate(regions, capacity, program_bytes):
    """First-fit by lifetime; initial weights remain live until their use."""
    if capacity < 1024 or capacity & (capacity - 1) or program_bytes > capacity:
        raise ValueError('invalid SRAM/program capacity')
    placed = []
    for region in sorted(regions, key=lambda r: (r.first, -r.size, r.name)):
        if region.size <= 0 or len(region.data) > region.size or region.first > region.last:
            raise ValueError(f'invalid region: {region.name}')
        size = align8(region.size)
        blockers = sorted((p['offset'], p['offset'] + p['size']) for p in placed
                          if not (p['last'] < region.first or region.last < p['first']))
        cursor = align8(program_bytes)
        for start, end in blockers:
            if cursor + size <= start:
                break
            cursor = max(cursor, end)
        if cursor + size > capacity:
            raise ValueError('model exceeds target SRAM')
        placed.append(dict(name=region.name, offset=cursor, size=size,
                           logical_bytes=region.size, first=region.first,
                           last=region.last, kind=region.kind))
    return placed


def regions_for_program(program, parameter_bytes):
    """Inputs and parameters persist across RUN; intermediates follow liveness."""
    n = len(program.layers)
    # Preserve the phase-2 FC fixture's post-run intermediate readback contract.
    # Spatial graphs use true liveness and stop after each layer for debugging.
    retain_all = not any(layer.op in ('Conv','MaxPool') for layer in program.layers)
    producer = {name: -1 for name in program.inputs}
    producer.update({layer.output: i for i, layer in enumerate(program.layers)})
    last = {name: n if name in program.outputs else producer.get(name, -1)
            for name in program.tensors}
    for i, layer in enumerate(program.layers):
        for name in layer.inputs:
            last[name] = max(last[name], i)
    regions = []
    for name, tensor in program.tensors.items():
        if name not in producer:
            raise ValueError(f'unsupported persistent tensor: {name}')
        regions.append(Region(name, prod(tensor.shape), -1 if retain_all else producer[name],
                              n if retain_all else last[name], 'tensor'))
    for i, payloads in enumerate(parameter_bytes):
        for name, data in payloads.items():
            regions.append(Region(f'{i}/{name}', len(data), -1, n, name, data))
    return regions


def report(layout, capacity, program_bytes, *, bsram_blocks=16):
    from memory_verifier import verify
    verified = verify(layout, capacity, program_bytes)
    # Eight independent 8-bit lanes, 4,096 deep at the selected capacity.
    # A Gowin BSRAM block stores 18,432 bits in this accounting model.
    lane_depth=capacity//8
    predicted=8*ceil(lane_depth*8/18432)
    if predicted!=bsram_blocks:raise ValueError('scratchpad block prediction differs from target')
    return dict(**verified, port_count=1, port_width_bits=64,
                byte_lanes=8, lane_depth=lane_depth, bsram_payload_bits=18432,
                predicted_bsram_blocks=predicted, nominal_bsram_blocks=bsram_blocks,
                note='BSRAM count is tied to the exact synthesized 32-KiB scratchpad')
