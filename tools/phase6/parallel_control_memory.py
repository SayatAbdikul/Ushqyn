"""Bounded, source-pinned control-memory experiment for the frozen B3 programs.

This is a trace/packing study, not an RTL implementation. It intentionally uses
only the small command binaries and Python's standard library.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct
from typing import NamedTuple


ROOT = Path(__file__).resolve().parents[2]
FROZEN = Path('/Users/sayat/.codex/worktrees/21f0/tinyML_accelerator')
SOURCES = {
    'kws': FROZEN / 'work/phase6/matched-baselines-v1/current-v4/fixtures/kws-pinned/commands.bin',
    'vww': FROZEN / 'work/phase6/generic-b3-board-v2/fixtures/vww-pinned/commands.bin',
}
EXPECTED_SHA256 = {
    'kws': '6f4499cd0b5f87cc49230c0afe4ffec07806741978660762cd7a5e5ffda3fe31',
    'vww': '62f8a4ed24b212dd69377539820712685b515b2bcd0f3a8a5b85f82d561a2e96',
}
OUTPUT = ROOT / 'work/phase6/parallel-control-memory-v1'
ROOT_KWS_ALIAS = ROOT / 'work/phase6/novelty-contracts-v1/fixtures/kws-pinned-compacted-fused-timed/commands.bin'
ROOT_VWW_PRIOR = ROOT / 'work/phase6/novelty-contracts-v1/fixtures/vww-pinned-compacted-fused-timed/commands.bin'
ROOT_VWW_PRIOR_SHA256 = 'c0074a16faac838ebd873ddc8b251dfd000d3864d8b8a9747f9c8eb37f6c1d84'
COMMAND = struct.Struct('<BBHIII')
LOOP = struct.Struct('<BBHHH')  # tag, kind (1 exact; 2 affine), body length, count, zero
DELTA = struct.Struct('<hhhH')  # signed strides for arg0/arg1/arg2; zero
LOOP_TAG = 0xFE
MAX_BODY = 32
OP_NAMES = {0: 'HALT', 1: 'DMA', 2: 'RUN', 3: 'WAIT'}


class Segment(NamedTuple):
    start: int
    body: int
    iterations: int
    kind: str
    deltas: tuple[tuple[int, int, int], ...]


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def commands_from(raw: bytes) -> list[tuple[int, int, int, int, int, int]]:
    if len(raw) % COMMAND.size:
        raise ValueError('partial command record')
    return list(COMMAND.iter_unpack(raw))


def affine_deltas(cmds: list[tuple[int, ...]], start: int, body: int):
    if start + 2 * body > len(cmds):
        return None
    result = []
    for j in range(body):
        a, b = cmds[start + j], cmds[start + body + j]
        if a[0] == 0 or a[:3] != b[:3]:
            return None
        diff = tuple(b[k] - a[k] for k in (3, 4, 5))
        if any(not -32768 <= x <= 32767 for x in diff):
            return None
        result.append(diff)
    return tuple(result)


def repeat_count(cmds: list[tuple[int, ...]], start: int, body: int,
                 deltas: tuple[tuple[int, int, int], ...]) -> int:
    n = 2
    while start + (n + 1) * body <= len(cmds):
        if any(
            cmds[start + n * body + j][:3] != cmds[start + j][:3]
            or any(cmds[start + n * body + j][k] != cmds[start + j][k] + n * deltas[j][k - 3]
                   for k in (3, 4, 5))
            for j in range(body)
        ):
            break
        n += 1
    return n


def compress(cmds: list[tuple[int, ...]]) -> tuple[bytes, list[Segment]]:
    """Dynamic program over adjacent exact or signed-16-bit-affine loop bodies."""
    count = len(cmds)
    cost = [0] * (count + 1)
    chosen: list[Segment | None] = [None] * count
    for i in range(count - 1, -1, -1):
        cost[i] = COMMAND.size + cost[i + 1]
        for body in range(1, min(MAX_BODY, (count - i) // 2) + 1):
            deltas = affine_deltas(cmds, i, body)
            if deltas is None:
                continue
            nmax = repeat_count(cmds, i, body, deltas)
            kind = 'exact' if all(x == (0, 0, 0) for x in deltas) else 'affine16'
            unit = COMMAND.size if kind == 'exact' else COMMAND.size + DELTA.size
            encoded = LOOP.size + body * unit
            for iterations in range(2, nmax + 1):
                after = i + body * iterations
                new_cost = encoded + cost[after]
                if new_cost < cost[i]:
                    cost[i] = new_cost
                    chosen[i] = Segment(i, body, iterations, kind, deltas)

    pieces = []
    segments = []
    i = 0
    while i < count:
        seg = chosen[i]
        if seg is None:
            pieces.append(COMMAND.pack(*cmds[i]))
            i += 1
            continue
        pieces.append(LOOP.pack(LOOP_TAG, 1 if seg.kind == 'exact' else 2,
                                seg.body, seg.iterations, 0))
        for j in range(seg.body):
            pieces.append(COMMAND.pack(*cmds[i + j]))
            if seg.kind == 'affine16':
                pieces.append(DELTA.pack(*seg.deltas[j], 0))
        segments.append(seg)
        i += seg.body * seg.iterations
    raw = b''.join(pieces)
    if len(raw) != cost[0]:
        raise AssertionError('dynamic-programming byte cost differs from encoder')
    return raw, segments


def expand(raw: bytes) -> bytes:
    output = bytearray()
    offset = 0
    while offset < len(raw):
        if raw[offset] != LOOP_TAG:
            if offset + COMMAND.size > len(raw):
                raise ValueError('truncated literal')
            cmd = COMMAND.unpack_from(raw, offset)
            if cmd[0] not in OP_NAMES:
                raise ValueError('unknown literal opcode')
            output.extend(raw[offset:offset + COMMAND.size])
            offset += COMMAND.size
            continue
        if offset + LOOP.size > len(raw):
            raise ValueError('truncated loop header')
        tag, kind, body, iterations, reserved = LOOP.unpack_from(raw, offset)
        if tag != LOOP_TAG or kind not in (1, 2) or not 1 <= body <= MAX_BODY or iterations < 2 or reserved:
            raise ValueError('invalid loop header')
        offset += LOOP.size
        base = []
        deltas = []
        for _ in range(body):
            if offset + COMMAND.size > len(raw):
                raise ValueError('truncated base command')
            cmd = COMMAND.unpack_from(raw, offset)
            offset += COMMAND.size
            if cmd[0] not in (1, 2, 3):
                raise ValueError('invalid loop body opcode')
            base.append(cmd)
            if kind == 2:
                if offset + DELTA.size > len(raw):
                    raise ValueError('truncated affine delta')
                a, b, c, zero = DELTA.unpack_from(raw, offset)
                if zero:
                    raise ValueError('nonzero affine reserved field')
                deltas.append((a, b, c))
                offset += DELTA.size
            else:
                deltas.append((0, 0, 0))
        if kind == 1 and any(delta != (0, 0, 0) for delta in deltas):
            raise AssertionError('exact loop has a stride')
        for iteration in range(iterations):
            for cmd, delta in zip(base, deltas):
                args = tuple(cmd[k] + iteration * delta[k - 3] for k in (3, 4, 5))
                if any(not 0 <= v <= 0xFFFFFFFF for v in args):
                    raise ValueError('affine argument overflow')
                output.extend(COMMAND.pack(*cmd[:3], *args))
        if len(output) > 32768:
            raise ValueError('expanded program exceeds original store')
    return bytes(output)


def self_test():
    # The round-trip condition covers all command fields, including flags and
    # WAIT ordering; these adversaries forbid unsafe cross-boundary matching.
    def c(op, a=0, b=0, n=0, flags=0):
        return (op, flags, 0, a, b, n)
    samples = [
        [c(1, a=8 * k, b=512, n=64) for k in range(12)] + [c(0)],
        [c(1, a=8 * k, b=0, n=8) if k % 2 == 0 else c(3, flags=2)
         for k in range(30)] + [c(0)],
        [c(1, a=8 * k, b=0, n=8, flags=k % 2) for k in range(16)] + [c(0)],
        [c(1, a=65528 * k, b=0, n=8) for k in range(4)] + [c(0)],
        [c(1, a=8 * k, b=0, n=8) for k in range(8)] + [c(3, flags=1), c(0)],
    ]
    for cmds in samples:
        packed, _ = compress(cmds)
        assert expand(packed) == b''.join(COMMAND.pack(*x) for x in cmds)
    for malformed in (b'\xfe', LOOP.pack(LOOP_TAG, 2, 1, 2, 0),
                      LOOP.pack(LOOP_TAG, 2, 0, 2, 0),
                      LOOP.pack(LOOP_TAG, 9, 1, 2, 0), b'\x01'):
        try:
            expand(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError('malformed program accepted')


def estimate_fetch(segments: list[Segment], total_commands: int) -> dict:
    loop_commands = sum(s.body * s.iterations for s in segments)
    literal = total_commands - loop_commands
    buffered = 2 * literal
    unbuffered = 2 * literal
    for s in segments:
        words = (COMMAND.size + (DELTA.size if s.kind == 'affine16' else 0)) // 8
        buffered += 1 + s.body * words
        unbuffered += 1 + s.body * words * s.iterations
    return {
        'original_64bit_reads': total_commands * 2,
        'buffered_body_64bit_reads': buffered,
        'unbuffered_body_64bit_reads': unbuffered,
        'minimum_extra_loop_advance_events': sum(s.iterations - 1 for s in segments),
        'body_buffer_peak_bytes': max((s.body * (COMMAND.size + (DELTA.size if s.kind == 'affine16' else 0)) for s in segments), default=0),
        'note': 'Idealized word-read counts; RTL port, decoder, arithmetic, setup, and Fmax costs unmeasured.',
    }


def block_count(memory_bytes: int) -> int:
    """Eight byte-write banks; each bank uses 2,048 x 8 bits per BSRAM SP."""
    if memory_bytes <= 0 or memory_bytes % 8:
        raise ValueError('memory must have positive eight-byte granularity')
    depth = memory_bytes // 8
    return 8 * ((depth + 2047) // 2048)


def measure(model: str, path: Path, expected_sha256: str) -> dict:
    raw = path.read_bytes()
    if sha(raw) != expected_sha256:
        raise ValueError(f'{model} fixture hash changed')
    cmds = commands_from(raw)
    if not cmds or cmds[-1][0] != 0 or any(c[0] not in OP_NAMES for c in cmds):
        raise ValueError('invalid frozen program opcodes/HALT')
    compressed, segments = compress(cmds)
    if expand(compressed) != raw:
        raise AssertionError('compressed B3 command stream differs')
    op_counts = {OP_NAMES[k]: v for k, v in sorted(Counter(c[0] for c in cmds).items())}
    dma_bytes = sum(c[5] for c in cmds if c[0] == 1)
    data_high_water = max((c[4] + c[5] for c in cmds if c[0] == 1), default=0)
    live_high_water = max((c[4] >> 16 for c in cmds if c[0] == 2), default=0)
    report = {
        'source': str(path), 'source_sha256': sha(raw),
        'commands': len(cmds), 'op_counts': op_counts,
        'raw_bytes': len(raw), 'compressed_bytes': len(compressed),
        'compressed_sha256': sha(compressed),
        'logical_reduction_bytes': len(raw) - len(compressed),
        'logical_reduction_percent': round(100 * (1 - len(compressed) / len(raw)), 3),
        'loop_count': len(segments),
        'loop_cover_commands': sum(s.body * s.iterations for s in segments),
        'longest_loop_iterations': max((s.iterations for s in segments), default=0),
        'max_loop_body_commands': max((s.body for s in segments), default=0),
        'loop_kinds': dict(Counter(s.kind for s in segments)),
        'dma_bytes_ordered_unchanged': dma_bytes,
        'data_sram_high_water_byte': max(data_high_water, live_high_water),
        'data_dma_high_water_byte': data_high_water,
        'data_run_live_end_byte': live_high_water,
        'program_bram_raw_64bit': block_count(len(raw)),
        'program_bram_compact_64bit': block_count(len(compressed)),
        'fetch': estimate_fetch(segments, len(cmds)),
        'segments': [dict(start=s.start, body=s.body, iterations=s.iterations, kind=s.kind,
                          encoded_bytes=LOOP.size + s.body * (COMMAND.size + (DELTA.size if s.kind == 'affine16' else 0)))
                     for s in segments],
        'round_trip_exact': True,
    }
    (OUTPUT / f'{model}.compact.bin').write_bytes(compressed)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print('self-test passed')
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    models = {name: measure(name, path, EXPECTED_SHA256[name]) for name, path in SOURCES.items()}
    if sha(ROOT_KWS_ALIAS.read_bytes()) != EXPECTED_SHA256['kws']:
        raise ValueError('root KWS Phase 6 fixture is no longer the sealed B3 alias')
    prior_vww = measure('root-vww-prior', ROOT_VWW_PRIOR, ROOT_VWW_PRIOR_SHA256)
    # One physical allocation is shared across model loads. The byte-bank
    # width imposes at least 8 SP blocks for any nonempty code store.
    max_raw = max(x['raw_bytes'] for x in models.values())
    max_compact = max(x['compressed_bytes'] for x in models.values())
    max_data = max(x['data_sram_high_water_byte'] for x in models.values())
    route = FROZEN / 'work/phase6/engine-candidate-rtl-v2/route27/phase6_uart_burst/impl/pnr/phase6_uart_burst.rpt.txt'
    route_text = route.read_text()
    for expression in (r'BSRAM\s+\|\s+38/46', r'--SP\s+\|\s+32', r'--SDPB\s+\|\s+6'):
        if re.search(expression, route_text) is None:
            raise ValueError(f'physical BSRAM accounting changed: {expression}')
    partitions = []
    for program_bytes, data_bytes in [(32768, 32768), (16384, 32768), (8192, 32768),
                                      (16384, 49152), (8192, 49152)]:
        partitions.append({
            'program_capacity_bytes': program_bytes,
            'data_capacity_bytes': data_bytes,
            'program_sp_blocks': block_count(program_bytes),
            'data_sp_blocks': block_count(data_bytes),
            'total_sp_blocks': block_count(program_bytes) + block_count(data_bytes),
            'raw_code_fits_both': max_raw <= program_bytes,
            'compact_code_fits_both': max_compact <= program_bytes,
            'existing_data_high_water_fits_both': max_data <= data_bytes,
        })
    report = {
        'schema': 1,
        'scope': 'frozen KWS/VWW B3 command-only trace; no FPGA implementation',
        'model_results': models,
        'root_phase6_fixture_cross_check': {
            'kws_alias_path': str(ROOT_KWS_ALIAS),
            'kws_alias_sha256': EXPECTED_SHA256['kws'],
            'kws_byte_identical_to_frozen_b3': True,
            'vww_prior_control': prior_vww,
            'vww_prior_control_scope': 'root compacted-fused Phase 6 program; not the strongest frozen B3 policy',
        },
        'physical_baseline': {
            'route_report': str(route),
            'route_report_sha256': sha(route.read_bytes()),
            'route_bram_total_used': 38, 'route_bram_available': 46,
            'route_sp_blocks': 32, 'route_sdpb_blocks': 6,
            'separate_program_bytes': 32768, 'separate_data_bytes': 32768,
            'inferred_sp_blocks_each': 16,
            'sequencer_rtl_sha256': sha((ROOT / 'rtl/v2/tile_sequencer.sv').read_bytes()),
            'scratchpad_rtl_sha256': sha((ROOT / 'rtl/v2/scratchpad.sv').read_bytes()),
            'inference_basis': 'RTL has two identically banked v2_scratchpad(32768) instances; report has 32 SP macros.',
        },
        'partitions': partitions,
        'min_program_sp_blocks_64bit_each_model_raw': {name: x['program_bram_raw_64bit'] for name, x in models.items()},
        'min_program_sp_blocks_64bit_each_model_compact': {name: x['program_bram_compact_64bit'] for name, x in models.items()},
        'joint_at_32_sp_blocks': {
            name: {
                'raw_program_sp_blocks': x['program_bram_raw_64bit'],
                'compact_program_sp_blocks': x['program_bram_compact_64bit'],
                'raw_data_sp_blocks_remaining': 32 - x['program_bram_raw_64bit'],
                'compact_data_sp_blocks_remaining': 32 - x['program_bram_compact_64bit'],
            } for name, x in models.items()
        },
        'self_test': 'passed',
    }
    (OUTPUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({
        'kws': {k: models['kws'][k] for k in ('raw_bytes', 'compressed_bytes', 'program_bram_raw_64bit', 'program_bram_compact_64bit')},
        'vww': {k: models['vww'][k] for k in ('raw_bytes', 'compressed_bytes', 'program_bram_raw_64bit', 'program_bram_compact_64bit')},
        'report': str(OUTPUT / 'report.json'),
    }, indent=2))


if __name__ == '__main__':
    main()
