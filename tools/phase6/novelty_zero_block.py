#!/usr/bin/env python3
"""Exact all-zero-point block opportunity screen for the frozen P6 schedules.

This is a boardless, value-aware lower-bound study, not an RTL speed claim.
It decodes every actual pointwise descriptor in the pinned 27 MHz three-pair
schedule, applies the corresponding compacted graph tensor and strip, and
counts eight-pixel blocks that are all equal to the input zero point.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'compiler'), str(ROOT / 'tools/phase6')]
from hardware_v2 import Descriptor
from integer_reference import evaluate
from output_pipeline_fusion import decode_fused
from run_boardless import load_model
from followup_graph import group_channels
from channel_compaction import compact_channels

CMD = struct.Struct('<BBHIII')
FIXTURE_ROOT = ROOT / 'work/phase6/strip-fusion-pair7-v1/full/fixtures'
FIXTURES = {
    'kws': 'kws-pinned-compacted-fused-timed',
    'vww': 'vww-pinned-compacted-strip3-7-11-timed',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schedule_pointwise_runs(folder):
    """Replay descriptor DMA writes to recover the *executed* pointwise runs."""
    schedule = json.loads((folder / 'schedule.json').read_text())
    code = (folder / 'commands.bin').read_bytes()
    image = (folder / 'payload.bin').read_bytes()
    sram = bytearray(32768)
    runs = []
    for index in range(len(code) // CMD.size):
        op, flags, reserved, a, b, size = CMD.unpack_from(code, index * CMD.size)
        assert reserved == 0
        if op == 1 and flags == 1:
            assert a + size <= len(image) and b + size <= len(sram)
            sram[b:b + size] = image[a:a + size]
        elif op == 2:
            try:
                descriptor, _ = decode_fused(bytes(sram[:64]))
            except ValueError:
                descriptor = Descriptor.decode(bytes(sram[:64]))
            contract = schedule['run_contracts'].get(str(index), {})
            if (descriptor.opcode == 4 and descriptor.kernel_h == descriptor.kernel_w == 1
                    and descriptor.stride_h == descriptor.stride_w == 1):
                assert 'layer' in contract, (folder, index)
                runs.append((index, descriptor, contract,
                             bytes(sram[descriptor.weight:descriptor.weight +
                                        descriptor.output_c * descriptor.row_stride])))
    return runs


def profile_run(program, values, index, descriptor, contract, raw_weights):
    layer_index = contract['layer']
    layer = program.layers[layer_index]
    assert layer.op == 'Conv' and layer.attributes.get('group', 1) == 1
    assert layer.parameters['weight'].shape[2:] == (1, 1)
    source_name = layer.inputs[0]
    source = values[source_name]
    zp = program.tensors[source_name].quantization.zero_point
    origin = contract.get('strip_origin_y', 0)
    assert source.shape[0] == 1 and source.shape[1] == descriptor.input_c
    assert origin + descriptor.input_h <= source.shape[2]
    assert descriptor.input_w == source.shape[3]
    piece = source[0, :, origin:origin + descriptor.input_h, :].reshape(descriptor.input_c, -1)
    plane = piece.shape[1]
    assert descriptor.outputs == descriptor.output_c * plane
    weights = layer.parameters['weight'][:descriptor.output_c, :, 0, 0].astype(np.int64)
    packed = np.frombuffer(raw_weights, np.int8).reshape(descriptor.output_c, descriptor.row_stride)
    assert np.array_equal(weights, packed[:, :descriptor.input_c])
    block_count = (plane + 7) // 8
    metadata_bits = descriptor.input_c * block_count
    zero_groups = 0
    numeric_zero_groups = 0
    weighted_skip_groups = 0
    physical_word_certified_groups = 0
    physical_word_certified_weighted_skips = 0
    already_zero_weight_groups = 0
    eligible_lane_mac = 0
    # The spatial pointwise datapath issues one eight-lane iteration for
    # each input channel, output channel, and spatial block. It includes a
    # final partial block, which this screen excludes from dynamic skipping.
    dynamic_mask = np.zeros_like(piece, dtype=bool)
    physical_mask = np.zeros_like(piece, dtype=bool)
    flat = piece.reshape(-1)
    for channel in range(descriptor.input_c):
        nonzero_outputs = int(np.count_nonzero(weights[:, channel]))
        for start in range(0, plane, 8):
            take = min(8, plane - start)
            chunk = piece[channel, start:start + take]
            if take < 8:
                continue
            if np.all(chunk == 0):
                numeric_zero_groups += 1
            if np.all(chunk == zp):
                zero_groups += 1
                dynamic_mask[channel, start:start + 8] = True
                weighted_skip_groups += nonzero_outputs
                already_zero_weight_groups += descriptor.output_c - nonzero_outputs
                eligible_lane_mac += 8 * nonzero_outputs
                # A producer can cheaply tag an aligned *physical* 64-bit
                # SRAM word during writeback. A logical eight-pixel group may
                # cross two such words when the channel plane is unaligned.
                absolute = channel * plane + start
                first = absolute // 8 * 8
                last = (absolute + 7) // 8 * 8
                if (last + 8 <= flat.size and
                        np.all(flat[first:first + 8] == zp) and
                        np.all(flat[last:last + 8] == zp)):
                    physical_word_certified_groups += 1
                    physical_word_certified_weighted_skips += nonzero_outputs
                    physical_mask[channel, start:start + 8] = True
    # The frozen RTL starts from corrected bias and multiplies raw bytes.
    # Skipping a zero-point block requires exactly zp*w in each affected lane.
    # Verify both accumulation forms against the independent centered oracle
    # for every pixel and every output in this executed descriptor.
    corrected_bias = layer.parameters['corrected_bias'][:descriptor.output_c].astype(np.int64)
    bias = layer.parameters['bias'][:descriptor.output_c].astype(np.int64)
    assert np.array_equal(corrected_bias, bias - zp * weights.sum(axis=1))
    raw = piece.astype(np.int64)
    centered = bias[:, None] + weights @ (raw - zp)
    dense = corrected_bias[:, None] + weights @ raw
    compensated = (corrected_bias[:, None] + weights @ np.where(dynamic_mask, 0, raw)
                   + weights @ (dynamic_mask.astype(np.int64) * zp))
    physical_compensated = (corrected_bias[:, None] + weights @ np.where(physical_mask, 0, raw)
                            + weights @ (physical_mask.astype(np.int64) * zp))
    assert np.array_equal(centered, dense)
    assert np.array_equal(centered, compensated)
    assert np.array_equal(centered, physical_compensated)
    assert np.max(np.abs(centered)) < 2**31
    zero_channel_runs = {}
    full_eight_channel_words = 0
    for start in range(0, plane - plane % 8, 8):
        active = dynamic_mask[:, start]
        run_length = 0
        for flag in (*active, False):
            if flag:
                run_length += 1
            elif run_length:
                zero_channel_runs[str(run_length)] = zero_channel_runs.get(str(run_length), 0) + 1
                run_length = 0
        for channel in range(0, descriptor.input_c - descriptor.input_c % 8, 8):
            full_eight_channel_words += int(np.all(active[channel:channel + 8]))
    iterations = descriptor.output_c * descriptor.input_c * block_count
    row = dict(command=index, layer=layer_index, descriptor_input_hw=[descriptor.input_h, descriptor.input_w],
               origin_y=origin, input_channels=descriptor.input_c, output_channels=descriptor.output_c,
               zero_point=zp, spatial_blocks=block_count, full_spatial_blocks=plane // 8,
               executed_channel_output_block_iterations=iterations,
               metadata_bits=metadata_bits, metadata_bytes=(metadata_bits + 7) // 8,
               physical_word_metadata_bits=(descriptor.input_c * plane + 7) // 8,
               quantized_zero_groups=zero_groups, numeric_zero_groups=numeric_zero_groups,
               weighted_skippable_iterations=weighted_skip_groups,
               physical_word_certified_groups=physical_word_certified_groups,
               physical_word_certified_weighted_skips=physical_word_certified_weighted_skips,
               zero_channel_run_lengths=zero_channel_runs,
               full_eight_channel_words=full_eight_channel_words,
               zero_weight_iterations_with_zero_input=already_zero_weight_groups,
               eligible_lane_mac=eligible_lane_mac,
               max_abs_centered_accumulator=int(np.max(np.abs(centered))),
               exact_compensation='zp * weight per output accumulator lane')
    return row


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    report = dict(schema=1, status='running', scope='pinned 27 MHz three-pair schedule, actual decoded pointwise descriptors; independent INT8 graph values; boardless only',
                  provenance={}, models={})
    for model in ('kws', 'vww'):
        original, pinned, manifest, sources = load_model(model)
        grouped, _, _ = group_channels(original)
        program, _, _, _ = compact_channels(grouped)
        fixture = FIXTURE_ROOT / FIXTURES[model]
        runs = schedule_pointwise_runs(fixture)
        report['provenance'][model] = dict(model_sources=sources,
            fixture={name: sha(fixture / name) for name in ('commands.bin', 'payload.bin', 'schedule.json')})
        model_result = dict(run_count=len(runs), samples={})
        for sample, data in (('pinned', pinned),
                             ('stress', np.random.default_rng(6157).integers(-128, 128, pinned.shape, dtype=np.int8))):
            values = evaluate(program, {program.inputs[0]: data})
            rows = [profile_run(program, values, *row) for row in runs]
            keys = ('executed_channel_output_block_iterations', 'metadata_bits',
                    'physical_word_metadata_bits', 'full_eight_channel_words',
                    'quantized_zero_groups', 'numeric_zero_groups',
                    'weighted_skippable_iterations', 'zero_weight_iterations_with_zero_input',
                    'physical_word_certified_groups', 'physical_word_certified_weighted_skips',
                    'eligible_lane_mac')
            totals = {k: sum(row[k] for row in rows) for k in keys}
            totals['metadata_bytes_sum_per_run'] = sum(row['metadata_bytes'] for row in rows)
            totals['weighted_skip_fraction'] = (totals['weighted_skippable_iterations'] /
                                                totals['executed_channel_output_block_iterations'])
            totals['physical_word_weighted_skip_fraction'] = (
                totals['physical_word_certified_weighted_skips'] /
                totals['executed_channel_output_block_iterations'])
            # This is deliberately optimistic: every eligible iteration has
            # at most an input fetch and a MAC to avoid, but exact compensation
            # still needs a weight-dependent accumulator update. Overlap and
            # metadata access are charged only in later RTL measurements.
            totals['optimistic_max_two_cycles_per_skip'] = 2 * totals['weighted_skippable_iterations']
            totals['optimistic_physical_word_two_cycles_per_skip'] = 2 * totals['physical_word_certified_weighted_skips']
            totals['break_even_lookup_cycles_per_iteration'] = 2 * totals['weighted_skip_fraction']
            model_result['samples'][sample] = dict(totals=totals, runs=rows)
        report['models'][model] = model_result
        (output / 'report.partial.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    report['status'] = 'passed-opportunity-screen'
    report['source_sha256'] = sha(Path(__file__))
    (output / 'report.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    print(json.dumps({m: {s: d['totals'] for s, d in row['samples'].items()}
                      for m, row in report['models'].items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'work/phase6/novelty-zero-block-v1')
    args = parser.parse_args()
    run(args.output)
