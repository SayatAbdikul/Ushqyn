#!/usr/bin/env python3
"""Read-only profitability gate for certified exact convolution early exit.

For two fixed test inputs, calculate conservative bounds on the remaining
centered INT8 products at 25%, 50%, and 75% of each filter.  Count a candidate
only if every possible remaining sum stays in INT32 and requantizes to the
same INT8 code.  This is an arithmetic-slot bound, not a hardware speedup.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'compiler'), str(ROOT/'tools/phase6')]
from integer_reference import evaluate, rounded
from run_boardless import load_model
from run_screening import save_json
from variants import check_frozen, sha


def windows_for_channel(x, layer, output_shape, channel, input_zero_point):
    weights = np.asarray(layer.parameters['weight'])
    kh, kw = weights.shape[2:]
    attrs = layer.attributes
    pt, pl, pb, pr = attrs.get('pads', [0, 0, 0, 0])
    sh, sw = attrs.get('strides', [1, 1])
    dh, dw = attrs.get('dilations', [1, 1])
    groups = attrs.get('group', 1)
    inputs_per_group = weights.shape[1]
    outputs_per_group = weights.shape[0] // groups
    first = (channel // outputs_per_group) * inputs_per_group
    selected = x[:, first:first+inputs_per_group].astype(np.int64) - input_zero_point
    pad = np.pad(selected, ((0, 0), (0, 0), (pt, pb), (pl, pr)))
    kernel_h = (kh-1)*dh+1
    kernel_w = (kw-1)*dw+1
    window = np.lib.stride_tricks.sliding_window_view(
        pad, (kernel_h, kernel_w), axis=(2, 3))
    window = window[:, :, ::sh, ::sw, ::dh, ::dw]
    if window.shape[2] != output_shape[2] or window.shape[3] != output_shape[3]:
        raise ValueError('convolution window geometry mismatch')
    return np.transpose(window, (0, 2, 3, 1, 4, 5)).reshape(-1, inputs_per_group*kh*kw)


def screen(program, input_value):
    values = evaluate(program, {program.inputs[0]: input_value})
    totals = dict(total_dense_mac_slots=0, certified_skip_slots=0,
                  certified_outputs=0, total_outputs=0, layers=[])
    for index, layer in enumerate(program.layers):
        if layer.op != 'Conv':
            continue
        x = values[layer.inputs[0]]
        y = values[layer.output]
        if x.ndim != 4 or y.ndim != 4:
            raise ValueError('expected NCHW Conv')
        weights = np.asarray(layer.parameters['weight'], dtype=np.int64)
        iq = program.tensors[layer.inputs[0]].quantization
        oq = program.tensors[layer.output].quantization
        max_abs = max(abs(-128 - iq.zero_point), abs(127 - iq.zero_point))
        layer_row = dict(index=index, outputs=0, dense_mac_slots=0,
                         certified_skip_slots=0, certified_outputs=0,
                         checkpoints={})
        for channel in range(weights.shape[0]):
            w = weights[channel].reshape(-1)
            # The selected compiler already removes wholly zero output filters.
            # Count only work that still reaches the hardware.
            if not np.any(w):
                continue
            k = len(w)
            centered = windows_for_channel(x, layer, y.shape, channel, iq.zero_point)
            n = len(centered)
            bias = int(layer.parameters['bias'][channel])
            multiplier = int(layer.parameters['multiplier'][channel])
            shift = int(layer.parameters['shift'][channel])
            actual = y[:, channel].reshape(-1)
            checkpoints = sorted({max(1, min(k, math.ceil(k*f))) for f in (.25, .5, .75)})
            undecided = np.ones(n, dtype=bool)
            for cut in checkpoints:
                remaining = int(np.abs(w[cut:]).sum()) * max_abs
                partial = bias + centered[:, :cut] @ w[:cut]
                low, high = partial - remaining, partial + remaining
                safe = (low >= -(1 << 31)) & (high <= (1 << 31)-1)
                if multiplier < 0:
                    lo_code = rounded(high * multiplier, shift, oq.zero_point)
                    hi_code = rounded(low * multiplier, shift, oq.zero_point)
                else:
                    lo_code = rounded(low * multiplier, shift, oq.zero_point)
                    hi_code = rounded(high * multiplier, shift, oq.zero_point)
                certified = safe & (lo_code == hi_code) & undecided
                if np.any(certified & (lo_code != actual)):
                    first = int(np.flatnonzero(certified & (lo_code != actual))[0])
                    full = bias + centered[first] @ w
                    raise ValueError(f'early-exit certificate disagrees with oracle '
                                     f'layer {index} channel {channel} cut {cut} '
                                     f'pixel {first}: range {low[first]}..{high[first]}, '
                                     f'full {full}, endpoint code {lo_code[first]}, '
                                     f'actual {actual[first]}')
                count = int(certified.sum())
                layer_row['checkpoints'][str(cut)] = layer_row['checkpoints'].get(str(cut), 0) + count
                layer_row['certified_outputs'] += count
                layer_row['certified_skip_slots'] += count * (k-cut)
                undecided &= ~certified
            layer_row['outputs'] += n
            layer_row['dense_mac_slots'] += n*k
        for key in ('outputs', 'dense_mac_slots', 'certified_skip_slots', 'certified_outputs'):
            aggregate = 'total_outputs' if key == 'outputs' else 'total_dense_mac_slots' if key == 'dense_mac_slots' else key
            totals[aggregate] += layer_row[key]
        totals['layers'].append(layer_row)
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check_frozen()
    report = dict(schema=1, status='running', physical_board=False,
                  scope='pinned and seeded stress arithmetic-slot profitability only',
                  models={})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, report)
    for name in ('kws', 'vww'):
        program, pinned, _, sources = load_model(name)
        row = dict(sources=sources, samples={})
        for label, value in (('pinned', pinned), ('stress',
            np.random.default_rng(6078).integers(-128, 128, pinned.shape, dtype=np.int8))):
            row['samples'][label] = screen(program, value)
            save_json(args.output, report)
        report['models'][name] = row
        save_json(args.output, report)
    report['status'] = 'passed-profitability-gate'
    save_json(args.output, report)
    for name, row in report['models'].items():
        print(name, {label: (v['certified_skip_slots'], v['total_dense_mac_slots'])
                     for label, v in row['samples'].items()}, flush=True)


if __name__ == '__main__':
    main()
