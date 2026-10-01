#!/usr/bin/env python3
"""Bounded, executable low-rank representation screen; no FPGA speed claims.

All factor weights are signed INT8. Latents use 8/16/24 bits with one explicit
ties-away power-of-two rounding and saturation. Wide second-stage products are
charged as ceil(width/8) INT8 limbs. Activation-aware reduced-rank regression is
a closed-form teacher fit on development activations, not end-to-end training.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'compiler'), str(ROOT / 'tools/phase6')]
from integer_reference import evaluate as oracle
from program_image import build_image, load_image
from quantization import Quantization, multiplier_shift, round_away
from run_boardless import load_model
from static_pipeline import Layer, Tensor, execute_layer

BASE = ROOT / 'work/phase6/representation-screen-v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def member(archive, name, folder):
    """One streaming decompression followed by mmap; never load a full image pool."""
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / (name.replace('/', '_') + '.npy')
    with zipfile.ZipFile(archive) as z:
        info = z.getinfo(name + '.npy')
        if not dest.exists() or dest.stat().st_size != info.file_size:
            with z.open(info) as src, dest.open('wb') as out:
                shutil.copyfileobj(src, out, 1 << 20)
    return np.load(dest, mmap_mode='r', allow_pickle=False)


def requant(acc, m, s, zp):
    acc = np.asarray(acc, np.int64)
    if np.any(acc < -(1 << 31)) or np.any(acc > (1 << 31) - 1):
        raise ArithmeticError('second-stage or dense accumulator outside INT32')
    m, s = np.asarray(m, np.int64), np.asarray(s, np.int64)
    product = acc * m
    rounding = np.where(s, np.left_shift(1, np.maximum(s - 1, 0)), 0)
    mag = (np.abs(product) + rounding) >> s
    return np.clip(np.where(product < 0, -mag, mag) + zp, -128, 127).astype(np.int8)


def shift_away(x, shift):
    x = np.asarray(x, np.int64)
    if not shift:
        return x
    mag = (np.abs(x) + (1 << (shift - 1))) >> shift
    return np.where(x < 0, -mag, mag)


def linear_matrix(program, layer, x):
    iq = program.tensors[layer.inputs[0]].quantization
    if layer.op == 'Gemm':
        return x.reshape(-1, x.shape[-1]).astype(np.int64) - iq.zero_point
    return x.transpose(0, 2, 3, 1).reshape(-1, x.shape[1]).astype(np.int64) - iq.zero_point


def linear_shape(program, layer, y):
    shape = program.tensors[layer.output].shape
    if layer.op == 'Gemm':
        return y.reshape((-1, shape[-1]))
    return y.reshape(shape[0], shape[2], shape[3], shape[1]).transpose(0, 3, 1, 2).copy()


def dense_linear(program, layer, x, weight=None):
    p = layer.parameters
    w = p['weight'].reshape(p['weight'].shape[0], -1) if weight is None else weight
    acc = linear_matrix(program, layer, x) @ w.astype(np.int64).T + p['bias']
    return linear_shape(program, layer, requant(acc, p['multiplier'], p['shift'],
                        program.tensors[layer.output].quantization.zero_point))


def eligible(layer):
    return layer.op == 'Gemm' or (layer.op == 'Conv' and
           layer.parameters['weight'].shape[2:] == (1, 1) and layer.attributes.get('group', 1) == 1)


def run(program, sample, changes=None, capture=False, counters=None):
    changes = changes or {}
    values = {n: a.copy() for n, a in program.constants.items()}
    values[program.inputs[0]] = sample
    observed = {}
    for i, layer in enumerate(program.layers):
        x = values[layer.inputs[0]]
        if eligible(layer):
            if capture:
                observed[i] = linear_matrix(program, layer, x)
            if i in changes:
                change = changes[i]
                if change['kind'] == 'pruning':
                    y = dense_linear(program, layer, x, change['weight'])
                else:
                    centered = linear_matrix(program, layer, x)
                    latent = centered @ change['aq'].astype(np.int64).T
                    latent = shift_away(latent, change['shift'])
                    lo, hi = -(1 << (change['width'] - 1)), (1 << (change['width'] - 1)) - 1
                    clips = int(np.count_nonzero((latent < lo) | (latent > hi)))
                    if counters is not None:
                        counters['latent_values'] += latent.size
                        counters['latent_clips'] += clips
                    latent = np.clip(latent, lo, hi)
                    acc = latent @ change['bq'].astype(np.int64).T + change['bias']
                    y = linear_shape(program, layer, requant(acc, change['m'], change['s'],
                                     program.tensors[layer.output].quantization.zero_point))
            else:
                y = dense_linear(program, layer, x)
        else:
            y = execute_layer(program, layer, values)
        values[layer.output] = y
    return values[program.outputs[0]], observed, values if capture else None


def spectra(program):
    rows = []
    for i, l in enumerate(program.layers):
        if not eligible(l):
            continue
        p = l.parameters
        wi = p['weight'].reshape(p['weight'].shape[0], -1).astype(np.float64)
        w = wi * p['weight_scales'][:, None]
        s = np.linalg.svd(w, compute_uv=False)
        si = np.linalg.svd(wi, compute_uv=False)
        energy, energy_i = np.cumsum(s*s) / np.sum(s*s), np.cumsum(si*si) / np.sum(si*si)
        co, ci = w.shape
        positions = math.prod(program.tensors[l.output].shape) // co
        rows.append({'layer': i, 'op': l.op, 'ci': ci, 'co': co, 'positions': positions,
                     'dense_macs': positions * ci * co, 'dense_weight_bytes': ci * co,
                     'break_even_rank': ci * co / (ci + co),
                     'dequantized_singular_values': s.tolist(),
                     'dequantized_rank_at_energy': {str(t): int(np.searchsorted(energy, t) + 1) for t in (.9, .95, .99)},
                     'integer_code_rank_at_energy': {str(t): int(np.searchsorted(energy_i, t) + 1) for t in (.9, .95, .99)}})
    return rows


def total_macs(program):
    return sum(math.prod(program.tensors[l.output].shape) * math.prod(l.parameters['weight'].shape[1:])
               for l in program.layers if l.op in ('Conv', 'Gemm'))


def factor(w, rank, covariance, method):
    if method == 'activation-regression':
        ridge = max(float(np.trace(covariance) / len(covariance)) * 1e-6, 1e-8)
        eigen, vectors = np.linalg.eigh(covariance + ridge * np.eye(len(covariance)))
        root = (vectors * np.sqrt(eigen)) @ vectors.T
        inverse = (vectors / np.sqrt(eigen)) @ vectors.T
        u, s, vt = np.linalg.svd(w @ root, full_matrices=False)
        b = u[:, :rank] * np.sqrt(s[:rank])
        a = (np.sqrt(s[:rank])[:, None] * vt[:rank]) @ inverse
    else:
        u, s, vt = np.linalg.svd(w, full_matrices=False)
        b = u[:, :rank] * np.sqrt(s[:rank])
        a = np.sqrt(s[:rank])[:, None] * vt[:rank]
    # Balance each rank component before global first-factor quantization.
    balance = np.sqrt(np.maximum(np.max(np.abs(b), axis=0), 1e-30) /
                      np.maximum(np.max(np.abs(a), axis=1), 1e-30))
    a *= balance[:, None]
    b /= balance[None, :]
    a_scale = max(float(np.max(np.abs(a))) / 127, 1e-20)
    b_scale = np.maximum(np.max(np.abs(b), axis=1) / 127, 1e-20)
    aq = np.clip(round_away(a / a_scale), -127, 127).astype(np.int8)
    bq = np.clip(round_away(b / b_scale[:, None]), -127, 127).astype(np.int8)
    reconstructed = (bq.astype(float) * b_scale[:, None]) @ (aq.astype(float) * a_scale)
    return aq, bq, a_scale, b_scale, reconstructed


def prepare(program, row, covariance, probes, rank, method, width):
    layer = program.layers[row['layer']]
    p = layer.parameters
    w = p['weight'].reshape(row['co'], row['ci']).astype(float) * p['weight_scales'][:, None]
    aq, bq, a_scale, b_scale, reconstructed = factor(w, rank, covariance, method)
    maximum = max((int(np.max(np.abs(x @ aq.astype(np.int64).T))) for x in probes), default=0)
    shift = max(0, int(math.ceil(math.log2(max(maximum, 1) / ((1 << (width - 1)) - 1)))))
    iq = program.tensors[layer.inputs[0]].quantization
    oq = program.tensors[layer.output].quantization
    # BN-folded models have zero/near-zero weight rows with nonzero biases.
    # Reserve half the INT32 range for bias and a representable output scale,
    # as the production quantizer also does for these channels.
    original_b_scale = b_scale.copy()
    physical_bias = p['bias'].astype(float) * p['weight_scales']
    minimum_effective = np.maximum(np.abs(physical_bias) / ((1 << 30) - 1),
                                   (oq.scale / iq.scale) * 2.**-31)
    b_scale = np.maximum(b_scale, minimum_effective / (a_scale * 2.**shift))
    bq = np.clip(round_away(bq.astype(float) * original_b_scale[:, None] / b_scale[:, None]),
                 -127, 127).astype(np.int8)
    reconstructed = (bq.astype(float) * b_scale[:, None]) @ (aq.astype(float) * a_scale)
    effective_scale = a_scale * b_scale * 2.**shift
    bias = round_away(physical_bias / effective_scale).astype(np.int64)
    # Same all-input bound as the software-v2 accumulator contract.
    stage1_bound = max(abs(-128 - iq.zero_point), abs(127 - iq.zero_point)) * np.abs(aq.astype(np.int64)).sum(axis=1)
    # The latent's analytical range is usually much narrower than its storage
    # type. Charge every possible input, rather than unconstrained type values.
    latent_bound = np.minimum(shift_away(stage1_bound, shift), (1 << (width - 1)))
    stage2_bound = np.abs(bq.astype(np.int64)) @ latent_bound + np.abs(bias)
    if np.any(stage1_bound > (1 << 31) - 1) or np.any(stage2_bound > (1 << 31) - 1):
        raise ArithmeticError('factor accumulator range proof failed')
    ms = [multiplier_shift(float(v * iq.scale / oq.scale)) for v in effective_scale]
    m = np.array([v[0] for v in ms], dtype='<i4')
    s = np.array([v[1] for v in ms], dtype=np.int8)
    return {'kind': 'factor', 'rank': rank, 'method': method, 'width': width, 'shift': shift,
            'aq': aq, 'bq': bq, 'a_scale': a_scale, 'b_scale': b_scale,
            'bias': bias, 'm': m, 's': s,
            'fit_relative_squared_error': float(np.sum(((w - reconstructed) @ covariance) * (w - reconstructed)) /
                                                max(np.sum((w @ covariance) * w), 1e-30)),
            'weight_relative_squared_error': float(np.sum((w - reconstructed)**2) / max(np.sum(w*w), 1e-30)),
            'dev_latent_max_before_shift': maximum,
            'stage1_worst_bound': int(np.max(stage1_bound)), 'stage2_worst_bound': int(np.max(stage2_bound))}


def cost(program, rows, changes):
    total = total_macs(program)
    useful = lanes = total
    dense_bytes = factor_bytes = 0
    extra_latent_transfer_bytes = 0
    details = []
    for row in rows:
        i = row['layer']
        if i not in changes:
            continue
        c = changes[i]
        ci, co, p = row['ci'], row['co'], row['positions']
        dense_bytes += ci * co
        if c['kind'] == 'pruning':
            k = c['retained_columns']
            new = p * co * k
            useful += new - p * ci * co
            lanes += new - p * ci * co
            factor_bytes += co * k
            details.append({'layer': i, 'retained_columns': k, 'dense_macs': p*ci*co,
                            'pruned_macs': new, 'weight_bytes': co*k})
            continue
        rank, width = c['rank'], c['width']
        first, second = p * ci * rank, p * co * rank
        useful += first + second - p * ci * co
        # Width is costed only on the second factor, which consumes the latent.
        limbs = math.ceil(width / 8)
        charged = first + limbs * second
        lanes += charged - p * ci * co
        nbytes = rank * (ci + co)
        factor_bytes += nbytes
        spill = 2 * p * rank * math.ceil(width / 8)
        extra_latent_transfer_bytes += spill
        # Separate factor calls on the same 8 spatial INT8 lanes; tile padding
        # charged explicitly. These are lane-work counts, never elapsed cycles.
        dense_lane_work = math.ceil(p/8) * co * ci
        factor_lane_work = math.ceil(p/8) * (rank*ci + limbs*co*rank)
        details.append({'layer': i, 'rank': rank, 'latent_width': width, 'latent_shift': c['shift'],
                        'factor_weight_bytes': nbytes, 'dense_macs': p*ci*co,
                        'factor_useful_macs': first+second, 'charged_int8_limb_products': charged,
                        'dense_8_spatial_lane_work': dense_lane_work,
                        'factor_8_spatial_lane_work': factor_lane_work,
                        'latent_unfused_write_read_bytes': spill,
                        'latent_fused_8pixel_local_bytes': 8*rank*math.ceil(width/8),
                        'all_factor_weights_8pixel_min_live_bytes': nbytes+8*(ci+co)+8*rank*math.ceil(width/8)+9*(rank+co),
                        'fit_relative_squared_error': c['fit_relative_squared_error'],
                        'weight_relative_squared_error': c['weight_relative_squared_error'],
                        'stage1_worst_bound': c['stage1_worst_bound'], 'stage2_worst_bound': c['stage2_worst_bound']})
    return {'scope': 'operation, parameter and logical-transfer counts; no measured cycle/latency/energy benefit',
            'dense_model_useful_macs': total, 'candidate_model_useful_macs': useful,
            'candidate_model_charged_int8_limb_products': lanes,
            'useful_mac_reduction_fraction': (total-useful)/total,
            'charged_limb_product_reduction_fraction': (total-lanes)/total,
            'changed_dense_weight_bytes': dense_bytes, 'changed_factor_weight_bytes': factor_bytes,
            'weight_payload_saving_bytes': dense_bytes-factor_bytes,
            'unfused_extra_latent_write_read_bytes': extra_latent_transfer_bytes,
            'fused_extra_latent_shared_port_bytes': 0,
            'fusion_incremental_shared_port_word_saving_upper_bound': math.ceil(extra_latent_transfer_bytes/8),
            'fusion_control': 'ordinary blocked factor-pair fusion can also eliminate this spill; no unique fusion advantage established',
            'layers': details}


def stratified(labels, count, rng):
    groups = [rng.permutation(np.flatnonzero(labels == y)) for y in np.unique(labels)]
    chosen = []
    step = 0
    while len(chosen) < min(count, len(labels)):
        for group in groups:
            if step < len(group) and len(chosen) < count:
                chosen.append(int(group[step]))
        step += 1
    return np.array(chosen, dtype=int)


def eval_class(program, features, labels, indices, configs):
    rows = {name: {'correct': 0, 'baseline_agreement': 0, 'latent_values': 0, 'latent_clips': 0} for name in configs}
    predictions = {name: [] for name in configs}
    for j, index in enumerate(indices):
        x = program.tensors[program.inputs[0]].quantization.encode(np.asarray(features[index]))
        base, _, _ = run(program, x)
        expected = int(np.argmax(base))
        for name, change in configs.items():
            y = base if not change else run(program, x, change, counters=rows[name])[0]
            pred = int(np.argmax(y))
            predictions[name].append(pred)
            rows[name]['correct'] += pred == int(labels[index])
            rows[name]['baseline_agreement'] += pred == expected
        if j % 32 == 0:
            print(f'evaluation {j+1}/{len(indices)} ({len(configs)} configurations)', flush=True)
    for name, row in rows.items():
        row.update(samples=len(indices), accuracy=row['correct']/len(indices),
                   prediction_agreement=row['baseline_agreement']/len(indices))
    return rows, predictions


def auc(labels, scores):
    positive = scores[labels == 1][:, None]
    negative = scores[labels == 0][None, :]
    return float(np.mean((positive > negative) + .5*(positive == negative)))


def eval_ad(program, features, recording_index, labels, recordings, configs):
    rows, score_rows = {}, {}
    oq = program.tensors[program.outputs[0]].quantization
    for name, change in configs.items():
        stats = {'latent_values': 0, 'latent_clips': 0}
        scores = []
        for record in recordings:
            indices = np.flatnonzero(recording_index == record)
            raw = np.asarray(features[indices])
            x = program.tensors[program.inputs[0]].quantization.encode(raw).reshape(len(raw), -1)
            y = run(program, x, change, counters=stats)[0]
            scores.append(float(np.mean((raw.reshape(len(raw), -1) - oq.decode(y))**2)))
        score_rows[name] = scores
        rows[name] = {'recordings': len(recordings), 'windows': int(sum(np.count_nonzero(recording_index == r) for r in recordings)),
                      'pooled_auc': auc(labels[recordings], np.array(scores)), **stats}
        print(name, rows[name], flush=True)
    return rows, score_rows


def export_int8(program, changes, path):
    output = copy.deepcopy(program)
    layers = []
    for i, layer in enumerate(output.layers):
        if i not in changes:
            layers.append(layer)
            continue
        c = changes[i]
        if c['kind'] != 'factor' or c['width'] != 8:
            raise ValueError('export requires INT8 latent factors')
        rank = c['rank']
        iq = output.tensors[layer.inputs[0]].quantization
        original_shape = output.tensors[layer.output].shape
        latent_name = layer.output + '/representation_latent'
        shape = (original_shape[0], rank, *original_shape[2:]) if layer.op == 'Conv' else (original_shape[0], rank)
        output.tensors[latent_name] = Tensor(latent_name, shape, Quantization(iq.scale*c['a_scale']*2.**c['shift'], 0),
                                            output.tensors[layer.output].layout)
        aq = c['aq'].reshape(rank, c['aq'].shape[1], 1, 1) if layer.op == 'Conv' else c['aq']
        ap = {'weight': aq, 'bias': np.zeros(rank, dtype='<i4'),
              'corrected_bias': (-iq.zero_point*c['aq'].astype(np.int64).sum(axis=1)).astype('<i4'),
              'weight_scales': np.full(rank, c['a_scale'], dtype='<f8'),
              'multiplier': np.ones(rank, dtype='<i4'), 'shift': np.full(rank, c['shift'], dtype=np.int8)}
        bq = c['bq'].reshape(c['bq'].shape[0], rank, 1, 1) if layer.op == 'Conv' else c['bq']
        bp = {'weight': bq, 'bias': c['bias'].astype('<i4'), 'corrected_bias': c['bias'].astype('<i4'),
              'weight_scales': c['b_scale'].astype('<f8'), 'multiplier': c['m'], 'shift': c['s']}
        layers.append(Layer(layer.op, list(layer.inputs), latent_name, copy.deepcopy(layer.attributes), ap))
        layers.append(Layer(layer.op, [latent_name], layer.output, copy.deepcopy(layer.attributes), bp))
    output.layers = layers
    output.provenance['representation_screen'] = {'source_sha256': digest(Path(__file__)), 'original_layers': sorted(changes),
                                                 'numerics': 'INT8 factors and INT8 latent; development-calibrated power-of-two latent scales'}
    data = build_image(output)
    path.write_bytes(data)
    return output, {'path': str(path.relative_to(ROOT)), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=('kws','vww','ad'), default=['kws','vww','ad'])
    parser.add_argument('--class-dev', type=int, default=48)
    parser.add_argument('--class-heldout', type=int, default=256)
    parser.add_argument('--ad-dev', type=int, default=48)
    args = parser.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'source_sha256': digest(Path(__file__)), 'models': {},
              'scope': __doc__, 'quality_gate': {'classification_loss_absolute': .01, 'auc_loss_absolute': .01},
              'investment_gate': {'charged_limb_product_reduction': .15, 'is_measured_latency_gate': False},
              'threads': 1, 'physical': 'not_run', 'native': 'not_run', 'end_to_end_training': 'not_run'}
    # Resume independent model runs without discarding completed reports. Each
    # export pins its own driver version, including earlier numerical repairs.
    for prior_name in ('kws','vww','ad'):
        prior_path = BASE/prior_name/'report.json'
        if prior_path.exists() and prior_name not in args.models:
            prior = json.loads(prior_path.read_text())
            exported_path = ROOT/prior['export']['path']
            prior['experiment_driver_sha256'] = load_image(exported_path.read_bytes()).provenance['representation_screen']['source_sha256']
            report['models'][prior_name] = prior
    for name in args.models:
        print('begin', name, flush=True)
        start = time.perf_counter()
        if name == 'ad':
            image = ROOT/'work/phase6/parallel-ad-v1/ad.uq2'
            program = load_image(image.read_bytes())
            pins = {str(image.relative_to(ROOT)): digest(image)}
            archive = ROOT/'work/phase0-ad/features/ad.accuracy.npz'
        else:
            program, pinned_input, _, pins = load_model(name)
            archive = ROOT/f'work/quality/{name}.accuracy.npz'
        manifest_path = ROOT/f'benchmarks/manifests/{name}.data.json'
        manifest = json.loads(manifest_path.read_text())
        archive_hash = digest(archive)
        if archive_hash != manifest['splits']['accuracy']['npz_sha256']:
            raise ValueError('evaluation archive differs from pinned manifest')
        folder = BASE/name
        folder.mkdir(exist_ok=True)
        features = member(archive, program.inputs[0], folder/'mapped')
        ids = member(archive, 'sample_ids', folder/'mapped')
        rng = np.random.default_rng(61001)
        if name == 'ad':
            labels = member(archive, 'recording_labels', folder/'mapped')
            recording_index = member(archive, 'recording_index', folder/'mapped')
            dev = stratified(labels, args.ad_dev, rng)
            heldout = np.setdiff1d(np.arange(len(labels)), dev)
            fit_indices = np.concatenate([np.flatnonzero(recording_index == r)[::25] for r in dev])
        else:
            labels = member(archive, 'labels', folder/'mapped')
            order = stratified(labels, args.class_dev + args.class_heldout, rng)
            dev, heldout = order[:args.class_dev], order[args.class_dev:]
            fit_indices = dev
        rows = spectra(program)
        cov = {r['layer']: np.zeros((r['ci'], r['ci']), float) for r in rows}
        probes = {r['layer']: [] for r in rows}
        for index in fit_indices:
            x = program.tensors[program.inputs[0]].quantization.encode(np.asarray(features[index]))
            if name == 'ad':
                x = x.reshape(1, -1)
            _, captures, values = run(program, x, capture=True)
            for i, matrix in captures.items():
                # Bound covariance/latent calibration to 32 spatial rows/sample.
                xx = matrix[np.linspace(0, len(matrix)-1, min(32, len(matrix)), dtype=int)]
                cov[i] += xx.astype(float).T @ xx.astype(float)
                probes[i].append(xx)
        # Check the accelerated dense implementation against the independent
        # oracle at every layer on real development samples.
        checks = []
        for index in fit_indices[:2]:
            x = program.tensors[program.inputs[0]].quantization.encode(np.asarray(features[index]))
            expected = oracle(program, {program.inputs[0]: x})
            _, _, values = run(program, x, capture=True)
            if any(not np.array_equal(values[l.output], expected[l.output]) for l in program.layers):
                raise AssertionError('dense evaluator differs from independent oracle')
            checks.append(int(index))
        configs = {'dense': {}}
        # Candidate ranks depend only on spectra; development quality picks one
        # policy. Held-out results never rank or alter configurations.
        for policy in ('energy99', 'energy95', 'aggressive'):
            for method in ('svd', 'activation-regression'):
                changes = {}
                for row in rows:
                    # Preserve narrow classifier/autoencoder bottleneck layers.
                    if min(row['ci'], row['co']) < 16:
                        continue
                    rank = (int(row['break_even_rank']*.75) if policy == 'aggressive' else
                            row['dequantized_rank_at_energy']['0.99' if policy == 'energy99' else '0.95'])
                    rank = max(1, rank)
                    if rank * (row['ci'] + row['co']) > .85 * row['ci'] * row['co']:
                        continue
                    changes[row['layer']] = prepare(program, row, cov[row['layer']], probes[row['layer']], rank, method, 8)
                if changes:
                    configs[f'{policy}/{method}/int8'] = changes
        costs = {k: cost(program, rows, v) for k, v in configs.items()}
        if name == 'ad':
            dev_quality, dev_predictions = eval_ad(program, features, recording_index, labels, dev, configs)
            metric = 'pooled_auc'
        else:
            dev_quality, dev_predictions = eval_class(program, features, labels, dev, configs)
            metric = 'accuracy'
        baseline_quality = dev_quality['dense'][metric]
        candidates = [k for k in configs if k != 'dense']
        passing = [k for k in candidates if dev_quality[k][metric] >= baseline_quality-.01 and
                   costs[k]['charged_limb_product_reduction_fraction'] >= .15]
        # If all gates fail, keep the best-quality candidate for an honest
        # held-out rejection check; never make a post-heldout second selection.
        pool = passing or candidates
        selected = max(pool, key=lambda k: (costs[k]['charged_limb_product_reduction_fraction'], dev_quality[k][metric])) if passing else max(pool, key=lambda k: (dev_quality[k][metric], costs[k]['charged_limb_product_reduction_fraction']))
        selected_changes = configs[selected]
        wide_configs, wide_rejections = {}, {}
        for width in (16,24):
            try:
                wide_configs[f'selected/int{width}'] = {i: prepare(program, next(r for r in rows if r['layer']==i), cov[i], probes[i],
                                                                 c['rank'], c['method'], width) for i,c in selected_changes.items()}
            except ArithmeticError as error:
                wide_rejections[f'selected/int{width}'] = {'status':'rejected-range-proof', 'reason':str(error),
                    'scope':'this width/rank/factor implementation with the INT32 accumulator contract; wider accumulators not implemented'}
        pruning = {}
        for row in rows:
            i = row['layer']
            if i not in selected_changes:
                continue
            budget = selected_changes[i]['rank']*(row['ci']+row['co'])
            keep = min(row['ci'], max(1, budget // row['co']))
            layer = program.layers[i]
            weight = layer.parameters['weight'].reshape(row['co'],row['ci']).copy()
            real_w = weight.astype(float)*layer.parameters['weight_scales'][:,None]
            importance = np.sum(real_w*real_w,axis=0)*np.diag(cov[i])
            retained = np.argsort(importance)[-keep:]
            dropped = np.setdiff1d(np.arange(row['ci']),retained)
            weight[:, dropped] = 0
            pruning[i] = {'kind':'pruning','weight':weight,'retained_columns':keep}
        heldout_configs = {'dense': {}, 'selected/int8': selected_changes, 'matched-column-pruning': pruning, **wide_configs}
        frozen = {'selected': selected, 'passed_development_gates': bool(passing),
                  'dev_indices': dev.tolist(), 'heldout_indices': heldout.tolist(),
                  'selection_rule': 'dev quality within .01 of dense plus >=15% charged limb-product saving; maximize saving; otherwise best dev quality',
                  'selected_ranks': {str(i):c['rank'] for i,c in selected_changes.items()}}
        save(folder/'selection-frozen-before-heldout.json', frozen)
        if name == 'ad':
            held_quality, held_predictions = eval_ad(program, features, recording_index, labels, heldout, heldout_configs)
        else:
            held_quality, held_predictions = eval_class(program, features, labels, heldout, heldout_configs)
        selected_program, exported = export_int8(program, selected_changes, folder/'selected-int8.uq2')
        export_checks=[]
        for index in fit_indices[:2]:
            x = program.tensors[program.inputs[0]].quantization.encode(np.asarray(features[index]))
            expected = run(program, x, selected_changes)[0]
            actual = oracle(selected_program, {selected_program.inputs[0]: x})[selected_program.outputs[0]]
            if not np.array_equal(expected, actual):
                raise AssertionError('exported INT8 factor program differs from executable factor reference')
            export_checks.append(int(index))
        all_costs = {k:cost(program,rows,v) for k,v in heldout_configs.items()}
        save(folder/'predictions.json', {'dev':dev_predictions,'heldout':held_predictions})
        row = {'source_pins':pins, 'archive_sha256':archive_hash, 'manifest_sha256':digest(manifest_path),
               'experiment_driver_sha256':digest(Path(__file__)),
               'spectra':rows, 'dense_macs':total_macs(program), 'development_quality':dev_quality,
               'development_costs':costs, 'selection':frozen, 'heldout_quality':held_quality,
               'heldout_costs':all_costs, 'dense_oracle_all_layer_checks':checks,
               'wide_rejections':wide_rejections,
               'int8_export_oracle_checks':export_checks, 'export':exported,
               'fit': {'type':'activation-weighted reduced-rank regression with ridge1e-6; teacher activations',
                       'samples':len(fit_indices),'max_spatial_rows_per_sample':32,
                       'source':'development partition of existing accuracy archive; separate from heldout'},
               'heldout_ids_sha256':hashlib.sha256('\n'.join(str(ids[i]) for i in (heldout if name!='ad' else np.flatnonzero(np.isin(recording_index,heldout)))).encode()).hexdigest(),
               'elapsed_seconds':time.perf_counter()-start}
        report['models'][name] = row
        save(folder/'report.json',row)
        report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=='darwin' else 1024)
        save(BASE/'report.json',report)
        print('completed',name,selected,held_quality,flush=True)
        del features, ids, labels, cov, probes, configs, selected_program
    report['status']='completed-executable-software-screen'
    report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=='darwin' else 1024)
    report['limitations'] = ['Existing accuracy pools were split into development and heldout for an exploratory screen; no official benchmark certification.',
        'Class heldout samples are stratified subsets. AD heldout uses every window of disjoint recordings.',
        'Closed-form activation regression is not end-to-end fine-tuning, QAT, pruning retraining or distillation.',
        'Wide-latent variants share factor weights/ranks but change latent rounding; limb costs exclude correction/control overhead.',
        'Port byte savings from fusion also apply to ordinary blocked factor-pair fusion; unique architectural advantage not established.',
        'No RTL modification, synthesis, route, FPGA execution, energy measurement or guaranteed speedup.']
    save(BASE/'report.json',report)


if __name__ == '__main__':
    main()
