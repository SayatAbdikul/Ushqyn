from itertools import product

import numpy as np
import pytest

from integer_reference import evaluate
from quantization import Quantization, quantize_parameters
from static_pipeline import Layer, Program, Tensor
from scheduler.contract import IllegalSchedule
from scheduler.spatial import Rect, Reduction, execute_segment, input_halo, validate_reduction


def spatial_fixture(seed=5, groups=1, dilation=1, zp=-37):
    rng = np.random.default_rng(seed)
    tensors = {'input': Tensor('input', (1, 2, 9, 8), Quantization(.07, zp), 'NCHW')}
    layers = []
    previous = 'input'
    specs = [('Conv', (2, 3), (2, 1), (1, 2, 1, 0)), ('Relu', None, None, None),
             ('Conv', (3, 2), (1, 1), (1, 0, 1, 1)), ('Clip', None, None, None)]
    for i, (op, kernel, stride, pads) in enumerate(specs):
        inp = tensors[previous]; q = Quantization(.09+i*.02, 19-i*7)
        name = f'layer_{i}'; attrs = {}; params = {}
        if op == 'Conv':
            kh, kw = kernel; sh, sw = stride; pt, pl, pb, pr = pads
            shape = (1, 2, (inp.shape[2]+pt+pb-(kh-1)*dilation-1)//sh+1,
                     (inp.shape[3]+pl+pr-(kw-1)*dilation-1)//sw+1)
            params = quantize_parameters(rng.normal(0, .15, (2, 2//groups, kh, kw)),
                                         rng.normal(0, .1, 2), inp.quantization, q)
            attrs = {'group': groups, 'strides': list(stride), 'pads': list(pads), 'dilations': [dilation]*2}
        else:
            shape = inp.shape
            if op == 'Clip':
                params['clip_bounds'] = np.array([-33, 51], np.int8)
        tensors[name] = Tensor(name, shape, q, 'NCHW')
        layers.append(Layer(op, [previous], name, attrs, params)); previous = name
    p = Program(tensors, layers, ['input'], [previous], {}, {})
    x = rng.integers(-128, 128, tensors['input'].shape, dtype=np.int8)
    return p, x


@pytest.mark.parametrize('groups,dilation,zp,cache', product((1, 2), (1, 2), (-128, -37, 127), (False, True)))
def test_fused_regions_equal_independent_centered_integer_oracle(groups, dilation, zp, cache):
    p, x = spatial_fixture(groups=groups, dilation=dilation, zp=zp)
    expected = evaluate(p, {'input': x})[p.outputs[0]]
    actual, counts = execute_segment(p, 0, 4, x, tile=(2, 3), cache=cache, reduction_chunk=3)
    np.testing.assert_array_equal(actual, expected)
    assert counts['layers']['3']['computed_elements'] == expected.size


def test_cache_reduces_recomputation_without_altering_quantized_values():
    p, x = spatial_fixture()
    no, a = execute_segment(p, 0, 4, x, tile=(2, 2), cache=False)
    yes, b = execute_segment(p, 0, 4, x, tile=(2, 2), cache=True)
    np.testing.assert_array_equal(no, yes)
    assert sum(r['macs'] for r in b['layers'].values()) < sum(r['macs'] for r in a['layers'].values())
    assert b['peak_retained_cache_bytes'] > 0 and a['peak_retained_cache_bytes'] == 0


def test_negative_ties_and_quantization_barrier_are_preserved():
    tensors = {n: Tensor(n, (1, 1, 1, 3), Quantization(1, 0), 'NCHW') for n in ('x', 'a', 'b')}
    params = dict(weight=np.ones((1, 1, 1, 1), np.int8), bias=np.array([0], np.int32),
                  corrected_bias=np.array([0], np.int32), multiplier=np.array([1]), shift=np.array([1]))
    p = Program(tensors, [Layer('Conv', ['x'], 'a', {}, params),
                         Layer('Conv', ['a'], 'b', {}, params)], ['x'], ['b'], {}, {})
    y, _ = execute_segment(p, 0, 2, np.array([[[[-1, -3, 3]]]], np.int8), tile=(1, 2))
    assert y.tolist() == [[[[-1, -1, 1]]]]  # Combining the two /2 rounds would differ.


def test_halo_includes_stride_dilation_and_asymmetric_padding():
    rect = Rect(1, 2, 4, 6)
    halo = input_halo(rect, (3, 2), (2, 1), (1, 3, 0, 2), (2, 3))
    coords = {(y*2-1+ky*2, x-3+kx*3) for y in range(1, 4) for x in range(2, 6)
              for ky in range(3) for kx in range(2)}
    assert halo == Rect(min(y for y,x in coords), min(x for y,x in coords),
                        max(y for y,x in coords)+1, max(x for y,x in coords)+1)


def test_rectangle_difference_exactly_covers_missing_pixels():
    outer = Rect(0, 0, 4, 5)
    for y, x in product(range(-1, 6), repeat=2):
        cached = Rect(y, x, y+2, x+3)
        pieces = outer.subtract(cached)
        pixels = [(iy, ix) for r in pieces for iy in range(r.y0, r.y1) for ix in range(r.x0, r.x1)]
        expected = {(iy, ix) for iy in range(4) for ix in range(5)
                    if not (cached.y0 <= iy < cached.y1 and cached.x0 <= ix < cached.x1)}
        assert len(pixels) == len(set(pixels)) and set(pixels) == expected


@pytest.mark.parametrize('parts', [(Reduction(0, 2, True), Reduction(1, 4)),
                                  (Reduction(0, 2, True), Reduction(3, 4)),
                                  (Reduction(0, 2, True),),
                                  (Reduction(0, 2), Reduction(2, 4)),
                                  (Reduction(0, 2, True), Reduction(2, 4, True))])
def test_invalid_partial_sum_contract_rejected(parts):
    with pytest.raises(IllegalSchedule):
        validate_reduction(parts, 4)


def test_bias_corruption_and_all_input_overflow_proof_rejected():
    p, x = spatial_fixture()
    p.layers[0].parameters['corrected_bias'][0] += 1
    with pytest.raises(IllegalSchedule, match='bias correction'):
        execute_segment(p, 0, 2, x)
    p, x = spatial_fixture(zp=0)
    p.layers[0].parameters['bias'][0] = 2**31-1
    p.layers[0].parameters['corrected_bias'][0] = 2**31-1
    with pytest.raises(IllegalSchedule, match='INT32 partial sum bound'):
        execute_segment(p, 0, 2, x)
