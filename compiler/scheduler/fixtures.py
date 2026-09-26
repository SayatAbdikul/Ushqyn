"""Seeded geometry/quantization fixture for integration tests, not a benchmark."""
import numpy as np

from quantization import Quantization, quantize_parameters
from static_pipeline import Layer, Program, Tensor


def wide_chain():
    rng = np.random.default_rng(6102)
    shapes = {'x': (1, 4, 14, 14), 'conv': (1, 80, 14, 14), 'relu': (1, 80, 14, 14),
              'pool': (1, 80, 1, 1), 'flat': (1, 80), 'y': (1, 3)}
    q = {n: Quantization(.04+i*.01, -17+i*6) for i, n in enumerate(shapes)}
    q['flat'] = q['pool']
    tensors = {n: Tensor(n, s, q[n], 'NCHW' if len(s) == 4 else 'row-major') for n, s in shapes.items()}
    layers = [Layer('Conv', ['x'], 'conv', {}, quantize_parameters(rng.normal(0, .1, (80, 4, 1, 1)), np.zeros(80), q['x'], q['conv'])),
              Layer('Relu', ['conv'], 'relu', {}, {}),
              Layer('GlobalAveragePool', ['relu'], 'pool', {}, {}),
              Layer('Flatten', ['pool'], 'flat', {}, {}),
              Layer('Gemm', ['flat'], 'y', {}, quantize_parameters(rng.normal(0, .1, (3, 80)), np.zeros(3), q['flat'], q['y']))]
    return Program(tensors, layers, ['x'], ['y'], {}, {}), rng.integers(-128, 128, shapes['x'], dtype=np.int8)
