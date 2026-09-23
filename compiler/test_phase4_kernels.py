"""Pinned audio/vision kernel shapes at the software-to-board boundary."""
import dataclasses
import numpy as np
import pytest
from onnx import helper as h

from hardware_v2 import Descriptor, lower
from integer_reference import evaluate
from static_pipeline import run
from test_static_pipeline import compile_case, model


def test_depthwise_pointwise_average_and_clip_lowering():
    rng = np.random.default_rng(20260923)
    x = rng.normal(size=(1, 4, 6, 8)).astype(np.float32)
    nodes = [
        h.make_node('Conv', ['x', 'dw', 'db'], ['d'], group=4,
                    kernel_shape=[3, 3], strides=[2, 2], pads=[0, 0, 1, 1]),
        h.make_node('Conv', ['d', 'pw', 'pb'], ['p'], kernel_shape=[1, 1]),
        h.make_node('AveragePool', ['p'], ['a'], kernel_shape=[3, 4], strides=[3, 4]),
        h.make_node('Clip', ['a', 'min', 'max'], ['y']),
    ]
    weights = dict(dw=rng.normal(size=(4, 1, 3, 3)), db=rng.normal(size=4),
                   pw=rng.normal(size=(3, 4, 1, 1)), pb=rng.normal(size=3),
                   min=np.array(-0.5), max=np.array(0.8))
    p = compile_case(model(nodes, weights, x.shape, {'y': [1, 3, 1, 1]}), x)
    blob, metadata = lower(p)
    assert [Descriptor.decode(blob[i*64:(i+1)*64]).opcode for i in range(4)] == [6, 4, 7, 8]
    assert metadata['memory']['predicted_bsram_blocks'] == 16
    qx = p.tensors['x'].quantization.encode(x)
    np.testing.assert_array_equal(run(p, {'x': qx}, quantized=True)['y'], evaluate(p, {'x': qx})['y'])
    d = Descriptor.decode(blob[:64])
    assert d.count == 9 and d.input_c == d.output_c == 4
    with pytest.raises(ValueError, match='depthwise'):
        dataclasses.replace(d, output_c=3, outputs=3*d.outputs//4).validate()


def test_kws_rectangular_and_global_average_geometry():
    conv = Descriptor(4, input=512, output=4096, weight=8192, params=16384,
                      count=40, outputs=4*6*2, row_stride=40, next_pc=64,
                      kernel_h=10, kernel_w=4, stride_h=2, stride_w=2,
                      pad_top=4, pad_bottom=5, pad_left=1, pad_right=1,
                      input_h=11, input_w=5, input_c=1, output_c=4)
    conv.validate()
    avg = Descriptor(7, input=512, output=4096, params=16384,
                     count=2*25*5, outputs=2, next_pc=64,
                     kernel_h=25, kernel_w=5, stride_h=25, stride_w=5,
                     input_h=25, input_w=5, input_c=2, output_c=2)
    avg.validate()
    wide = dataclasses.replace(avg, opcode=6, count=1,
                               input_h=1,input_w=1,input_c=256,output_c=256,
                               kernel_h=1,kernel_w=1,stride_h=1,stride_w=1,
                               outputs=256,weight=8192,row_stride=8)
    wide.validate()
    with pytest.raises(ValueError, match='average pool'):
        dataclasses.replace(avg, pad_top=1, outputs=2).validate()
