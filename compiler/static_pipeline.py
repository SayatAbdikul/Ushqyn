"""Calibrated, typed graph compiler and integer software execution (numerics v2).

The software VM container is never flashed directly. Hardware lowering of its
supported graph subset is implemented separately in ``hardware_v2.py``.
"""
from dataclasses import dataclass, asdict
import hashlib
import numpy as np
import onnx
from onnx import numpy_helper, helper
from onnx.reference import ReferenceEvaluator
from canonicalize import canonicalize, shapes, attrs
from quantization import Quantization, from_range, quantize_parameters, multiplier_shift, requantize


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple
    quantization: Quantization
    layout: str


@dataclass
class Layer:
    op: str
    inputs: list
    output: str
    attributes: dict
    parameters: dict


@dataclass
class Program:
    tensors: dict
    layers: list
    inputs: list
    outputs: list
    constants: dict
    provenance: dict


def digest(model):
    return hashlib.sha256(model.SerializeToString()).hexdigest()


def calibrate(model, samples, sample_ids):
    """Min/max calibration of canonical float intermediates; no test data here."""
    model = canonicalize(model)
    sm = shapes(model)
    input_names = [i.name for i in model.graph.input
                   if i.name not in {t.name for t in model.graph.initializer}]
    names = list(dict.fromkeys(input_names + [n.output[0] for n in model.graph.node]))
    exposed = onnx.ModelProto()
    exposed.CopyFrom(model)
    del exposed.graph.output[:]
    exposed.graph.output.extend(helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, sm[n]) for n in names)
    evaluator = ReferenceEvaluator(exposed)
    ranges = {n: [float('inf'), float('-inf')] for n in names}
    ids = list(sample_ids)
    if not ids or len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("nonempty, unique calibration IDs required")
    hashes = []
    for index, sample in enumerate(samples):
        if set(sample) != set(input_names):
            raise ValueError("calibration input names mismatch")
        feed = {}
        h = hashlib.sha256()
        for name in input_names:
            arr = np.asarray(sample[name], dtype=np.float32)
            if arr.shape != sm[name] or not np.isfinite(arr).all():
                raise ValueError("invalid calibration shape/values")
            feed[name] = arr
            h.update(name.encode()); h.update(arr.astype('<f4').tobytes())
        hashes.append(h.hexdigest())
        for name, arr in zip(names, evaluator.run(None, feed)):
            if not np.isfinite(arr).all():
                raise ValueError(f"nonfinite float intermediate: {name}")
            ranges[name][0] = min(ranges[name][0], float(np.min(arr)))
            ranges[name][1] = max(ranges[name][1], float(np.max(arr)))
    if len(hashes) != len(ids):
        raise ValueError("calibration samples and IDs differ in length")
    return {'schema': 2, 'model_sha256': digest(model), 'method': 'minmax-including-zero',
            'sample_ids': ids, 'sample_sha256': hashes, 'ranges': ranges}


def compile_static(model, calibration, target='software-v2'):
    if target != 'software-v2':
        raise ValueError("compile_static emits software-v2; lower supported FPGA graphs separately")
    model = canonicalize(model)
    if calibration.get('schema') != 2 or calibration.get('model_sha256') != digest(model):
        raise ValueError("calibration version/model hash mismatch")
    ids, hashes = calibration.get('sample_ids', []), calibration.get('sample_sha256', [])
    if not ids or len(set(ids)) != len(ids) or len(hashes) != len(ids):
        raise ValueError("invalid calibration provenance")
    sm = shapes(model)
    raw = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    inputs = [x.name for x in model.graph.input if x.name not in raw]
    needed = inputs + [n.output[0] for n in model.graph.node]
    tensors = {}
    for name in needed:
        if name not in calibration['ranges']:
            raise ValueError(f"missing calibrated range: {name}")
        tensors[name] = Tensor(name, sm[name], from_range(*calibration['ranges'][name]),
                               'ONNX-axis-order' if len(sm[name]) == 4 else 'row-major')
    layers, constants = [], {}
    for node in model.graph.node:
        a = attrs(node)
        a = {k: v.decode() if isinstance(v, bytes) else v for k, v in a.items()}
        data_inputs = [node.input[0]]
        params = {}
        out = tensors[node.output[0]]
        if node.op_type in ('Conv','MaxPool','AveragePool','GlobalAveragePool'):
            t = tensors[node.input[0]]
            tensors[t.name] = Tensor(t.name,t.shape,t.quantization,'NCHW')
            tensors[out.name] = out = Tensor(out.name,out.shape,out.quantization,'NCHW')
        elif node.op_type == 'Transpose' and a.get('perm') in ([0,3,1,2],[0,2,3,1]):
            source_layout,dest_layout = ('NHWC','NCHW') if a['perm']==[0,3,1,2] else ('NCHW','NHWC')
            t = tensors[node.input[0]]
            tensors[t.name] = Tensor(t.name,t.shape,t.quantization,source_layout)
            tensors[out.name] = out = Tensor(out.name,out.shape,out.quantization,dest_layout)
        for name in node.input:
            if node.op_type == 'Add' and name in raw:
                q = from_range(float(raw[name].min()), float(raw[name].max()))
                tensors[name] = Tensor(name, sm[name], q, 'row-major')
                constants[name] = q.encode(raw[name])
        iq = tensors[node.input[0]].quantization
        if node.op_type in ('Gemm','Conv'):
            params = quantize_parameters(raw[node.input[1]], raw[node.input[2]], iq, out.quantization)
        elif node.op_type == 'Add':
            data_inputs = list(node.input)
            for name in data_inputs:
                if name in raw:
                    q = from_range(float(raw[name].min()), float(raw[name].max()))
                    tensors[name] = Tensor(name, sm[name], q, 'row-major')
                    constants[name] = q.encode(raw[name])
            # Add has one common fixed-point denominator and one final rounding.
            ratios = [tensors[x].quantization.scale / out.quantization.scale for x in data_inputs]
            pairs = [multiplier_shift(r) for r in ratios]
            shift = min(s for _, s in pairs)
            multipliers = [int(round(r * (1 << shift))) for r in ratios]
            if any(m <= 0 or m > (1 << 31) - 1 for m in multipliers):
                raise ValueError("Add ratio cannot be represented")
            params = {'add_multiplier': np.array(multipliers, dtype='<i4'), 'add_shift': np.array([shift], dtype=np.int8)}
        elif node.op_type in ('Flatten','Reshape','Identity','Transpose'):
            tensors[out.name] = Tensor(out.name, out.shape, iq, out.layout)
        elif node.op_type == 'Clip':
            bounds = [int(iq.encode(raw[name]).item()) for name in node.input[1:]]
            params = {'clip_bounds': np.asarray(bounds,dtype=np.int8)}
        layers.append(Layer(node.op_type, data_inputs, node.output[0], a, params))
    return Program(tensors, layers, inputs, [o.name for o in model.graph.output], constants,
                   {'target': target, 'numerics': 2, 'calibration': calibration})


def _rescale(value, source, dest):
    m, s = multiplier_shift(source.scale / dest.scale)
    return requantize(np.asarray(value, dtype=np.int64) - source.zero_point, m, s, dest.zero_point)


def execute_layer(program, layer, values):
    x = values[layer.inputs[0]]
    iq = program.tensors[layer.inputs[0]].quantization
    out = program.tensors[layer.output]
    oq, a, p = out.quantization, layer.attributes, layer.parameters
    if layer.op in ('Gemm','Conv'):
        w = p['weight'].astype(np.int64)
        if layer.op == 'Gemm':
            acc = x.astype(np.int64) @ w.T + p['corrected_bias']
            result = np.empty(out.shape, dtype=np.int8)
            for c in range(w.shape[0]):
                result[:, c] = requantize(acc[:, c], p['multiplier'][c], p['shift'][c], oq.zero_point)
            return result
        sh, sw = a.get('strides', [1,1]); dh, dw = a.get('dilations', [1,1])
        pt, pl, pb, pr = a.get('pads', [0,0,0,0])
        padded = np.pad(x.astype(np.int64), ((0,0),(0,0),(pt,pb),(pl,pr)), constant_values=iq.zero_point)
        _, oc, oh, ow = out.shape
        _, icg, kh, kw = w.shape
        groups = a.get('group', 1)
        result = np.empty(out.shape, dtype=np.int8)
        for c in range(oc):
            start = (c // (oc // groups)) * icg
            acc = np.full((oh, ow), int(p['corrected_bias'][c]), dtype=np.int64)
            for ky in range(kh):
                for kx in range(kw):
                    patch = padded[0, start:start+icg, ky*dh:ky*dh+oh*sh:sh, kx*dw:kx*dw+ow*sw:sw]
                    acc += np.einsum('ihw,i->hw', patch, w[c,:,ky,kx])
            result[0,c] = requantize(acc, p['multiplier'][c], p['shift'][c], oq.zero_point)
        return result
    if layer.op == 'Relu':
        return _rescale(np.maximum(x, iq.zero_point), iq, oq)
    if layer.op == 'Clip':
        return _rescale(np.clip(x,*p['clip_bounds']),iq,oq)
    if layer.op in ('Flatten','Reshape','Identity'):
        return x.reshape(out.shape).copy()
    if layer.op == 'Transpose':
        return x.transpose(a.get('perm', list(reversed(range(x.ndim))))).copy()
    if layer.op == 'Add':
        total = 0
        for name, m in zip(layer.inputs, p['add_multiplier']):
            total = total + (values[name].astype(np.int64) - program.tensors[name].quantization.zero_point) * int(m)
        s = int(p['add_shift'][0])
        mag = (np.abs(total) + (1 << (s-1))) >> s if s else np.abs(total)
        return np.clip(np.where(total < 0, -mag, mag) + oq.zero_point, -128, 127).astype(np.int8)
    if layer.op in ('MaxPool','AveragePool','GlobalAveragePool'):
        _, channels, oh, ow = out.shape
        kh, kw = a.get('kernel_shape', x.shape[2:])
        sh, sw = a.get('strides', [1,1])
        pt, pl, pb, pr = a.get('pads', [0,0,0,0])
        result = np.empty(out.shape, dtype=np.int8)
        for y in range(oh):
            for z in range(ow):
                y0, x0 = y*sh-pt, z*sw-pl
                patch = x[:,:,max(y0,0):min(y0+kh,x.shape[2]),max(x0,0):min(x0+kw,x.shape[3])]
                if not patch.size:
                    raise ValueError("pool window contains no input")
                if layer.op == 'MaxPool':
                    result[:,:,y,z] = _rescale(patch.max(axis=(2,3)), iq, oq)
                else:
                    count = kh*kw if a.get('count_include_pad',0) else patch.shape[2]*patch.shape[3]
                    m,s = multiplier_shift(iq.scale / (oq.scale * count))
                    centered = (patch.astype(np.int64)-iq.zero_point).sum(axis=(2,3))
                    result[:,:,y,z] = requantize(centered,m,s,oq.zero_point)
        return result
    raise ValueError(f"unsupported VM operation: {layer.op}")


def run(program, inputs, *, quantized=False):
    if set(inputs) != set(program.inputs):
        raise ValueError("input names mismatch")
    values = {name: arr.copy() for name, arr in program.constants.items()}
    for name, value in inputs.items():
        t = program.tensors[name]
        arr = np.asarray(value)
        if arr.shape != tuple(t.shape) or (quantized and arr.dtype != np.int8):
            raise ValueError("input shape/type mismatch")
        values[name] = arr.copy() if quantized else t.quantization.encode(arr)
    for layer in program.layers:
        values[layer.output] = execute_layer(program, layer, values)
    return {name: values[name] for name in program.outputs}
