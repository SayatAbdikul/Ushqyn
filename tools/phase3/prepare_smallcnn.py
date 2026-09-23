#!/usr/bin/env python3
"""Pin trained SmallCNN, disjoint calibration and independent integer cases."""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import onnx
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'compiler'))
from smallcnn import SmallCNN
from static_pipeline import calibrate, compile_static
from hardware_v2 import lower
from integer_reference import evaluate
from program_image import build_image


def idx(path):
    data = path.read_bytes()
    dims = data[3]
    shape = struct.unpack('>' + str(dims) + 'I', data[4:4+4*dims])
    return np.frombuffer(data, offset=4+4*dims, dtype=np.uint8).reshape(shape)


def float_input(image):
    return (image.astype(np.float32)-128).reshape(1,1,28,28)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT/'work/phase3/smallcnn')
    p.add_argument('--jobs', type=int, default=1000)
    p.add_argument('--quality-only', action='store_true', help='save full-set predictions without large intermediate tensors')
    a = p.parse_args()
    if not 1 <= a.jobs <= 10000: raise ValueError('job count')
    a.output.mkdir(parents=True, exist_ok=True)
    weights_path = ROOT/'compiler/small_cnn_weights.pth'
    network = SmallCNN().eval()
    network.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    with torch.no_grad():
        torch.onnx.export(network, torch.zeros(1,1,28,28), a.output/'source.onnx',
                          input_names=['input'], output_names=['output'], opset_version=14,
                          dynamo=False)
    model = onnx.load(a.output/'source.onnx')
    raw = ROOT/'compiler/data/MNIST/raw'
    train = idx(raw/'train-images-idx3-ubyte')
    test = idx(raw/'t10k-images-idx3-ubyte')
    labels = idx(raw/'t10k-labels-idx1-ubyte')
    calibration_hashes={hashlib.sha256(x.tobytes()).digest() for x in train[:64]}
    if any(hashlib.sha256(x.tobytes()).digest() in calibration_hashes for x in test):
        raise ValueError('calibration and full test images overlap')
    calibration = calibrate(model, [{'input':float_input(x)} for x in train[:64]],
                            [f'mnist/train/{i}' for i in range(64)])
    program = compile_static(model, calibration)
    image, meta = lower(program)
    inputs, outputs = [], []
    intermediates = {name:[] for name in program.tensors if name!='input'}
    float_correct = 0
    with torch.no_grad():
        for i, raw_image in enumerate(test[:a.jobs]):
            x = float_input(raw_image)
            q = program.tensors['input'].quantization.encode(x)
            result = evaluate(program, {'input':q})
            if not a.quality_only:inputs.append(q)
            outputs.append(result[program.outputs[0]])
            if not a.quality_only:
                for name in intermediates:intermediates[name].append(result[name])
            float_correct += int(network(torch.from_numpy(x)).argmax().item()==labels[i])
    arr_out=np.array(outputs)
    if a.quality_only:
        np.savez_compressed(a.output/'quality-predictions.npz',outputs=arr_out,labels=labels[:a.jobs])
    else:
        np.savez_compressed(a.output/'checks.npz', inputs=np.array(inputs),outputs=arr_out,
                            labels=labels[:a.jobs],**{'layer_'+n:np.array(v) for n,v in intermediates.items()})
    meta.update(image_sha256=hashlib.sha256(image).hexdigest(),
                weights_sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256((a.output/'source.onnx').read_bytes()).hexdigest(),
                calibration_ids=calibration['sample_ids'],
                dataset_sha256={n:hashlib.sha256((raw/n).read_bytes()).hexdigest()
                                for n in ['train-images-idx3-ubyte','t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte']},
                jobs=a.jobs, float_correct=float_correct,
                integer_correct=int(np.sum(np.argmax(arr_out.reshape(a.jobs,-1),axis=1)==labels[:a.jobs])),
                macs=sum(int(np.prod(l.parameters['weight'].shape))*
                         (int(np.prod(program.tensors[l.output].shape[2:])) if l.op=='Conv' else 1)
                         for l in program.layers if l.op in ('Conv','Gemm')))
    (a.output/'board.bin').write_bytes(image)
    (a.output/'software.uq2').write_bytes(build_image(program))
    (a.output/'board.json').write_text(json.dumps(meta,indent=2)+'\n')
    (a.output/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    print(json.dumps({k:meta[k] for k in ('used_bytes','macs','jobs','float_correct','integer_correct')},indent=2))
    print('operations:',[layer.op for layer in program.layers])


if __name__=='__main__':main()
