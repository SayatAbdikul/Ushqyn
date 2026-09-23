#!/usr/bin/env python3
"""Build corrected trained MLP fixture with disjoint MNIST training calibration."""
import argparse,hashlib,json,struct,sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh,TensorProto as T
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'compiler'))
from static_pipeline import calibrate,compile_static
from program_image import build_image
from hardware_v2 import lower
from integer_reference import evaluate as independent_execute


def idx(path):
    data=path.read_bytes();dims=data[3];shape=struct.unpack('>'+str(dims)+'I',data[4:4+4*dims]);return np.frombuffer(data,offset=4+4*dims,dtype=np.uint8).reshape(shape)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'work/phase2/mlp');p.add_argument('--jobs',type=int,default=1000);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    weights_path=ROOT/'compiler/digit_model_weights.pth';weights=torch.load(weights_path,map_location='cpu',weights_only=True)
    nodes=[];initializers=[];src='input'
    for i in range(1,4):
        w=weights[f'fc{i}.weight'].numpy();b=weights[f'fc{i}.bias'].numpy()
        initializers += [nh.from_array(w,f'w{i}'),nh.from_array(b,f'b{i}')]
        dst=f'fc{i}';nodes.append(h.make_node('Gemm',[src,f'w{i}',f'b{i}'],[dst],transB=1));src=dst
        if i<3:nodes.append(h.make_node('Relu',[src],[f'relu{i}']));src=f'relu{i}'
    model=h.make_model(h.make_graph(nodes,'trained_mlp',[h.make_tensor_value_info('input',T.FLOAT,[1,784])],[h.make_tensor_value_info('fc3',T.FLOAT,[1,10])],initializers),opset_imports=[h.make_opsetid('',14)])
    onnx.save(model,a.output/'source.onnx')
    raw=ROOT/'compiler/data/MNIST/raw';train=idx(raw/'train-images-idx3-ubyte');test=idx(raw/'t10k-images-idx3-ubyte');labels=idx(raw/'t10k-labels-idx1-ubyte')
    def prep(x):return ((x.astype(np.float32)/255.-.1307)/.3081).reshape(1,784)
    cal=calibrate(model,[{'input':prep(x)} for x in train[:64]],[f'mnist/train/{i}' for i in range(64)])
    program=compile_static(model,cal);image,meta=lower(program)
    inputs=[];outputs=[];layers={n:[] for n in program.tensors if n!='input'}
    for i,x in enumerate(test[:a.jobs]):
        q=program.tensors['input'].quantization.encode(prep(x));values=independent_execute(program,{'input':q})
        inputs.append(q);outputs.append(values['fc3'])
        for name in layers:layers[name].append(values[name])
    np.savez_compressed(a.output/'checks.npz',inputs=np.array(inputs),outputs=np.array(outputs),labels=labels[:a.jobs],**{'layer_'+n:np.array(v) for n,v in layers.items()})
    meta.update(image_sha256=hashlib.sha256(image).hexdigest(),weights_sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),source_sha256=hashlib.sha256((a.output/'source.onnx').read_bytes()).hexdigest(),macs=sum(int(np.prod(l.parameters['weight'].shape)) for l in program.layers if l.op=='Gemm'),fixture_jobs=a.jobs,software_accuracy=float(np.mean(np.argmax(np.array(outputs)[:,0],axis=1)==labels[:a.jobs])),calibration_ids=cal['sample_ids'],dataset_sha256={n:hashlib.sha256((raw/n).read_bytes()).hexdigest() for n in ['train-images-idx3-ubyte','t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte']})
    (a.output/'board.bin').write_bytes(image);(a.output/'software.uq2').write_bytes(build_image(program));(a.output/'board.json').write_text(json.dumps(meta,indent=2)+'\n');(a.output/'calibration.json').write_text(json.dumps(cal,indent=2)+'\n');print(json.dumps({k:v for k,v in meta.items() if k in ['used_bytes','macs','software_accuracy','image_sha256']}))
