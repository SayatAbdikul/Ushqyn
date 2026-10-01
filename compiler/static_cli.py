"""Compile a static ONNX graph using explicitly identified calibration samples.

python compiler/static_cli.py model.onnx calibration.npz program.uq2
NPZ must contain sample_ids (Unicode) and one float32 [samples,*shape] array
per graph input. Calibration is saved beside the image for reproducibility.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import onnx
from static_pipeline import calibrate, compile_static
from program_image import write_image


def load_calibration_samples(model, path):
    """Load each NPZ member once and yield views into the shared input arrays."""
    with np.load(path, allow_pickle=False) as data:
        ids = data['sample_ids'].tolist()
        input_names = [n.name for n in model.graph.input
                       if n.name not in {t.name for t in model.graph.initializer}]
        inputs = {name: data[name] for name in input_names}
    if any(len(value) != len(ids) for value in inputs.values()):
        raise ValueError('calibration input count mismatch')
    samples = ({name: inputs[name][i] for name in input_names}
               for i in range(len(ids)))
    return ids, samples


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',type=Path)
    parser.add_argument('calibration_npz',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--target',default='software-v2',choices=['software-v2','tang-nano-20k'])
    parser.add_argument('--capacity',type=int,default=8*1024*1024)
    args=parser.parse_args()
    model=onnx.load(args.model)
    ids,samples=load_calibration_samples(model,args.calibration_npz)
    calibration=calibrate(model,samples,ids)
    program=compile_static(model,calibration,target=args.target)
    write_image(program,args.output,capacity=args.capacity,target=args.target)
    args.output.with_suffix('.calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    print(f'{args.output}: software VM image; current FPGA RTL does not execute v2')


if __name__=='__main__':main()
