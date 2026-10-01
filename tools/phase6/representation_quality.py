#!/usr/bin/env python3
"""Frozen representation candidates on untouched classification examples.

CPU float64 convolution evaluates integer products exactly within a proved
2**53 absolute-sum bound; requantization uses integer ties-away arithmetic.
Every layer is checked against the independent integer oracle on real, random
and endpoint inputs before evaluation. This is software quality, not FPGA time.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
import zipfile
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'compiler'), str(ROOT/'tools/phase6')]
from program_image import load_image, build_image
from quantization import multiplier_shift
from integer_reference import evaluate
from run_boardless import load_model
from representation_screen import eligible, member, digest

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.mkldnn.enabled = False
BASE = ROOT/'work/phase6/representation-quality-v1'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def as_tensor(a, dtype=torch.int64):
    return torch.tensor(np.asarray(a).copy(), dtype=dtype)


def rq(acc, multiplier, shift, zero_point):
    if bool(torch.any((acc < -(1<<31)) | (acc > (1<<31)-1))):
        raise ArithmeticError('INT32 accumulator overflow')
    m, s = as_tensor(multiplier), as_tensor(shift)
    if acc.ndim == 4:
        m, s = m.reshape(1,-1,1,1), s.reshape(1,-1,1,1)
    else:
        m, s = m.reshape(1,-1), s.reshape(1,-1)
    product = acc*m
    rounding = torch.where(s > 0, torch.bitwise_left_shift(torch.ones_like(s), torch.clamp(s-1,min=0)), 0)
    magnitude = torch.bitwise_right_shift(torch.abs(product)+rounding,s)
    return torch.clamp(torch.where(product < 0,-magnitude,magnitude)+zero_point,-128,127).to(torch.int8)


class BatchVM:
    def __init__(self, program):
        self.program = program
        self.weights = {}
        self.bounds = {}
        for i,l in enumerate(program.layers):
            if l.op in ('Conv','Gemm'):
                w = l.parameters['weight'].astype(np.int64)
                b = l.parameters['corrected_bias'].astype(np.int64)
                bound = 128*np.abs(w).reshape(len(w),-1).sum(axis=1)+np.abs(b)
                if np.any(bound >= 1<<53):
                    raise ArithmeticError('float64 cannot exactly represent integer partial sums')
                self.bounds[str(i)] = int(bound.max())
                self.weights[i] = (as_tensor(w,torch.float64),as_tensor(b,torch.float64))

    def run(self, x, capture=False):
        p = self.program
        values = {p.inputs[0]:as_tensor(x,torch.int8)}
        values.update({n:as_tensor(a,torch.int8) for n,a in p.constants.items()})
        captured = {}
        remaining = {}
        for l in p.layers:
            for n in l.inputs:
                remaining[n] = remaining.get(n,0)+1
        batch = len(x)
        for i,l in enumerate(p.layers):
            v = values[l.inputs[0]]
            iq = p.tensors[l.inputs[0]].quantization
            oq = p.tensors[l.output].quantization
            a,par = l.attributes,l.parameters
            if l.op in ('Conv','Gemm'):
                w,b = self.weights[i]
                if l.op == 'Gemm':
                    acc = v.to(torch.float64) @ w.T+b
                else:
                    pt,pl,pb,pr = a.get('pads',[0,0,0,0])
                    padded = F.pad(v.to(torch.float64),(pl,pr,pt,pb),value=iq.zero_point)
                    acc = F.conv2d(padded,w,b,stride=a.get('strides',[1,1]),
                                   dilation=a.get('dilations',[1,1]),groups=a.get('group',1))
                if not bool(torch.all(acc == torch.round(acc))):
                    raise ArithmeticError('noninteger convolution result')
                y = rq(acc.to(torch.int64),par['multiplier'],par['shift'],oq.zero_point)
            elif l.op in ('Relu','Clip'):
                if l.op == 'Relu':
                    center = torch.clamp(v.to(torch.int64),min=iq.zero_point)-iq.zero_point
                else:
                    lo,hi = map(int,par['clip_bounds'])
                    center = torch.clamp(v.to(torch.int64),lo,hi)-iq.zero_point
                m,s = multiplier_shift(iq.scale/oq.scale)
                y = rq(center,[m],[s],oq.zero_point)
            elif l.op == 'Transpose':
                y = v.permute(a.get('perm',list(reversed(range(v.ndim))))).contiguous()
            elif l.op in ('Flatten','Reshape','Identity'):
                y = v.reshape((batch,*p.tensors[l.output].shape[1:])).clone()
            elif l.op in ('MaxPool','AveragePool','GlobalAveragePool'):
                kh,kw = a.get('kernel_shape',v.shape[2:])
                stride = a.get('strides',[1,1])
                pt,pl,pb,pr = a.get('pads',[0,0,0,0])
                if a.get('ceil_mode',0):
                    raise ValueError('ceil pool unsupported by accelerated evaluator')
                if l.op == 'MaxPool':
                    yy = F.max_pool2d(F.pad(v.to(torch.float64),(pl,pr,pt,pb),value=-128),
                                      (kh,kw),stride).to(torch.int64)-iq.zero_point
                    m,s = multiplier_shift(iq.scale/oq.scale)
                    y = rq(yy,[m],[s],oq.zero_point)
                else:
                    centered = F.pad(v.to(torch.float64)-iq.zero_point,(pl,pr,pt,pb),value=0)
                    sums = F.avg_pool2d(centered,(kh,kw),stride,divisor_override=1).to(torch.int64)
                    if a.get('count_include_pad',0):
                        counts = torch.full_like(sums[:1,:1],kh*kw)
                    else:
                        valid = F.pad(torch.ones((1,1,*v.shape[2:]),dtype=torch.float64),(pl,pr,pt,pb))
                        counts = F.avg_pool2d(valid,(kh,kw),stride,divisor_override=1).to(torch.int64)
                    y = torch.empty_like(sums,dtype=torch.int8)
                    for count in torch.unique(counts).tolist():
                        if count <= 0:
                            raise ValueError('empty pool')
                        m,s = multiplier_shift(iq.scale/(oq.scale*count))
                        candidate = rq(sums,[m],[s],oq.zero_point)
                        y = torch.where(counts == count,candidate,y)
            else:
                raise ValueError(f'unsupported accelerated op: {l.op}')
            if y.shape != (batch,*p.tensors[l.output].shape[1:]):
                raise ValueError(f'output geometry: {i}/{l.op}/{tuple(y.shape)}')
            values[l.output] = y
            if capture:
                captured[l.output] = y.numpy().copy()
            for n in l.inputs:
                remaining[n] -= 1
                if not remaining[n] and n not in p.outputs:
                    del values[n]
        return values[p.outputs[0]].numpy().copy(),captured


def verify_vm(vm, features, indices):
    p = vm.program
    q = p.tensors[p.inputs[0]].quantization
    shape = p.tensors[p.inputs[0]].shape
    real = [q.encode(np.asarray(features[i])) for i in indices[:2]]
    rng = np.random.default_rng(61002)
    cases = real+[rng.integers(-128,128,shape,dtype=np.int8),
                  np.full(shape,-128,np.int8),np.full(shape,127,np.int8)]
    checked = 0
    for x in cases:
        oracle = evaluate(p,{p.inputs[0]:x})
        _,actual = vm.run(x,capture=True)
        for l in p.layers:
            if not np.array_equal(oracle[l.output],actual[l.output]):
                raise AssertionError(f'accelerated oracle mismatch: {l.op}/{l.output}')
            checked += 1
    return {'cases':len(cases),'exact_layer_checks':checked,'integer_partial_sum_bounds':vm.bounds}


def paired_interval(labels, dense, candidate):
    rng = np.random.default_rng(61003)
    dist = np.zeros(4000,dtype=float)
    for label in np.unique(labels):
        ix = labels == label
        delta = (candidate[ix] == labels[ix]).astype(int)-(dense[ix] == labels[ix]).astype(int)
        counts = np.array([np.count_nonzero(delta == k) for k in (-1,0,1)])
        draws = rng.multinomial(len(delta),counts/len(delta),size=len(dist))
        dist += (draws[:,2]-draws[:,0])/len(labels)
    return {'method':'paired bootstrap stratified at original untouched label proportions; 4000 draws',
            'delta_accuracy':float(np.mean(candidate == labels)-np.mean(dense == labels)),
            'delta_95_percentile':np.quantile(dist,[.025,.975]).tolist()}


def verify_mapped(archive,name,path):
    h = hashlib.sha256()
    with zipfile.ZipFile(archive) as z, z.open(name+'.npy') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):
            h.update(chunk)
    actual = digest(path)
    if actual != h.hexdigest():
        raise ValueError('cached NPY does not match archive member')
    return actual


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models',nargs='+',choices=('kws','vww'),default=['kws','vww'])
    ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--limit',type=int,default=0,help='smoke limit; zero evaluates all untouched examples')
    args = ap.parse_args()
    if not 1 <= args.batch <= 32 or args.limit < 0:
        raise ValueError('invalid bounded settings')
    BASE.mkdir(parents=True,exist_ok=True)
    for name in args.models:
        folder = BASE/name
        folder.mkdir(exist_ok=True)
        original,_,_,pins = load_model(name)
        src = ROOT/'work/phase6/representation-screen-v1'/name
        before = json.loads((src/'report.json').read_text())
        factor_path = src/'selected-int8.uq2'
        if digest(factor_path) != before['export']['sha256']:
            raise ValueError('frozen factor changed')
        programs = {'dense':original,'factor':load_image(factor_path.read_bytes()),
                    'pruning_refit':load_image((src/'matched-column-pruning-refit.uq2').read_bytes())}
        input_name = original.inputs[0]
        archive = ROOT/f'work/quality/{name}.accuracy.npz'
        manifest = json.loads((ROOT/f'benchmarks/manifests/{name}.data.json').read_text())
        if digest(archive) != manifest['splits']['accuracy']['npz_sha256']:
            raise ValueError('archive provenance changed')
        features = member(archive,input_name,src/'mapped')
        labels = member(archive,'labels',src/'mapped')
        mapping = {key:verify_mapped(archive,key,src/'mapped'/(key+'.npy')) for key in (input_name,'labels')}
        seen = set(before['selection']['dev_indices']) | set(before['selection']['heldout_indices'])
        untouched = np.array([i for i in range(len(features)) if i not in seen])
        if args.limit:
            untouched = untouched[:args.limit]
        vms = {key:BatchVM(p) for key,p in programs.items()}
        checks = {key:verify_vm(vm,features,before['selection']['dev_indices']) for key,vm in vms.items()}
        record = {'status':'running','model':name,'scope':'frozen programs; examples excluded from earlier dev and heldout screens',
                  'source_sha256':digest(Path(__file__)),'source_pins':pins,'archive_sha256':digest(archive),
                  'cached_member_sha256':mapping,'factor_sha256':digest(factor_path),
                  'pruning_refit_sha256':digest(src/'matched-column-pruning-refit.uq2'),
                  'excluded_seen_indices':sorted(seen),'evaluated_indices':untouched.tolist(),
                  'quality_gate_absolute_loss':.01,'batch':args.batch,'threads':1,'smoke':bool(args.limit),
                  'oracle_checks':checks,'predictions':{k:[] for k in programs},'labels':[]}
        began = time.monotonic()
        for offset in range(0,len(untouched),args.batch):
            ix = untouched[offset:offset+args.batch]
            raw = np.asarray(features[ix]).reshape((len(ix),*original.tensors[input_name].shape[1:]))
            x = original.tensors[input_name].quantization.encode(raw)
            for key,vm in vms.items():
                y,_ = vm.run(x)
                record['predictions'][key].extend(np.argmax(y.reshape(len(ix),-1),axis=1).tolist())
            record['labels'].extend(np.asarray(labels[ix]).tolist())
            if offset % (args.batch*32) == 0:
                print(name,offset+len(ix),'/',len(untouched),f'{time.monotonic()-began:.1f}s',flush=True)
                save(folder/'report.partial.json',record)
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak = peak if sys.platform == 'darwin' else peak*1024
            if peak > 1<<30:
                raise MemoryError('observed RSS exceeds 1GiB guard')
        yy = np.array(record['labels'])
        preds = {k:np.array(v) for k,v in record['predictions'].items()}
        record['quality'] = {k:{'count':len(yy),'correct':int(np.count_nonzero(v == yy)),
                                'accuracy':float(np.mean(v == yy)),
                                'label_counts':{str(c):int(np.count_nonzero(yy == c)) for c in np.unique(yy)}}
                             for k,v in preds.items()}
        record['intervals'] = {k:paired_interval(yy,preds['dense'],v) for k,v in preds.items() if k != 'dense'}
        record['elapsed_seconds'] = time.monotonic()-began
        record['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
        record['status'] = 'passed-software-quality-evaluation'
        save(folder/('smoke.json' if args.limit else 'report.json'),record)
        print(name,record['quality'],record['intervals'],flush=True)


if __name__ == '__main__':
    main()
