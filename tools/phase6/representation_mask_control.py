#!/usr/bin/env python3
"""Reconstruct the frozen dev-selected column-mask control; no new selection.

The zero-column software executable validates quality. Real arithmetic savings
would additionally require gather/propagated compaction, absent from native RTL.
"""
import copy
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from representation_quality import (ROOT, BatchVM, digest, load_model, member,
                                    save, verify_vm, paired_interval)
from program_image import build_image

BASE = ROOT/'work/phase6/representation-quality-v1'


def main():
    for name in ('kws','vww'):
        src = ROOT/'work/phase6/representation-screen-v1'/name
        report = json.loads((src/'report.json').read_text())
        p,_,_,pins = load_model(name)
        features = member(ROOT/f'work/quality/{name}.accuracy.npz',p.inputs[0],src/'mapped')
        labels = member(ROOT/f'work/quality/{name}.accuracy.npz','labels',src/'mapped')
        chosen = {int(k):v for k,v in report['selection']['selected_ranks'].items()}
        vm = BatchVM(p)
        covariance = {i:np.zeros((p.layers[i].parameters['weight'].shape[1],)*2) for i in chosen}
        for index in report['selection']['dev_indices']:
            x = p.tensors[p.inputs[0]].quantization.encode(np.asarray(features[index]))
            _,values = vm.run(x,capture=True)
            values[p.inputs[0]] = x
            for i in chosen:
                l = p.layers[i]
                v = values[l.inputs[0]].astype(np.int64)-p.tensors[l.inputs[0]].quantization.zero_point
                matrix = v.reshape(-1,v.shape[-1]) if l.op == 'Gemm' else v.transpose(0,2,3,1).reshape(-1,v.shape[1])
                xx = matrix[np.linspace(0,len(matrix)-1,min(32,len(matrix)),dtype=int)].astype(float)
                covariance[i] += xx.T@xx
        masked = copy.deepcopy(p)
        kept = {}
        savings = 0
        for i,rank in chosen.items():
            l = masked.layers[i]
            w = l.parameters['weight'].reshape(l.parameters['weight'].shape[0],-1).copy()
            co,ci = w.shape
            keep = min(ci,max(1,rank*(ci+co)//co))
            rw = w.astype(float)*l.parameters['weight_scales'][:,None]
            importance = np.sum(rw*rw,axis=0)*np.diag(covariance[i])
            retained = np.argsort(importance)[-keep:]
            w[:,np.setdiff1d(np.arange(ci),retained)] = 0
            l.parameters['weight'] = w.reshape(l.parameters['weight'].shape)
            corrected = l.parameters['bias'].astype(np.int64)-masked.tensors[l.inputs[0]].quantization.zero_point*w.astype(np.int64).sum(axis=1)
            if np.any(corrected < -(1<<31)) or np.any(corrected > (1<<31)-1):
                raise ArithmeticError('mask corrected bias outside INT32')
            l.parameters['corrected_bias'] = corrected.astype('<i4')
            positions = np.prod(masked.tensors[l.output].shape)//co
            savings += int(positions*co*(ci-keep))
            kept[str(i)] = retained.tolist()
        folder = BASE/name
        folder.mkdir(parents=True,exist_ok=True)
        image = folder/'frozen-column-mask.uq2'
        image.write_bytes(build_image(masked))
        mask_vm = BatchVM(masked)
        checks = verify_vm(mask_vm,features,report['selection']['dev_indices'])
        prior = json.loads((src/'predictions.json').read_text())['heldout']['matched-column-pruning']
        old_ix = report['selection']['heldout_indices']
        got = []
        for offset in range(0,len(old_ix),8):
            ix = np.asarray(old_ix[offset:offset+8])
            raw = np.asarray(features[ix]).reshape((len(ix),*p.tensors[p.inputs[0]].shape[1:]))
            y,_ = mask_vm.run(p.tensors[p.inputs[0]].quantization.encode(raw))
            got.extend(np.argmax(y.reshape(len(ix),-1),axis=1).tolist())
        if got != prior:
            raise AssertionError('mask reconstruction differs from original frozen control predictions')
        seen = set(report['selection']['dev_indices']) | set(old_ix)
        indices = np.array([i for i in range(len(features)) if i not in seen])
        # Close the broad mmap after the scattered development/old-subset
        # reads. Reopening it during sequential evaluation bounds mapped
        # resident pages as well as heap allocations.
        features._mmap.close()
        features = np.load(src/'mapped'/(p.inputs[0]+'.npy'),mmap_mode='r',allow_pickle=False)
        result = {'status':'running','scope':__doc__,'source_sha256':digest(Path(__file__)),
                  'batch_vm_source_sha256':digest(ROOT/'tools/phase6/representation_quality.py'),
                  'source_pins':pins,'mask_image_sha256':digest(image),'retained_columns':kept,
                  'prior_subset_exact_predictions':len(got),'oracle_checks':checks,
                  'evaluated_indices':indices.tolist(),'hypothetical_compacted_macs_saved':savings,
                  'predictions':[],'labels':[]}
        start = time.monotonic()
        for offset in range(0,len(indices),8):
            ix = indices[offset:offset+8]
            raw = np.asarray(features[ix]).reshape((len(ix),*p.tensors[p.inputs[0]].shape[1:]))
            y,_ = mask_vm.run(p.tensors[p.inputs[0]].quantization.encode(raw))
            result['predictions'].extend(np.argmax(y.reshape(len(ix),-1),axis=1).tolist())
            result['labels'].extend(np.asarray(labels[ix]).tolist())
            if (offset//8+1) % 8 == 0:
                features._mmap.close()
                features = np.load(src/'mapped'/(p.inputs[0]+'.npy'),mmap_mode='r',allow_pickle=False)
            if offset % 512 == 0:
                print(name,'mask',offset+len(ix),'/',len(indices),flush=True)
                save(folder/'mask-report.partial.json',result)
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
            if peak > 1<<30:
                raise MemoryError('observed RSS exceeds1GiB guard')
        quality = json.loads((folder/'report.json').read_text())
        if quality['evaluated_indices'] != result['evaluated_indices'] or quality['labels'] != result['labels']:
            raise AssertionError('mask/factor quality populations differ')
        yy = np.array(result['labels']);pred = np.array(result['predictions'])
        result.update(status='passed-frozen-mask-quality',quality={'count':len(yy),
            'correct':int(np.count_nonzero(pred==yy)),'accuracy':float(np.mean(pred==yy))},
            dense_comparison=paired_interval(yy,np.array(quality['predictions']['dense']),pred),
            factor_comparison=paired_interval(yy,pred,np.array(quality['predictions']['factor'])),
            elapsed_seconds=time.monotonic()-start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
        save(folder/'mask-report.json',result)
        print(name,'mask',result['quality'],result['factor_comparison'],flush=True)


if __name__ == '__main__':
    main()
