#!/usr/bin/env python3
"""Validate static engine predictions and a geometry-only burst DMA fit."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'compiler'))
from hardware_v2 import Descriptor
from phase4_cost import engine_cycles


def metrics(records):
    errors=np.array([abs(r['predicted_cycles']-r['cycles'])/r['cycles']*100 for r in records])
    return {'samples':len(records),'median_absolute_percent_error':float(np.median(errors)),
            'p95_absolute_percent_error':float(np.percentile(errors,95)),
            'worst_absolute_percent_error':float(max(errors))}


def dma_features(record):
    n=record['length_bytes'];a=record['external_offset']
    return [1,(n+7)//8,(a%64+n+63)//64,int((a+n-1)%64>=56),int(a%64!=0)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernels',type=Path,required=True)
    parser.add_argument('--dma',type=Path,required=True)
    parser.add_argument('--kws',type=Path,required=True)
    parser.add_argument('--vww',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    inputs={k:json.loads(getattr(args,k).read_text()) for k in ('kernels','dma','kws','vww')}
    hashes={r['bitstream_sha256'] for r in inputs.values()}
    if len(hashes)!=1 or any(r['status']!='passed' or not r['physical_board'] for r in inputs.values()):
        raise ValueError('requires successful evidence from one physical bitstream')
    kernel=inputs['kernels']
    if not kernel['prospective_holdout']:
        raise ValueError('expected prospective holdout geometry')
    predictions_path=args.kernels.with_suffix('.predictions.json')
    if hashlib.sha256(predictions_path.read_bytes()).hexdigest()!=kernel['predictions_sha256']:
        raise ValueError('prediction manifest changed after execution')
    rows=[]
    for case in kernel['cases']:
        predicted=engine_cycles(Descriptor(**case['descriptor']))
        if predicted!=case['predicted_cycles']:
            raise ValueError('predictor changed after the prospective test')
        rows.append({'label':case['label'],'cycles':float(np.median([r['elapsed'] for r in case['runs']])),
                     'predicted_cycles':predicted})
    model_rows=[]
    for name in ('kws','vww'):
        mode=inputs[name]['modes']['sequential']
        prediction=sum(engine_cycles(Descriptor.decode(bytes.fromhex(t['descriptor_hex'])))
                       for t in mode['schedule']['tiles'])
        model_rows.append({'model':name,'cycles':float(np.median([r['engine_busy_cycles'] for r in mode['runs']])),
                           'predicted_cycles':prediction})
    held_lengths={7,9,65,256,4096}
    dma={}
    for direction in ('to_sram','from_sram'):
        source=[r for r in inputs['dma']['records'] if r['direction']==direction]
        fit=[r for r in source if r['length_bytes'] not in held_lengths]
        hold=[r for r in source if r['length_bytes'] in held_lengths]
        coefficients=np.linalg.lstsq(np.array([dma_features(r) for r in fit]),
                                     np.array([r['dma_cycles'] for r in fit]),rcond=None)[0]
        predicted=[dict(r,cycles=r['dma_cycles'],predicted_cycles=max(1.,float(np.dot(coefficients,dma_features(r)))))
                   for r in hold]
        dma[direction]={'coefficients':coefficients.tolist(),'validation_error':metrics(predicted),
                        'validation_configurations':len({(r['length_bytes'],r['external_offset']) for r in hold}),
                        'validation_predictions':predicted}
    engine_error=metrics(rows);model_error=metrics(model_rows)
    checks=[engine_error,model_error]+[r['validation_error'] for r in dma.values()]
    passed=all(r['median_absolute_percent_error']<=10 and r['p95_absolute_percent_error']<=20 for r in checks)
    result={'status':'passed' if passed else 'failed','bitstream_sha256':next(iter(hashes)),
            'engine':{'method':'static descriptor/cache address walk; zero fitted coefficients',
                      'prospective_holdout_error':engine_error,'predictions':rows,
                      'full_model_uncontended_error':model_error,'full_model_predictions':model_rows},
            'dma':dma,'dma_feature_names':['intercept','8_byte_beats','64_byte_lines',
                                           'ends_in_last_beat','unaligned_line_start'],
            'split_policy':'DMA hold out lengths 7,9,65,256,4096 before fitting; engine prospective fixed shape seed 250926',
            'source_sha256':{str(getattr(args,k).relative_to(ROOT) if getattr(args,k).is_absolute() else getattr(args,k)):
                              hashlib.sha256(getattr(args,k).read_bytes()).hexdigest() for k in inputs},
            'predictor_sha256':hashlib.sha256((ROOT/'compiler/phase4_cost.py').read_bytes()).hexdigest(),
            'limits':['engine predictor assumes uncontended SRAM; overlap contention is separately measured',
                      'DMA fit is calibrated for this 64-byte burst adapter and nominal clock',
                      'no calibrated energy model: physical power instrumentation unavailable']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':result['status'],'engine':engine_error,'model':model_error,
                      'dma':{d:r['validation_error'] for d,r in dma.items()}}))
    if not passed: raise SystemExit(1)


if __name__=='__main__':main()
