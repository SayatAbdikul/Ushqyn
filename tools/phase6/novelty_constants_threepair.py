#!/usr/bin/env python3
"""Compose final-constant folding with the frozen three-pair VWW strip schedule.

Uses the exact existing strip splicers for layers 3–6, 11–14 and 7–10,
starting from the folded/compacted full-model fixture. All hardware is the
already routed pooled 27 MHz core; this script never programs the board.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'compiler'),str(ROOT/'tools/phase6')]

from integer_reference import evaluate
from novelty_constants import model,verify_sample
from variants import check_frozen,sha
import strip_fusion_vww as pair3
import strip_fusion_pair11 as pair11
import strip_fusion_pair7 as pair7

BASE=ROOT/'work/phase6/novelty-constants-v1'
OUT=BASE/'threepair'
NATIVE=ROOT/'work/phase6/pool-timing-v1/native/Vv2_tiled_host_bridge'
BASELINE=ROOT/'work/phase6/strip-fusion-pair7-v1/full'


def identical_block(compacted,block,start):
    for index,local in enumerate(block.layers):
        whole=compacted.layers[start+index]
        if (local.op!=whole.op or local.attributes!=whole.attributes or
                block.tensors[local.inputs[0]].shape!=compacted.tensors[whole.inputs[0]].shape or
                block.tensors[local.output].shape!=compacted.tensors[whole.output].shape or
                set(local.parameters)!=set(whole.parameters)):
            raise ValueError(f'changed layer {start+index} in strip composition')
        for key,value in local.parameters.items():
            if not np.array_equal(value,whole.parameters[key]):
                raise ValueError(f'changed layer {start+index} parameter {key}')


def run(output=OUT):
    check_frozen();output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    source_report=json.loads((BASE/'fused/report.json').read_text())
    proof_report=json.loads((BASE/'report.json').read_text())
    cert=BASE/'certificate.json'
    if (source_report['status']!='passed' or proof_report['status']!='passed-replay' or
            source_report['prepared_report_sha256']!=sha(BASE/'report.json') or
            proof_report['certificate_sha256']!=sha(cert)):
        raise ValueError('source folded fixture or certificate changed')
    native_report=json.loads((NATIVE.parent/'report.json').read_text())
    baseline=json.loads((BASELINE/'report.json').read_text())
    if (native_report['status']!='passed' or sha(NATIVE)!=native_report['executable_sha256'] or
            baseline['status']!='passed-native' or
            baseline['selected_native_executable_sha256']!=native_report['executable_sha256']):
        raise ValueError('latest three-pair baseline/native executable identity changed')
    original,pinned,grouped,folded,compacted,maps,group_maps,_,_,certificate,_=model()
    _,_,block3,_,_=pair3.block_program()
    _,block11,_,_=pair11.model_block()
    _,block7,_,_=pair7.model_block()
    for start,block in ((3,block3),(11,block11),(7,block7)):
        identical_block(compacted,block,start)
    c3,p3,r3=pair3.build_candidate(block3)
    c11,p11,r11=pair11.build(block11)
    c7,p7,r7=pair7.build(block7)
    report=dict(schema=1,status='running',physical_board=False,
        scope='final constant propagation composed with existing exact three-pair strip schedule; native 27MHz pooled core, no board run',
        source_report_sha256=sha(BASE/'fused/report.json'),
        certificate_sha256=sha(cert),
        baseline_report_sha256=sha(BASELINE/'report.json'),
        pooled_native_executable_sha256=sha(NATIVE),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (
            Path(__file__),ROOT/'tools/phase6/novelty_constants.py',
            ROOT/'tools/phase6/strip_fusion_vww.py',
            ROOT/'tools/phase6/strip_fusion_pair11.py',
            ROOT/'tools/phase6/strip_fusion_pair7.py')},
        fixtures={})
    source_rows={row['name']:row for row in source_report['results']}
    for sample,snapshots in (('pinned',False),('pinned',True),('stress',True)):
        label=f'vww-{sample}-known-compacted-fused-'+('check' if snapshots else 'timed')
        source=BASE/'fused/fixtures'/label
        for filename,digest in source_rows[label]['files'].items():
            if sha(source/filename)!=digest:
                raise ValueError('source fixture changed')
        value=(pinned if sample=='pinned' else
               np.random.default_rng(6257).integers(-128,128,pinned.shape,dtype=np.int8))
        oracle=verify_sample(original,folded,compacted,maps,group_maps,certificate,value)
        host_input=oracle[compacted.layers[0].output]
        local=[]
        for start,block,engine,code,payload,record in (
            (3,block3,pair3,c3,p3,r3),(11,block11,pair11,c11,p11,r11),
            (7,block7,pair7,c7,p7,r7)):
            source_value=oracle[compacted.layers[start].inputs[0]]
            expected=evaluate(block,{block.inputs[0]:source_value})
            for i,layer in enumerate(block.layers):
                if not np.array_equal(expected[layer.output],oracle[compacted.layers[start+i].output]):
                    raise ValueError(f'block {start}: original INT8 layer mismatch')
            result=(engine.replay_candidate(block,source_value,expected[block.outputs[0]],code,payload,record)
                    if start==3 else engine.replay(block,source_value,expected[block.outputs[0]],code,payload,record))
            local.append(dict(pair_start=start,verification=result))
        stem=label.replace('known-compacted-fused','known-compacted-strip3-7-11')
        first=output/'stage3'/stem
        second=output/'stage11'/stem
        final=output/'fixtures'/stem
        a=pair3.splice_full_fixture(block3,grouped,compacted,source,first,
                                    c3,p3,r3,host_input,False)
        b=pair11.splice(first,second,c11,p11,r11,host_input,False)
        c=pair7.splice(second,final,c7,p7,r7,host_input,False)
        expected_final=oracle[compacted.outputs[0]].tobytes()
        if (final/'output.bin').read_bytes()!=expected_final:
            raise ValueError('composed schedule has changed public logits')
        native=[]
        for seed in (0,6063):
            path=output/f'{stem}-s{seed}.json'
            subprocess.run([str(NATIVE),str(final),str(seed),str(path)],check=True)
            result=json.loads(path.read_text())
            if (result['status']!='passed' or
                    result['tensor_checks']!=(30 if snapshots else 1)):
                raise ValueError('composed full-model RTL exactness failed')
            native.append(dict(seed=seed,**result))
        report['fixtures'][stem]=dict(name=stem,sample=sample,snapshots=snapshots,
            local_replay=local,splices=[a['splice'],b['splice'],c['splice']],
            files=c['files'],native=native)
        (output/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
        print(stem,[r['elapsed_cycles'] for r in native],flush=True)
    baseline_name='vww-pinned-compacted-strip3-7-11-timed'
    candidate_name='vww-pinned-known-compacted-strip3-7-11-timed'
    base=baseline['fixtures'][baseline_name]['native']
    cand=report['fixtures'][candidate_name]['native']
    report['matched_native_comparison']={str(seed):dict(
        baseline_cycles=base[i]['elapsed_cycles'],
        candidate_cycles=cand[i]['elapsed_cycles'],
        saved_cycles=base[i]['elapsed_cycles']-cand[i]['elapsed_cycles'],
        latency_reduction_fraction=(base[i]['elapsed_cycles']-cand[i]['elapsed_cycles'])/
                                   base[i]['elapsed_cycles']) for i,seed in enumerate((0,6063))}
    report['status']='passed-native'
    (output/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    check_frozen();return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    result=run(args.output)
    print(result['status'],result['matched_native_comparison'])
