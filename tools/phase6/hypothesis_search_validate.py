#!/usr/bin/env python3
"""Valid quantized counterexample and final evidence decision for the search screen."""
import argparse
import copy
import json
from pathlib import Path

from hypothesis_search import ROOT,OUT,StateKeys,baseline,pointwise_fixture,lower,replay_stack,save


def run(directory=OUT):
    output=Path(directory).resolve()
    report=json.loads((output/'report.json').read_text());cost=json.loads((output/'cost.json').read_text())
    winners=json.loads((output/'winners.json').read_text())
    if any(r['status']!='passed' for r in (report,cost,winners)):raise ValueError('incomplete screen')
    for name,expected in report['source_sha256'].items():
        if baseline.sha(ROOT/name)!=expected:raise ValueError('source changed')
    program,source=pointwise_fixture();modified=copy.deepcopy(program)
    parameters=modified.layers[0].parameters
    parameters['weight'][-1]=0
    # The independent centered oracle uses bias; the hardware uses corrected
    # bias. Update the correction after changing a weight row, preserving the
    # exact documented raw-MAC versus centered arithmetic relationship.
    parameters['corrected_bias'][-1]=parameters['bias'][-1]
    cfg=dict(start=0,stop=2,kind='rectangle',h=4,w=5,mode=1,retain=True,prefetch=True)
    rows=[]
    for label,p in (('original',program),('zero-tail',modified)):
        summary,artifact=lower(p,cfg)
        if artifact is None:raise AssertionError('counterexample unexpectedly infeasible')
        proof=replay_stack(p,0,2,source,*artifact)
        directory=output/'valid-counterexample'/label;directory.mkdir(parents=True,exist_ok=True)
        code,payload,record=artifact
        (directory/'commands.bin').write_bytes(code);(directory/'payload.bin').write_bytes(payload)
        save(directory/'schedule.json',record)
        rows.append(dict(label=label,outcome=summary,replay=proof,
            key=StateKeys(p,report['source_sha256'],exact_halo=True).key(cfg),
            directory=str(directory.relative_to(ROOT)),files={name:baseline.sha(directory/name)
                for name in ('commands.bin','payload.bin','schedule.json')}))
    if rows[0]['outcome']==rows[1]['outcome'] or rows[0]['key']==rows[1]['key']:
        raise AssertionError('unsafe shape-key example ineffective')
    upstream='work/phase6/defines-source/DeFiNES-7097d6090dc22321e44ce91434e7cc23b065864f/classes/stages/DepthFirstStage.py'
    decision=dict(status='passed',physical_board=False,source_sha256={
        'tools/phase6/hypothesis_search_validate.py':baseline.sha(Path(__file__)),
        upstream:baseline.sha(ROOT/upstream)},
        proof_files={str((output/name).relative_to(ROOT)):baseline.sha(output/name)
                     for name in ('report.json','cost.json','winners.json')},
        valid_shape_only_counterexample=dict(configuration=cfg,quantization_correction_recomputed=True,
            both_original_integer_oracles_pass=True,rows=rows),
        engineering_decision='Keep as an optional compiler search improvement; do not present as accelerator architecture novelty.',
        novelty_decision='Not established: DeFiNES already memoizes tile cost and caching-level searches; current addition is conservative inactive-mode canonicalization.',
        prior_code=[dict(path=upstream,lines=[456,472,475,800],capability='tile cost memoization keyed by cache and geometry arguments'),
                    dict(path=upstream,lines=[809,825,827,1028],capability='memoized cache-level search')],
        proof_scope='All 33,544 recorded generic catalogue rows and top24 proxy paths are unchanged; 96 held-out real lowerings, 258 synthetic oracle replays, and all 4,968 avoided lowerings were independently rerun.',
        improvement_scope='15.3% KWS /14.7% VWW fewer lowering calls; measured removed work minus key overhead estimates 9.0s/30.5s saved. No inference speedup.',
        pivot='A next claim needs more than memoization: a provably sound capacity/cost bound or state abstraction that skips distinct lowerings, plus preserved optimum versus exhaustive search and an advantage over equally implemented prior methods.',
        limitations=['The entire real search was not timed end-to-end in both implementations; removed work minus key time is an estimate.',
                     'Only empty-arena standalone states are supported; arbitrary SRAM contents/live buffers are not proven equivalent.',
                     'Source-frozen quantized arrays and complete ABI identity prevent reuse across incompatible programs.',
                     'Do not interpret approximate kernel proxy rankings as newly measured FPGA performance.'])
    save(output/'decision.json',decision)
    print(json.dumps({k:decision[k] for k in ('status','engineering_decision','novelty_decision')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=OUT)
    run(parser.parse_args().directory)
