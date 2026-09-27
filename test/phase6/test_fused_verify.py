"""Negative controls for the independent fused descriptor/graph replay."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'compiler'),str(ROOT/'tools/phase6')]
from followup_graph import group_channels, check_oracles
from run_boardless import load_model
from scheduler.fused_verify import replay_fused


@pytest.mark.parametrize('model',['kws','vww'])
def test_replay_rejects_corrupt_map_and_wrong_activation_edge(model):
    root=ROOT/f'work/phase6/experiments-v1/fused-activation-v1/fixtures/{model}-pinned-grouped-fused-timed'
    schedule=json.loads((root/'schedule.json').read_text())
    code=(root/'commands.bin').read_bytes();payload=(root/'payload.bin').read_bytes()
    original,pinned,_,_=load_model(model)
    grouped,mappings,_=group_channels(original)
    oracle=check_oracles(original,grouped,mappings,pinned)

    def replay(image,contracts):
        return replay_fused(grouped,code,image,{grouped.inputs[0]:pinned},
            run_contracts=contracts,constant_contracts=schedule['constant_contracts'],
            final_output=schedule['final_output'],snapshot_regions=schedule['snapshot_regions'],oracle=oracle)

    assert replay(payload,schedule['run_contracts'])['status']=='passed'
    stage=next(s for s in schedule['stages'] if s.get('fused_activation_layer') is not None
               and s['loads'][-1]['bytes']==256)
    load=stage['loads'][-1]
    corrupted=bytearray(payload);corrupted[load['ext']]^=1
    with pytest.raises(ValueError,match='unready or incorrect operand bytes'):
        replay(bytes(corrupted),schedule['run_contracts'])
    contracts=copy.deepcopy(schedule['run_contracts'])
    contract=next(c for c in contracts.values() if 'fused_activation_layer' in c)
    contract['fused_activation_layer']+=1
    with pytest.raises(ValueError,match='immediately following activation'):
        replay(payload,contracts)
