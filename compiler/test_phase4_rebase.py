"""Fail-closed checks for rebasing pinned calibration ranges."""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools/phase4'))
from rebase_calibration import check_inventory


@pytest.mark.parametrize('name', ['kws', 'vww'])
def test_only_constant_name_drift_is_accepted(name):
    old = json.loads((ROOT/f'benchmarks/manifests/{name}.canonical-inventory.json').read_text())
    new = copy.deepcopy(old)
    constant = next(t for t in new['tensors'] if t['constant'] and
                    any(t['name'] in op['inputs'] for op in new['operators']))
    prior = constant['name']
    constant['name'] = 'renamed_constant_for_test'
    for op in new['operators']:
        op['inputs'] = [constant['name'] if item == prior else item
                        for item in op['inputs']]
    assert check_inventory(old, new)[prior] == constant['name']
    changed = copy.deepcopy(new)
    tensor = next(t for t in changed['tensors'] if t['name'] == constant['name'])
    tensor['shape'][0] += 1
    with pytest.raises(ValueError, match='constant tensor geometry'):
        check_inventory(old, changed)
    changed = copy.deepcopy(new)
    changed['operators'][0]['op'] = 'Different'
    with pytest.raises(ValueError, match='operator 0'):
        check_inventory(old, changed)
    changed = copy.deepcopy(new)
    changed['operators'][0]['inputs'][0] = constant['name']
    with pytest.raises(ValueError, match='nonconstant input'):
        check_inventory(old, changed)
