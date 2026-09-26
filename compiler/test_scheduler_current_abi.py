import json
from pathlib import Path

import pytest

from scheduler.abi_verify import replay
from scheduler.contract import IllegalSchedule
from scheduler.current_abi import CostModel, optimize
from scheduler.fixtures import wide_chain

ROOT = Path(__file__).resolve().parents[1]


def model():
    return CostModel(json.loads((ROOT/'docs/research/evidence/phase4/physical-sequence-costs.json').read_text()))


def test_tuning_and_cost_ablation_produce_verified_deterministic_bytecode():
    program, x = wide_chain()
    artifacts, report = optimize(program, model())
    again, _ = optimize(program, model())
    assert report['exact']['objective'] == report['beam']['objective']
    assert report['beam']['bound_gap_fraction'] == 0
    assert artifacts['prefer16-serial']['components']['tiles'] > artifacts['full32-serial']['components']['tiles']
    for policy, candidate in artifacts.items():
        assert candidate['commands'] == again[policy]['commands']
        assert candidate['payload'] == again[policy]['payload']
        assert replay(program, candidate['commands'], candidate['payload'], {'x': x})['status'] == 'passed'
    assert artifacts['prefer16-prefetch']['schedule']['prefetch_payload_bytes'] > 0


def test_changed_cost_source_rejected():
    record = model().record.copy()
    record['predictor_sha256'] = '0'*64
    with pytest.raises(IllegalSchedule, match='changed engine cost'):
        CostModel(record)


def test_tiny_budget_falls_back_to_legal_materialized_schedule():
    p, x = wide_chain()
    artifacts, report = optimize(p, model(), max_expansions=1)
    assert report['beam']['stop_reason'] == 'expansion_budget'
    assert report['beam']['path'] == artifacts['full32-serial']['path']
    assert replay(p, artifacts['mixed-serial']['commands'], artifacts['mixed-serial']['payload'], {'x': x})['status'] == 'passed'
