"""Boardless gates for the isolated B1/B2 VWW finalist campaign."""

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/phase6'))
import screen_b1b2_vww_tuning as tuning  # noqa: E402


def test_preflight_binds_frozen_finalists_and_common_image():
    plan = tuning.prepare()
    assert plan['status'] == 'passed-preflight'
    assert plan['planned'] == 20
    assert plan['image']['sha256'] == tuning.shared.IMAGE_SHA
    assert plan['clock_hz'] == 27_000_000
    assert set(plan['variants']) == set(tuning.VARIANTS)
    assert all(plan['variants'][left]['policy'] != plan['variants'][right]['policy']
               for left, right in zip(plan['block_order'], plan['block_order'][1:]))
    reference = plan['variants']['B1_selected']['fixtures']['pinned_timed']['files']
    alternative = plan['variants']['B1_full_prefetch']['fixtures']['pinned_timed']['files']
    assert reference['commands.bin'] != alternative['commands.bin']
    for sample in ('pinned_timed', 'stress_check'):
        assert len({variant['fixtures'][sample]['files']['input.bin']
                    for variant in plan['variants'].values()}) == 1
        assert len({variant['fixtures'][sample]['files']['output.bin']
                    for variant in plan['variants'].values()}) == 1
    for variant in plan['variants'].values():
        fixtures = variant['fixtures']
        for fixture in fixtures.values():
            assert sorted(row['stall_seed'] for row in fixture['native']) == [0, 6063]
            assert all(row['status'] == 'passed' for row in fixture['native'])


def test_summary_requires_all_exact_runs_and_selects_lower_median():
    plan = tuning.prepare()
    medians = {'B1_selected': 110, 'B1_full_prefetch': 100,
               'B2_selected': 80, 'B2_half_prefetch': 90}
    rows = []
    for name in tuning.VARIANTS:
        for kind, repeat in [('stress', 0), ('warmup', 0),
                             ('timed', 0), ('timed', 1), ('timed', 2)]:
            cycles = medians[name]
            rows.append(dict(variant=name, kind=kind, repeat=repeat,
                bitstream_sha256=plan['image']['sha256'],
                core_clock_hz=plan['clock_hz'], output_hex='00', expected_hex='00',
                elapsed_cycles=cycles,
                device_latency_ms=cycles/plan['clock_hz']*1000))
    summary = tuning.summarize(rows)
    assert summary['physically_fastest'] == {
        'B1': 'B1_full_prefetch', 'B2': 'B2_selected'}
    with pytest.raises(ValueError, match='twenty'):
        tuning.summarize(rows[:-1])
    rows[0]['output_hex'] = 'ff'
    with pytest.raises(ValueError, match='logits'):
        tuning.summarize(rows)
