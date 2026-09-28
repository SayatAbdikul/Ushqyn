"""Boardless gates for the executable 8x16+8x8 VWW policy point."""

import sys
from pathlib import Path

import pytest


ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/phase6'))
import screen_b3_x_tiles as screen  # noqa: E402


def test_preflight_pins_image_model_and_all_intermediates():
    plan=screen.prepare()
    assert plan['status']=='passed-preflight'
    assert plan['planned']==10
    assert plan['image']['sha256']==screen.shared.IMAGE_SHA
    assert plan['clock_hz']==27_000_000
    x=plan['variants']['x_tiles']['fixtures']
    baseline=plan['variants']['full_width']['fixtures']
    assert x['pinned_timed']['files']['input.bin']==baseline['pinned_timed']['files']['input.bin']
    assert x['stress_check']['files']['input.bin']==baseline['stress_check']['files']['input.bin']
    for sample in ('pinned_timed','stress_check'):
        assert x[sample]['files']['output.bin']==baseline[sample]['files']['output.bin']
        assert sorted(row['stall_seed'] for row in x[sample]['native'])==[0,6063]
    assert all(row['tensor_checks']==8 for row in x['stress_check']['native'])
    assert x['pinned_timed']['command_count']<2048
    assert x['stress_check']['command_count']<2048


def test_summary_requires_ten_exact_records():
    plan=screen.prepare()
    rows=[]
    for variant,cycles in (('full_width',100),('x_tiles',120)):
        for kind,repeat in (('stress',0),('warmup',0),
                            ('timed',0),('timed',1),('timed',2)):
            rows.append(dict(variant=variant,kind=kind,repeat=repeat,
                bitstream_sha256=plan['image']['sha256'],
                core_clock_hz=plan['clock_hz'],output_hex='00',expected_hex='00',
                elapsed_cycles=cycles,
                device_latency_ms=cycles/plan['clock_hz']*1000))
    summary=screen.summarize(rows)
    assert summary['full_width_over_x_tiles_speedup']==1.2
    with pytest.raises(ValueError,match='ten'):
        screen.summarize(rows[:-1])
    rows[0]['output_hex']='ff'
    with pytest.raises(ValueError,match='logits'):
        screen.summarize(rows)
