"""Reject missing/duplicated physical timing evidence before drawing conclusions."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/phase6'))
from screen_writeback import summarize


def records():
    rows = [dict(kind='correctness', variant='writeback', model=model,
                 fixture=f'{model}/{sample}', tensor_checks=[{}])
            for model in ('kws', 'vww') for sample in ('pinned', 'stress')]
    rows += [dict(kind='warmup' if repeat == 0 else 'timed', variant=variant,
                  model=model, repeat=repeat, elapsed_cycles=cycles)
             for variant, cycles in (('spatial', 1000), ('writeback', 900))
             for model in ('kws', 'vww') for repeat in range(11)]
    return rows


def test_matched_improvement():
    result = summarize(records())
    assert result['stable'] and result['improves_both_models']
    assert result['comparisons']['kws']['speedup'] == pytest.approx(1000/900)


@pytest.mark.parametrize('damage', ['missing', 'duplicate_timing', 'duplicate_correctness'])
def test_incomplete_evidence_rejected(damage):
    rows = records()
    if damage == 'missing': rows.pop()
    elif damage == 'duplicate_timing': rows[-1] = copy.deepcopy(rows[-2])
    else: rows[0] = copy.deepcopy(rows[1])
    with pytest.raises(ValueError): summarize(rows)


def test_unstable_or_slower_candidate_is_not_accepted():
    rows = records()
    rows[-1]['elapsed_cycles'] = 950
    assert not summarize(rows)['stable']
    for row in rows:
        if row['variant'] == 'writeback': row['elapsed_cycles'] = 1100
    assert not summarize(rows)['improves_both_models']
