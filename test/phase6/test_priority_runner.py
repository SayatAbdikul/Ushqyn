"""Failure and continuation safeguards for priority board campaigns."""
from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/phase6'))
import run_priority as priority


def confirmation_records():
    report = json.loads((ROOT / 'work/phase6/screening-v1/report.json').read_text())
    rows = []
    for session in (2, 3):
        for variant in ('baseline', 'spatial'):
            for model in ('kws', 'vww'):
                cycles = report['summary']['timings'][variant][
                    f'{model}-pinned-resident-half-overlap-timed']['median_cycles']
                for repeat in range(11):
                    rows.append(dict(session=session, variant=variant, model=model,
                                     kind='timed' if repeat else 'warmup', repeat=repeat,
                                     elapsed_cycles=cycles))
    return rows, report


def test_confirmation_accepts_repeatable_matched_improvement():
    rows, report = confirmation_records()
    assert priority.confirmation_summary(rows, report)['passed']


def test_confirmation_rejects_missing_and_duplicate_repeats():
    rows, report = confirmation_records()
    with pytest.raises(ValueError): priority.confirmation_summary(rows[:-1], report)
    rows[-1]['repeat'] = 9
    with pytest.raises(ValueError): priority.confirmation_summary(rows, report)


def test_confirmation_rejects_session_drift():
    rows, report = confirmation_records()
    for row in rows:
        if row['session'] == 3: row['elapsed_cycles'] *= 1.1
    assert not priority.confirmation_summary(rows, report)['passed']


def test_confirmation_rejects_lost_speedup():
    rows, report = confirmation_records()
    for row in rows:
        if row['variant'] == 'spatial': row['elapsed_cycles'] *= 2
    assert not priority.confirmation_summary(rows, report)['passed']


def test_persisted_samples_detect_corruption(tmp_path):
    path = tmp_path / 'records.jsonl'
    with path.open('w') as stream: priority.append_row(stream, {'sample_index': 7})
    assert priority.read_rows(path) == [{'sample_index': 7}]
    path.write_text(path.read_text().replace('7', '8', 1))
    with pytest.raises(ValueError, match='corrupted record'): priority.read_rows(path)


def prefix():
    sample = dict(id='sample-0', feature_sha256='feature', label=1)
    row = dict(sample_index=0, workload='kws', sample_id='sample-0', feature_sha256='feature',
               label=1, bitstream_sha256='image', oracle='independent-centered-integer',
               output_hex='0102', expected_hex='0102', prediction=1, elapsed_cycles=100,
               engine_cycles=80, dma_cycles=40, overlap_cycles=25, command_index=6)
    schedule = dict(final_output={'bytes': 2}, command_count=4)
    return row, [sample], schedule, {'entry': 3}


def test_accuracy_resume_accepts_exact_prefix():
    row, samples, schedule, location = prefix()
    priority.check_prefix([row], 'kws', samples, 'image', schedule, location)


@pytest.mark.parametrize('field,value', [('sample_index', 1), ('feature_sha256', 'other'),
    ('expected_hex', '0103'), ('prediction', 0), ('bitstream_sha256', 'other'), ('command_index', 5)])
def test_accuracy_resume_rejects_wrong_prefix(field, value):
    row, samples, schedule, location = prefix(); row[field] = value
    with pytest.raises(ValueError): priority.check_prefix([row], 'kws', samples, 'image', schedule, location)


def test_screening_source_evidence_remains_valid():
    assert priority.accepted_screening()['status'] == 'passed-screening'
