"""Physical-campaign gates: corrupted evidence must fail before device access."""

import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/phase6'))
import matched_baselines_board as runner


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    folder = tmp_path / 'fixture'
    folder.mkdir()
    contents = {'commands.bin': bytes(16), 'payload.bin': bytes(512),
                'input.bin': bytes(490), 'output.bin': bytes(range(12)),
                'checks.txt': b'496 output.bin\n'}
    for name, contents_bytes in contents.items():
        (folder / name).write_bytes(contents_bytes)
    schedule = dict(command_count=1, final_output=dict(ext=496, bytes=12),
                    program_sha256=runner.sha(folder / 'commands.bin'),
                    image_sha256=runner.sha(folder / 'payload.bin'),
                    snapshots_enabled=False, snapshot_regions={})
    (folder / 'schedule.json').write_text(json.dumps(schedule))
    raw = dict(directory='fixture', files={p.name: runner.sha(p) for p in folder.iterdir()},
               replay=dict(status='passed'), native=[])
    for seed in (0, 6063):
        report = tmp_path / f'native-{seed}.json'
        row = dict(status='passed', physical_board=False, stall_seed=seed, tensor_checks=1,
                   elapsed_cycles=10, engine_cycles=8, dma_cycles=2, overlap_cycles=1)
        report.write_text(json.dumps(row))
        raw['native'].append(dict(row, report=report.name, report_sha256=runner.sha(report)))
    return raw, dict(files=copy.deepcopy(raw['files'])), folder


def test_native_and_file_gate_passes(fixture):
    raw, reference, _ = fixture
    result = runner.validate_fixture(raw, 'kws', 'pinned_timed', reference)
    assert result['command_count'] == 1


def test_rejects_changed_native_report(fixture):
    raw, reference, folder = fixture
    (folder.parent / raw['native'][0]['report']).write_text('{}')
    with pytest.raises(ValueError, match='native report changed'):
        runner.validate_fixture(raw, 'kws', 'pinned_timed', reference)


def test_rejects_missing_memory_seed(fixture):
    raw, reference, _ = fixture
    raw['native'].pop()
    with pytest.raises(ValueError, match='two native memory seeds'):
        runner.validate_fixture(raw, 'kws', 'pinned_timed', reference)


def test_rejects_changed_public_input(fixture):
    raw, reference, folder = fixture
    (folder / 'input.bin').write_bytes(bytes([1]) * 490)
    raw['files']['input.bin'] = runner.sha(folder / 'input.bin')
    with pytest.raises(ValueError, match='public input.bin differs'):
        runner.validate_fixture(raw, 'kws', 'pinned_timed', reference)


def test_rejects_snapshots_in_timing_fixture(fixture):
    raw, reference, folder = fixture
    path = folder / 'schedule.json'
    schedule = json.loads(path.read_text())
    schedule['snapshots_enabled'] = True
    path.write_text(json.dumps(schedule))
    raw['files']['schedule.json'] = runner.sha(path)
    with pytest.raises(ValueError, match='diagnostic snapshots'):
        runner.validate_fixture(raw, 'kws', 'pinned_timed', reference)


def test_rejects_false_replay(fixture):
    raw, reference, _ = fixture
    raw['replay']['status'] = 'failed'
    with pytest.raises(ValueError, match='independent command replay'):
        runner.validate_fixture(raw, 'kws', 'pinned_timed', reference)


def test_block_order_is_balanced_interleaved_and_reproducible():
    policies = ('B1', 'B2', 'B4')
    blocks = runner.make_blocks(20260928, policies)
    assert blocks == runner.make_blocks(20260928, policies)
    assert sorted((row['policy'], row['model']) for row in blocks) == sorted(
        (policy, model) for policy in policies for model in runner.MODELS)
    assert all(a['policy'] != b['policy'] for a, b in zip(blocks, blocks[1:]))


def physical_rows():
    rows = []
    for policy in ('B1', 'B2', 'B4'):
        for model in runner.MODELS:
            for index, kind in enumerate(('stress', 'warmup', 'timed', 'timed', 'timed')):
                rows.append(dict(policy=policy, model=model, kind=kind,
                    repeat=max(0, index - 2), bitstream_sha256=runner.IMAGE_SHA,
                    output_hex='01', expected_hex='01', elapsed_cycles=27_000,
                    device_latency_ms=1.0, wall_seconds=0.01,
                    duplicate_commands_of_B4=policy == 'B4',
                    duplicate_program_and_payload_of_B4=policy == 'B4'))
    return rows


def test_summary_uses_active_policies_and_actual_27mhz_clock():
    rows = physical_rows()
    result = runner.summarize(rows, ('B1', 'B2', 'B4'))
    assert result['policies']['B4']['vww']['median_device_latency_ms'] == 1.0
    rows[-1]['device_latency_ms'] = 27_000 / 20_250_000 * 1000
    with pytest.raises(ValueError, match='clock or exactness changed'):
        runner.summarize(rows, ('B1', 'B2', 'B4'))


def test_summary_marks_duplicate_b4_as_nonindependent():
    rows = physical_rows()
    for row in rows:
        if row['policy'] == 'B2':
            row['duplicate_commands_of_B4'] = True
            row['duplicate_program_and_payload_of_B4'] = True
    result = runner.summarize(rows, ('B1', 'B2', 'B4'))
    comparison = result['comparisons']['B4_versus_B2']
    assert comparison['independent_performance_evidence'] is False
    assert comparison['duplicate_program_models'] == ['kws', 'vww']


def test_nested_model_hashes_are_checked(fixture):
    raw, _, folder = fixture
    pins = {'kws': {'fixture/input.bin': raw['files']['input.bin']}}
    assert runner.checked_hash_tree(pins, 'models') == pins['kws']
    (folder / 'input.bin').write_bytes(b'changed model input')
    with pytest.raises(ValueError, match='source changed'):
        runner.checked_hash_tree(pins, 'models')


@pytest.fixture
def selection_fixture(fixture):
    raw, _, folder = fixture
    files = raw['files']
    candidates = {}
    for name, cycles in (('000', 10), ('111', 20)):
        candidates[name] = dict(files=files, worst_seed_cycles=cycles + 2,
            native=[dict(stall_seed=seed, status='passed', elapsed_cycles=cycles + offset)
                    for seed, offset in ((0, 0), (6063, 2))])
    path = folder.parent / 'selection.json'
    selection = dict(status='passed-native', selection='000', variants=candidates)
    path.write_text(json.dumps(selection))
    record = dict(independent_policy_generation=True, selection_report='selection.json',
                  selection_report_sha256=runner.sha(path), fixed_fallback_models=['kws'])
    fixtures = {sample: dict(files=copy.deepcopy(files))
                for sample in ('pinned_timed', 'stress_check')}
    models = {model: dict(fixtures=copy.deepcopy(fixtures)) for model in runner.MODELS}
    reference = dict(fixture_names={'kws': {'pinned': 'kp', 'stress': 'ks'}},
                     fixtures={name: dict(files=files) for name in ('kp', 'ks')})
    return record, models, reference, path, selection


def test_selection_digest_argmin_and_winner_fixture_bind(selection_fixture):
    record, models, reference, _, _ = selection_fixture
    result = runner.checked_b3_selection(record, models, reference)
    assert result['winner'] == '000'
    models['vww']['fixtures']['pinned_timed']['files']['commands.bin'] = '0' * 64
    with pytest.raises(ValueError, match='winner fixture differs'):
        runner.checked_b3_selection(record, models, reference)


def test_selection_rejects_changed_report(selection_fixture):
    record, models, reference, path, _ = selection_fixture
    path.write_text('{}')
    with pytest.raises(ValueError, match='selection report changed'):
        runner.checked_b3_selection(record, models, reference)


def test_selection_rejects_nonwinning_candidate(selection_fixture):
    record, models, reference, path, selection = selection_fixture
    selection['selection'] = '111'
    path.write_text(json.dumps(selection))
    record['selection_report_sha256'] = runner.sha(path)
    with pytest.raises(ValueError, match='declared argmin'):
        runner.checked_b3_selection(record, models, reference)
