"""Exact provenance and measurement-boundary gates for the cache board screen."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/phase6'))
import screen_b3_cache as runner


@pytest.fixture
def replay_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner.shared, 'ROOT', tmp_path)
    files = {name: str(index) * 64 for index, name in enumerate(
        ('commands.bin', 'payload.bin', 'input.bin', 'output.bin', 'schedule.json'))}
    result = dict(status='passed', model='vww', fixture='fixture',
                  fixture_files_sha256=files, oracle_output_sha256=files['output.bin'],
                  replay=dict(status='passed'))
    path = tmp_path / 'replay.json'
    path.write_text(json.dumps(result))
    raw = dict(directory='fixture', files=dict(files),
               replay=dict(status='passed', report='replay.json', report_sha256=runner.sha(path)))
    return raw, result, path


def test_symbolic_report_pins_actual_fixture(replay_fixture):
    raw, _, _ = replay_fixture
    assert runner.checked_symbolic(raw)['report'] == 'replay.json'
    raw['files']['commands.bin'] = 'f' * 64
    with pytest.raises(ValueError, match='fixture differs: commands.bin'):
        runner.checked_symbolic(raw)


def test_symbolic_status_without_report_is_rejected(replay_fixture):
    raw, _, _ = replay_fixture
    raw['replay'] = dict(status='native-only')
    with pytest.raises(ValueError, match='has not passed'):
        runner.checked_symbolic(raw)


def test_changed_symbolic_report_is_rejected(replay_fixture):
    raw, _, path = replay_fixture
    path.write_text('{}')
    with pytest.raises(ValueError, match='report changed'):
        runner.checked_symbolic(raw)


def test_symbolic_wrong_oracle_is_rejected(replay_fixture):
    raw, result, path = replay_fixture
    result['oracle_output_sha256'] = 'a' * 64
    path.write_text(json.dumps(result))
    raw['replay']['report_sha256'] = runner.sha(path)
    with pytest.raises(ValueError, match='oracle output differs'):
        runner.checked_symbolic(raw)


def test_traffic_counts_descriptor_loads_and_activation_writes(tmp_path):
    commands = ((1, 1, 0, 100, 0, 128), (1, 1, 0, 1000, 128, 3072),
                (2, 0, 0, 0, 0, 0), (1, 0, 0, 2000, 4096, 64), (0, 0, 0, 0, 0, 0))
    (tmp_path / 'commands.bin').write_bytes(b''.join(runner.CMD.pack(*row) for row in commands))
    result = runner.traffic(tmp_path)
    assert result == dict(command_count=5, external_read_bytes=3200,
                          external_write_bytes=64, engine_dispatches=1)


def test_summary_uses_actual_clock_and_correct_speedup_direction():
    rows = []
    for variant, cycles in (('recompute', 27_000), ('vertical_cache', 54_000)):
        for index, kind in enumerate(('stress', 'warmup', 'timed', 'timed', 'timed')):
            rows.append(dict(variant=variant, kind=kind, repeat=max(0, index - 2),
                bitstream_sha256=runner.shared.IMAGE_SHA, output_hex='01', expected_hex='01',
                elapsed_cycles=cycles, device_latency_ms=cycles / runner.shared.CLOCK_HZ * 1000,
                wall_seconds=0.1))
    result = runner.summarize(rows)
    assert result['cache_throughput_speedup'] == 0.5
    assert result['cache_latency_change_fraction'] == 1.0
    assert result['variants']['vertical_cache']['median_device_latency_ms'] == 2.0
    rows[-1]['device_latency_ms'] = 54_000 / 20_250_000 * 1000
    with pytest.raises(ValueError, match='clock or image differs'):
        runner.summarize(rows)
