"""Boardless safeguards for the physical campaign runner."""
import importlib.util
from pathlib import Path
import struct
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('screening', ROOT / 'tools/phase6/run_screening.py')
screening = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screening)
sys.path.insert(0, str(ROOT / 'tools/phase2'))


def registers(elapsed=100, engine=80, dma=40, overlap=25, index=3):
    return bytes(4) + b'SEQ4' + struct.pack('<6I', elapsed, engine, dma, overlap, index, 0)


@pytest.mark.parametrize('kwargs', [dict(index=2), dict(elapsed=0), dict(overlap=41),
                                   dict(overlap=0), dict(engine=101)])
def test_rejects_partial_execution_and_inconsistent_counters(kwargs):
    with pytest.raises(AssertionError):
        screening.check_counters(registers(**kwargs), 4)


def test_counters_accept_overlap_and_report_device_time():
    result = screening.check_counters(registers(), 4)
    assert result['elapsed_cycles'] == 100
    assert result['device_latency_ms'] == pytest.approx(100 / 20250)


def test_changed_artifact_is_rejected(tmp_path):
    path = tmp_path / 'commands.bin'; path.write_bytes(b'valid')
    digest = screening.sha(path)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact hash mismatch'):
        screening.verify_files(tmp_path, {'commands.bin': digest})


class FakeBoard:
    def __init__(self, output=b'\x01', timeout=False, protocol_errors=0):
        self.output = output; self.timeout = timeout
        self.protocol_errors = protocol_errors; self.launches = 0

    def write_external(self, address, data):
        pass

    def write(self, address, data):
        self.launches += 1
        if self.timeout:
            raise TimeoutError('ambiguous launch acknowledgement')

    def exchange(self, cmd):
        return b'\0\0' + struct.pack('<9I', *([0] * 8 + [self.protocol_errors]))

    def read(self, address, length):
        return registers()

    def read_external(self, address, length):
        return self.output


def fixture(tmp_path):
    (tmp_path / 'input.bin').write_bytes(b'\0')
    (tmp_path / 'output.bin').write_bytes(b'\x01')
    (tmp_path / 'checks.txt').write_text('0 output.bin\n')
    return {'command_count': 4}


@pytest.mark.parametrize('actual', [b'\x02', b''])
def test_board_output_mismatch_is_not_hidden(tmp_path, actual):
    board = FakeBoard(actual)
    with pytest.raises(screening.OutputMismatch):
        screening.execute(board, tmp_path, fixture(tmp_path), 1)
    assert board.launches == 1


def test_ambiguous_launch_is_never_retried(tmp_path):
    board = FakeBoard(timeout=True)
    with pytest.raises(TimeoutError):
        screening.execute(board, tmp_path, fixture(tmp_path), 1)
    assert board.launches == 1


def test_protocol_error_is_not_a_success(tmp_path):
    with pytest.raises(AssertionError, match='protocol error'):
        screening.execute(FakeBoard(protocol_errors=1), tmp_path, fixture(tmp_path), 1)


def test_complete_execution_checks_output(tmp_path):
    result = screening.execute(FakeBoard(), tmp_path, fixture(tmp_path), 1)
    assert result['tensor_checks'][0]['sha256'] == screening.sha(tmp_path / 'output.bin')


def test_frozen_campaign_has_exact_counts_and_correctness_first():
    plan = screening.make_plan()
    seen_timing = False; correctness = timed = warmups = 0
    for block in plan['blocks']:
        if block['stage'] == 'correctness':
            assert not seen_timing
            correctness += len(block['fixtures'])
        else:
            seen_timing = True
            assert correctness == 24
            warmups += len(block['fixtures'])
            timed += 10 * len(block['fixtures'])
    assert (correctness, warmups, timed) == (24, 12, 120)
