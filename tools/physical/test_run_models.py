"""Ensure physical-result collection cannot turn bad outputs into passing runs."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_models import ROOT, TARGET, digest, execute_fixture, load_fixture
from collect_evidence import validate_phase


class Link:
    def __init__(self, bad_output=False, bad_counter=False):
        self.memory = bytearray(TARGET['memory_bytes'])
        self.bad_output = bad_output
        self.bad_counter = bad_counter

    def capabilities(self):
        return len(self.memory)

    def exchange(self, command):
        return b''

    def write(self, address, data):
        self.memory[address:address + len(data)] = data

    def read(self, address, length):
        return bytes(self.memory[address:address + length])

    def run(self, pc):
        self.memory[16:18] = bytes([7, 9 if self.bad_output else 8])
        return dict(busy=False, error=0, useful_macs=2, elapsed=10 if not self.bad_counter else 11,
                    compute_cycles=4, wait_cycles=3, control_cycles=3,
                    protocol_errors=0)


class PhysicalRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = Path(self.temp.name)
        image = bytes(TARGET['memory_bytes'])
        (self.fixture / 'board.bin').write_bytes(image)
        self.metadata = dict(target=TARGET['name'],
                             target_manifest_sha256=digest(ROOT / 'hardware/targets/tang_nano_20k_v2.json'),
                             image_sha256=hashlib.sha256(image).hexdigest(),
                             inputs={'input': 0}, outputs={'output': 16}, entry=0, macs=2)
        (self.fixture / 'board.json').write_text(json.dumps(self.metadata))
        np.savez(self.fixture / 'checks.npz', inputs=np.array([[1, 2]], np.int8),
                 outputs=np.array([[7, 8]], np.int8), labels=np.array([1], np.uint8))

    def test_exact_run_records_accuracy_separately(self):
        result = {}
        execute_fixture(Link(), self.fixture, 1, result)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['completed_jobs'], 1)
        self.assertEqual(result['correct'], 1)
        self.assertEqual(result['records'][0]['output_hex'], '0708')

    def test_same_class_wrong_logit_is_failure(self):
        result = {}
        with self.assertRaisesRegex(AssertionError, 'integer output mismatch'):
            execute_fixture(Link(bad_output=True), self.fixture, 1, result)
        self.assertEqual(result['integer_mismatches'], 1)
        self.assertEqual(result['completed_jobs'], 0)
        self.assertEqual(result['first_mismatch']['actual_hex'], '0709')

    def test_bad_counter_does_not_count_as_completed(self):
        result = {}
        with self.assertRaisesRegex(AssertionError, 'counters do not reconcile'):
            execute_fixture(Link(bad_counter=True), self.fixture, 1, result)
        self.assertEqual(result['completed_jobs'], 0)

    def test_changed_image_is_rejected_before_execution(self):
        (self.fixture / 'board.bin').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'image digest mismatch'):
            load_fixture(self.fixture, 1)

    def test_archive_revalidates_raw_logits_despite_pass_flag(self):
        result = {}
        execute_fixture(Link(), self.fixture, 1, result)
        validate_phase(result)
        result['records'][0]['output_hex'] = '0709'  # Same class, wrong logit.
        with self.assertRaisesRegex(ValueError, 'archived integer output mismatch'):
            validate_phase(result)

    def test_archive_rejects_wrong_accuracy_summary(self):
        result = {}
        execute_fixture(Link(), self.fixture, 1, result)
        result['correct'] = 0
        result['accuracy'] = 0
        with self.assertRaisesRegex(ValueError, 'archived accuracy differs'):
            validate_phase(result)


if __name__ == '__main__':
    unittest.main()
