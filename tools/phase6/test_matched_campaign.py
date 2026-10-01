"""Read-only hardware preflight and malformed-artifact campaign regressions.

Tests copy fixture files into a temporary workspace directory; no historical
evidence or board state is modified. Full-artifact cases skip on checkouts that
have not provisioned the sealed selected image and fixtures.
"""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matched_campaign as mc


class PathBoundaryTests(unittest.TestCase):
    def test_local_rejects_escaping_checkout(self):
        with self.assertRaises(ValueError):
            mc.local('../../../../outside')

    def test_fixture_hash_paths_cannot_escape_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'inside').mkdir()
            outside = root/'outside'
            outside.write_bytes(b'x')
            with self.assertRaises(ValueError):
                mc.verify(root/'inside', {'../outside':mc.sha(outside)})


@unittest.skipUnless((mc.REFERENCE/'plan.json').exists(), 'sealed work artifacts not provisioned')
class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = mc.read(mc.REFERENCE/'plan.json')
        cls.label = cls.plan['fixture_names']['kws']['pinned']
        cls.original = cls.plan['fixtures'][cls.label]

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='campaign-test-', dir=mc.BASE.parent)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        for name in self.original['files']:
            target = self.directory/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(mc.REFERENCE_FIXTURES/self.label/name, target)
        self.files = {name:mc.sha(self.directory/name) for name in self.original['files']}
        self.native = copy.deepcopy(self.original['native'])
        self.bind_native()

    def bind_native(self):
        for row in self.native:
            row['fixture_files'] = dict(self.files)
            row['executable_sha256'] = mc.NATIVE_SHA

    def call_fixture(self):
        return mc.fixture(mc.relative(self.directory), self.files, 'kws', 'pinned',
                          self.native, self.original['verification'])

    def rewrite_schedule(self, change):
        schedule = mc.read(self.directory/'schedule.json')
        change(schedule)
        mc.save(self.directory/'schedule.json', schedule)
        self.files['schedule.json'] = mc.sha(self.directory/'schedule.json')
        self.bind_native()

    def test_valid_frozen_hardware_and_reference_relocate(self):
        contract = mc.hardware_contract()
        self.assertEqual(contract['clock_hz'], 27_000_000)
        self.assertEqual(contract['current_checkout'], str(mc.ROOT))
        reference = mc.reference_policy()
        self.assertEqual(set(reference), {'kws','vww'})
        self.assertEqual(set(reference['kws']), {'pinned','stress'})

    def test_bound_unchanged_fixture_passes(self):
        self.assertEqual(self.call_fixture()['files'], self.files)

    def test_changed_fixture_bytes_rejected(self):
        (self.directory/'commands.bin').write_bytes(b'broken')
        with self.assertRaises((ValueError, KeyError)):
            self.call_fixture()

    def test_native_without_artifact_identity_rejected(self):
        for row in self.native:
            row.pop('fixture_files', None)
        with self.assertRaises((ValueError, KeyError)):
            self.call_fixture()

    def test_native_from_another_fixture_rejected(self):
        self.native[0]['fixture_files']['commands.bin'] = '0'*64
        with self.assertRaises((ValueError, KeyError)):
            self.call_fixture()

    def test_native_from_another_engine_rejected(self):
        self.native[0]['executable_sha256'] = '0'*64
        with self.assertRaises((ValueError, KeyError)):
            self.call_fixture()

    def test_missing_native_cycles_rejected(self):
        self.native[0].pop('elapsed_cycles')
        with self.assertRaises((ValueError, KeyError)):
            self.call_fixture()

    def test_snapshot_in_timed_fixture_rejected(self):
        self.rewrite_schedule(lambda s:s.update(snapshot_regions={'0':{'ext':0,'bytes':1}},
                                               snapshots_enabled=True))
        with self.assertRaises((ValueError, KeyError)):
            self.call_fixture()

    def test_native_checks_must_include_final_output(self):
        (self.directory/'checks.txt').write_text('0 input.bin\n')
        self.files['checks.txt'] = mc.sha(self.directory/'checks.txt')
        self.bind_native()
        with self.assertRaises((ValueError, KeyError)):
            self.call_fixture()

    def test_programmed_image_is_exactly_the_verified_image(self):
        # Keep the mock plan/report internally sealed while changing only the
        # intended programming path. The actual image files remain untouched.
        original_read, original_sha = mc.read, mc.sha
        plan = copy.deepcopy(self.plan)
        plan['image']['file'] = 'work/phase6/not-the-verified-image.fs'
        report = original_read(mc.REFERENCE/'report.json')
        seal = original_read(mc.REFERENCE/'seal.json')
        plan_hash, report_hash = 'a'*64, 'b'*64
        report['plan_sha256'] = plan_hash
        seal['plan_sha256'], seal['report_sha256'] = plan_hash, report_hash
        values = {mc.REFERENCE/'plan.json':plan, mc.REFERENCE/'report.json':report,
                  mc.REFERENCE/'seal.json':seal}
        def read(path):
            return copy.deepcopy(values[Path(path)]) if Path(path) in values else original_read(path)
        def sha(path):
            if Path(path) == mc.REFERENCE/'plan.json': return plan_hash
            if Path(path) == mc.REFERENCE/'report.json': return report_hash
            return original_sha(path)
        with patch.object(mc, 'read', read), patch.object(mc, 'sha', sha):
            with self.assertRaises((ValueError, KeyError, FileNotFoundError)):
                mc.hardware_contract()


if __name__ == '__main__':
    unittest.main()
