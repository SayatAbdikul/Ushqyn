"""Phase 5 plan invariants and evidence-tamper regressions."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from audit import audit
from schedule import ROOT, build_jobs, create_plan, draw_indices, load_split


class Phase5Test(unittest.TestCase):
    def test_frozen_balanced_switch_plan(self):
        manifests = {
            name: load_split(ROOT / f'benchmarks/manifests/{name}.data.json')[0]
            for name in ('kws', 'vww')
        }
        jobs = list(build_jobs(manifests))
        self.assertEqual(len(jobs), 10000)
        self.assertEqual([job['job'] for job in jobs], list(range(10000)))
        self.assertTrue(all(job['workload'] == ('kws' if i % 2 == 0 else 'vww')
                            for i, job in enumerate(jobs)))
        self.assertEqual(len({job['sample_id'] for job in jobs[::2]}), 4890)
        self.assertEqual(len({job['sample_id'] for job in jobs[1::2]}), 5000)
        self.assertEqual(create_plan()['jsonl_sha256'],
                         '6ad5398fd6983947fe8ef5e85e2a261bdd54195876a6e243e063446d86a64dbe')

    def test_small_split_repeats_only_after_full_pass(self):
        selection = draw_indices(3, 8, 19)
        self.assertEqual(set(selection[:3]), {0, 1, 2})
        self.assertEqual(set(selection[3:6]), {0, 1, 2})
        with self.assertRaises(ValueError):
            draw_indices(0, 8, 19)

    def test_split_rejects_same_feature_in_calibration(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'data.json'
            path.write_text(json.dumps({'splits': {
                'calibration': {'count': 1, 'records': [
                    {'id': 'cal', 'feature_sha256': 'a' * 64}]},
                'accuracy': {'count': 1, 'records': [
                    {'id': 'eval', 'feature_sha256': 'a' * 64}]},
            }}))
            with self.assertRaisesRegex(ValueError, 'overlap'):
                load_split(path)

    def test_audit_rejects_changed_release(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [
                'docs/research/evidence/phase4/summary.json',
                'docs/research/evidence/phase4/inventory-audit.json',
                'hardware/releases/phase4/tinyml_v4_kernels.fs',
            ]
            for name in ('kws', 'vww'):
                paths.extend(f'benchmarks/manifests/{name}.{kind}.json'
                             for kind in ('canonical-inventory', 'data', 'static-quality'))
            for name in paths:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            report = audit(root)
            self.assertFalse(report['g5_certified'])
            self.assertEqual(report['schedule_jobs'], 10000)
            self.assertFalse(report['checks']['integrated_sdram_controller'])
            with (root / paths[2]).open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'bitstream hash mismatch'):
                audit(root)


if __name__ == '__main__':
    unittest.main()
