"""The one-flash plan must be reproducible from frozen strict manifests."""
import copy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/phase6'))
import screen_matched_finalists as finalists


class CombinedFinalistPreflightTest(unittest.TestCase):
    def test_frozen_manifest_reprepare_and_dedup(self):
        plan=finalists.prepare()
        self.assertEqual(plan['status'],'passed-preflight')
        self.assertEqual(plan['planned'],40)
        self.assertEqual(plan['programming_operations'],1)
        self.assertEqual(plan['uart_sessions'],1)
        self.assertEqual(len(plan['variants']),8)
        self.assertEqual(plan['aliases']['recompute'],'B4')
        self.assertEqual(finalists.reprepare(plan),plan)

    def test_optional_x_tile_reprepare_and_dedup(self):
        timed=ROOT/'work/phase6/matched-baselines-v1/b3-x-tiles-v2/report.json'
        plan=finalists.prepare(x_manifest=timed)
        self.assertEqual(plan['status'],'passed-preflight')
        self.assertEqual(plan['planned'],45)
        self.assertEqual(plan['programming_operations'],1)
        self.assertEqual(plan['uart_sessions'],1)
        self.assertEqual(len(plan['variants']),9)
        self.assertEqual(plan['aliases']['recompute'],'B4')
        self.assertEqual(plan['aliases']['full_width'],'B4')
        self.assertEqual(plan['aliases']['x_tiles'],'x_tiles')
        self.assertNotEqual(
            plan['variants']['B4']['fixtures']['pinned_timed']['files']['commands.bin'],
            plan['variants']['x_tiles']['fixtures']['pinned_timed']['files']['commands.bin'])
        self.assertEqual(plan['variants']['x_tiles']['fixtures']['stress_check']['native'][0]['tensor_checks'],8)
        self.assertEqual(finalists.reprepare(plan),plan)

    def test_x_manifest_must_be_frozen_timed_report(self):
        diagnostic=ROOT/'work/phase6/matched-baselines-v1/b3-x-diagnostics-v1/report.json'
        with self.assertRaisesRegex(ValueError,'frozen timed report'):
            finalists.prepare(x_manifest=diagnostic)

    def test_reprepare_rejects_manifest_set_change(self):
        plan=finalists.prepare()
        altered=copy.deepcopy(plan)
        altered['source_manifests'].pop('b3_cache')
        with self.assertRaisesRegex(ValueError,'manifest set changed'):
            finalists.reprepare(altered)


if __name__=='__main__':
    unittest.main()
