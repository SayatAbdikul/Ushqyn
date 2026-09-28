"""Board preflight must bind the distinct tile schedules to exact replay."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/phase6'))
import screen_b3_spatial as spatial


class SpatialPreflightTest(unittest.TestCase):
    def test_exact_candidates_pass_without_device_access(self):
        plan=spatial.prepare()
        self.assertEqual(plan['status'],'passed-preflight')
        self.assertEqual(plan['planned'],15)
        self.assertEqual(set(plan['variants']),{'B4','h8','h12'})
        hashes={plan['variants'][name]['fixtures']['pinned_timed']['files']['commands.bin']
                for name in plan['variants']}
        self.assertEqual(len(hashes),3)

    def test_replay_hash_mutation_is_rejected(self):
        raw=json.loads(spatial.MANIFEST.read_text())
        tampered=copy.deepcopy(raw)
        tampered['candidates']['8']['replay']['report_sha256']='0'*64
        with tempfile.TemporaryDirectory(dir=spatial.BASE) as folder:
            path=Path(folder)/'tampered.json'
            path.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError,'symbolic replay report changed'):
                spatial.prepare(path)


if __name__=='__main__':
    unittest.main()
