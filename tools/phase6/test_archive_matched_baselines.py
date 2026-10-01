"""Bounded archive tests; every artifact lives in a temporary synthetic checkout."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.phase6 import archive_matched_baselines as archive


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "sealed"

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = archive.encoded(value) if isinstance(value, (dict, list)) else value
        path.write_bytes(raw)
        return archive.digest(raw)

    def basic(self):
        source = self.write("compiler/policy.py", b"return_exact = True\n")
        files = {name: self.write("work/run/fixture/" + name, value) for name, value in {
            "commands.bin": b"\x00\x01\x02", "schedule.json": {"status": "passed"},
            "checks.txt": b"output.bin 1 1\n"}.items()}
        catalogue = self.write("work/run/catalogue.json", {"status": "enumerated", "candidates": []})
        native = {"status": "passed", "elapsed_cycles": 12, "stall_seed": 0,
                  "fixture_files": files}
        self.write("work/run/fixture/native-s0.json", native)
        self.write("work/run/report.json", {"status": "passed", "compiler_sources": {
            "compiler/policy.py": source}, "catalogue_file": "work/run/catalogue.json",
            "catalogue_sha256": catalogue, "candidate": {"directory": "work/run/fixture",
            "files": files, "native": native}})
        return "work/run/report.json"

    def build(self, report=None, **kwargs):
        return archive.build(self.root, self.output, reports=[report or self.basic()], **kwargs)

    def test_recursive_deterministic_seal_and_workspace(self):
        report = self.basic()
        first = self.build(report)
        before = {str(p.relative_to(self.output)): p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        self.assertEqual(first, self.build(report))
        after = {str(p.relative_to(self.output)): p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        by_path = {row["workspace_path"]: row for row in first["artifacts"].values()}
        self.assertIn("compiler/policy.py", by_path)
        self.assertIn("work/run/catalogue.json", by_path)
        self.assertIn("work/run/fixture/native-s0.json", by_path)
        self.assertEqual(by_path["work/run/fixture/commands.bin"]["storage"], "hash_only")
        self.assertEqual(first["limitations"], [])
        result = archive.verify(self.output, self.root, workspace=True)
        self.assertEqual(result["status"], "verified")

    def test_include_fixtures_embeds_bytes_but_not_executable(self):
        report = self.basic()
        value = json.loads((self.root / report).read_bytes())
        binary = self.write("work/native/executable", b"ELF content")
        commands = value["candidate"]["files"]["commands.bin"]
        # A source-map pin is visited first; the later fixture binding must
        # upgrade only the command file from hash-only to embedded content.
        value["source_sha256"] = {"work/native/executable": binary,
            "work/run/fixture/commands.bin": commands}
        self.write(report, value)
        manifest = self.build(report, include_fixtures=True)
        by_path = {row["workspace_path"]: row for row in manifest["artifacts"].values()}
        self.assertEqual(by_path["work/run/fixture/commands.bin"]["storage"], "gzip")
        self.assertEqual(by_path["work/native/executable"]["storage"], "hash_only")
        self.assertEqual(archive.verify(self.output, self.root, workspace=True)["status"], "verified")

    def test_legacy_native_requires_exact_declared_file_map(self):
        report = self.basic()
        value = json.loads((self.root / report).read_bytes())
        native = json.loads(json.dumps(value["candidate"]["native"]))
        legacysha = self.write("work/legacy-native/report.json", {"status": "passed", "results": [native]})
        value["source_sha256"] = {"work/legacy-native/report.json": legacysha}
        self.write(report, value)
        manifest = self.build(report, include_fixtures=True)
        self.assertEqual(manifest["limitations"], [])
        native["fixture_files"] = {**native["fixture_files"], "missing.bin": "d" * 64}
        legacysha = self.write("work/legacy-native/report.json", {"status": "passed", "results": [native]})
        value["source_sha256"]["work/legacy-native/report.json"] = legacysha
        # Keep the declared candidate map unchanged so the legacy row's extra
        # hash cannot bind to a merely similar fixture.
        self.write(report, value)
        other = archive.build(self.root, self.root / "second", reports=[report], include_fixtures=True)
        self.assertTrue(any(r["kind"] == "unresolved_fixture_directory" for r in other["limitations"]))

    def test_corrupt_payload_and_manifest_rejected(self):
        manifest = self.build()
        row = next(row for row in manifest["artifacts"].values() if row["storage"] == "gzip")
        path = self.output / row["archive"]
        original = path.read_bytes()
        path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "gzip seal"):
            archive.verify(self.output, self.root)
        path.write_bytes(original)
        path = self.output / "manifest.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "manifest seal"):
            archive.verify(self.output, self.root)

    def test_missing_and_changed_dependencies_are_explicit(self):
        report = self.basic()
        path = self.root / report
        value = json.loads(path.read_bytes())
        value["source_sha256"] = {"missing/source.py": "1" * 64, "compiler/policy.py": "2" * 64}
        self.write(report, value)
        result = self.build(report)
        self.assertEqual(len(result["limitations"]), 2)
        self.assertEqual(archive.verify(self.output, self.root)["status"], "verified")
        with self.assertRaisesRegex(ValueError, "workspace dependencies missing or changed"):
            archive.verify(self.output, self.root, workspace=True)

    def test_absolute_relocation_and_outside_root_rejection(self):
        report = self.basic()
        path = self.root / report
        value = json.loads(path.read_bytes())
        sha = value["compiler_sources"].pop("compiler/policy.py")
        value["compiler_sources"]["/former/checkout/compiler/policy.py"] = sha
        value["source_sha256"] = {"../../escape.py": "3" * 64}
        self.write(report, value)
        manifest = self.build(report, relocate_roots=["/former/checkout"])
        self.assertEqual([row["kind"] for row in manifest["limitations"]], ["unresolved_path"])
        self.assertIn("compiler/policy.py@" + sha, manifest["artifacts"])
        self.assertEqual(archive.verify(self.output, self.root, workspace=True)["status"], "verified")

    def test_historical_source_kept_separate_from_current(self):
        report = self.basic()
        historical = self.root / "docs/research/evidence/old"
        archive.build(self.root, historical, reports=[report])
        self.write("compiler/policy.py", b"return_exact = 'changed'\n")
        result = self.build(report)
        row = next(row for row in result["artifacts"].values() if row["workspace_path"] == "compiler/policy.py")
        self.assertEqual(row["workspace_policy"], "historical_snapshot")
        self.assertEqual(result["limitations"], [])
        self.assertEqual(archive.verify(self.output, self.root, workspace=True)["historical_source_snapshots"], 1)

    def test_incomplete_report_and_oversize_explicit_report_fail(self):
        self.write("report.json", {"status": "running"})
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.build("report.json")
        self.write("report.json", {"status": "passed", "description": "x" * 200})
        with self.assertRaisesRegex(ValueError, "explicit input cannot be embedded"):
            self.build("report.json", max_file_bytes=100)

    def test_hash_budget_does_not_read_huge_binary(self):
        report = self.basic()
        self.write("work/run/fixture/commands.bin", b"x" * 4096)
        # The stale digest must remain explicitly unavailable, not silently updated.
        manifest = self.build(report, max_hash_bytes=2000)
        row = next(row for row in manifest["artifacts"].values() if row["workspace_path"].endswith("commands.bin"))
        self.assertEqual(row["storage"], "unavailable")
        self.assertEqual(row["reason"], "hash-budget")

    def test_prior_files_use_prior_directory_and_nested_plan_hash_is_not_a_path(self):
        current = self.write("work/current/commands.bin", b"current")
        previous = self.write("work/prior/commands.bin", b"prior")
        self.write("report.json", {"status": "passed", "directory": "work/current",
            "files": {"commands.bin": current}, "prior_directory": "work/prior",
            "prior_files": {"commands.bin": previous}, "comparison": {
                "physical_board": True, "plan_sha256": "4" * 64}})
        manifest = self.build("report.json")
        self.assertEqual(manifest["limitations"], [])
        self.assertIn("work/prior/commands.bin@" + previous, manifest["artifacts"])
        self.assertFalse(any(row["workspace_path"] == "plan.json" for row in manifest["artifacts"].values()))

    def test_legacy_labeled_fixture_directory_and_upstream_source(self):
        revision = "a" * 40
        upstream = "work/phase6/defines-source/DeFiNES-" + revision
        source = self.write(upstream + "/classes/stage.py", b"class Stage: pass\n")
        fixture = self.write("work/run/fixtures/tiny/checks.txt", b"exact\n")
        self.write("work/run/report.json", {"status": "passed", "upstream": {
            "revision": revision, "files": [{"path": "classes/stage.py", "sha256": source}]},
            "results": [{"label": "tiny", "files": {"checks.txt": fixture}}]})
        manifest = self.build("work/run/report.json")
        self.assertEqual(manifest["limitations"], [])
        self.assertIn(upstream + "/classes/stage.py@" + source, manifest["artifacts"])
        self.assertIn("work/run/fixtures/tiny/checks.txt@" + fixture, manifest["artifacts"])

    def physical(self):
        folder = "work/physical/"
        image = self.write("image.fs", b"image")
        plan = {"status": "prepared", "hardware": {"image": {"file": "image.fs", "sha256": image}},
                "planned": 2, "repeats": 1}
        ph = self.write(folder + "plan.json", plan)
        rows = [{"policy": "B1", "model": "kws", "kind": kind, "output_hex": "aa",
                 "expected_hex": "aa", "input_readback_verified": True, "bitstream_sha256": image,
                 "elapsed_cycles": 123} for kind in ("stress", "timed")]
        signed = [{**row, "record_sha256": archive.digest(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())} for row in rows]
        rh = self.write(folder + "records.jsonl", b"".join(archive.encoded(row).replace(b"\n", b"") + b"\n" for row in signed))
        log = self.write(folder + "program.log", b"programmed\n")
        report = {"status": "passed-matched-short-screen", "physical_board": True, "plan_sha256": ph,
                  "records_sha256": rh, "program_log_sha256": log, "completed": 2, "records": rows,
                  "summary": {"B1": {"kws": {"median_cycles": 123}},
                              "B2": {"kws": {"measured_policy": "B1", "median_cycles": 123}}}}
        rph = self.write(folder + "report.json", report)
        self.write(folder + "seal.json", {"plan_sha256": ph, "records_sha256": rh, "report_sha256": rph})
        return folder + "report.json"

    def test_physical_seal_signed_rows_and_alias_medians(self):
        manifest = self.build(self.physical())
        self.assertEqual(manifest["checks"][0]["physical_records"], 2)
        self.assertEqual(archive.verify(self.output, self.root, workspace=True)["status"], "verified")

    def test_physical_invalid_signature_rejected(self):
        report = self.physical()
        records = self.root / "work/physical/records.jsonl"
        lines = records.read_text().splitlines()
        row = json.loads(lines[0]); row["record_sha256"] = "f" * 64
        records.write_text(json.dumps(row) + "\n" + lines[1] + "\n")
        value = json.loads((self.root / report).read_bytes())
        value["records_sha256"] = archive.hash_file(records)
        reportsha = self.write(report, value)
        self.write("work/physical/seal.json", {"report_sha256": reportsha,
            "plan_sha256": value["plan_sha256"], "records_sha256": value["records_sha256"]})
        with self.assertRaisesRegex(ValueError, "signature mismatch"):
            self.build(report)


if __name__ == "__main__":
    unittest.main()
