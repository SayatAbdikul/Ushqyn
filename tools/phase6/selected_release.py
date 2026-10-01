#!/usr/bin/env python3
"""Verify, materialize, and simulate the selected board-tested Phase 6 release.

Only materialize/native write local work products. No command accesses JTAG,
UART, or the physical board. Paths derive from this checkout, not a worktree.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "hardware/releases/phase6/selected"
MANIFEST = RELEASE / "manifest.json"
WORK = ROOT / "work/phase6/selected-release"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_manifest():
    record = json.loads(MANIFEST.read_text())
    if record["schema"] != 1 or record["clock_hz"] != 27_000_000:
        raise ValueError("unexpected selected release contract")
    return record


def pinned(path, expected):
    if sha(path) != expected:
        raise ValueError(f"identity mismatch: {path.relative_to(ROOT)}")


def gzip_identity(path):
    digest = hashlib.sha256()
    size = 0
    with gzip.open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def archived_bytes(entry):
    path = ROOT / entry["path"]
    pinned(path, entry["gzip_sha256"])
    raw = gzip.decompress(path.read_bytes())
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"archive identity mismatch: {entry['path']}")
    return raw


def verify():
    record = load_manifest()
    image = record["bitstream"]
    pinned(ROOT / image["path"], image["gzip_sha256"])
    if gzip_identity(ROOT / image["path"]) != (image["sha256"], image["bytes"]):
        raise ValueError("selected image decompression identity mismatch")
    sources = record["physical_sources"]
    if len(sources) != 18:
        raise ValueError("physical source inventory must have 18 entries")
    for name, entry in sources.items():
        pinned(ROOT / name, entry["sha256"])
    pinned(ROOT / record["build_recipe"]["path"], record["build_recipe"]["sha256"])
    counts = {}
    for key, fixture in record["fixtures"].items():
        folder = ROOT / fixture["path"]
        actual = {p.name: sha(p) for p in folder.iterdir() if p.is_file()}
        if actual != fixture["files"]:
            raise ValueError(f"fixture inventory differs: {key}")
        program = (folder / "commands.bin").read_bytes()
        if not program or len(program) % 16 or len(program) > 32 * 1024:
            raise ValueError(f"program alignment/capacity violation: {key}")
        if (folder / "payload.bin").stat().st_size > 8 * 1024 * 1024:
            raise ValueError(f"external-memory capacity violation: {key}")
        checks = []
        for line in (folder / "checks.txt").read_text().splitlines():
            address, name = line.split()
            expected = folder / name
            if name not in fixture["files"] or int(address) < 0 or int(address) + expected.stat().st_size > 8 * 1024 * 1024:
                raise ValueError(f"invalid tensor check: {key}/{name}")
            checks.append({"address": int(address), "file": name, "bytes": expected.stat().st_size})
        if not checks or len(checks) != fixture["tensor_checks"]:
            raise ValueError(f"tensor-check count changed: {key}")
        counts[key] = checks
    evidence = record["physical_evidence"]
    raw = {key: archived_bytes(entry) for key, entry in evidence["files"].items()}
    parsed = {key: json.loads(value) for key, value in raw.items() if key != "records"}
    seal, plan, report = (parsed[key] for key in ("seal", "plan", "report"))
    for key in ("plan", "report", "records"):
        if seal[key + "_sha256"] != hashlib.sha256(raw[key]).hexdigest():
            raise ValueError("physical evidence seal mismatch")
    if report["status"] != "passed-short-screen" or report["completed"] != 14 or report["planned"] != 14:
        raise ValueError("physical evidence is incomplete")
    if plan["bitstream_sha256"] != image["sha256"] or plan["clock_hz"] != record["clock_hz"]:
        raise ValueError("physical campaign uses another image/clock")
    historical_sources = plan["route"]["expected_gprj_sources"]
    if {entry["original_path"]: entry["sha256"] for entry in sources.values()} != historical_sources:
        raise ValueError("promoted physical sources differ from tested route inventory")
    rows = []
    for line in raw["records"].splitlines():
        row = json.loads(line)
        signature = row.pop("record_sha256")
        canonical = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(canonical).hexdigest() != signature:
            raise ValueError("physical record signature mismatch")
        rows.append(row)
    if rows != report["records"] or len(rows) != 14:
        raise ValueError("physical report differs from signed records")
    for index, row in enumerate(rows):
        fixture = record["fixtures"][row["model"] + "-" + row["sample"]]
        if row["fixture_files"] != fixture["files"] or row["output_hex"] != row["expected_hex"]:
            raise ValueError("physical output/fixture identity mismatch")
        if row["bitstream_sha256"] != image["sha256"] or row["order_index"] != index:
            raise ValueError("physical image/ordering mismatch")
        if row["output_hex"] != (ROOT / fixture["path"] / "output.bin").read_bytes().hex():
            raise ValueError("physical oracle differs from selected fixture")
        if row["sample"] == "stress":
            expected_checks = [{"file": check["file"], "bytes": check["bytes"],
                                "sha256": fixture["files"][check["file"]]}
                               for check in counts[row["model"] + "-stress"]]
            if row["stress_tensor_checks"] != expected_checks:
                raise ValueError("physical stress tensor coverage differs from selected checks")
        if row["kind"] == "timed":
            if index == 0 or rows[index - 1]["kind"] != "warmup" or rows[index - 1]["load_id"] != row["load_id"] or row["load_seconds"] != 0:
                raise ValueError("physical timed sample lacks immediate same-load warmup")
        elif row["load_seconds"] <= 0:
            raise ValueError("physical stress/warmup lacks reload")
    for model in ("kws", "vww"):
        timed = [r["elapsed_cycles"] for r in rows if r["model"] == model and r["kind"] == "timed"]
        if len(timed) != 3 or statistics.median(timed) != record["board_medians"][model]["cycles"]:
            raise ValueError("board median differs from raw samples")
    return {"status": "passed-selected-release-verification", "physical_sources": 18,
            "fixture_checks": counts, "bitstream_sha256": image["sha256"],
            "physical_records": len(rows), "manifest_sha256": sha(MANIFEST),
            "physical_board_accessed": False}


def materialize():
    record = load_manifest()
    for target, entry in ((ROOT / record["generated_ip"]["path"], record["generated_ip"]),
                          (WORK / "selected_27mhz.fs", record["bitstream"])):
        if target.exists():
            pinned(target, entry["sha256"])
            continue
        source = ROOT / entry.get("archive_path", entry["path"])
        pinned(source, entry["gzip_sha256"])
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        with gzip.open(source, "rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, 1 << 20)
        pinned(temporary, entry["sha256"])
        temporary.replace(target)
    return verify()


def native():
    before = verify()
    record = load_manifest()
    source_names = ["rtl/v2/target_pkg.sv", "rtl/v2/requantizer.sv", record["engine"],
                    *["rtl/v2/" + name for name in ("scratchpad.sv", "tile_dma.sv", "tiled_core.sv",
                                                     "command.sv", "tile_sequencer.sv", "tiled_host_bridge.sv")]]
    harness_name = "test/phase6/native.cpp"
    pinned(ROOT / harness_name, record["native_harness_sha256"])
    names = set(source_names + [harness_name, "tools/phase6/selected_release.py", str(MANIFEST.relative_to(ROOT)),
                               record["build_recipe"]["path"], *record["physical_sources"]])
    pins = {name: sha(ROOT / name) for name in sorted(names)}
    build = WORK / "native-build"
    build.mkdir(parents=True, exist_ok=True)
    validation = RELEASE / "validation"
    validation.mkdir(exist_ok=True)
    report_path = validation / "native-report.json"
    report = {"status": "running", "physical_board": False, "scope": "Exact sequencer/engine/SRAM/DMA with abstract fixed/stalled external RAM; not physical SDRAM latency",
              "build_jobs": 2, "source_sha256": pins, "release_verification": before, "results": []}
    save(report_path, report)
    command = ["verilator", "--cc", "--exe", "--build", "-j", "2", "--public-flat-rw", "-Wno-fatal",
               "--top-module", "v2_tiled_host_bridge", "--Mdir", str(build),
               *[str(ROOT / name) for name in source_names], str(ROOT / harness_name)]
    started = time.monotonic()
    try:
        with (build / "build.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
        executable = build / "Vv2_tiled_host_bridge"
        report.update(build_seconds=time.monotonic() - started, executable_sha256=sha(executable),
                      build_log_sha256=sha(build / "build.log"), tool_version=subprocess.check_output(["verilator", "--version"], text=True).strip())
        for key, fixture in sorted(record["fixtures"].items()):
            folder = ROOT / fixture["path"]
            for seed in (0, 6063):
                output = validation / f"{key}-s{seed}.json"
                with (build / f"{key}-s{seed}.log").open("w") as log:
                    subprocess.run([str(executable), str(folder), str(seed), str(output)], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
                row = json.loads(output.read_text())
                expected = fixture["native_expected"][str(seed)]
                for name in ("elapsed_cycles", "engine_cycles", "dma_cycles", "overlap_cycles", "tensor_checks"):
                    if row[name] != expected[name]:
                        raise ValueError(f"native reference counter mismatch: {key} seed={seed} {name}")
                if row["status"] != "passed":
                    raise ValueError("native output check failed")
                row.update(fixture=key, fixture_files=fixture["files"], report_sha256=sha(output),
                           report=str(output.relative_to(ROOT)))
                report["results"].append(row)
                save(report_path, report)
                print(f"{key} seed {seed}: {row['elapsed_cycles']} cycles, {row['tensor_checks']} exact tensor checks", flush=True)
        for name, expected in pins.items():
            pinned(ROOT / name, expected)
        report.update(status="passed-selected-release-native", completed=8, release_verification_after=verify())
    except Exception as error:
        report.update(status="failed-selected-release-native", error=str(error))
        save(report_path, report)
        raise
    save(report_path, report)
    return {"status": report["status"], "completed": report["completed"], "report": str(report_path.relative_to(ROOT))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("verify", "materialize", "native"))
    args = parser.parse_args()
    result = globals()[args.stage]()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
