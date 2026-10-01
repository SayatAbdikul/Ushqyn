#!/usr/bin/env python3
"""Bounded, isolated physical counterfactuals for the paired pointwise engine.

The style-only variants are synthesis probes, not correctness-proved RTL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "work/phase6/physical-resource-screen-v1"
ENGINE = ROOT / "work/phase6/paired-pw-rtl-v3/engine.sv"
RECIPE = ROOT / "work/phase6/pool-timing-v1/build27.tcl"
GOWIN = Path("/Applications/GowinIDE.app/Contents/Resources/Gowin_EDA/IDE")
LIMIT_KIB = 8 * 1024 * 1024
TIMEOUT_SEC = 600


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variant_source(name: str) -> str:
    source = ENGINE.read_text()
    if name == "no-weight-cache":
        old = '''wire cached_weight=((op==OP_CONV||op==OP_DWCONV)&&
        count<=(spatial_pw&&!paired_pw?256:128)&&weight_cache_valid[weight_cache_index]);
    wire pair_odd_cached=weight_cache_valid[weight_cache_index];'''
        assert source.count(old) == 1
        return source.replace(old, "wire cached_weight=1'b0;\n    wire pair_odd_cached=1'b0;")
    target = {
        "block-weight": "logic [63:0] weight_cache[0:31]",
        "block-activation": "logic [63:0] activation_tile[0:15]",
    }[name]
    old = '(* syn_ramstyle = "distributed_ram" *) ' + target
    assert source.count(old) == 1
    return source.replace(old, '(* syn_ramstyle = "block_ram" *) ' + target)


def prepare(name: str) -> Path:
    directory = BASE / name
    directory.mkdir(parents=True, exist_ok=True)
    engine = directory / "engine.sv"
    new_source = variant_source(name)
    if engine.exists() and engine.read_text() != new_source:
        raise ValueError("existing engine source differs")
    engine.write_text(new_source)
    recipe = directory / "build27.tcl"
    if recipe.exists() and recipe.read_bytes() != RECIPE.read_bytes():
        raise ValueError("existing route recipe differs")
    recipe.write_bytes(RECIPE.read_bytes())
    input_path = directory / "inputs.json"
    data = {
        "mode": name,
        "interpretation": (
            "synthesis-only RAM-style counterfactual; no RTL correctness claim"
            if name.startswith("block-") else
            "cache-removal area/traffic counterfactual; no RTL correctness or board timing claim"
        ),
        "v3_engine_sha256": digest(ENGINE),
        "candidate_engine_sha256": digest(engine),
        "recipe_sha256": digest(recipe),
        "gowin_sha256": digest(GOWIN / "bin/gw_sh"),
        "memory_limit_kib": LIMIT_KIB,
        "timeout_sec": TIMEOUT_SEC,
    }
    if input_path.exists() and json.loads(input_path.read_text()) != data:
        raise ValueError("existing experiment inputs differ")
    input_path.write_text(json.dumps(data, indent=2) + "\n")
    return directory


def tree_memory_kib(pid: int) -> int:
    output = subprocess.check_output(
        ["ps", "-axo", "pid=,ppid=,rss="], text=True, stderr=subprocess.DEVNULL
    )
    children: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for line in output.splitlines():
        try:
            child, parent, size = map(int, line.split())
        except ValueError:
            continue
        children.setdefault(parent, []).append(child)
        rss[child] = size
    stack = [pid]
    total = 0
    seen: set[int] = set()
    while stack:
        child = stack.pop()
        if child in seen:
            continue
        seen.add(child)
        total += rss.get(child, 0)
        stack.extend(children.get(child, []))
    return total


def run(name: str) -> dict:
    directory = prepare(name)
    report_path = directory / "report.json"
    if report_path.exists():
        raise ValueError("recorded experiment is immutable")
    build_dir = directory / "build"
    build_dir.mkdir(exist_ok=True)
    environment = dict(
        os.environ,
        DYLD_FRAMEWORK_PATH=str(GOWIN / "lib"),
        DYLD_LIBRARY_PATH=str(GOWIN / "lib"),
        PHASE6_ENGINE=str(directory / "engine.sv"),
    )
    start = time.monotonic()
    peak_kib = 0
    stopped_reason = None
    with (directory / "route.log").open("w") as log:
        process = subprocess.Popen(
            [str(GOWIN / "bin/gw_sh"), str(directory / "build27.tcl")],
            cwd=build_dir, env=environment, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            try:
                memory = tree_memory_kib(process.pid)
            except (OSError, subprocess.CalledProcessError):
                stopped_reason = "memory monitor unavailable"
                os.killpg(process.pid, signal.SIGTERM)
                break
            peak_kib = max(peak_kib, memory)
            if peak_kib > LIMIT_KIB:
                stopped_reason = "8 GiB RSS limit exceeded"
                os.killpg(process.pid, signal.SIGTERM)
                break
            if time.monotonic() - start > TIMEOUT_SEC:
                stopped_reason = "600 second timeout"
                os.killpg(process.pid, signal.SIGTERM)
                break
            time.sleep(0.5)
        try:
            returncode = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait()
    log_text = (directory / "route.log").read_text(errors="replace")
    synthesis_log = build_dir / "phase6_uart_burst/impl/gwsynthesis/phase6_uart_burst.log"
    synthesis_text = synthesis_log.read_text(errors="replace") if synthesis_log.exists() else ""
    over = re.search(r"number\((\d+)\((\d+) LUTs, (\d+) ALUs, (\d+) ROM16s, (\d+) SSRAMs\)\).*resource limit\((\d+)\)", log_text + synthesis_text)
    routed_report = build_dir / "phase6_uart_burst/impl/pnr/phase6_uart_burst.rpt.txt"
    route_text = routed_report.read_text(errors="replace") if routed_report.exists() else ""
    timing_path = build_dir / "phase6_uart_burst/impl/pnr/phase6_uart_burst_tr_content.html"
    timing_text = timing_path.read_text(errors="replace") if timing_path.exists() else ""
    resources = {}
    for key in ("Logic", "Register", "CLS", "BSRAM", "DSP"):
        match = re.search(rf"^\s*{key}\s*\|\s*([\d.]+)/([\d.]+)", route_text, re.M)
        if match:
            resources[key] = {"used": float(match[1]), "available": float(match[2])}
    fmax = re.search(r"<td>27\.000\(MHz\)</td>\s*<td[^>]*>([\d.]+)\(MHz\)</td>", timing_text)
    setup = re.search(r"Numbers of Setup Violated Endpoints</td>\s*<td[^>]*>(\d+)</td>", timing_text)
    hold = re.search(r"Numbers of Hold Violated Endpoints</td>\s*<td[^>]*>(\d+)</td>", timing_text)
    routed_fmax = float(fmax[1]) if fmax else None
    setup_violations = int(setup[1]) if setup else None
    hold_violations = int(hold[1]) if hold else None
    route_pass = (
        returncode == 0 and routed_fmax is not None and routed_fmax >= 27 and
        setup_violations == hold_violations == 0 and
        len(resources) == 5 and
        all(x["used"] <= x["available"] for x in resources.values())
    )
    data = {
        **json.loads((directory / "inputs.json").read_text()),
        "status": "stopped" if stopped_reason else "passed-route" if route_pass else "build-failed" if returncode else "failed-route-gate",
        "stopped_reason": stopped_reason,
        "returncode": returncode,
        "seconds": round(time.monotonic() - start, 2),
        "peak_tree_rss_kib": peak_kib,
        "route_log_sha256": digest(directory / "route.log"),
        "synthesis_failure": None if over is None else {
            "logic": int(over[1]), "luts": int(over[2]), "alus": int(over[3]),
            "rom16": int(over[4]), "ssram": int(over[5]), "limit": int(over[6]),
        },
        "routed_resources": resources,
        "routed_core_fmax_mhz": routed_fmax,
        "setup_violations": setup_violations,
        "hold_violations": hold_violations,
    }
    report_path.write_text(json.dumps(data, indent=2) + "\n")
    return data


def traffic_bound() -> dict:
    """Exact extra weight-word requests implied by disabling the cache.

    Every frozen pointwise descriptor in this profile has count <= 128, so
    both the selected one-output mode and paired mode cache its weight words.
    The no-cache variant fetches one word for every channel, reduction block,
    and eight-pixel tile. This does not model backpressure or whole-model time.
    """
    path = ROOT / "work/phase6/port-fabric-model-v1/report.json"
    source = json.loads(path.read_text())
    models = {}
    for model in ("kws", "vww"):
        rows = []
        for row in source["model"][model]["descriptor_rows"]:
            geometry = row["geometry"]
            count = int(geometry["count"])
            channels = int(geometry["output_c"])
            outputs = int(geometry["outputs"])
            assert 0 < count <= 128 and channels > 0 and outputs % channels == 0
            pixels = outputs // channels
            tiles = (pixels + 7) // 8
            words_per_channel = (count + 7) // 8
            cached = channels * words_per_channel
            uncached = tiles * cached
            rows.append({
                "command": row["command"], "pixels": pixels,
                "channels": channels, "reduction": count,
                "cached_64bit_weight_reads": cached,
                "uncached_64bit_weight_reads": uncached,
                "extra_64bit_weight_reads": uncached - cached,
            })
        models[model] = {
            "pointwise_descriptors": len(rows),
            "extra_64bit_weight_reads": sum(row["extra_64bit_weight_reads"] for row in rows),
            "rows": rows,
        }
    result = {
        "source": str(path.relative_to(ROOT)), "source_sha256": digest(path),
        "scope": "pointwise weight requests only; not whole-model native cycles",
        "models": models,
    }
    output = BASE / "no-weight-cache-traffic-bound.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "traffic"))
    parser.add_argument("name", choices=("block-weight", "block-activation", "no-weight-cache"), nargs="?")
    args = parser.parse_args()
    if args.action != "traffic" and args.name is None:
        parser.error("name required for prepare/run")
    data = traffic_bound() if args.action == "traffic" else prepare(args.name) if args.action == "prepare" else run(args.name)
    print(data if isinstance(data, Path) else json.dumps(data, indent=2))
