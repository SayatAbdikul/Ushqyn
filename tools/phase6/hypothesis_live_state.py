#!/usr/bin/env python3
"""Falsify a small live-INT32/port/overlap novelty hypothesis.

This is an abstract, exhaustively solved four-task example, NOT a Tang timing
model, DeFiNES reproduction, or COSMA implementation. It deliberately compares
an imagined scratchpad accumulator with the selected RTL's register lifetime.
Run with Python's standard library; artifacts are written below work/phase6.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler"))
from scheduler.contract import (Buffer, Certificate, IllegalSchedule, Memory,
                                Problem, Reservation, Resource, Task)
from scheduler.search import exact_search
from scheduler.verify import verify


def make_problem(capacity=40, accumulator_bytes=32, accumulator_memory="scratch", ports=1):
    """Eight elementwise outputs, explicit accumulator and quantization nodes.

    Tick durations and access windows are deliberately synthetic. The initial
    input and constant parameters are preloaded; only the next input's DMA is
    included, identically for every policy. The next input remains live at exit.
    """
    return Problem(
        (Memory("scratch", capacity), Memory("accumulator_registers", 32), Memory("external", 8)),
        (Resource("engine", 1), Resource("dma", 1), Resource("scratch_port", ports)),
        (Buffer("input", "scratch", 8, 8, initial=True),
         Buffer("acc32", accumulator_memory, accumulator_bytes, 8),
         Buffer("q8", "scratch", 8, 8),
         Buffer("output", "scratch", 8, 8, retain=True),
         Buffer("next_source", "external", 8, 8, initial=True),
         Buffer("next_input", "scratch", 8, 8, retain=True)),
        (Task("mac", "compute", 4, ("input",), ("acc32",),
              "eight-i32:127*(x+37)+300",
              (Reservation("engine", 0, 4), Reservation("scratch_port", 0, 2))),
         Task("quant", "requantize", 2, ("acc32",), ("q8",),
              "round-away(acc/256)-11:sat-i8",
              (Reservation("engine", 0, 2), Reservation("scratch_port", 1, 2))),
         Task("consume", "compute", 1, ("q8",), ("output",),
              "round-away(3*(q+11)/2)+5:sat-i8",
              (Reservation("engine", 0, 1), Reservation("scratch_port", 0, 1))),
         Task("prefetch", "dma", 2, ("next_source",), ("next_input",),
              "exact-eight-byte-copy",
              (Reservation("dma", 0, 2), Reservation("scratch_port", 0, 2)))))


def independent_oracle(problem, limit=9):
    """Enumerate all starts and aligned placements without search/verifier code."""
    tasks = {t.name: t for t in problem.tasks}
    memories = {m.name: m.capacity for m in problem.memories}
    resources = {r.name: r.capacity for r in problem.resources}
    duration = {t.name: t.duration for t in problem.tasks}
    tried = 0
    for horizon in range(1, limit + 1):
        for mac, quant, consume, prefetch in itertools.product(range(horizon), repeat=4):
            starts = dict(mac=mac, quant=quant, consume=consume, prefetch=prefetch)
            ends = {n: starts[n]+duration[n] for n in starts}
            if max(ends.values()) != horizon or ends["mac"] > quant or ends["quant"] > consume:
                continue
            tried += 1
            occupancy = {n: [0]*horizon for n in resources}
            for n, task in tasks.items():
                for r in task.reservations:
                    for tick in range(starts[n]+r.begin, starts[n]+r.end):
                        occupancy[r.resource][tick] += r.units
            if any(max(v) > resources[n] for n, v in occupancy.items()):
                continue
            life = {}
            for b in problem.buffers:
                writers = [n for n, t in tasks.items() if b.name in t.writes]
                begin = starts[writers[0]] if writers else 0
                readers = [n for n, t in tasks.items() if b.name in t.reads]
                end = horizon if b.retain else max([ends[n] for n in writers+readers]+[0])
                life[b.name] = (begin, end)
            # Aggregate capacity is only a rejection bound, never a placement proof.
            if any(sum(b.size for b in problem.buffers if b.memory == m and
                       life[b.name][0] <= tick < life[b.name][1]) > cap
                   for m, cap in memories.items() for tick in range(horizon)):
                continue
            buffers = sorted(problem.buffers, key=lambda b: (-b.size, b.name))
            occupied = {n: [0]*horizon for n in memories}
            offsets = {}

            def place(index):
                if index == len(buffers):
                    return dict(offsets)
                b = buffers[index]
                begin, end = life[b.name]
                for offset in range(0, memories[b.memory]-b.size+1, b.alignment):
                    mask = ((1 << b.size)-1) << offset
                    cells = occupied[b.memory]
                    if any(cells[tick] & mask for tick in range(begin, end)):
                        continue
                    for tick in range(begin, end):
                        cells[tick] |= mask
                    offsets[b.name] = offset
                    result = place(index+1)
                    if result is not None:
                        return result
                    for tick in range(begin, end):
                        cells[tick] ^= mask
                return None

            offsets = place(0)
            if offsets is not None:
                return {"objective_ticks": horizon, "starts": starts, "offsets": offsets,
                        "dependency_legal_timelines_examined": tried}
    raise AssertionError("example is not feasible within declared horizon")


def arithmetic_checks():
    def quant(n, shift, zp):
        q, r = divmod(abs(n), 1 << shift)
        q += int(2*r >= 1 << shift)
        return min(127, max(-128, (-q if n < 0 else q)+zp))

    checks, failures = 0, []
    for x in range(-128, 128):
        centered = 127*(x+37)+300
        corrected_bias = 300+37*127
        raw = 127*x+corrected_bias
        assert centered == raw and -(1 << 31) <= raw < 1 << 31
        q = quant(raw, 8, -11)
        final = quant(3*(q+11), 1, 5)
        narrow = quant(min(127, max(-128, raw)), 8, -11)
        if narrow != q:
            failures.append({"input": x, "int32": raw, "q8": q,
                             "illegally_narrowed_q8": narrow, "final": final})
        checks += 1
    assert failures
    return {"all_int8_inputs_checked": checks,
            "raw_corrected_equals_centered": True,
            "premature_i8_narrowing_mismatches": len(failures),
            "first_mismatch": failures[0],
            "scope": "scalar arithmetic proof; no RTL latency claim"}


def inspect_frozen_engine():
    base = ROOT / "docs/research/evidence/phase6/novelty-nonstream-v1"
    manifest = json.loads((base / "manifest.json").read_text())
    path = "work/phase6/pool-timing-v1/engine.sv"
    # Use the sealed source so this check does not depend on untracked work files.
    archive = base / "artifacts" / (path + ".gz")
    raw = gzip.decompress(archive.read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    # The archive manifest schema is deliberately not otherwise coupled here.
    assert digest in json.dumps(manifest)
    source = raw.decode()
    assert "pw_acc[0:7],pw_next[0:7]" in source
    assert "acc<=pw_acc[1][31:0]" in source and "Q_STREAM:begin" in source
    lines = source.splitlines()
    selected = [{"line": i+1, "text": s.strip()} for i, s in enumerate(lines)
                if "pw_acc[0:7],pw_next[0:7]" in s or "acc<=pw_acc[1][31:0]" in s]
    return {"archive": str(archive.relative_to(ROOT)), "sha256": digest,
            "lines": selected,
            "finding": "Selected spatial accumulators are internal registers through requantization; no descriptor-visible INT32 spill/reload operation is established.",
            "qualification": "Source inspection only; register storage does not prove the toy access durations are physical."}


def run(output):
    scenarios = {
        "wrong_int8_scratch_40": make_problem(accumulator_bytes=8),
        "hypothetical_int32_scratch_40": make_problem(),
        "hypothetical_int32_scratch_48": make_problem(capacity=48),
        "hypothetical_int32_scratch_40_two_ports": make_problem(ports=2),
        "current_style_register_accumulator": make_problem(accumulator_memory="accumulator_registers"),
    }
    results = {}
    output.mkdir(parents=True, exist_ok=True)
    for name, problem in scenarios.items():
        found = exact_search(problem, 9, max_expansions=2_000_000, seconds=20)
        assert found["optimal"], (name, found)
        checked = verify(problem, found["certificate"])
        independent = independent_oracle(problem)
        assert checked["makespan_ticks"] == independent["objective_ticks"]
        oracle_cert = Certificate(problem.digest(), independent["starts"], independent["offsets"])
        verify(problem, oracle_cert)
        results[name] = {"problem": asdict(problem), "certificate": asdict(found["certificate"]),
                         "verification": checked, "independent_oracle": independent,
                         "search_expansions": found["expansions"]}
    wrong = results["wrong_int8_scratch_40"]["certificate"]
    actual = scenarios["hypothetical_int32_scratch_40"]
    rejected = Certificate(actual.digest(), wrong["starts"], wrong["offsets"])
    try:
        verify(actual, rejected)
    except IllegalSchedule as e:
        rejection = str(e)
    else:
        raise AssertionError("INT8-only certificate incorrectly accepted for INT32")
    report = {
        "schema": 1,
        "status": "passed_negative_research_screen",
        "hypothesis": "Explicit accumulator lifetime and port reservations create a scheduling capability unavailable to a suitably expanded existing schedule/allocation policy.",
        "decision": "reject_as_novelty_evidence",
        "units": "synthetic ticks, not board cycles",
        "arithmetic": arithmetic_checks(), "frozen_engine": inspect_frozen_engine(),
        "scenarios": results, "invalid_narrow_certificate_rejection": rejection,
        "interpretation": [
            "Using final INT8 bytes for an INT32 temporary understates live storage and can admit illegal early DMA.",
            "An exact fixed-task joint schedule/allocation oracle accounts for the interaction without a new algorithm.",
            "The current engine holds these accumulators in registers; treating them as scratchpad residents invents a current bottleneck.",
            "Neither DeFiNES nor COSMA was executed by this script; its oracle is a representability counterargument, not a named-baseline result.",
            "A published-policy gap is not established; do not extrapolate toy ticks to KWS/VWW latency."
        ],
        "next_hypothesis": {
            "claim": "A reusable executable certificate and compact live-frontier representation may reduce search cost while preserving near-optimal legal schedules.",
            "gate": "First compare equal executable catalogues and accurate resource costs; require exact-small-instance agreement, held-out bounded search quality and end-to-end measured benefit over tuned baselines.",
            "status": "proposal_only_not_tested_here"
        },
        "sources": [
            {"title": "DeFiNES pinned source and prior code audit", "path": "docs/research/evidence/phase0/prior-code.json", "revision": "7097d6090dc22321e44ce91434e7cc23b065864f"},
            {"title": "COSMA III-A through III-C", "url": "https://arxiv.org/html/2311.18246v1", "finding": "Atomic operator timesteps and contiguous live-tensor placement; expanding a quantize node and accumulator edge is conceptually possible. Continuous port timing would be an adaptation, not a reproduced result."}
        ],
    }
    sources = [Path(__file__), ROOT / "compiler/scheduler/contract.py", ROOT / "compiler/scheduler/search.py", ROOT / "compiler/scheduler/verify.py"]
    report["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"status": report["status"], "decision": report["decision"],
                      "objectives_ticks": {k: v["verification"]["makespan_ticks"] for k, v in results.items()},
                      "report": str(output / "report.json")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "work/phase6/hypothesis-v1")
    run(parser.parse_args().output.resolve())
