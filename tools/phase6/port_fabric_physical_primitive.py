"""Isolated route screen for independently addressed 32-bit scratchpad groups.

This intentionally changes read behavior on a live engine address condition.
It is a physical primitive screen, not an inference-correct accelerator.
Only the scratchpad and its immediate wiring differ from the routed co-issue
image. No board, native simulation, or large workload is run here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path("/Users/sayat/.codex/worktrees/21f0/tinyML_accelerator")
OUT = ROOT / "work/phase6/port-fabric-physical-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"expected one replacement, found {source.count(old)}: {old[:80]}")
    return source.replace(old, new)


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_scratch = SOURCE_ROOT / "rtl/v2/scratchpad.sv"
    source_core = SOURCE_ROOT / "rtl/v2/tiled_core.sv"
    source_build = SOURCE_ROOT / "work/phase6/engine-candidate-rtl-v2/build27.tcl"
    source_engine = SOURCE_ROOT / "work/phase6/engine-candidate-rtl-v2/engine.sv"
    source_route = SOURCE_ROOT / "work/phase6/engine-candidate-rtl-v2/route27/report.json"
    for path in (source_scratch, source_core, source_build, source_engine, source_route):
        if not path.is_file():
            raise FileNotFoundError(path)

    scratch = source_scratch.read_text()
    scratch = replace_once(
        scratch,
        "    input logic [23:0] addr,\n",
        "    input logic [23:0] addr,\n"
        "    input logic split_read,\n"
        "    input logic [23:0] split_lo_addr, split_hi_addr,\n",
    )
    scratch = replace_once(
        scratch,
        "        (* syn_ramstyle = \"block_ram\" *) logic [7:0] ram[0:MEM_BYTES/8-1];\n"
        "        always_ff @(posedge clk) begin\n",
        "        (* syn_ramstyle = \"block_ram\" *) logic [7:0] ram[0:MEM_BYTES/8-1];\n"
        "        wire [WORD_BITS-1:0] group_index = (split_read && !wr) ?\n"
        "            (lane < 4 ? split_lo_addr[WORD_BITS+2:3] : split_hi_addr[WORD_BITS+2:3]) :\n"
        "            addr[WORD_BITS+2:3];\n"
        "        always_ff @(posedge clk) begin\n",
    )
    scratch = replace_once(
        scratch,
        "ram[addr[WORD_BITS+2:3]]",
        "ram[group_index]",
    ) if scratch.count("ram[addr[WORD_BITS+2:3]]") == 1 else scratch.replace(
        "ram[addr[WORD_BITS+2:3]]", "ram[group_index]"
    )
    if scratch.count("ram[group_index]") != 2:
        raise ValueError("expected exactly two bank accesses")

    core = source_core.read_text()
    core = replace_once(
        core,
        "        .clk(clk), .rst_n(rst_n), .req(sreq), .wr(swr), .addr(saddr),\n",
        "        .clk(clk), .rst_n(rst_n), .req(sreq), .wr(swr), .addr(saddr),\n"
        "        .split_read(grant_engine && !ewr && eaddr[3]),\n"
        "        .split_lo_addr(eaddr), .split_hi_addr(daddr),\n",
    )

    (OUT / "scratchpad_2x32.sv").write_text(scratch)
    (OUT / "tiled_core_2x32.sv").write_text(core)

    build = source_build.read_text()
    build = replace_once(
        build,
        f"set root {{{SOURCE_ROOT}}}",
        f"set root {{{SOURCE_ROOT}}}\nset physical {{{OUT}}}",
    )
    build = replace_once(
        build,
        "    rtl/v2/scratchpad.sv\n",
        "",
    )
    build = replace_once(
        build,
        "    rtl/v2/tiled_core.sv\n",
        "",
    )
    build = replace_once(
        build,
        "add_file $::env(PHASE6_ENGINE)\n",
        f"add_file [file join $physical scratchpad_2x32.sv]\n"
        f"add_file [file join $physical tiled_core_2x32.sv]\n"
        f"add_file [file join $root work/phase6/engine-candidate-rtl-v2/engine.sv]\n",
    )
    (OUT / "build27.tcl").write_text(build)

    route = OUT / "route27"
    route.mkdir(exist_ok=True)
    inputs = {
        "scope": "nonfunctional physical primitive; independently addressed 2x32-bit scratchpad read groups",
        "base_route": str(source_route),
        "base_route_sha256": sha(source_route),
        "source_engine_sha256": sha(source_engine),
        "source_scratchpad_sha256": sha(source_scratch),
        "source_core_sha256": sha(source_core),
        "candidate_scratchpad_sha256": sha(OUT / "scratchpad_2x32.sv"),
        "candidate_core_sha256": sha(OUT / "tiled_core_2x32.sv"),
        "build_sha256": sha(OUT / "build27.tcl"),
        "run": [
            "cd " + str(route),
            "/Applications/GowinIDE.app/Contents/Resources/Gowin_EDA/IDE/bin/gw_sh "
            + str(OUT / "build27.tcl"),
        ],
    }
    (OUT / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
    print(json.dumps(inputs, indent=2))


def report() -> None:
    source_route = SOURCE_ROOT / "work/phase6/engine-candidate-rtl-v2/route27/report.json"
    base = json.loads(source_route.read_text())
    pnr = OUT / "route27/phase6_uart_burst/impl/pnr"
    routed_file = pnr / "phase6_uart_burst.rpt.txt"
    if not routed_file.is_file():
        raise FileNotFoundError(routed_file)
    routed = routed_file.read_text()
    resources: dict[str, dict[str, float]] = {}
    for key in ("CLS", "BSRAM", "DSP", "Logic", "Register"):
        match = re.search(rf"^\s*{key}\s*\|\s*([\d.]+)/([\d.]+)", routed, re.M)
        if not match:
            raise ValueError(f"missing routed resource {key}")
        resources[key] = {"used": float(match[1]), "available": float(match[2])}
    fmax = re.search(r"Fmax\s*=\s*([\d.]+)\s*MHz", routed)
    if not fmax:
        raise ValueError("missing routed Fmax")
    fmax_mhz = float(fmax[1])
    original = base["resources"]
    result = {
        "status": "passed-route" if fmax_mhz >= 27 and all(
            resources[k]["used"] <= resources[k]["available"] for k in resources
        ) else "failed-fit-or-timing",
        "scope": "nonfunctional physical primitive only; no model outputs or latency claim",
        "base": {"fmax_mhz": base["routed_core_fmax_mhz"], "resources": original},
        "candidate": {"fmax_mhz": fmax_mhz, "resources": resources},
        "delta": {
            "fmax_mhz": fmax_mhz - base["routed_core_fmax_mhz"],
            **{k: resources[k]["used"] - original[k]["used"] for k in resources},
        },
        "route_report": str(routed_file),
        "route_report_sha256": sha(routed_file),
        "inputs_sha256": sha(OUT / "inputs.json"),
    }
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "report"))
    action = parser.parse_args().action
    prepare() if action == "prepare" else report()
