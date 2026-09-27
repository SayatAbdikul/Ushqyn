#!/usr/bin/env python3
"""Reproduce the exact 4-pixel x 2-channel pointwise expansion gate.

This is a lane-scheduling bound, not an RTL, route, or physical speed claim.
The existing independent INT8-oracle prototype establishes arithmetic
equivalence; this script counts only the remaining nonconstant filters.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

from variants import ROOT, check_frozen, sha
from run_boardless import load_model
from run_screening import save_json

OUTPUT = ROOT / "docs/research/evidence/phase6/dual-lane-gate-v1.json"
ALGORITHM = ROOT / "work/phase6/experiments-v1/algorithms/report.json"
CONSTANT_FIXTURES = ROOT / "work/phase6/constant-sibling-v1/fixtures.json"
CONSTANT_SCREEN = ROOT / "work/phase6/constant-screen-v1/report.json"
ROUTE = ROOT / "docs/research/evidence/phase6/combined-next-v1.json"


def adjacent_pairs(active):
    """Pair only neighboring live channels within each contiguous run."""
    result = run = 0
    for live in (*active, False):
        if live:
            run += 1
        else:
            result += run // 2
            run = 0
    return result


def mac_steps(area, reductions, live, pairs):
    """One cycle per eight products, with unmatched channels on 8x1."""
    assert 0 <= 2 * pairs <= live
    return reductions * (pairs * math.ceil(area / 4)
                         + (live - 2 * pairs) * math.ceil(area / 8))


def verify_prototype(report, model, layer_index, output_bytes, dense_macs):
    rows = {(row["sample"], row["layer"], row["mode"]): row
            for row in report["models"][model]["operators"]}
    for sample in ("pinned", "seeded-stress"):
        for mode in ("pointwise-8x1-padded", "pointwise-4x2-padded"):
            row = rows[sample, layer_index, mode]
            assert row["status"] == "passed-arithmetic"
            assert row["output_bytes"] == output_bytes
            assert row["useful_mac_products"] == dense_macs


def run():
    check_frozen()
    prototype = json.loads(ALGORITHM.read_text())
    if (prototype["status"] != "passed-prototype-arithmetic" or
            prototype["source_sha256"] != sha(ROOT / "tools/phase6/algorithm_experiments.py")):
        raise ValueError("Exact 4x2 arithmetic prototype is stale")
    fixtures = json.loads(CONSTANT_FIXTURES.read_text())
    screened = json.loads(CONSTANT_SCREEN.read_text())
    route = json.loads(ROUTE.read_text())
    if screened["status"] != "passed-short-screen" or screened["output_mismatches"]:
        raise ValueError("Constant-filter physical baseline is invalid")
    if route["status"] != "passed-boardless-route":
        raise ValueError("Resource reference is not a routed design")
    clock = route["route"]["target_clock_mhz"]
    models = {}
    skipped_dense_macs = {}
    for name in ("kws", "vww"):
        program, _, _, pins = load_model(name)
        layers = []
        totals = {"dense_8x1_mac_steps": 0, "live_8x1_mac_steps": 0,
                  "live_4x2_any_pair_mac_steps": 0,
                  "live_4x2_adjacent_pair_mac_steps": 0,
                  "8x1_weight_byte_selection_events": 0,
                  "4x2_weight_byte_selection_events": 0,
                  "live_useful_macs": 0, "constant_removed_dense_macs": 0,
                  "output_bytes_unchanged": 0}
        for index, layer in enumerate(program.layers):
            if layer.op != "Conv" or layer.parameters["weight"].shape[2:] != (1, 1):
                continue
            weight = layer.parameters["weight"]
            if layer.attributes.get("group", 1) != 1:
                continue
            output = program.tensors[layer.output].shape
            area, reductions, channels = int(np.prod(output[2:])), weight.shape[1], weight.shape[0]
            active = (~np.all(weight == 0, axis=(1, 2, 3))).tolist()
            live = sum(active)
            pairs_any, pairs_adjacent = live // 2, adjacent_pairs(active)
            dense_macs = area * reductions * channels
            verify_prototype(prototype, name, index, area * channels, dense_macs)
            dense_steps = mac_steps(area, reductions, channels, 0)
            live_steps = mac_steps(area, reductions, live, 0)
            any_steps = mac_steps(area, reductions, live, pairs_any)
            adjacent_steps = mac_steps(area, reductions, live, pairs_adjacent)
            dual_weight_selections = reductions * (2 * pairs_any * math.ceil(area / 4)
                                                   + (live - 2 * pairs_any) * math.ceil(area / 8))
            entry = {
                "layer": index, "spatial_pixels": area, "input_channels": reductions,
                "output_channels": channels, "live_channels": live,
                "constant_channels": channels - live, "arbitrary_live_pairs": pairs_any,
                "adjacent_live_pairs": pairs_adjacent,
                "dense_8x1_mac_steps": dense_steps, "live_8x1_mac_steps": live_steps,
                "live_4x2_any_pair_mac_steps": any_steps,
                "live_4x2_adjacent_pair_mac_steps": adjacent_steps,
                "8x1_weight_byte_selection_events": live_steps,
                "4x2_weight_byte_selection_events": dual_weight_selections,
                "optimistic_saved_steps": live_steps - any_steps,
                "adjacent_saved_steps": live_steps - adjacent_steps,
            }
            layers.append(entry)
            for key in totals:
                if key in entry:
                    totals[key] += entry[key]
            totals["live_useful_macs"] += area * reductions * live
            totals["constant_removed_dense_macs"] += area * reductions * (channels - live)
            totals["output_bytes_unchanged"] += area * channels
        skipped_dense_macs[name] = totals["constant_removed_dense_macs"]
        saved = totals["live_8x1_mac_steps"] - totals["live_4x2_any_pair_mac_steps"]
        adjacent_saved = totals["live_8x1_mac_steps"] - totals["live_4x2_adjacent_pair_mac_steps"]
        latency = screened["summary"][name]["median_ms"]
        saved_ms = saved / (1000 * clock)
        models[name] = {"layers": layers, "totals": totals,
                        "ideal_arbitrary_pair_step_saving": saved,
                        "adjacent_pair_step_saving": adjacent_saved,
                        "ideal_arbitrary_pair_ms_at_target_clock": saved_ms,
                        "physical_constant_screen_median_ms": latency,
                        "speedup_if_only_these_steps_disappeared": latency / (latency - saved_ms),
                        "boundary": "No credit for cache, layout, control, parameter, output, or clock effects."}
    # Check the live-filter count against the independently generated lowering.
    constant_case = next(case for case in fixtures["fixtures"] if case["name"] == "vww-pinned-constant-sibling-timed")
    if constant_case["constants"]["skipped_dense_macs"] != skipped_dense_macs["vww"]:
        raise ValueError("Constant-filter compiler and model weights disagree")
    assert skipped_dense_macs["kws"] == 0
    geometric = math.prod(models[name]["speedup_if_only_these_steps_disappeared"]
                          for name in ("kws", "vww")) ** 0.5
    resources = route["route"]["resources"]
    available = route["route"]["available"]
    free_bsram = available["BSRAM"] - resources["BSRAM"]
    free_cls = available["CLS"] - resources["CLS"]
    timing_margin = route["route"]["routed_fmax_mhz"] - clock
    report = {
        "schema": 1, "status": "no-go-for-4x2-lane-utilization-expansion",
        "scope": "Boardless arithmetic-step gate for frozen KWS/VWW with exact constant filters skipped; no 4x2 RTL or physical speedup claim.",
        "models": models,
        "geometric_mean_speedup_if_only_mac_step_saving": geometric,
        "previous_expansion_trigger_geometric_mean": 1.03,
        "decision": "The optimistic live-filter 4x2 MAC-step saving is far below the 3% two-model expansion trigger, while a dual weight/parameter path would add implementation and timing risk. Defer RTL/route/board for this mapping; padded planes and 16-MAC architecture are separate experiments.",
        "qualification": "The step calculation is a bound on lane-utilization benefit, not on all possible redesign side effects. It optimistically pairs arbitrary live channels; the adjacent-only case is reported separately. Weight-byte selection events are internal datapath demands, not SDRAM transactions. Output quantization and byte writes remain serialized.",
        "resource_reference": {
            "route_candidate": route["candidate"]["engine"],
            "target_clock_mhz": clock,
            "routed_fmax_mhz": route["route"]["routed_fmax_mhz"],
            "timing_margin_mhz": timing_margin,
            "free_cls": free_cls,
            "free_bsram": free_bsram,
            "used_dsp_equivalents": resources["DSP"],
            "free_dsp_equivalents": available["DSP"] - resources["DSP"],
        },
        "sixteen_mac_separate_gate": {
            "additional_products_per_cycle": 8,
            "hypothetical_dsp_equivalents_if_one_per_added_mac": resources["DSP"] + 8,
            "fits_raw_dsp_count_under_that_unverified_mapping": resources["DSP"] + 8 <= available["DSP"],
            "conclusion": ("Raw DSP count does not rule out 16 MACs, but this route has only "
                           f"{free_bsram} free BSRAM, {free_cls} free CLS and "
                           f"{timing_margin:.3f} MHz clock margin. A second weight stream, "
                           "delivery/control, accumulation, and routed timing need a separate "
                           "exact RTL prototype. A 4x2 mapping still uses only eight products "
                           "per cycle and is not a 16-MAC experiment."),
        },
        "provenance": {
            "script_sha256": sha(Path(__file__)),
            "algorithm_report_sha256": sha(ALGORITHM),
            "algorithm_source_sha256": sha(ROOT / "tools/phase6/algorithm_experiments.py"),
            "constant_fixture_sha256": sha(CONSTANT_FIXTURES),
            "constant_screen_sha256": sha(CONSTANT_SCREEN),
            "route_evidence_sha256": sha(ROUTE),
            "model_pins": {name: load_model(name)[3] for name in ("kws", "vww")},
        },
    }
    save_json(OUTPUT, report)
    check_frozen()
    print(json.dumps({"status": report["status"],
                      "kws_saved_steps": models["kws"]["ideal_arbitrary_pair_step_saving"],
                      "vww_saved_steps": models["vww"]["ideal_arbitrary_pair_step_saving"],
                      "geometric_mean": geometric,
                      "output": str(OUTPUT.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    run()
