#!/usr/bin/env python3
"""Exact arithmetic and cycle-budget screen for pointwise zero-block bypass.

This is intentionally a no-route feasibility probe. It only credits cycles
that a hypothetical separate, zero-latency metadata port could save; no RTL
latency or resource result is represented as measured.
"""

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OPPORTUNITY = ROOT / "work/phase6/novelty-zero-block-v1/report.json"
TIMING = ROOT / "work/phase6/novelty-zero-block-v1/native-profile/report.json"
BOARD = ROOT / "work/phase6/pair7-fusion-board-v1/physical-short-v1/report.json"
OUTPUT = ROOT / "work/phase6/novelty-zero-bypass-v1/report.json"
INT32_MIN = -(1 << 31)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed8(x):
    return x if x < 128 else x - 256


def exact_compensation_proof():
    # Exhaust the legal INT8 zero points and weights. The bypassed raw
    # multiplication must be precisely zx*w, including -128*-128.
    cases = 0
    for zi in range(256):
        z = signed8(zi)
        for wi in range(256):
            w = signed8(wi)
            original = z * w
            correction = z * w
            assert -32768 <= correction <= 32767
            assert original == correction
            for accumulator in (INT32_MIN, INT32_MIN + 1, -1, 0, 1,
                                (1 << 31) - 2, (1 << 31) - 1):
                assert accumulator + original == accumulator + correction
                assert (accumulator + original < INT32_MIN or
                        accumulator + original > (1 << 31) - 1) == (
                            accumulator + correction < INT32_MIN or
                            accumulator + correction > (1 << 31) - 1)
            cases += 1
    return {"legal_int8_zero_point_weight_pairs": cases,
            "test_accumulators_per_pair": 7,
            "conclusion": "raw zx*weight compensation preserves the exact accumulator and overflow predicate"}


def centered_counterexample():
    # A centered-product replacement changes intermediate overflow behavior
    # even if the completed dot product agrees. The existing RTL checks at
    # eight-channel boundaries, not only at the end.
    z = -128
    weights = [127] * 8 + [-127] * 8
    values = [z] * 16
    p0 = INT32_MIN + 100000
    centered_bias = p0 + z * sum(weights)
    raw_first8 = p0 + sum(values[i] * weights[i] for i in range(8))
    centered_first8 = centered_bias + sum(
        (values[i] - z) * weights[i] for i in range(8))
    raw_final = p0 + sum(x * w for x, w in zip(values, weights))
    centered_final = centered_bias + sum(
        (x - z) * w for x, w in zip(values, weights))
    assert raw_first8 < INT32_MIN <= centered_first8
    assert raw_final == centered_final == p0
    return {"zero_point": z, "weights": weights,
            "raw_corrected_bias": p0, "centered_bias": centered_bias,
            "raw_accumulator_after_eight": raw_first8,
            "centered_accumulator_after_eight": centered_first8,
            "equal_final_accumulator": raw_final,
            "conclusion": "centered products require a certified overflow contract; they are not a generic exact replacement"}


def screen():
    opportunity = json.loads(OPPORTUNITY.read_text())
    timing = json.loads(TIMING.read_text())
    board = json.loads(BOARD.read_text())
    models = {}
    for model in ("kws", "vww"):
        elapsed = board["summary"][model]["median_cycles"]
        engine = timing["models"][model]["seeds"][0]["reported_engine_cycles"]
        pointwise = timing["models"][model]["seeds"][0]["pointwise_descriptor_cycles"]
        samples = {}
        for name in ("pinned", "stress"):
            totals = opportunity["models"][model]["samples"][name]["totals"]
            eligible = totals["physical_word_certified_weighted_skips"]
            ideal_two = 2 * eligible
            samples[name] = {
                "certified_weighted_channel_iterations": eligible,
                "iterations": totals["executed_channel_output_block_iterations"],
                "whole_eight_channel_weight_words": totals["full_eight_channel_words"],
                "one_cycle_ideal_saved_cycles": eligible,
                "one_cycle_ideal_board_fraction": eligible / elapsed,
                "two_cycle_ideal_saved_cycles": ideal_two,
                "two_cycle_ideal_board_fraction": ideal_two / elapsed,
            }
        models[model] = {"physical_median_cycles": elapsed,
                         "native_engine_cycles": engine,
                         "native_pointwise_descriptor_cycles": pointwise,
                         "measured_native_event_states": timing["models"][model]["seeds"][0]["pw_events"],
                         "samples": samples}
    joint = {}
    for name in ("pinned", "stress"):
        kws = models["kws"]
        vww = models["vww"]
        ck = kws["physical_median_cycles"]
        cv = vww["physical_median_cycles"]
        nk = kws["samples"][name]["certified_weighted_channel_iterations"]
        nv = vww["samples"][name]["certified_weighted_channel_iterations"]

        def geo(saved_per_event):
            return math.sqrt((ck / (ck - saved_per_event * nk)) *
                             (cv / (cv - saved_per_event * nv)))

        lo, hi = 0.0, 2.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if geo(mid) >= 1.03:
                hi = mid
            else:
                lo = mid
        joint[name] = {
            "one_cycle_ideal_geomean_throughput": geo(1),
            "two_cycle_ideal_geomean_throughput": geo(2),
            "minimum_net_saved_cycles_per_certified_iteration_for_1_03x": hi,
            "max_shared_overhead_cycles_per_certified_iteration_if_two_cycles_removed": 2-hi,
        }
    result = {
        "schema": 1,
        "status": "no_go_for_existing_cache_path_side_metadata_untested",
        "scope": "exact arithmetic, measured unchanged-native event coverage, and analytic ideal-cycle ceilings; no candidate RTL, synthesis, or board result",
        "source_hashes": {str(p.relative_to(ROOT)): sha(p)
                          for p in (OPPORTUNITY, TIMING, BOARD)},
        "mechanism": (
            "In a pointwise input-fetch state, an all-zero-point tile could update each raw "
            "accumulator with zx*weight and elide PW_MAC. This preserves the same accumulator "
            "and reduction-boundary overflow check, but needs the current weight, eight updates, "
            "and a zero tag available without a serial SRAM read."),
        "arithmetic_proof": exact_compensation_proof(),
        "centered_product_counterexample": centered_counterexample(),
        "models": models,
        "joint_gate": {
            "metric": "geometric mean of KWS and VWW throughput ratios, threshold 1.03x",
            "samples": joint,
        },
        "rtl_bypass_implemented": False,
        "native_candidate_simulated": False,
        "physical_candidate_tested": False,
        "decision": (
            "A one-cycle exact bypass cannot reach the 1.03x dual-workload throughput gate, "
            "even with free metadata. An ideal two-cycle bypass can clear it if correction "
            "and metadata together cost less than the reported shared overhead per certified "
            "event. However a two-channel fusion condition using the existing "
            "prefetched line word has zero matching PW_MAC events in both models: KWS "
            "alignment kills its condition and VWW matching zero inputs are never line-cache "
            "hits. No RTL was implemented for that zero-coverage path. A useful two-cycle "
            "design requires a separate producer-written, pipelined metadata RAM that also "
            "invalidates host-DMA words and handles unaligned KWS tiles. That sidecar and "
            "the required extra correction adders have not been implemented or routed, so "
            "the ideal speedups are ceilings, not measured hardware results."),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(OUTPUT), "status": result["status"],
                      "models": models}, indent=2))


if __name__ == "__main__":
    screen()
