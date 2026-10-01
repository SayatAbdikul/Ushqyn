#!/usr/bin/env python3
"""Bounded, exact range screen for hypothetical stored MAC continuation state.

The production engine keeps its 32-bit accumulator in a register. This script
does not change RTL or claim a routed resource saving. It computes tight
coefficient-wise interval bounds for every reduction prefix of each frozen
INT8 MAC and evaluates conservative 64-bit-port storage layouts.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler"))
from program_image import load_image
from quantization import requantize
from static_pipeline import compile_static, execute_layer

OUT = ROOT / "work/phase6/parallel-partial-sum-width-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def align8(n: int) -> int:
    return (n + 7) & ~7


def signed_width(low: int, high: int) -> int:
    assert low <= high
    w = 1
    while low < -(1 << (w - 1)) or high > (1 << (w - 1)) - 1:
        w += 1
    return w


def lane_bytes(width: int) -> int:
    """Native byte lanes keep each signed value inside one 64-bit word."""
    for n in (1, 2, 4):
        if width <= 8 * n:
            return n
    raise AssertionError(f"INT32 contract violated: {width} bits")


def interval_prefix(weight: np.ndarray, bias: int,
                    xlow: int = -128, xhigh: int = 127) -> tuple[np.ndarray, np.ndarray]:
    """Exact extrema for independent input codes, including prefix zero."""
    weight = np.asarray(weight, dtype=np.int64).reshape(-1)
    first = weight * xlow
    second = weight * xhigh
    low = np.concatenate(([bias], bias + np.cumsum(np.minimum(first, second))))
    high = np.concatenate(([bias], bias + np.cumsum(np.maximum(first, second))))
    return low, high


def brute_force_proof() -> dict:
    """Check every prefix and code combination of a genuinely exhaustive case."""
    weight = np.array([2, -3, 0, 1], dtype=np.int8)
    bias = -7
    codes = tuple(range(-2, 3))
    low, high = interval_prefix(weight, bias, codes[0], codes[-1])
    checked = 0
    for prefix in range(len(weight) + 1):
        values = [bias + sum(int(x) * int(w) for x, w in zip(xs, weight[:prefix]))
                  for xs in itertools.product(codes, repeat=prefix)]
        assert min(values) == int(low[prefix]) and max(values) == int(high[prefix])
        width = signed_width(int(low[prefix]), int(high[prefix]))
        assert all(-(1 << (width - 1)) <= v < (1 << (width - 1)) for v in values)
        checked += len(values)
    for width in range(1, 33):
        for contexts in range(1, 66):
            assert straddle_count(width, contexts) == sum(
                ((i * width) % 64) + width > 64 for i in range(contexts))
    return {"weights": weight.tolist(), "bias": bias, "codes": list(codes),
            "prefixes": len(weight) + 1, "exhaustive_assignments": checked,
            "low": low.tolist(), "high": high.tolist(),
            "straddle_formula_cases": 32 * 65, "passed": True}


def load_programs() -> dict:
    results = {}
    for model in ("kws", "vww"):
        source = ROOT / f"work/phase4/{model}-logits.onnx"
        calibration = ROOT / f"work/phase4/{model}-calibration-rebased.json"
        fixture = json.loads((ROOT / f"work/phase4/rtl-{model}/manifest.json").read_text())
        if sha(source) != fixture["source_onnx_sha256"] or sha(calibration) != fixture["calibration_sha256"]:
            raise ValueError(f"{model}: source/calibration differs from frozen physical fixture")
        program = compile_static(onnx.load(source), json.loads(calibration.read_text()))
        results[model] = (program, {"source_sha256": sha(source),
                                    "calibration_sha256": sha(calibration),
                                    "frozen_fixture": str((ROOT / f"work/phase4/rtl-{model}/manifest.json").relative_to(ROOT))})
    ad_image = ROOT / "work/phase6/parallel-ad-v1/ad.uq2"
    if ad_image.is_file():
        results["ad"] = (load_image(ad_image.read_bytes()),
                          {"source_sha256": sha(ROOT / "work/ad-float.onnx"),
                           "software_image_sha256": sha(ad_image),
                           "scope": "new held-out AD static-INT8 image; see parallel-ad-v1"})
    return results


def store_cost(widths: list[int], spatial: int, kind: str) -> dict:
    channels = len(widths)
    if kind == "always32":
        logical_bits = 32 * channels * spatial
        data_bytes = align8(4 * channels * spatial)
        metadata_bytes = 0
    elif kind == "layer_static":
        maxwidth = max(widths)
        logical_bits = maxwidth * channels * spatial
        data_bytes = align8(lane_bytes(maxwidth) * channels * spatial)
        metadata_bytes = 8  # one width/address descriptor in SRAM
    elif kind in ("channel_static", "prefix_phase"):
        logical_bits = sum(widths) * spatial
        # Each channel plane has a 64-bit-aligned base; signed lanes are 8/16/32.
        data_bytes = sum(align8(lane_bytes(w) * spatial) for w in widths)
        # Width byte plus 32-bit base per channel, stored in aligned arrays.
        metadata_bytes = align8(channels) + align8(4 * channels)
    else:
        raise ValueError(kind)
    total_bytes = align8(data_bytes + metadata_bytes)
    if kind in ("always32", "layer_static"):
        dense_data_bytes = align8(math.ceil(logical_bits / 8))
        dense_straddles = straddle_count(widths[0], channels * spatial)
    else:
        # A channel plane starts at an aligned 64-bit word; contexts can still
        # straddle later word boundaries and need multiple port transactions.
        dense_data_bytes = sum(align8(math.ceil(w * spatial / 8)) for w in widths)
        dense_straddles = sum(straddle_count(w, spatial) for w in widths)
    dense_total = align8(dense_data_bytes + metadata_bytes)
    return {"logical_bits": logical_bits, "bit_tight_lower_bound_bytes": math.ceil(logical_bits / 8),
            "data_bytes_64bit_port": data_bytes, "metadata_bytes": metadata_bytes,
            "total_bytes_64bit_port": total_bytes,
            "fits_16k_scratch_region": total_bytes <= 16384,
            "fits_full_32k_scratch": total_bytes <= 32768,
            "hypothetical_separate_eight_lane_blocks": separate_blocks(total_bytes),
            "dense_bitpack": {"data_bytes_aligned8": dense_data_bytes,
                              "total_bytes_including_metadata": dense_total,
                              "contexts": channels * spatial,
                              "cross_word_contexts": dense_straddles,
                              "uncoalesced_read_write_port_transactions_per_checkpoint":
                                  2 * (channels * spatial + dense_straddles),
                              "fits_16k_scratch_region": dense_total <= 16384,
                              "fits_full_32k_scratch": dense_total <= 32768,
                              "hypothetical_separate_eight_lane_blocks": separate_blocks(dense_total)}}


def straddle_count(width: int, contexts: int) -> int:
    if contexts == 0:
        return 0
    period = 64 // math.gcd(64, width)
    count_per_period = sum((i * width) % 64 + width > 64 for i in range(period))
    cycles, rest = divmod(contexts, period)
    return cycles * count_per_period + sum((i * width) % 64 + width > 64 for i in range(rest))


def separate_blocks(total_bytes: int) -> int | None:
    """Conservative inferred-RTL mode, not a routed result for a new RAM.

    Eight byte-lane arrays and power-of-two depth as in the current 32-KiB
    scratchpad. The existing instance empirically uses 16 BSRAM blocks.
    """
    if total_bytes <= 0:
        return 0
    capacity = max(1024, 1 << (total_bytes - 1).bit_length())
    if capacity > 32768:
        return None
    return 8 * math.ceil(capacity / 18432)


def layer_screen(model: str, idx: int, layer, program) -> dict:
    weights = np.asarray(layer.parameters["weight"], dtype=np.int8)
    corr = layer.parameters["corrected_bias"]
    rows = weights.reshape(weights.shape[0], -1)
    channels, reduction = rows.shape
    output_shape = tuple(program.tensors[layer.output].shape)
    spatial = math.prod(output_shape) // channels
    all_widths = []
    final_widths = []
    phase_widths: dict[int, list[int]] = {t: [] for t in list(range(0, reduction, 8)) + [reduction]}
    overall_low, overall_high = None, None
    channels_requiring_32 = 0
    for c in range(channels):
        low, high = interval_prefix(rows[c], int(corr[c]))
        assert np.all(low >= -(1 << 31)) and np.all(high <= (1 << 31) - 1)
        widths = [signed_width(int(a), int(b)) for a, b in zip(low, high)]
        all_widths.append(max(widths))
        final_widths.append(widths[-1])
        channels_requiring_32 += max(widths) > 16
        overall_low = int(low.min()) if overall_low is None else min(overall_low, int(low.min()))
        overall_high = int(high.max()) if overall_high is None else max(overall_high, int(high.max()))
        for t in phase_widths:
            phase_widths[t].append(widths[t])
    layer_widths = [max(all_widths)] * channels
    layouts = {"always32": store_cost([32] * channels, spatial, "always32"),
               "layer_static": store_cost(layer_widths, spatial, "layer_static"),
               "channel_static": store_cost(all_widths, spatial, "channel_static")}
    phase = [(t, store_cost(ws, spatial, "prefix_phase"))
             for t, ws in phase_widths.items()]
    phase_best = min(phase[1:] if len(phase) > 1 else phase,
                     key=lambda pair: pair[1]["dense_bitpack"]["total_bytes_including_metadata"])
    phase_worst = max(phase, key=lambda pair: pair[1]["total_bytes_64bit_port"])
    # Width changes require all saved contexts to be read and rewritten once
    # at the checkpoint. This is a byte lower bound, not a cycle prediction.
    repack_bytes = sum(phase[i - 1][1]["data_bytes_64bit_port"] + phase[i][1]["data_bytes_64bit_port"]
                       for i in range(1, len(phase))
                       if phase_widths[phase[i - 1][0]] != phase_widths[phase[i][0]])
    return {"index": idx, "op": layer.op, "output": layer.output,
            "output_shape": list(output_shape), "channels": channels,
            "spatial_outputs_per_channel": spatial, "reduction": reduction,
            "global_safe_interval": [overall_low, overall_high],
            "safe_signed_bits": {"per_layer": max(all_widths),
                                 "per_channel_min": min(all_widths),
                                 "per_channel_max": max(all_widths),
                                 "per_channel_mean": round(float(np.mean(all_widths)), 4),
                                 "final_only_max": max(final_widths),
                                 "channels_over_16_bits": channels_requiring_32},
            "layouts": layouts,
            "prefix_phase": {"step_products": 8, "checkpoints": len(phase),
                             "best_checkpoint": phase_best[0], "best": phase_best[1],
                             "final_checkpoint": reduction, "final": phase[-1][1],
                             "worst_checkpoint": phase_worst[0], "worst": phase_worst[1],
                             "minimum_repacking_byte_traffic": repack_bytes,
                             "width_changes": sum(phase_widths[phase[i - 1][0]] != phase_widths[phase[i][0]]
                                                  for i in range(1, len(phase)))}}


def validate_real_weights(program, model: str) -> dict:
    """Seeded one-layer integer run and explicit raw-prefix check on actual weights."""
    layer_index = next(i for i, layer in enumerate(program.layers) if layer.op in ("Conv", "Gemm"))
    layer = program.layers[layer_index]
    input_name = program.inputs[0]
    rng = np.random.default_rng(20260928 + len(model))
    source = rng.integers(-128, 128, size=program.tensors[input_name].shape, dtype=np.int8)
    values = {input_name: source}
    for predecessor in program.layers[:layer_index]:
        values[predecessor.output] = execute_layer(program, predecessor, values)
    x = values[layer.inputs[0]]
    output = execute_layer(program, layer, values)
    w = layer.parameters["weight"]
    checks = []
    for c in sorted({0, w.shape[0] // 2, w.shape[0] - 1}):
        if layer.op == "Gemm":
            xv = x.reshape(-1).astype(np.int64)
            out_at = int(output[0, c])
        else:
            # Test the first spatial output, including any zero-point padding.
            a = layer.attributes
            sh, sw = a.get("strides", [1, 1])
            dh, dw = a.get("dilations", [1, 1])
            pt, pl, _, _ = a.get("pads", [0, 0, 0, 0])
            _, icg, kh, kw = w.shape
            group = a.get("group", 1)
            group_first = (c // (w.shape[0] // group)) * icg
            xv = []
            for ic in range(icg):
                for ky in range(kh):
                    for kx in range(kw):
                        iy, ix = -pt + ky * dh, -pl + kx * dw
                        xv.append(int(x[0, group_first + ic, iy, ix])
                                  if 0 <= iy < x.shape[2] and 0 <= ix < x.shape[3]
                                  else program.tensors[layer.inputs[0]].quantization.zero_point)
            xv = np.asarray(xv, dtype=np.int64)
            out_at = int(output[0, c, 0, 0])
        weight = w[c].reshape(-1).astype(np.int64)
        bias = int(layer.parameters["corrected_bias"][c])
        actual_prefix = bias + np.concatenate(([0], np.cumsum(xv * weight)))
        low, high = interval_prefix(weight, bias)
        assert np.all(actual_prefix >= low) and np.all(actual_prefix <= high)
        oq = program.tensors[layer.output].quantization
        expected = int(requantize(np.asarray([actual_prefix[-1]], dtype=np.int64),
                                  layer.parameters["multiplier"][c],
                                  layer.parameters["shift"][c], oq.zero_point)[0])
        assert expected == out_at
        checks.append({"channel": c, "prefixes_checked": len(actual_prefix),
                       "final_accumulator": int(actual_prefix[-1]),
                       "output_code": out_at})
    return {"model": model, "layer_index": layer_index, "input_kind": "seeded INT8",
            "checks": checks, "passed": True}


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    programs = load_programs()
    result = {"schema": 1,
              "scope": "analytical width/packing screen only; no RTL, route, or board measurement",
              "method": "exact interval extrema for corrected_bias + sum(raw_int8*weight_int8) at every reduction prefix; padded Conv positions are conservatively covered",
              "hardware_contract": {"current_accumulator_location": "32-bit engine register, not scratchpad RAM",
                                    "scratchpad": "32 KiB, one synchronous 64-bit request port, eight byte lanes",
                                    "current_scratchpad_bsram": 16,
                                    "selected_design_bsram": 38, "device_bsram": 46,
                                    "block_model": "hypothetical separate RAM with same eight-lane RTL organization, power-of-two depth, 18,432-bit primitive nominal capacity; not a routed result",
                                    "block_model_source": "compiler/memory_planner.py and observed routed 16-block 32-KiB scratchpad",
                                    "single_port_access": "one aligned 64-bit word per read or write; 8/16/32-bit lanes avoid cross-word element access"},
              "proof_test": brute_force_proof(), "model": {}, "real_weight_validation": []}
    for model, (program, provenance) in programs.items():
        layers = [layer_screen(model, i, layer, program)
                  for i, layer in enumerate(program.layers) if layer.op in ("Conv", "Gemm")]
        result["model"][model] = {"provenance": provenance,
                                  "mac_layers": len(layers), "layers": layers,
                                  "layers_with_per_channel_byte_saving": sum(
                                      l["layouts"]["channel_static"]["total_bytes_64bit_port"]
                                      < l["layouts"]["always32"]["total_bytes_64bit_port"]
                                      for l in layers),
                                  "layers_with_dense_per_channel_byte_saving": sum(
                                      l["layouts"]["channel_static"]["dense_bitpack"]["total_bytes_including_metadata"]
                                      < l["layouts"]["always32"]["total_bytes_64bit_port"]
                                      for l in layers)}
        result["real_weight_validation"].append(validate_real_weights(program, model))
    (OUT / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str((OUT / "report.json").relative_to(ROOT)),
                      "models": {n: {"layers": m["mac_layers"],
                                     "byte_saving_layers_native_lanes": m["layers_with_per_channel_byte_saving"],
                                     "byte_saving_layers_dense_bitpack": m["layers_with_dense_per_channel_byte_saving"]}
                                 for n, m in result["model"].items()},
                      "brute_force_test": result["proof_test"]["passed"],
                      "real_weight_checks": sum(len(v["checks"]) for v in result["real_weight_validation"])},
                     indent=2))
    return result


if __name__ == "__main__":
    main()
