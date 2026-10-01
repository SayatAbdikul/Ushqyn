#!/usr/bin/env python3
"""Bounded split-bank operand screen against the frozen paired PW control.

This is a transaction and optimistic serialized-service model, not RTL timing.
The candidate gives the two existing four-byte bank groups independent word
addresses during an *engine-owned* grant. Its 8-pixel activation tile is two
4-pixel x 2-output-channel MAC issues using the same eight multipliers. Each
four-bank group has a 128-entry, single-read/single-write tagged halfword cache.
No DMA/engine overlap, output-write coalescing, extra MACs, or extra ports are
assumed. The model explicitly charges two weight scalars per candidate issue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from parallel_operand_delivery import simulate


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "work/phase6/port-fabric-model-v1"
DEFAULT_PROFILE = DEFAULT_OUTPUT / "profiles"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate(fields: dict) -> dict[str, int]:
    count = int(fields["count"])
    channels = int(fields["output_c"])
    outputs = int(fields["outputs"])
    area = outputs // channels
    if count != int(fields["input_c"]) or area != int(fields["input_h"]) * int(fields["input_w"]):
        raise ValueError("only dense unstrided PW descriptors are modeled")
    if count > 128:
        raise ValueError("128 entries per halfword bank do not fit input channels")
    if channels <= 0 or outputs % channels:
        raise ValueError("invalid descriptor geometry")
    result = {k: 0 for k in (
        "activation_tiles", "mac_issues", "weight_operand_bytes",
        "weight_cache_read_ops", "weight_cache_service_slots",
        "weight_register_reuses",
        "cache_read_bank_ops", "cache_read_service_slots", "cache_hit_bank_ops",
        "cache_miss_bank_ops", "scratchpad_read_service_slots",
        "scratchpad_read_bytes", "scratchpad_read_64b_equivalents",
        "serialized_activation_service_slots", "split_cross_word_tiles",
        "split_one_grant_cross_word_tiles", "split_two_grant_tiles",
        "fallback_odd_channel_tiles", "bank0_misses", "bank1_misses",
    )}
    # One 32-bit halfword entry per input channel in each physically separate
    # four-bank group. A tag is the 64-bit physical word address. Retaining
    # entries across output pairs matches the one-word control's lifetime.
    tags = [[-1] * 128, [-1] * 128]
    start = int(fields["input"])
    for oc in range(0, channels, 2):
        pair = oc + 1 < channels
        for pixel in range(0, area, 8):
            take = min(8, area - pixel)
            result["activation_tiles"] += 1
            if pair:
                result["mac_issues"] += count * ((take + 3) // 4)
                result["weight_operand_bytes"] += 2 * count * ((take + 3) // 4)
                # Each pair of scalars is fetched from two separately read
                # cache banks once, then held across both four-pixel issues.
                result["weight_cache_read_ops"] += 2 * count
                result["weight_cache_service_slots"] += count
                result["weight_register_reuses"] += 2 * count * (((take + 3) // 4) - 1)
            else:
                # Exact same one-output, 8-pixel fallback as the paired
                # control. Its activation demand is separately charged below.
                result["fallback_odd_channel_tiles"] += 1
                result["mac_issues"] += count
                result["weight_operand_bytes"] += count
                result["weight_cache_read_ops"] += count
                result["weight_cache_service_slots"] += count
            for ic in range(count):
                addr = start + ic * area + pixel
                if not pair:
                    # The one-output fallback needs a full 64-bit cache line.
                    # We still read it through the two independent half banks;
                    # this can only help the candidate, never penalize it.
                    pass
                halves = range(addr // 4, (addr + take - 1) // 4 + 1)
                ops = [0, 0]
                misses = [0, 0]
                distinct_words = {half // 2 for half in halves}
                if len(distinct_words) > 1:
                    result["split_cross_word_tiles"] += 1
                for half in halves:
                    bank = half & 1
                    tag = half // 2
                    ops[bank] += 1
                    if tags[bank][ic] == tag:
                        result["cache_hit_bank_ops"] += 1
                    else:
                        result["cache_miss_bank_ops"] += 1
                        result[f"bank{bank}_misses"] += 1
                        misses[bank] += 1
                        tags[bank][ic] = tag
                cache_slots = max(ops)
                miss_slots = max(misses)
                result["cache_read_bank_ops"] += sum(ops)
                result["cache_read_service_slots"] += cache_slots
                result["scratchpad_read_service_slots"] += miss_slots
                result["scratchpad_read_bytes"] += 4 * sum(misses)
                result["scratchpad_read_64b_equivalents"] += (sum(misses) + 1) // 2
                result["serialized_activation_service_slots"] += cache_slots + miss_slots
                if cache_slots > 2 or miss_slots > 2:
                    raise AssertionError("an eight-byte tile needs at most two grants per bank")
                if len(distinct_words) > 1 and cache_slots == 1:
                    result["split_one_grant_cross_word_tiles"] += 1
                if cache_slots > 1:
                    result["split_two_grant_tiles"] += 1
    result["output_byte_writes"] = outputs
    result["output_write_service_slots"] = outputs
    result["parameter_64b_reads"] = 2 * channels
    result["weight_64b_reads"] = channels * ((count + 7) // 8)
    result["lut_64b_reads"] = 32
    # Weights are fetched once into the existing two-channel cache banks.
    # The two 8-bit weight operands are registered before an 8-MAC issue;
    # they count above even though the external fill traffic is unchanged.
    result["fixed_64b_read_slots"] = (
        8 + result["parameter_64b_reads"] + result["weight_64b_reads"] +
        result["lut_64b_reads"]
    )
    result["all_serial_service_slots"] = (
        result["serialized_activation_service_slots"] +
        result["output_write_service_slots"] +
        2 * result["fixed_64b_read_slots"]
    )
    result["nonoverlap_service_plus_issue_score"] = (
        result["all_serial_service_slots"] + result["mac_issues"]
    )
    return result


def baseline(fields: dict, policy: str) -> dict[str, int]:
    original = simulate(fields, policy)
    count = int(fields["count"])
    channels = int(fields["output_c"])
    outputs = int(fields["outputs"])
    original["weight_operand_bytes"] = original["mac_issues"]
    original["weight_cache_read_ops"] = original["mac_issues"]
    original["weight_cache_service_slots"] = original["mac_issues"]
    original["weight_register_reuses"] = 0
    original["output_byte_writes"] = outputs
    original["output_write_service_slots"] = outputs
    original["parameter_64b_reads"] = 2 * channels
    original["weight_64b_reads"] = channels * ((count + 7) // 8)
    original["lut_64b_reads"] = 32
    original["fixed_64b_read_slots"] = (
        8 + original["parameter_64b_reads"] + original["weight_64b_reads"] +
        original["lut_64b_reads"]
    )
    original["all_serial_service_slots"] = (
        original["serial_access_score"] + original["output_write_service_slots"] +
        2 * original["fixed_64b_read_slots"]
    )
    original["nonoverlap_service_plus_issue_score"] = (
        original["all_serial_service_slots"] + original["mac_issues"]
    )
    return original


def stress() -> dict[str, dict]:
    cases = {
        "aligned_128_pixel_plane": dict(input=8192, input_h=16, input_w=8,
                                        input_c=64, count=64, outputs=128 * 64,
                                        output_c=64),
        "kws_odd_125_pixel_plane": dict(input=8192, input_h=25, input_w=5,
                                         input_c=64, count=64, outputs=125 * 64,
                                         output_c=64),
        "half_bank_seam_36_pixel_plane": dict(input=8196, input_h=6, input_w=6,
                                              input_c=60, count=60,
                                              outputs=36 * 106, output_c=106),
        "all_phases_9_pixel_plane": dict(input=8195, input_h=3, input_w=3,
                                         input_c=31, count=31, outputs=9 * 22,
                                         output_c=22),
        "odd_output_channel_tail": dict(input=4099, input_h=19, input_w=7,
                                        input_c=63, count=63, outputs=133 * 17,
                                        output_c=17),
    }
    result = {}
    for label, fields in cases.items():
        base = baseline(fields, "pair_one_word")
        split = candidate(fields)
        result[label] = {
            "fields": fields,
            "paired_activation_score": base["serial_access_score"],
            "split_activation_score": split["serialized_activation_service_slots"],
            "paired_mac_issues": base["mac_issues"],
            "split_mac_issues": split["mac_issues"],
            "split_one_grant_cross_word_tiles": split["split_one_grant_cross_word_tiles"],
            "split_cross_word_tiles": split["split_cross_word_tiles"],
            "split_fallback_odd_channel_tiles": split["fallback_odd_channel_tiles"],
        }
        if base["mac_issues"] != split["mac_issues"]:
            # The 4x2 mapping can have more issues on a short spatial tail.
            # These extra issue slots cannot be omitted from the decision.
            result[label]["extra_issue_slots"] = split["mac_issues"] - base["mac_issues"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sources = {}
    models = {}
    for model in ("kws", "vww"):
        path = args.profile_dir / f"{model}-s0-enriched.json"
        stress_path = args.profile_dir / f"{model}-s6063-enriched.json"
        sources[str(path)] = sha(path)
        sources[str(stress_path)] = sha(stress_path)
        profile = json.loads(path.read_text())
        stress_profile = json.loads(stress_path.read_text())
        if profile["stall_seed"] != 0 or stress_profile["stall_seed"] != 6063:
            raise ValueError("unexpected seed")
        rows = []
        for src, twin in zip(profile["descriptors"], stress_profile["descriptors"], strict=True):
            if src["descriptor_hex"] != twin["descriptor_hex"] or src["family"] != twin["family"]:
                raise ValueError("seed descriptor mismatch")
            if src["family"] != "pointwise_conv":
                continue
            fields = src["descriptor"]
            check = baseline(fields, "one_word")
            if (check["first_reads"] != src["pw_first_word_reads"] or
                    check["cross_reads"] != src["pw_cross_reads"] or
                    check["mac_issues"] != src["states"].get("PW_MAC", 0)):
                raise ValueError(f"frozen trace mismatch at {model} command {src['command']}")
            if (check["external_read_words"] + check["fixed_64b_read_slots"] != src["accepted_reads"] or
                    check["output_byte_writes"] != src["accepted_writes"] or
                    check["weight_64b_reads"] != src["weight_miss"] or
                    src["states"].get("F_L_REQ", 0) != 32):
                raise ValueError(f"frozen fixed traffic mismatch at {model} command {src['command']}")
            rows.append({
                "command": src["command"], "geometry": fields,
                "paired_one_word": baseline(fields, "pair_one_word"),
                "paired_two_word": baseline(fields, "pair_two_word"),
                "split_four_bank": candidate(fields),
            })
        totals = {}
        for policy in ("paired_one_word", "paired_two_word", "split_four_bank"):
            keys = rows[0][policy]
            totals[policy] = {k: sum(r[policy][k] for r in rows) for k in keys}
        models[model] = {
            "pointwise_descriptors": len(rows), "descriptor_rows": rows,
            "totals": totals,
            "frozen_native_device_cycles": profile["sequencer_cycles"],
            "frozen_native_dma_cycles": profile["dma_cycles"],
            "frozen_native_engine_cycles": profile["engine_cycles"],
        }
    output = {
        "status": "passed", "source_sha256": sources,
        "script_sha256": sha(Path(__file__)),
        "model": models,
        "stress": stress(),
        "hardware_contract": {
            "mac_lanes": 8,
            "paired_control_mapping": "8 pixels x 1 output per issue, 2 output accumulator banks",
            "candidate_mapping": "4 pixels x 2 outputs per issue, two 4-pixel issues per 8-pixel tile",
            "candidate_weight_operands_per_issue": 2,
            "candidate_weight_cache": "two 16x64 one-read banks; both weight scalars registered once for each 8-pixel tile and held across its two 4-pixel x 2-output MAC issues; total 32x64 unchanged",
            "candidate_activation_cache": "two 128x54-bit halfword banks (32 data + 21 tag + valid), one synchronous read and one write per bank; index=input channel",
            "candidate_activation_cache_bits": 2 * 128 * 54,
            "paired_control_activation_cache_bits": 252 * 86,
            "scratchpad": "same eight byte-write BSRAM banks; independent word address per four-bank group; one engine-or-DMA grant at a time",
            "scratchpad_data_bram_blocks": 16,
            "scratchpad_program_bram_blocks": 16,
            "candidate_dma_interface": "full-width 64-bit DMA uses both groups at the same address; no DMA/engine split-grant overlap credited",
            "output_writes": "one 8-bit masked write per output element, identical for all policies; no free coalescing or concurrent requantization",
            "fixed_reads": "8 descriptor words, 32 activation-LUT words, 2 parameter words/output channel, ceil(input_c/8) weight words/output channel; sum plus observed PW words equals accepted_reads in all 20 frozen descriptors",
        },
        "evidence_level": "counterfactual address/port-demand screen; native seed trace validates only the one-word baseline; candidate cycles and Fmax are not measured",
        "limitations": [
            "Serialized service score is cache-bank read slots plus scratchpad request/response slots, with low/high banks optimistically paired within each tile; it is not a cycle lower bound because pipeline overlap and schedule dependencies are omitted.",
            "DMA cycles are reported but added to neither policy's pointwise service score; DMA has exclusive grants and is unchanged under the candidate.",
            "The 32-bit split requires scratchpad, engine interface, arbiter mux, and activation-cache RTL changes; equal SRAM bit capacity does not establish routed CLS/Fmax.",
            "Weight and output traffic are charged equally; candidate dual-weight cache-port timing, quantizer scheduling, and bank conflict control require RTL. The candidate consumes two weight scalars per 4x2 issue, holds them across the two half-tiles, and never receives free cache reads.",
            "The separately reported service-plus-issue score assumes zero overlap between operands and MACs; it is not native latency and is not added to the measured board result.",
            "Only frozen KWS/VWW pointwise descriptors are analyzed; non-pointwise engine work and AD are not candidate-transformed.",
        ],
        "decision": "Do not build full split-bank candidate RTL on these data. Optimistic all-in pointwise serial-service gains are only about 3-4 percent versus the paired control before clock, logic, and DMA-interface costs. Reconsider only if routed paired RTL reveals a materially larger crossing-word bottleneck that this mechanism uniquely removes.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    lines = ["# Four-bank split operand service screen", "",
             "All candidate numbers are address/port estimates, not measured cycles. DMA retains the full scratchpad grant.", "",
             "| Model | Policy | Activation SRAM bytes | Activation cache read slots | Activation SRAM read slots | Serial activation score | Score vs paired one-word | All-in serial service incl. fixed reads/writes | MAC issue slots |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, model in models.items():
        totals = model["totals"]
        ref = totals["paired_one_word"]["serial_access_score"]
        for label in ("paired_one_word", "paired_two_word", "split_four_bank"):
            t = totals[label]
            if label == "split_four_bank":
                byte_count = t["scratchpad_read_bytes"]
                cache_slots = t["cache_read_service_slots"]
                sram_slots = t["scratchpad_read_service_slots"]
                score = t["serialized_activation_service_slots"]
            else:
                byte_count = t["external_read_bytes"]
                cache_slots = t["first_accesses"] + t["cross_accesses"]
                sram_slots = t["external_read_words"]
                score = t["serial_access_score"]
            lines.append(f"| {name.upper()} | {label} | {byte_count:,} | {cache_slots:,} | {sram_slots:,} | {score:,} | {100*(score/ref-1):+.3f}% | {t['all_serial_service_slots']:,} | {t['mac_issues']:,} |")
    lines += ["", "A split request can fetch the upper half of one 64-bit word and the lower half of the next in one grant. Each half has one independent cache read and one independent scratchpad byte-bank address. All output bytes and both scalar weights are charged. The service score adds synchronous cache lookup slots and accepted SRAM read slots; observed RTL may overlap some of these, so it cannot be subtracted from board cycles. The candidate consumes two weights per 4×2 issue and registers them across both four-pixel halves; cache-bank read operations remain explicit in report.json.",
              "", "The fixed descriptor, parameter, and weight traffic, output writes, DMA ownership, eight MAC lanes, and numerical semantics are the same across policies. Equal resource accounting is a design constraint, not a routed result.",
              "", "| Model | Command | Geometry (H×W, IC→OC) | Paired one-word score | Split-bank score | Change | Cross-word tiles handled in one bank slot |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for name, model in models.items():
        for row in model["descriptor_rows"]:
            fields = row["geometry"]
            ref = row["paired_one_word"]["serial_access_score"]
            split = row["split_four_bank"]
            score = split["serialized_activation_service_slots"]
            lines.append(
                f"| {name.upper()} | {row['command']} | {fields['input_h']}×{fields['input_w']}, "
                f"{fields['input_c']}→{fields['output_c']} | {ref:,} | {score:,} | "
                f"{100*(score/ref-1):+.2f}% | {split['split_one_grant_cross_word_tiles']:,}/"
                f"{split['split_cross_word_tiles']:,} |"
            )
    lines += ["", "Synthetic address stress covers aligned planes, KWS-style odd planes, 4-byte seams, all eight word phases and odd output-channel tails. The frozen VWW rows themselves include short 6×6 and 3×3 planes and odd channel counts. See report.json for the bounded stress counters and hashes.",
              "", "**Decision:** Do not build full split-bank candidate RTL on these data. The optimistic all-in pointwise service gain is only about 4.2% KWS and 3.4% VWW before clock, logic, and interface costs. The 4×2 mapping separately saves 8,392 VWW issue slots on short spatial tails; that lane packing is conventional. No benefit is modeled for AD's unchanged Gemm/Relu graph. The fixed 16 KiB program partition and published full-width/2D VWW schedules address separate costs and are not credited as split-bank wins.",
              "", "The model is not a cycle lower bound: it serializes cache lookup and SRAM request/response but ignores pipeline overlap and dependent stalls. The actual route would need a second scratchpad-group address, a new engine/arbiter mux, two tagged cache banks, two weight-cache read banks and a fragment shifter. The DMA retains the entire scratchpad grant and all 64-bit DMA transfers retain both groups at one address. See report.json for every descriptor, source hashes and exact constraints."]
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
