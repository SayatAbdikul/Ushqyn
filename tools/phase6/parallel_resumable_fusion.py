#!/usr/bin/env python3
"""Bounded, resource-charged study of a resumable fused DW/PW execution order.

This is a software trace/port-work model, not a new ABI, RTL, routed result or
latency prediction. It deliberately gives the proposed order optimistic
parameter residency and eight-byte single-port service. The numerical replay
uses the frozen quantized KWS/VWW models and preserves each INT8 boundary.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "compiler"), str(ROOT / "tools/phase6")]

from followup_graph import group_channels
from integer_reference import evaluate
from phase4_compile import _parameter_rows
from quantization import requantize
from run_boardless import load_model
from static_pipeline import Program

OUT = ROOT / "work/phase6/parallel-resumable-fusion-v1"
PORT_BYTES = 8  # rtl/v2/scratchpad.sv: one synchronous 64-bit port
CONTEXT_CYCLES = 16  # assumed; sweep below is the actual sensitivity test
METADATA_BYTES = 256  # charged to both alternatives, not a fitted RTL value
ACC_REGISTER_BYTES = 4  # RTL v2_engine has one scalar INT32 accumulator register


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass(frozen=True)
class Pair:
    name: str
    model: str
    producer_index: int
    consumer_index: int
    kind: str
    h: int
    w: int
    source_channels: int
    producer_channels: int
    consumer_channels: int
    producer_kernel: tuple[int, int]
    producer_strides: tuple[int, int]
    producer_parameter_bytes: int
    consumer_parameter_bytes: int
    consumer_kernel: tuple[int, int]
    source_sha256: str

    @property
    def pixels(self) -> int:
        return self.h * self.w


def load_pair(name: str, model: str, producer_index: int, consumer_index: int) -> Pair:
    program, _, _, sources = load_model(model)
    if model == "vww":
        program, _, _ = group_channels(program)
    a, b = program.layers[producer_index], program.layers[consumer_index]
    assert b.inputs[0] == program.layers[consumer_index - 1].output
    assert all(program.layers[i].op in ("Relu", "Clip") for i in range(producer_index + 1, consumer_index))
    x = program.tensors[a.inputs[0]].shape
    y = program.tensors[a.output].shape
    z = program.tensors[b.output].shape
    wa, pa = _parameter_rows(program, a)
    wb, pb = _parameter_rows(program, b)
    akw = tuple(np.asarray(a.parameters["weight"]).shape[-2:])
    bkw = tuple(np.asarray(b.parameters["weight"]).shape[-2:])
    kind = "DW_to_PW" if a.attributes.get("group", 1) == y[1] and bkw == (1, 1) else "PW_to_DW"
    assert kind == "DW_to_PW" or (akw == (1, 1) and b.attributes.get("group", 1) == z[1])
    return Pair(name, model, producer_index, consumer_index, kind, y[2], y[3], x[1], y[1], z[1],
                akw, tuple(a.attributes.get("strides", [1, 1])), len(wa) + len(pa), len(wb) + len(pb),
                bkw, sources[f"work/phase4/{model}-logits.onnx"])


def tile_widths(width: int) -> list[int]:
    return sorted({v for v in (1, 2, 4, 8, 16, 24, 32, width) if v <= width and width % v == 0})


def source_load_bytes(pair: Pair, width: int) -> int:
    """Conservative row-local halo service; clipped/padded edges are ignored.

    This intentionally overcharges both policies equally. For non-stride-one
    spatial producer it expands the requested source rectangle by stride.
    """
    kh, kw = pair.producer_kernel
    _, sw = pair.producer_strides
    return pair.h * (pair.w // width) * pair.source_channels * kh * ((width - 1) * sw + kw)


def source_live_bytes(pair: Pair, width: int) -> int:
    kh, kw = pair.producer_kernel
    _, sw = pair.producer_strides
    return pair.source_channels * kh * ((width - 1) * sw + kw)


def candidate(pair: Pair, budget: int, width: int, channel_chunk: int,
              context_cycles: int, *, continuation: bool) -> dict | None:
    """Count a full row-tiled schedule, with all state fitting the chosen SRAM.

    Both policies compute the same MACs and use the same input/producer/output
    bytes. Ordinary blocked fusion stores the entire quantized producer tile
    and finishes one consumer output at a time in its scalar INT32 register. The
    continuation stores a smaller producer channel chunk, but keeps every
    consumer INT32 partial sum live until all channel chunks complete.
    """
    if pair.kind != "DW_to_PW":
        return None
    c, m, p = pair.producer_channels, pair.consumer_channels, pair.pixels
    if continuation and (channel_chunk >= c or c % channel_chunk):
        return None
    if not continuation:
        channel_chunk = c
    chunks = ceildiv(c, channel_chunk)
    tiles = pair.h * (pair.w // width)
    partial = 4 * m * width if continuation else 0
    producer = channel_chunk * width
    output = m * width
    common = source_live_bytes(pair, width) + producer + output + METADATA_BYTES
    state = common + partial + (0 if continuation else ACC_REGISTER_BYTES)
    full_parameters = pair.producer_parameter_bytes + pair.consumer_parameter_bytes
    # A context alternates over one time-shared datapath; if the combined
    # parameters do not fit, reserve the largest active context and reload
    # parameters at every tile. This is optimistic for both policies.
    if continuation:
        active_parameters = max(ceildiv(pair.producer_parameter_bytes * channel_chunk, c),
                                ceildiv(pair.consumer_parameter_bytes * channel_chunk, c), 16)
    else:
        # The ordinary output-stationary control can finish one consumer
        # output channel at a time with its scalar register accumulator. Its
        # whole PW parameter tensor need not fit in the SRAM at once.
        active_parameters = max(pair.producer_parameter_bytes,
                                ceildiv(pair.consumer_parameter_bytes, m), 16)
    resident = state + full_parameters <= budget
    peak = state + (full_parameters if resident else active_parameters)
    if peak > budget:
        return None
    parameter_load = full_parameters if resident else full_parameters * tiles
    src = source_load_bytes(pair, width)
    producer_write = c * p
    producer_read = m * c * p  # no fictitious free consumer broadcast
    output_service = 2 * m * p  # engine SRAM write and final DMA read
    partial_read_write = 8 * m * p * (chunks - 1) if continuation else 0
    # External input/parameter transfer consumes the same single SRAM port
    # as engine reads; count one write plus one engine read per byte. This is
    # a proxy because real 64-bit word reuse and stalls are not fitted here.
    port_bytes = 2 * src + producer_write + producer_read + output_service \
        + partial_read_write + 2 * parameter_load
    port_words = ceildiv(port_bytes, PORT_BYTES)
    macs = pair.pixels * (pair.producer_channels * math.prod(pair.producer_kernel)
                          + pair.consumer_channels * pair.producer_channels)
    ideal_mac_slots = ceildiv(macs, 8)
    switches = 2 * tiles * chunks
    context_charge = context_cycles * switches
    return dict(width=width, channel_chunk=channel_chunk, channel_chunks=chunks,
                spatial_tiles=tiles, peak_sram_bytes=peak, partial_sum_bytes=partial,
                parameters_resident=resident, parameter_load_bytes=parameter_load,
                source_load_bytes=src, producer_write_bytes=producer_write,
                producer_read_bytes=producer_read, output_service_bytes=output_service,
                partial_sum_read_write_bytes=partial_read_write,
                total_port_bytes=port_bytes, port_words=port_words,
                ideal_mac_slots=ideal_mac_slots, context_switches=switches,
                context_charge_cycles=context_charge,
                optimistic_parallel_lower_bound=max(port_words, ideal_mac_slots) + context_charge,
                serialized_service_proxy=port_words + ideal_mac_slots + context_charge)


def exact_consumer_continuations(model: str, producer_index: int, consumer_index: int) -> dict:
    """Replay real quantized DW→PW values across multiple suspended sums."""
    program, _, _, _ = load_model(model)
    if model == "vww":
        program, _, _ = group_channels(program)
    layers = copy.deepcopy(program.layers[producer_index:consumer_index + 1])
    names = [layers[0].inputs[0]] + [layer.output for layer in layers]
    block = Program({n: copy.deepcopy(program.tensors[n]) for n in names}, layers,
                    [names[0]], [names[-1]], {}, copy.deepcopy(program.provenance))
    with np.load(ROOT / f"work/phase4/rtl-{model}/expected.npz") as frozen:
        source = frozen[f"layer_{producer_index - 1}"].copy()
    values = evaluate(block, {block.inputs[0]: source})
    consumer = layers[-1]
    source_values = values[consumer.inputs[0]]
    reference = values[consumer.output]
    weights = consumer.parameters["weight"][:, :, 0, 0].astype(np.int64)
    c = source_values.shape[1]
    m = reference.shape[1]
    p = reference.shape[2] * reference.shape[3]
    xs = source_values.reshape(c, p).T.astype(np.int64)
    corrected = consumer.parameters["corrected_bias"].astype(np.int64)
    out_q = block.tensors[consumer.output].quantization
    checks = []
    for chunk in sorted({1, 2, 4, max(1, c // 2)}):
        if c % chunk:
            continue
        sums = np.broadcast_to(corrected, (p, m)).copy()
        for begin in range(0, c, chunk):
            sums += xs[:, begin:begin + chunk] @ weights[:, begin:begin + chunk].T
            assert np.all(sums >= -(1 << 31)) and np.all(sums < (1 << 31))
            # Explicit serialize/reload of the proposed persistent INT32
            # state, without preserving Python's hidden accumulator object.
            if begin + chunk < c:
                sums = sums.astype("<i4").copy().astype(np.int64)
        out = np.empty((p, m), np.int8)
        for channel in range(m):
            out[:, channel] = requantize(sums[:, channel],
                consumer.parameters["multiplier"][channel],
                consumer.parameters["shift"][channel], out_q.zero_point)
        actual = out.T.reshape(reference.shape)
        assert np.array_equal(actual, reference), (model, producer_index, consumer_index, chunk)
        checks.append(dict(channel_chunk=chunk, exact_outputs=int(actual.size),
                           output_sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
    return dict(model=model, producer_index=producer_index, consumer_index=consumer_index,
                quantization_boundaries="producer Conv -> Relu -> consumer Conv, retained unchanged",
                checks=checks, input_sha256=hashlib.sha256(source.tobytes()).hexdigest())


def exact_channelwise_pw_to_dw() -> dict:
    """Independently replay the later quantized pointwise -> depthwise case."""
    model = "vww"
    producer_index, consumer_index = 9, 11
    program, _, _, _ = load_model(model)
    program, _, _ = group_channels(program)
    layers = copy.deepcopy(program.layers[producer_index:consumer_index + 1])
    names = [layers[0].inputs[0]] + [layer.output for layer in layers]
    block = Program({n: copy.deepcopy(program.tensors[n]) for n in names}, layers,
                    [names[0]], [names[-1]], {}, copy.deepcopy(program.provenance))
    # Grouped channel order entering layer 9 can differ from the ungrouped
    # frozen NPZ. Obtain it by evaluating only the grouped prefix, using the
    # frozen first-layer input. This is small (24x24 at the target pair).
    with np.load(ROOT / "work/phase4/rtl-vww/expected.npz") as frozen:
        start = frozen["layer_8"].copy()
    # At this cut the grouped layer 8 ordering may differ; the independent
    # arithmetic validation is about the consumer boundary, so generate the
    # block's own deterministic input in that same channel order.
    rng = np.random.default_rng(60928)
    start = rng.integers(-128, 128, start.shape, dtype=np.int8)
    values = evaluate(block, {block.inputs[0]: start})
    dw = layers[-1]
    quantized = values[dw.inputs[0]]
    reference = values[dw.output]
    weights = dw.parameters["weight"].astype(np.int64)
    raw_bias = dw.parameters["corrected_bias"].astype(np.int64)
    iq = block.tensors[dw.inputs[0]].quantization
    oq = block.tensors[dw.output].quantization
    pads = dw.attributes.get("pads", [0, 0, 0, 0])
    strides = dw.attributes.get("strides", [1, 1])
    kh, kw = weights.shape[-2:]
    output = np.empty_like(reference)
    for c in range(output.shape[1]):
        padded = np.pad(quantized[0, c].astype(np.int64),
                        ((pads[0], pads[2]), (pads[1], pads[3])),
                        constant_values=iq.zero_point)
        acc = np.full(reference.shape[2:], int(raw_bias[c]), np.int64)
        for ky in range(kh):
            for kx in range(kw):
                acc += padded[ky:ky + strides[0] * reference.shape[2]:strides[0],
                              kx:kx + strides[1] * reference.shape[3]:strides[1]] * int(weights[c, 0, ky, kx])
        assert np.all(acc >= -(1 << 31)) and np.all(acc < (1 << 31))
        output[0, c] = requantize(acc, dw.parameters["multiplier"][c],
                                   dw.parameters["shift"][c], oq.zero_point)
    assert np.array_equal(output, reference)
    return dict(model=model, producer_index=producer_index,
                consumer_index=consumer_index, channelwise_no_cross_channel_partials=True,
                exact_outputs=int(output.size),
                output_sha256=hashlib.sha256(output.tobytes()).hexdigest(),
                input_kind="seeded INT8 in grouped-channel order; not a pinned accuracy sample")


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    pairs = [load_pair("kws-dw3-pw5", "kws", 3, 5),
             load_pair("vww-dw3-pw5-early", "vww", 3, 5),
             load_pair("vww-pw9-dw11", "vww", 9, 11),
             load_pair("vww-dw23-pw25-heldout", "vww", 23, 25)]
    assert [p.kind for p in pairs] == ["DW_to_PW", "DW_to_PW", "PW_to_DW", "DW_to_PW"]
    controls = json.loads((ROOT / "work/phase6/strip-fusion-vww-v1/report.json").read_text())
    assert controls["status"] == "passed-native"
    assert controls["candidate"]["bridge_intermediate_external_bytes"] == 0
    rows = []
    for pair in pairs:
        if pair.kind == "PW_to_DW":
            rows.append(dict(pair=pair.name, kind=pair.kind, conclusion="channelwise consumer has no cross-channel INT32 continuation; ordinary quantized line/window fusion is the direct control",
                             consumer_partial_sum_bytes=0, peak_producer_full_map_bytes=pair.producer_channels * pair.pixels))
            continue
        for budget in (8192, 16384, 32768):
            ordinary = []
            resumed = []
            for width in tile_widths(pair.w):
                base = candidate(pair, budget, width, pair.producer_channels, CONTEXT_CYCLES, continuation=False)
                if base is not None:
                    ordinary.append(base)
                for chunk in (1, 2, 4, 8, 16, 32, 64):
                    result = candidate(pair, budget, width, chunk, CONTEXT_CYCLES, continuation=True)
                    if result is not None:
                        resumed.append(result)
            key = lambda r: (r["serialized_service_proxy"], r["peak_sram_bytes"], r["width"], r["channel_chunk"])
            best_ordinary = min(ordinary, key=key) if ordinary else None
            best_resumed = min(resumed, key=key) if resumed else None
            paired_ordinary = (candidate(pair, budget, best_resumed["width"], pair.producer_channels,
                                         CONTEXT_CYCLES, continuation=False) if best_resumed else None)
            rows.append(dict(pair=pair.name, kind=pair.kind, budget_bytes=budget,
                             ordinary_candidate_count=len(ordinary), continuation_candidate_count=len(resumed),
                             ordinary=best_ordinary, continuation=best_resumed,
                             ordinary_at_continuation_width=paired_ordinary,
                             continuation_vs_ordinary_serialized_proxy_ratio=(best_resumed["serialized_service_proxy"] / best_ordinary["serialized_service_proxy"] if best_resumed and best_ordinary else None)))
    exact = [exact_consumer_continuations("kws", 3, 5), exact_consumer_continuations("vww", 3, 5)]
    channelwise_exact = exact_channelwise_pw_to_dw()
    sensitivity = []
    for cc in (0, 4, 16, 64):
        pair = pairs[1]
        for continuation in (False, True):
            options = [candidate(pair, 32768, width, chunk, cc, continuation=continuation)
                       for width in tile_widths(pair.w)
                       for chunk in ((1, 2, 4) if continuation else (pair.producer_channels,))]
            options = [x for x in options if x is not None]
            best = min(options, key=lambda x: x["serialized_service_proxy"])
            sensitivity.append(dict(context_cycles=cc, policy="resumable" if continuation else "ordinary",
                                    serialized_service_proxy=best["serialized_service_proxy"],
                                    width=best["width"], channel_chunk=best["channel_chunk"]))
    report = dict(schema=1, status="passed-model", physical_board=False,
                  scope="exact quantized consumer replay plus analytical single-port service; no RTL implementation or measured speedup",
                  model_parameters=dict(port_bytes=PORT_BYTES, context_cycles=CONTEXT_CYCLES,
                                        metadata_bytes=METADATA_BYTES, ordinary_acc_register_bytes=ACC_REGISTER_BYTES),
                  source_hashes={str(p.relative_to(ROOT)): sha(p) for p in
                                 (ROOT / "rtl/v2/scratchpad.sv", ROOT / "rtl/v2/engine.sv",
                                  ROOT / "work/phase6/strip-fusion-vww-v1/report.json",
                                  ROOT / "work/phase4/kws-calibration-rebased.json",
                                  ROOT / "work/phase4/vww-calibration-rebased.json",
                                  ROOT / "work/phase4/rtl-kws/manifest.json",
                                  ROOT / "work/phase4/rtl-vww/manifest.json",
                                  ROOT / "work/phase4/rtl-kws/expected.npz",
                                  ROOT / "work/phase4/rtl-vww/expected.npz")},
                  model_pairs=[p.__dict__ for p in pairs], exact_quantization_replay=exact,
                  exact_channelwise_replay=channelwise_exact,
                  synthetic_residual_fork_join_stress=dict(
                      scope="shape-only two-branch fork with two PW consumers and an eventual join; no Add RTL or exact replay",
                      source_geometry="KWS DW3 output, 64 channels x 125 pixels",
                      branch_output_channels=[64, 64],
                      ordinary_producer_retained_bytes=64 * 125,
                      resumable_partial_sum_bytes=4 * 125 * (64 + 64),
                      resumable_partial_read_write_bytes_for_32_channel_chunks=8 * 125 * (64 + 64),
                      result="resumable state alone exceeds 32 KiB across full geometry; spatial tiling is required"),
                  preexisting_executable_control=dict(model="VWW layers 3–6", status=controls["status"],
                      external_bridge_bytes=controls["candidate"]["bridge_intermediate_external_bytes"],
                      activation_dma_bytes_baseline=controls["activation_dma_bytes"]["baseline"],
                      activation_dma_bytes_strip=controls["activation_dma_bytes"]["candidate"],
                      block_baseline_cycles_seed0=controls["samples"][0]["native"]["baseline"][0]["elapsed_cycles"],
                      block_strip_cycles_seed0=controls["samples"][0]["native"]["candidate"][0]["elapsed_cycles"],
                      image=controls["selected_native_executable_sha256"],
                      provenance="older selected Phase-6 native image; not directly comparable with newest B3/co-issue board image"),
                  rows=rows, context_sensitivity=sensitivity,
                  limitations=["The context-cycle charge is assumed and swept, not measured for a proposed RTL.",
                               "The service proxy serializes ideal MAC slots and 64-bit SRAM word work; it is not a clock-accurate latency or energy model.",
                               "Parameter residency and row halo accounting are optimistic; bank packing, DMA setup, source rereads, and RTL timing may be worse.",
                               "The comparator is ordinary output-stationary blocked fusion with quantized producer tiles and one INT32 accumulator register, not just non-fused materialization.",
                               "The held-out VWW layer-23/25 geometry checks capacity/traffic only; exact numerical replay is limited to KWS, early VWW and a later PW-to-DW pair.",
                               "The residual fork/join row is synthetic sizing only because the frozen KWS/VWW graphs are linear and current RTL has no Add lowering."])
    (output / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    report = run(args.output)
    for row in report["rows"]:
        if row["kind"] == "DW_to_PW":
            print(row["pair"], row["budget_bytes"], "ordinary", row["ordinary"] and row["ordinary"]["serialized_service_proxy"],
                  "resumable", row["continuation"] and row["continuation"]["serialized_service_proxy"], flush=True)
    print(args.output / "report.json", flush=True)


if __name__ == "__main__":
    main()
