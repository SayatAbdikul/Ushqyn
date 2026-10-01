#!/usr/bin/env python3
"""Bounded residual tile-lifetime screen; analytical service, exact small replay.

No residual Add is lowered by the current FPGA compiler. The large shapes below
come from pinned linear KWS/VWW models and are representative *synthetic* residual
blocks. This script does not estimate routed latency or claim a hardware result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "compiler"), str(ROOT / "tools/phase6")]

from integer_reference import evaluate
from quantization import Quantization, multiplier_shift, quantize_parameters
from run_boardless import load_model
from static_pipeline import Layer, Program, Tensor

OUTPUT = ROOT / "work/phase6/residual-ownership-screen-v1"
SRAM_BYTES = 32768
PORT_BYTES = 8
METADATA_BYTES = 256  # common descriptor/address staging assumption
LANES = 8


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def word_bytes(n: int) -> int:
    return PORT_BYTES * ceildiv(n, PORT_BYTES)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Rect:
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def area(self) -> int:
        return (self.y1 - self.y0) * (self.x1 - self.x0)

    def grow(self, halo: int, h: int, w: int) -> "Rect":
        return Rect(max(0, self.y0 - halo), min(h, self.y1 + halo),
                    max(0, self.x0 - halo), min(w, self.x1 + halo))

    def intersects(self, other: "Rect") -> bool:
        return self.y0 < other.y1 and other.y0 < self.y1 and self.x0 < other.x1 and other.x0 < self.x1


@dataclass(frozen=True)
class Case:
    name: str
    source_model: str
    source_layer: int
    kind: str  # basic = 3x3 -> 3x3; inverted = 1x1 -> 3x3 DW -> 1x1
    channels: int
    h: int
    w: int
    expansion: int = 1

    @property
    def expanded_channels(self) -> int:
        return self.channels * self.expansion

    @property
    def tensor_bytes(self) -> int:
        return self.channels * self.h * self.w


def tiles(case: Case, th: int, tw: int) -> list[Rect]:
    return [Rect(y, min(case.h, y + th), x, min(case.w, x + tw))
            for y in range(0, case.h, th) for x in range(0, case.w, tw)]


def parameter_stage(case: Case) -> int:
    """One eight-output weight/scale/bias row, not all network parameters."""
    c, e = case.channels, case.expanded_channels
    rows = ([LANES * (9 * c + 9)] if case.kind == "basic" else
            [LANES * (c + 9), LANES * (9 + 9), LANES * (e + 9)])
    return word_bytes(max(rows))


def tile_geometry(case: Case, tile: Rect) -> dict[str, int]:
    c, e = case.channels, case.expanded_channels
    source_halo = 2 if case.kind == "basic" else 1
    source = tile.grow(source_halo, case.h, case.w)
    inner = tile.grow(1, case.h, case.w)
    if case.kind == "basic":
        a = c * inner.area
        d = 0
        macs = 9 * c * c * (inner.area + tile.area)
    else:
        a = e * inner.area
        d = e * tile.area
        macs = e * c * inner.area + 9 * e * tile.area + e * c * tile.area
    return dict(source_bytes=c * source.area, intermediate_bytes=a,
                depthwise_bytes=d, output_bytes=c * tile.area, macs=macs)


def full_geometry(case: Case) -> dict[str, int]:
    c, e, p = case.channels, case.expanded_channels, case.h * case.w
    if case.kind == "basic":
        return dict(intermediate_bytes=c * p, depthwise_bytes=0,
                    output_bytes=c * p, macs=18 * c * c * p)
    return dict(intermediate_bytes=e * p, depthwise_bytes=e * p,
                output_bytes=c * p, macs=(2 * e * c + 9 * e) * p)


def last_halo_uses(case: Case, ts: list[Rect]) -> list[int]:
    halo = 2 if case.kind == "basic" else 1
    return [max(i for i, output in enumerate(ts)
                if output.grow(halo, case.h, case.w).intersects(source_tile))
            for source_tile in ts]


def evaluate_tile_choice(case: Case, th: int, tw: int) -> dict:
    ts = tiles(case, th, tw)
    geom = [tile_geometry(case, t) for t in ts]
    source_bytes = case.tensor_bytes
    source_stream_bytes = sum(g["source_bytes"] for g in geom)
    a_bytes = sum(g["intermediate_bytes"] for g in geom)
    d_bytes = sum(g["depthwise_bytes"] for g in geom)
    output_bytes = source_bytes
    macs = sum(g["macs"] for g in geom)
    stage = parameter_stage(case) + METADATA_BYTES
    working = [word_bytes(g["source_bytes"]) + word_bytes(g["intermediate_bytes"])
               + word_bytes(g["depthwise_bytes"]) + word_bytes(g["output_bytes"]) for g in geom]
    branch_work = [word_bytes(g["intermediate_bytes"]) + word_bytes(g["depthwise_bytes"])
                   + word_bytes(g["output_bytes"]) for g in geom]
    stream_peak = stage + max(working)
    resident_direct_peak = stage + word_bytes(source_bytes) + max(branch_work)

    # A fixed tile order makes these lifetimes knowable at compile time. Both
    # static overwrite and runtime ownership use this exact same eligibility.
    last = last_halo_uses(case, ts)
    pending: dict[int, int] = {}
    overwrite_peak = 0
    pending_peak = 0
    deferred_bytes = 0
    for i, g in enumerate(geom):
        pending_before = sum(pending.values())
        overwrite_peak = max(overwrite_peak, stage + word_bytes(source_bytes)
                             + pending_before + branch_work[i])
        if last[i] > i:
            pending[i] = word_bytes(g["output_bytes"])
            deferred_bytes += g["output_bytes"]
        for j in list(pending):
            if last[j] <= i:
                del pending[j]
        pending_peak = max(pending_peak, sum(pending.values()))
    assert not pending
    token_bytes = word_bytes(ceildiv(len(ts), 8))  # one live/owned bit per tile

    # Each access below is serialized through one 64-bit scratchpad port.
    # One engine read per resident byte is an optimistic minimum; real MAC
    # operand/cache traffic, DRAM setup and arbitration may cost more.
    materialization = 2 * a_bytes + 2 * d_bytes + 5 * output_bytes
    common_source_reads = source_stream_bytes
    stream_port_bytes = materialization + common_source_reads + source_stream_bytes
    resident_port_bytes = materialization + common_source_reads + source_bytes
    # Deferred result must be copied from a temporary to the old source slot.
    # Charge one 64-bit read and one write; token updates are 64-bit RMWs.
    overwrite_copy_bytes = 2 * deferred_bytes
    owner_token_port_bytes = 2 * PORT_BYTES * len(ts)
    ideal_mac_slots = ceildiv(macs, LANES)

    def row(name: str, peak: int, port_bytes: int, ext_input: int,
            *, defers_output: bool, extra: dict | None = None) -> dict:
        service = ceildiv(port_bytes, PORT_BYTES) + ideal_mac_slots
        return dict(policy=name, tile=[th, tw], tile_count=len(ts),
                    peak_scratchpad_bytes=peak, fits_32k=peak <= SRAM_BYTES,
                    source_external_read_bytes=ext_input,
                    output_external_write_bytes=output_bytes,
                    external_activation_bytes=ext_input + output_bytes,
                    source_halo_repeat_bytes=source_stream_bytes - source_bytes,
                    intermediate_materialized_bytes=a_bytes + d_bytes,
                    deferred_output_bytes=deferred_bytes if defers_output else 0,
                    scratchpad_port_min_bytes=port_bytes,
                    scratchpad_port_min_words=ceildiv(port_bytes, PORT_BYTES),
                    ideal_mac_slots=ideal_mac_slots,
                    optimistic_serial_service=service,
                    extra=extra or {})

    return dict(
        streamed=row("ordinary_blocked_streamed", stream_peak, stream_port_bytes,
                     source_stream_bytes, defers_output=False),
        resident_direct=row("static_resident_inplace_add", resident_direct_peak,
                            resident_port_bytes, source_bytes, defers_output=False,
                            extra={"add_overwrites_temporary_output_tile": True,
                                   "output_tile_dma_before_reuse": True}),
        static_overwrite=row("static_last_use_source_overwrite", overwrite_peak,
                             resident_port_bytes + overwrite_copy_bytes, source_bytes,
                             defers_output=True, extra={"pending_peak_bytes": pending_peak,
                                                   "copy_bytes": overwrite_copy_bytes}),
        ownership_zero_overhead=row("ownership_zero_overhead_lower_bound", overwrite_peak,
                                    resident_port_bytes + overwrite_copy_bytes,
                                    source_bytes, defers_output=True,
                                    extra={"pending_peak_bytes": pending_peak,
                                           "copy_bytes": overwrite_copy_bytes,
                                           "runtime_tracking_cost": "set to zero as an optimistic lower bound"}),
        ownership=row("runtime_tile_ownership", overwrite_peak + token_bytes,
                      resident_port_bytes + overwrite_copy_bytes + owner_token_port_bytes,
                      source_bytes, defers_output=True,
                      extra={"pending_peak_bytes": pending_peak,
                             "copy_bytes": overwrite_copy_bytes,
                             "token_scratchpad_bytes": token_bytes,
                             "token_port_bytes": owner_token_port_bytes}),
        macs=macs,
    )


def full_tensor_row(case: Case) -> dict:
    g = full_geometry(case)
    peak = (word_bytes(case.tensor_bytes) + word_bytes(g["intermediate_bytes"])
            + word_bytes(g["depthwise_bytes"]) + word_bytes(g["output_bytes"])
            + parameter_stage(case) + METADATA_BYTES)
    return dict(policy="full_tensor_liveness", peak_scratchpad_bytes=peak,
                fits_32k=peak <= SRAM_BYTES,
                external_activation_bytes=2 * case.tensor_bytes if peak <= SRAM_BYTES else None,
                ideal_mac_slots=ceildiv(g["macs"], LANES),
                note="If this exceeds 32 KiB, full-map scratchpad residency is infeasible; no unmodeled SDRAM spill is credited.")


def best(rows: list[dict]) -> dict | None:
    feasible = [r for r in rows if r["fits_32k"]]
    return min(feasible, key=lambda r: (r["optimistic_serial_service"],
                                        r["peak_scratchpad_bytes"], r["tile"])) if feasible else None


def add_params(q1: Quantization, q2: Quantization, qo: Quantization) -> dict:
    ratios = [q1.scale / qo.scale, q2.scale / qo.scale]
    shift = min(multiplier_shift(r)[1] for r in ratios)
    ms = [int(round(r * (1 << shift))) for r in ratios]
    assert all(0 < m <= (1 << 31) - 1 for m in ms)
    return dict(add_multiplier=np.array(ms, dtype="<i4"),
                add_shift=np.array([shift], dtype=np.int8))


def rounded_clipped(product: int, shift: int, zp: int) -> np.int8:
    denominator = 1 << shift
    q, r = divmod(abs(int(product)), denominator)
    q += int(2 * r >= denominator)
    signed = -q if product < 0 else q
    return np.int8(min(127, max(-128, signed + zp)))


def manual_add(left: np.ndarray, right: np.ndarray, ql: Quantization,
               qr: Quantization, qo: Quantization, params: dict) -> np.ndarray:
    out = np.empty_like(left)
    m0, m1 = (int(v) for v in params["add_multiplier"])
    shift = int(params["add_shift"][0])
    for idx in np.ndindex(left.shape):
        total = (int(left[idx]) - ql.zero_point) * m0 + (int(right[idx]) - qr.zero_point) * m1
        out[idx] = rounded_clipped(total, shift, qo.zero_point)
    return out


def conv_roi(src: np.ndarray, src_origin: tuple[int, int], full_hw: tuple[int, int],
             rect: Rect, iq: Quantization, oq: Quantization, params: dict,
             *, group: int, pad: int) -> np.ndarray:
    w = params["weight"]
    cout, cin_group, kh, kw = w.shape
    out = np.empty((1, cout, rect.y1 - rect.y0, rect.x1 - rect.x0), np.int8)
    oc_per_group = cout // group
    for oc in range(cout):
        cstart = (oc // oc_per_group) * cin_group
        for oy in range(rect.y0, rect.y1):
            for ox in range(rect.x0, rect.x1):
                acc = int(params["bias"][oc])
                for ci in range(cin_group):
                    for ky in range(kh):
                        for kx in range(kw):
                            iy, ix = oy + ky - pad, ox + kx - pad
                            if 0 <= iy < full_hw[0] and 0 <= ix < full_hw[1]:
                                ly, lx = iy - src_origin[0], ix - src_origin[1]
                                assert 0 <= ly < src.shape[2] and 0 <= lx < src.shape[3]
                                value = int(src[0, cstart + ci, ly, lx])
                            else:
                                value = iq.zero_point
                            acc += (value - iq.zero_point) * int(w[oc, ci, ky, kx])
                assert -(1 << 31) <= acc < (1 << 31)
                out[0, oc, oy - rect.y0, ox - rect.x0] = rounded_clipped(
                    acc * int(params["multiplier"][oc]), int(params["shift"][oc]), oq.zero_point)
    return out


def make_fixture(kind: str) -> tuple[Program, np.ndarray]:
    rng = np.random.default_rng(6292026 if kind == "basic" else 6302026)
    c, h, w = 4, 7, 7
    shape = (1, c, h, w)
    qx, qa, qd, qb, qy = (Quantization(.1, -7), Quantization(.2, 3),
                         Quantization(.3, -2), Quantization(.4, 1), Quantization(.5, -3))
    tensors = {"x": Tensor("x", shape, qx, "NCHW")}
    layers: list[Layer] = []

    def conv(inp: str, output: str, ic: int, oc: int, kernel: int,
             group: int, qi: Quantization, qo: Quantization) -> None:
        weight = rng.uniform(-.02, .02, (oc, ic // group, kernel, kernel))
        bias = rng.uniform(-.025, .025, oc)
        params = quantize_parameters(weight, bias, qi, qo)
        tensors[output] = Tensor(output, (1, oc, h, w), qo, "NCHW")
        layers.append(Layer("Conv", [inp], output,
                            {"pads": [kernel // 2] * 4, "strides": [1, 1],
                             "dilations": [1, 1], "group": group}, params))

    if kind == "basic":
        conv("x", "a", c, c, 3, 1, qx, qa)
        conv("a", "b", c, c, 3, 1, qa, qb)
    else:
        e = 3 * c
        conv("x", "e", c, e, 1, 1, qx, qa)
        tensors["er"] = Tensor("er", (1, e, h, w), qa, "NCHW")
        layers.append(Layer("Relu", ["e"], "er", {}, {}))
        conv("er", "d", e, e, 3, e, qa, qd)
        tensors["dr"] = Tensor("dr", (1, e, h, w), qd, "NCHW")
        layers.append(Layer("Relu", ["d"], "dr", {}, {}))
        conv("dr", "b", e, c, 1, 1, qd, qb)
    tensors["y"] = Tensor("y", shape, qy, "NCHW")
    layers.append(Layer("Add", ["b", "x"], "y", {}, add_params(qb, qx, qy)))
    program = Program(tensors, layers, ["x"], ["y"], {},
                      {"synthetic": True, "kind": kind, "purpose": "exact tile replay only"})
    x = rng.integers(-100, 101, shape, dtype=np.int16).astype(np.int8)
    x.reshape(-1)[:9] = np.array([-128, -127, -3, -1, 0, 1, 3, 126, 127], np.int8)
    return program, x


def replay_fixture(kind: str) -> dict:
    program, x = make_fixture(kind)
    oracle = evaluate(program, {"x": x})["y"]
    q = {name: t.quantization for name, t in program.tensors.items()}
    params = {layer.output: layer.parameters for layer in program.layers}
    h, w = x.shape[2:]
    matches = []
    for th, tw in ((1, 1), (2, 3), (4, 4), (7, 7)):
        tiled = np.empty_like(oracle)
        for tile in tiles(Case("fixture", "synthetic", -1, kind, 4, h, w,
                               3 if kind == "inverted" else 1), th, tw):
            source = tile.grow(2 if kind == "basic" else 1, h, w)
            xs = x[:, :, source.y0:source.y1, source.x0:source.x1]
            inner = tile.grow(1, h, w)
            if kind == "basic":
                a = conv_roi(xs, (source.y0, source.x0), (h, w), inner,
                             q["x"], q["a"], params["a"], group=1, pad=1)
                b = conv_roi(a, (inner.y0, inner.x0), (h, w), tile,
                             q["a"], q["b"], params["b"], group=1, pad=1)
            else:
                e = conv_roi(xs, (source.y0, source.x0), (h, w), inner,
                             q["x"], q["e"], params["e"], group=1, pad=0)
                e = np.maximum(e, q["e"].zero_point).astype(np.int8)
                d = conv_roi(e, (inner.y0, inner.x0), (h, w), tile,
                             q["er"], q["d"], params["d"], group=12, pad=1)
                d = np.maximum(d, q["d"].zero_point).astype(np.int8)
                b = conv_roi(d, (tile.y0, tile.x0), (h, w), tile,
                             q["dr"], q["b"], params["b"], group=1, pad=0)
            skip = x[:, :, tile.y0:tile.y1, tile.x0:tile.x1]
            result = manual_add(b, skip, q["b"], q["x"], q["y"], params["y"])
            tiled[:, :, tile.y0:tile.y1, tile.x0:tile.x1] = result
        assert np.array_equal(tiled, oracle), (kind, th, tw)
        matches.append(dict(tile=[th, tw], exact_values=int(tiled.size),
                            output_sha256=hashlib.sha256(tiled.tobytes()).hexdigest()))
    return dict(kind=kind, shape=list(x.shape), synthetic_parameters=True,
                source_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                unique_output_values=int(np.unique(oracle).size),
                saturated_output_values=int(np.count_nonzero((oracle == -128) | (oracle == 127))),
                oracle="compiler/integer_reference.py independent centered integer graph",
                matches=matches)


def add_rounding_edges() -> dict:
    """Exercise negative ties, positive ties, and both saturation endpoints."""
    qi = Quantization(.5, 0)
    qo = Quantization(1., 0)
    left = np.array([-128, -3, -1, 1, 3, 127, 127, -128, 0], np.int8).reshape(1, 1, 1, 9)
    right = np.array([0, 0, 0, 0, 0, 0, 127, -128, 0], np.int8).reshape(1, 1, 1, 9)
    params = add_params(qi, qi, qo)
    tensors = {name: Tensor(name, left.shape, q, "NCHW")
               for name, q in (("left", qi), ("right", qi), ("sum", qo))}
    program = Program(tensors, [Layer("Add", ["left", "right"], "sum", {}, params)],
                      ["left", "right"], ["sum"], {}, {"synthetic": True})
    expected = evaluate(program, {"left": left, "right": right})["sum"]
    direct = manual_add(left, right, qi, qi, qo, params)
    assert np.array_equal(direct, expected)
    assert direct.reshape(-1).tolist() == [-64, -2, -1, 1, 2, 64, 127, -128, 0]
    return dict(inputs=9, exact=True, expected=direct.reshape(-1).tolist(),
                common_shift=int(params["add_shift"][0]))


def run(output: Path = OUTPUT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    programs = {n: load_model(n)[0] for n in ("kws", "vww")}
    cases = [Case("vww-basic-32x24x24", "vww", 13, "basic", 32, 24, 24),
             Case("vww-inverted-32x24x24-e6", "vww", 13, "inverted", 32, 24, 24, 6),
             Case("kws-basic-64x25x5", "kws", 3, "basic", 64, 25, 5),
             Case("vww-basic-64x12x12", "vww", 17, "basic", 64, 12, 12)]
    for case in cases:
        assert programs[case.source_model].tensors[
            programs[case.source_model].layers[case.source_layer].output].shape == (
                1, case.channels, case.h, case.w)
    rows = []
    for case in cases:
        candidates = [evaluate_tile_choice(case, th, tw)
                      for th in (2, 3, 4, 5, 6, 8, 12)
                      for tw in (2, 3, 4, 5, 6, 8, 12)
                      if th <= case.h and tw <= case.w]
        policies = {key: best([row[key] for row in candidates])
                    for key in ("streamed", "resident_direct", "static_overwrite",
                                "ownership_zero_overhead", "ownership")}
        static = policies["static_overwrite"]
        owner = policies["ownership"]
        # Compare same tile, not just separately optimized winners.
        same_tile = all(row["ownership_zero_overhead"]["optimistic_serial_service"] ==
                        row["static_overwrite"]["optimistic_serial_service"] and
                        row["ownership_zero_overhead"]["peak_scratchpad_bytes"] ==
                        row["static_overwrite"]["peak_scratchpad_bytes"] and
                        row["ownership"]["optimistic_serial_service"] >=
                        row["static_overwrite"]["optimistic_serial_service"] and
                        row["ownership"]["peak_scratchpad_bytes"] >=
                        row["static_overwrite"]["peak_scratchpad_bytes"]
                        for row in candidates)
        assert same_tile
        rows.append(dict(case=asdict(case), full_tensor=full_tensor_row(case),
                         policy_best_feasible=policies,
                         all_tiles_static_dominates_ownership=same_tile,
                         best_static_overwrite_vs_owner_service=(
                             static["optimistic_serial_service"] - owner["optimistic_serial_service"]
                             if static and owner else None),
                         tile_candidates=len(candidates)))
    report = dict(schema=1, status="passed-model", physical_board=False,
                  hardware_add_supported=False,
                  scope="Representative synthetic residual graphs at real frozen tensor shapes; exact small INT8 replay plus optimistic analytical memory/port/MAC service, no native RTL or board result.",
                  assumptions=dict(scratchpad_bytes=SRAM_BYTES, one_port_bytes=PORT_BYTES,
                                   shared_engine_dma_port=True, mac_lanes=LANES,
                                   common_metadata_bytes=METADATA_BYTES,
                                   parameter_stage="one eight-output weight/bias/scale row; larger/full parameter residency not credited",
                                   add="one common denominator and one signed ties-away rounding; output saturates only after sum",
                                   tile_order="fixed row-major; source last use includes all branch halo reads"),
                  sources={str(p.relative_to(ROOT)): sha(p) for p in (
                      ROOT / "rtl/v2/scratchpad.sv", ROOT / "docs/specs/quantization.md",
                      ROOT / "compiler/integer_reference.py", ROOT / "compiler/static_pipeline.py",
                      ROOT / "tools/phase6/residual_ownership_screen.py",
                      ROOT / "work/phase4/kws-logits.onnx", ROOT / "work/phase4/vww-logits.onnx")},
                  exact_small_replay=[replay_fixture("basic"), replay_fixture("inverted")],
                  exact_add_rounding_edges=add_rounding_edges(),
                  cases=rows,
                  conclusion="NO-GO for runtime tile-ownership novelty under a fixed known graph/order: compile-time last-use schedule dominates it at identical capacity/port, and ordinary blocked fusion already avoids full-map intermediates.",
                  limitations=[
                      "Residual blocks are synthetic: pinned KWS/VWW graphs are linear, so model-layer correctness or real residual quality is not established.",
                      "The exact integer replay is a small synthetic fixture, not full-model inference; the MobileNetV2-style fixture uses ReLU instead of ReLU6.",
                      "The port/MAC service score is an optimistic analytical proxy, not cycles, energy, routed area, or a proven latency bound.",
                      "Only one eight-output parameter row is reserved; the model does not claim all weights fit in scratchpad.",
                      "Every 64-bit read/write is serialized; additional cache misses, DMA setup, alignment, bank mapping, arbitration and control timing can worsen any policy.",
                      "A dynamic graph or unpredictable tile order could make ownership necessary, but no such workload or hardware exists in this project."])
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    report = run()
    for row in report["cases"]:
        print(row["case"]["name"])
        for name, result in row["policy_best_feasible"].items():
            print(" ", name, None if result is None else
                  (result["tile"], result["peak_scratchpad_bytes"],
                   result["external_activation_bytes"], result["optimistic_serial_service"]))
    print(report["conclusion"])
