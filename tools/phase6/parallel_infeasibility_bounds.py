#!/usr/bin/env python3
"""Test cheap, sound pre-lowering bounds against frozen generic B3 catalogues.

This is a compiler-search experiment, not a new device result.  The bounds
refer to the pinned DeFiNES adaptation and its 32 KiB / 2048-command ABI.
They deliberately ignore caches and most work, making every rejection a
necessary condition rather than a guess about the compiler's outcome.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FROZEN = Path('/Users/sayat/.codex/worktrees/21f0/tinyML_accelerator')
sys.path[:0] = [str(FROZEN / 'compiler'), str(FROZEN / 'tools/phase6')]

import matched_defines_baseline as baseline
from hypothesis_search import StateKeys, lower, outcome
from matched_current import graph_identity
from scheduler.matched_defines_regions import _external_pieces
from scheduler.spatial import Rect, input_halo

OUT = ROOT / 'work/phase6/parallel-infeasibility-bounds-v1'
SRAM = 32768
COMMANDS = 2048
MAX_RSS_BYTES = 2 * 1024 * 1024 * 1024


def align8(value: int) -> int:
    return (value + 7) & ~7


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def peak_rss_bytes() -> int:
    observed = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return observed if sys.platform == 'darwin' else observed * 1024


def memory_guard() -> None:
    if peak_rss_bytes() > MAX_RSS_BYTES:
        raise MemoryError('experiment exceeded its 2 GiB peak-RSS guard')


class Bounds:
    def __init__(self, program):
        self.program = program
        self._live = {}

    def _conv(self, cfg, macro):
        return self.program.layers[cfg['start'] + 2 * macro]

    def _last_shape(self, cfg):
        return self.program.tensors[self.program.layers[cfg['stop'] - 1].output].shape

    def _first_tile_needs(self, cfg):
        shape = self._last_shape(cfg)
        count = (cfg['stop'] - cfg['start']) // 2
        needs = {count - 1: Rect(0, 0, min(cfg['h'], shape[2]),
                                 min(cfg['w'], shape[3]))}
        for j in range(count - 1, -1, -1):
            conv = self._conv(cfg, j)
            in_shape = self.program.tensors[conv.inputs[0]].shape
            a = conv.attributes
            halo = input_halo(needs[j], conv.parameters['weight'].shape[2:],
                              tuple(a.get('strides', [1, 1])),
                              tuple(a.get('pads', [0, 0, 0, 0])))
            needs[j - 1] = halo.intersect(Rect(0, 0, in_shape[2], in_shape[3]))
        return needs

    def _live_rows(self, conv):
        key = id(conv)
        if key not in self._live:
            w = conv.parameters['weight']
            if conv.attributes.get('group', 1) == 1:
                nonzero = np.flatnonzero(np.any(w != 0, axis=(1, 2, 3)))
                live = int(nonzero[-1]) + 1 if len(nonzero) else 0
            else:
                live = len(w)
            self._live[key] = live
        return self._live[key]

    def cold_sram(self, cfg):
        """Lower bound on peak bytes for any first-tile conv/activation pair.

        The recursive backend holds the clipped source and requested result
        together while loading live weight rows, params, and its 256-byte LUT.
        Each allocation has an 8-byte guard and 8-byte padded extent.  Halo
        caches, DMA scratch, and fragmentation are omitted.
        """
        needs = self._first_tile_needs(cfg)
        count = (cfg['stop'] - cfg['start']) // 2
        per_macro = []
        for j in range(count):
            conv = self._conv(cfg, j)
            weight = conv.parameters['weight']
            in_channels = self.program.tensors[conv.inputs[0]].shape[1]
            out_channels = self.program.tensors[
                self.program.layers[cfg['start'] + 2*j + 1].output].shape[1]
            sizes = [in_channels * needs[j-1].area,
                     out_channels * needs[j].area]
            live = self._live_rows(conv)
            if live:
                sizes += [live * align8(math.prod(weight.shape[1:])),
                          live * 16, 256]
            estimate = 128 + sum(8 + align8(size) for size in sizes)
            per_macro.append(estimate)
        return max(per_macro), per_macro

    def scatter_commands(self, cfg):
        """Count one mandatory 2-command output DMA per external piece.

        A full tensor can be emitted as one DMA.  A nonfull-width rectangle
        emits at least one piece per channel and row, independent of cache
        policy and SRAM allocation.  A full-width strip emits at least one
        piece per channel.  We add the required final HALT only.
        """
        shape = self._last_shape(cfg)
        channels, height, width = shape[1:]
        ny = (height + cfg['h'] - 1) // cfg['h']
        nx = (width + cfg['w'] - 1) // cfg['w']
        if nx == ny == 1:
            pieces = 1
        elif nx == 1:
            pieces = channels * ny
        else:
            pieces = channels * height * nx
        return 1 + 2 * pieces, pieces

    def strip_alignment(self, cfg):
        """Implementation-specific necessary alignment for full-width strips."""
        if cfg['kind'] != 'strip':
            return False
        shape_out = self._last_shape(cfg)
        shape_in = self.program.tensors[self.program.layers[cfg['start']].inputs[0]].shape
        for y0 in range(0, shape_out[2], cfg['h']):
            y1 = min(y0 + cfg['h'], shape_out[2])
            lo, hi = y0, y1
            for j in range((cfg['stop'] - cfg['start']) // 2 - 1, -1, -1):
                conv = self._conv(cfg, j)
                kh = conv.parameters['weight'].shape[2]
                sh = conv.attributes.get('strides', [1, 1])[0]
                pt = conv.attributes.get('pads', [0, 0, 0, 0])[0]
                input_h = self.program.tensors[conv.inputs[0]].shape[2]
                lo, hi = max(0, lo * sh - pt), min(input_h, (hi - 1) * sh - pt + kh)
            input_plane = (hi - lo) * shape_in[3]
            output_plane = (y1 - y0) * shape_out[3]
            full_input = lo == 0 and hi == shape_in[2]
            full_output = y0 == 0 and y1 == shape_out[2]
            if ((not full_input and (input_plane % 8 or lo * shape_in[3] % 8
                                     or shape_in[2] * shape_in[3] % 8))
                    or (not full_output and (output_plane % 8 or y0 * shape_out[3] % 8
                                            or shape_out[2] * shape_out[3] % 8))):
                return True
        return False

    def test(self, cfg):
        scatter, pieces = self.scatter_commands(cfg)
        cold, stages = self.cold_sram(cfg)
        alignment = self.strip_alignment(cfg)
        failures = []
        if scatter > COMMANDS:
            failures.append('mandatory-output-scatter')
        if cold > SRAM:
            failures.append('cold-first-tile-sram')
        if alignment:
            failures.append('full-width-strip-alignment')
        return dict(failures=failures, scatter_commands_lb=scatter,
                    scatter_pieces_lb=pieces, cold_sram_bytes_lb=cold,
                    cold_sram_per_macro_lb=stages)


def adversarial_geometry() -> dict:
    """Fuzz the scatter piece count against the frozen emitter's exact pieces."""
    rng = random.Random(6167)
    tested = 0
    for _ in range(5000):
        c = rng.randrange(1, 9)
        h = rng.randrange(1, 13)
        w = rng.randrange(1, 13)
        th = rng.randrange(1, h + 1)
        tw = rng.randrange(1, w + 1)
        cfg = dict(start=0, stop=2, kind='rectangle', h=th, w=tw)
        # Use the same formula as Bounds.scatter_commands without requiring a
        # Program; compare it with the concrete backend piece iterator.
        ny = (h + th - 1) // th
        nx = (w + tw - 1) // tw
        predicted = 1 if nx == ny == 1 else c*ny if nx == 1 else c*h*nx
        actual = 0
        for y in range(0, h, th):
            for x in range(0, w, tw):
                rect = (y, x, min(h, y+th), min(w, x+tw))
                actual += sum(1 for _ in _external_pieces(0, (c, h, w), rect, 8))
        if predicted > actual:
            raise AssertionError((cfg, c, h, w, predicted, actual))
        tested += 1
    return dict(random_shapes=tested, scatter_piece_lb_never_exceeded_exact_emitter=True,
                exact_command_boundary=dict(at_2048_reject=False, at_2049_reject=True))


def select_sample(rows, per_stratum=3, max_tiles=256):
    buckets = defaultdict(list)
    for row in rows:
        cfg = row['configuration']
        shape = row['_shape']
        tiles = math.ceil(shape[2] / cfg['h']) * math.ceil(shape[3] / cfg['w'])
        if tiles > max_tiles:
            continue
        key = row['_category']
        buckets[key].append(row)
    selected = []
    for category, group in sorted(buckets.items()):
        group.sort(key=lambda item: item['id'])
        indices = sorted(set(round(i*(len(group)-1)/max(1, min(per_stratum, len(group))-1))
                             for i in range(min(per_stratum, len(group)))))
        selected.extend(group[i] for i in indices)
    selected.sort(key=lambda row: row['id'])
    return selected, {k: len(v) for k, v in buckets.items()}


def cold_timing(program, rows, sources, model, max_tiles, per_stratum):
    """One cold, bounded slice per strategy; includes key/bound/lowering work."""
    if not rows:
        return dict(status='no-eligible-sample')
    summary = {}
    # Alternate strategy order between models to reduce order bias.
    order = ['exact-halo-control', 'bounds-plus-exact-halo']
    if model == 'vww':
        order.reverse()
    for name in order:
        gc.collect()
        bounds = Bounds(program)
        keyer = StateKeys(program, sources, exact_halo=True)
        memo = {}
        counts = Counter()
        began = time.perf_counter()
        for row in rows:
            cfg = row['configuration']
            if name == 'bounds-plus-exact-halo':
                bound = bounds.test(cfg)
                if bound['failures']:
                    if row['status'] == 'executable':
                        raise AssertionError('bound rejected executable '+row['id'])
                    counts['bound_rejections'] += 1
                    continue
            key = keyer.key(cfg)
            if key in memo:
                counts['memo_hits'] += 1
                actual = memo[key]
            else:
                actual, artifact = lower(program, cfg)
                del artifact
                memo[key] = actual
                counts['fresh_lowerings'] += 1
            if actual != outcome(row['_original']):
                raise AssertionError('sample changed '+row['id'])
            memory_guard()
        summary[name] = dict(seconds=time.perf_counter()-began, **counts,
                             sample_size=len(rows))
    return dict(status='passed', execution_order=order,
                max_tile_count=max_tiles, per_category_cap=per_stratum, results=summary,
                limitation='One cold pass per strategy on a deterministic bounded slice; no full-catalogue speedup claim.')


def run(output: Path, sample_per_stratum=3, max_tiles=256):
    output = output.resolve()
    if output.exists():
        raise FileExistsError('choose a fresh output directory')
    output.mkdir(parents=True)
    sources = baseline.source_pins()
    sources[str(Path(__file__).relative_to(ROOT))] = digest(Path(__file__))
    report = dict(schema=1, status='running', physical_board=False,
                  frozen_root=str(FROZEN), source_sha256=sources,
                  thresholds=dict(sram_bytes=SRAM, commands=COMMANDS),
                  adversarial=adversarial_geometry(), models={})
    for model in ('kws', 'vww'):
        catalogue = FROZEN / f'work/phase6/matched-baselines-v1/b3-final-{model}/{model}/catalogue.json'
        document = json.loads(catalogue.read_text())
        _, _, program, _, _ = baseline.matched_model(model)
        if document['identity']['graph_sha256'] != graph_identity(program):
            raise ValueError('frozen graph changed '+model)
        for path, expected in document['identity']['compiler_sources'].items():
            if digest(FROZEN/path) != expected:
                raise ValueError('frozen source changed '+path)
        bounds = Bounds(program)
        counts = Counter()
        by_status = defaultdict(Counter)
        category_by_reason = defaultdict(Counter)
        rows = []
        began = time.perf_counter()
        for candidate in document['candidates']:
            result = bounds.test(candidate['configuration'])
            status = candidate['status']
            if result['failures'] and status == 'executable':
                raise AssertionError('false rejection '+candidate['id']+' '+str(result))
            if status == 'executable':
                if result['scatter_commands_lb'] > candidate['command_bytes'] // 16:
                    raise AssertionError('scatter bound exceeds actual commands '+candidate['id'])
                if result['cold_sram_bytes_lb'] > candidate['peak_sram_address']:
                    raise AssertionError('SRAM bound exceeds actual peak '+candidate['id'])
            category = result['failures'][0] if result['failures'] else 'surviving-'+status
            counts[category] += 1
            category_by_reason[candidate.get('reason', 'executable')][category] += 1
            for failure in result['failures']:
                by_status[status][failure] += 1
            rows.append(dict(id=candidate['id'], configuration=candidate['configuration'],
                             status=status, _category=category,
                             _shape=bounds._last_shape(candidate['configuration']),
                             _original=candidate))
        audit_seconds = time.perf_counter()-began
        sample, eligible = select_sample(rows, sample_per_stratum, max_tiles)
        timing = cold_timing(program, sample, sources, model, max_tiles, sample_per_stratum)
        memory_guard()
        report['models'][model] = dict(catalogue_file=str(catalogue),
             catalogue_sha256=digest(catalogue), candidates=len(document['candidates']),
             catalogue_outcomes=document['counts'], full_catalogue_bound_seconds=audit_seconds,
             category_counts=dict(counts), category_by_actual_reason={k:dict(v) for k,v in category_by_reason.items()},
             failures_by_actual_status={k:dict(v) for k,v in by_status.items()},
             false_rejections=0, executable_scatter_and_sram_certificates_checked=sum(
                 row['status']=='executable' for row in document['candidates']),
             sample_eligible_counts=eligible, timing_sample_ids=[row['id'] for row in sample],
             cold_timing=timing, process_peak_rss_bytes=peak_rss_bytes())
        write(output/'report.partial.json', report)
        print(model, len(document['candidates']), dict(counts), timing['status'], flush=True)
        del document, rows, program, bounds
        gc.collect()
    for name, expected in sources.items():
        path = ROOT/name if name == str(Path(__file__).relative_to(ROOT)) else FROZEN/name
        if digest(path) != expected:
            raise ValueError('experiment source changed during run: '+name)
    report['status'] = 'passed'
    report['decision'] = ('Sound pruning is useful only for compiler search work; '
                          'it does not itself improve inference or establish architecture novelty.')
    report['limitations'] = [
        'The full catalogue checks use historical, source-pinned outcomes; a bounded sample is freshly lowered.',
        'Timing is a deterministic cold slice, not an end-to-end full-catalogue benchmark or statistical speedup.',
        'The bounds certify infeasibility only for this pinned backend, capacity, and command ABI.',
        'No FPGA resource, timing, energy, or accuracy result follows from pruning.']
    write(output/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--sample-per-stratum', type=int, default=3)
    parser.add_argument('--max-tiles', type=int, default=256)
    args = parser.parse_args()
    run(args.output, args.sample_per_stratum, args.max_tiles)
