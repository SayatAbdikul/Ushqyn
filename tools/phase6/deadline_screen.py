#!/usr/bin/env python3
"""Bounded discrete-job scheduling screen using exact native command traces.

The selected co-issue engine is unchanged. Additional interruption points are
created only by splitting independent output channels/rows in command fixtures.
Arithmetic and total command timing are checked in native RTL. Scheduling,
checkpoint transport and interruption control remain explicit cost proxies.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import resource
import struct
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path('/Users/sayat/.codex/worktrees/21f0/tinyML_accelerator')
BASE = ROOT / 'work/phase6/deadline-screen-v1'
sys.path.insert(0, str(ROOT / 'compiler'))
from hardware_v2 import Descriptor

PARENT_SHA = '9f934dcc27ae891e4c7b8b84bcc4b65e82d903492c1867572784fbe1cd9484ee'
MEM = 32768
UART_BAUD = 750000  # Selected route27/report.json; older target default is 115200.


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def commands(data):
    if len(data) % 16 or not data:
        raise ValueError('invalid command stream')
    rows = list(struct.iter_unpack('<BBHIII', data))
    if rows[-1] != (0, 0, 0, 0, 0, 0):
        raise ValueError('missing HALT')
    return rows


def pack(rows):
    return b''.join(struct.pack('<BBHIII', *r) for r in rows)


def mask(base, length):
    if not (0 <= base <= base + length <= MEM):
        raise ValueError('SRAM range')
    return ((1 << length) - 1) << base


def intervals(bits):
    result = []
    while bits:
        first = (bits & -bits).bit_length() - 1
        shifted = bits >> first
        length = (shifted ^ (shifted + 1)).bit_length() - 1
        result.append((first, length))
        bits &= ~mask(first, length)
    return result


def dma_extent(bits):
    """8-byte granules sufficient for a sparse byte set; no double counting."""
    words = 0
    for start, length in intervals(bits):
        lo, hi = start // 8, (start + length + 7) // 8
        words |= ((1 << (hi - lo)) - 1) << lo
    return words.bit_count() * 8


def decode(raw):
    raw = bytearray(raw)
    fusion = int.from_bytes(raw[6:8], 'little')
    raw[6:8] = b'\0\0'
    d = Descriptor.decode(bytes(raw))
    d.validate()
    return d, fusion


def descriptor_accesses(d, fusion, pc):
    """Semantic operand bytes; independent channels, exact spatial footprints.

    Padded weight lanes are excluded; descriptor fetch and epilogue table are
    included. Reading irrelevant bytes in a physical SRAM word does not make
    them semantic checkpoint state.
    """
    reads = mask(pc, 64) | mask(d.next_pc, 64)
    if fusion & 1:
        reads |= mask((fusion >> 1) * 8, 256)
    if d.opcode in (4, 5, 6, 7):
        oh = (d.input_h + d.pad_top + d.pad_bottom - d.kernel_h) // d.stride_h + 1
        ow = (d.input_w + d.pad_left + d.pad_right - d.kernel_w) // d.stride_w + 1
        plane = d.input_h * d.input_w
        footprint = 0
        for oy in range(oh):
            for ky in range(d.kernel_h):
                y = oy * d.stride_h - d.pad_top + ky
                if not 0 <= y < d.input_h:
                    continue
                for ox in range(ow):
                    x = ox * d.stride_w - d.pad_left
                    lo, hi = max(0, x), min(d.input_w, x + d.kernel_w)
                    if hi > lo:
                        footprint |= mask(y * d.input_w + lo, hi - lo)
        for channel in range(d.input_c):
            reads |= footprint << (d.input + channel * plane)
    else:
        reads |= mask(d.input, d.count)
    if d.opcode in (1, 4, 6):
        rows = d.outputs if d.opcode == 1 else d.output_c
        for row in range(rows):
            reads |= mask(d.weight + row * d.row_stride, d.count)
        reads |= mask(d.params, rows * 16)
    elif d.opcode in (2, 5, 7, 8):
        reads |= mask(d.params, 16)
    return reads, mask(d.output, d.outputs)


def analyze_fixture(folder):
    """Independent byte-version replay and backward SRAM liveness.

    This proves the saved bytes preserve future operand *versions*. Arithmetic
    is verified separately by the unchanged native RTL and fixture oracle.
    """
    rows = commands((folder / 'commands.bin').read_bytes())
    ext = bytearray((folder / 'payload.bin').read_bytes())
    inp = (folder / 'input.bin').read_bytes()
    ext[:len(inp)] = inp
    sram = bytearray(MEM)
    # Unique token per external byte; output writes create new independent tokens.
    ext_version = list(range(len(ext)))
    versions = [-1] * MEM
    next_token = len(ext)
    effects, states, descriptors = [], [], {}
    pending_dma, pending_engine = None, None

    def finish_dma():
        nonlocal pending_dma
        if pending_dma is None:
            return
        direction, a, b, c = pending_dma
        if direction:
            sram[b:b+c] = ext[a:a+c]
            versions[b:b+c] = ext_version[a:a+c]
        else:
            if min(versions[b:b+c]) < 0:
                raise ValueError('store reads uninitialized state')
            ext_version[a:a+c] = versions[b:b+c]
        pending_dma = None

    def finish_engine():
        nonlocal pending_engine, next_token
        if pending_engine is not None:
            d = pending_engine
            versions[d.output:d.output+d.outputs] = range(next_token, next_token+d.outputs)
            next_token += d.outputs
            pending_engine = None

    for index, (op, flags, reserved, a, b, c) in enumerate(rows):
        reads, writes = 0, 0
        if reserved:
            raise ValueError('reserved bits')
        if op == 1:
            if pending_dma:
                raise ValueError('overlapping DMA')
            if not 0 <= a <= a+c <= len(ext):
                raise ValueError('external range')
            if flags:
                writes = mask(b, c)
            else:
                reads = mask(b, c)
            pending_dma = flags, a, b, c
        elif op == 2:
            if pending_dma or pending_engine:
                raise ValueError('RUN without quiescence')
            d, fusion = decode(sram[a:a+64])
            if d.opcode == 0 or Descriptor.decode(bytes(sram[d.next_pc:d.next_pc+64])).opcode:
                raise ValueError('screen supports one descriptor followed by HALT')
            reads, writes = descriptor_accesses(d, fusion, a)
            for start, size in intervals(reads):
                if min(versions[start:start+size]) < 0:
                    raise ValueError('uninitialized operand')
            pending_engine = d
            descriptors[index] = {'raw_hex': bytes(sram[a:a+64]).hex(),
                                  'descriptor': d.__dict__, 'fusion': fusion}
        elif op == 3:
            if flags & 1:
                finish_engine()
            if flags & 2:
                finish_dma()
        elif op == 0:
            if pending_dma or pending_engine:
                raise ValueError('HALT with pending unit')
        else:
            raise ValueError('unknown command')
        effects.append((reads, writes))
        if op == 3 and pending_dma is None and pending_engine is None:
            # Immutable/source activation bytes already backed in private SDRAM
            # need only reload; outputs not yet committed need save and reload.
            backed = set(ext_version)
            dirty = 0
            for addr, token in enumerate(versions):
                if token >= 0 and token not in backed:
                    dirty |= 1 << addr
            states.append({'next_index': index+1, 'dirty_mask': dirty})
    live = 0
    live_at = {len(rows): 0}
    for index in range(len(rows)-1, -1, -1):
        read, write = effects[index]
        live = read | (live & ~write)
        live_at[index] = live
    for state in states:
        bits = live_at[state['next_index']]
        dirty = state.pop('dirty_mask') & bits
        # Independently check that arbitrary foreign-job overwrites of every
        # unsaved byte cannot reach a later read before that byte is replaced.
        corrupt = ((1 << MEM)-1) & ~bits
        for read, write in effects[state['next_index']:]:
            if read & corrupt:
                raise ValueError('checkpoint omits a future operand byte')
            corrupt &= ~write
        state.update(live_bytes=bits.bit_count(), live_dma_bytes=dma_extent(bits),
                     dirty_bytes=dirty.bit_count(), save_dma_bytes=dma_extent(dirty),
                     live_intervals=intervals(bits))
    return {'commands': rows, 'descriptors': descriptors, 'boundaries': states,
            'program_bytes': len(rows)*16,
            'dma_payload_bytes': sum(r[5] for r in rows if r[0] == 1)}


def build_trace():
    selected = REFERENCE / 'work/phase6/engine-candidate-rtl-v2'
    report_path = selected / 'native/report.json'
    report = json.loads(report_path.read_text())
    if report['status'] != 'passed' or sha(selected / 'engine.sv') != PARENT_SHA:
        raise ValueError('selected native evidence identity')
    sources = [ROOT / 'rtl/v2' / n for n in ('target_pkg.sv', 'requantizer.sv')]
    sources += [selected / 'engine.sv']
    sources += [ROOT / 'rtl/v2' / n for n in ('scratchpad.sv', 'tile_dma.sv',
                'tiled_core.sv', 'command.sv', 'tile_sequencer.sv', 'tiled_host_bridge.sv')]
    for p in sources:
        key = ('work/phase6/engine-candidate-rtl-v2/engine.sv' if p.name == 'engine.sv'
               else str(p.relative_to(ROOT)))
        if sha(p) != report['source_sha256'][key]:
            raise ValueError('native source mismatch: ' + key)
    harness = (ROOT / 'test/phase6/native.cpp').read_text()
    harness = harness.replace('static Request hold;', '''static Request hold;
struct Trace { unsigned index, elapsed, engine, dma, eb, db; };
static std::vector<Trace> trace;
static bool previous_busy=false;
static unsigned previous_index=0;
static void record_trace(){
    auto* r=d.rootp;
    bool busy=r->v2_tiled_host_bridge__DOT__seq_busy;
    unsigned index=r->v2_tiled_host_bridge__DOT__sequencer__DOT__command_index;
    if((busy&&(!previous_busy||index!=previous_index))||(!busy&&previous_busy))
        trace.push_back({index,r->v2_tiled_host_bridge__DOT__seq_elapsed,
            r->v2_tiled_host_bridge__DOT__seq_engine_cycles,
            r->v2_tiled_host_bridge__DOT__seq_dma_cycles,
            r->v2_tiled_host_bridge__DOT__engine_busy,
            r->v2_tiled_host_bridge__DOT__dma_busy});
    previous_busy=busy; previous_index=index;
}''')
    harness = harness.replace('d.clk=1;d.eval();cycles++;return out;',
                              'd.clk=1;d.eval();cycles++;record_trace();return out;')
    site = '    std::cout<<argv[1]'
    harness = harness.replace(site, '''    std::ofstream tf(std::string(argv[3])+".trace.json");
    tf<<"[";
    for(unsigned i=0;i<trace.size();i++){
        if(i)tf<<",";
        auto t=trace[i];
        tf<<"{\\"index\\":"<<t.index<<",\\"elapsed\\":"<<t.elapsed
          <<",\\"engine_cycles\\":"<<t.engine<<",\\"dma_cycles\\":"<<t.dma
          <<",\\"engine_busy\\":"<<t.eb<<",\\"dma_busy\\":"<<t.db<<"}";
    }
    tf<<"]\\n";
''' + site)
    path = BASE / 'trace.cpp'
    path.write_text(harness)
    build = BASE / 'native-build'
    build.mkdir(exist_ok=True)
    with (build / 'build.log').open('w') as log:
        subprocess.run(['verilator', '--cc', '--exe', '--build', '-j', '2',
                        '--public-flat-rw', '-Wno-fatal', '--top-module',
                        'v2_tiled_host_bridge', '--Mdir', str(build),
                        *map(str, sources), str(path)], cwd=ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    exe = build / 'Vv2_tiled_host_bridge'
    save(BASE / 'native-build/identity.json', {
        'engine_sha256': PARENT_SHA, 'selected_report_sha256': sha(report_path),
        'sources_sha256': {str(p): sha(p) for p in sources},
        'harness_sha256': sha(path), 'executable_sha256': sha(exe)})
    return exe, report


def run_native(exe, folder, label, seed):
    dest = BASE / 'native' / f'{label}-s{seed}.json'
    dest.parent.mkdir(exist_ok=True)
    subprocess.run([str(exe), str(folder), str(seed), str(dest)], cwd=ROOT,
                   check=True, capture_output=True, text=True, timeout=120)
    result = json.loads(dest.read_text())
    trace = json.loads(Path(str(dest)+'.trace.json').read_text())
    rows = commands((folder / 'commands.bin').read_bytes())
    if result['status'] != 'passed' or len(trace) != len(rows)+1:
        raise ValueError('native trace coverage')
    if [t['index'] for t in trace[:-1]] != list(range(len(rows))):
        raise ValueError('native trace missing command')
    if trace[0]['elapsed'] != 0 or trace[-1]['elapsed'] != result['elapsed_cycles']:
        raise ValueError('native trace elapsed total')
    return result, trace


def split_fixture(source, destination, analysis, trace, target_cycles):
    """Split whole output channels/rows, retaining all original operand bytes."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('input.bin', 'output.bin', 'checks.txt'):
        (destination / name).write_bytes((source / name).read_bytes())
    # AD layer-snapshot files are also checked; no model arrays are loaded.
    for line in (source / 'checks.txt').read_text().splitlines():
        name = line.split()[1]
        (destination / name).write_bytes((source / name).read_bytes())
    payload = bytearray((source / 'payload.bin').read_bytes())
    rows, changes = [], []
    original = analysis['commands']
    for index, row in enumerate(original):
        if row[0] != 2:
            rows.append(row)
            continue
        info = analysis['descriptors'][index]
        d = Descriptor(**info['descriptor'])
        if d.opcode == 1:
            channels, plane = d.outputs, 1
        elif d.opcode in (4, 6):
            channels, plane = d.output_c, d.outputs // d.output_c
        else:
            rows.append(row)
            continue
        end = next(j for j in range(index+1, len(original))
                   if original[j][0] == 3 and original[j][1] & 1)
        duration = trace[end+1]['elapsed'] - trace[index]['elapsed']
        group = max(1, min(channels, int(target_cycles * channels / duration)))
        if d.opcode == 6:
            # The existing descriptor ABI requires every input base to be
            # eight-byte aligned; channel planes need not be eight bytes long.
            alignment = 8 // math.gcd(d.input_h*d.input_w, 8)
            group = max(alignment, group // alignment * alignment)
        if group >= channels:
            rows.append(row)
            continue
        # Added WAIT3 gives a legal quiescent point even if a previous prefetch
        # remains pending. Preserve original overlap after the last split RUN.
        rows.append((3, 3, 0, 0, 0, 0))
        groups = []
        for first in range(0, channels, group):
            count = min(group, channels-first)
            q = copy.copy(d)
            q.output += first * plane
            q.outputs = count * plane
            q.weight += first * d.row_stride
            q.params += first * 16
            if q.opcode != 1:
                q.output_c = count
            if q.opcode == 6:
                q.input_c = count
                q.input += first * d.input_h * d.input_w
            q.validate()
            encoded = bytearray(q.encode())
            encoded[6:8] = int(info['fusion']).to_bytes(2, 'little')
            payload.extend(bytes((-len(payload)) % 8))
            ext = len(payload)
            payload.extend(encoded + Descriptor(0).encode())
            rows += [(1, 1, 0, ext, row[3], 128), (3, 2, 0, 0, 0, 0), row]
            if first+count < channels:
                rows.append((3, 3, 0, 0, 0, 0))
            groups.append(count)
        changes.append({'original_command': index, 'opcode': d.opcode,
                        'original_channels': channels, 'groups': groups,
                        'original_native_run_interval_cycles': duration})
    program = pack(rows)
    if len(program) > MEM or len(payload) > 8*1024*1024:
        raise ValueError('split fixture does not fit current program/external memory')
    (destination / 'commands.bin').write_bytes(program)
    (destination / 'payload.bin').write_bytes(payload)
    save(destination / 'changes.json', {'target_cycles': target_cycles, 'changes': changes,
                                      'program_bytes': len(program), 'payload_bytes': len(payload)})
    return changes


def segments(analysis, trace):
    boundaries = {s['next_index']: s for s in analysis['boundaries']}
    result, previous = [], 0
    for next_index, state in sorted(boundaries.items()):
        event = trace[next_index]
        if event['engine_busy'] or event['dma_busy']:
            raise ValueError('symbolic/native boundary disagreement')
        if event['elapsed'] > previous:
            result.append(dict(state, cycles=event['elapsed']-previous))
            previous = event['elapsed']
    final = trace[-1]['elapsed']
    if final > previous:
        result.append({'cycles': final-previous, 'next_index': len(trace)-1,
                       'live_bytes': 0, 'live_dma_bytes': 0,
                       'dirty_bytes': 0, 'save_dma_bytes': 0})
    if sum(s['cycles'] for s in result) != final:
        raise ValueError('segment conservation')
    return result


def ad_timed_fixture(source, destination):
    """Remove diagnostic layer snapshots, keeping the actual inference program."""
    destination.mkdir(parents=True, exist_ok=True)
    # The source AD plan has two 640-byte external activation slots. Stores
    # beyond those slots are diagnostic snapshots, not inference dependencies.
    original = commands((source / 'commands.bin').read_bytes())
    rows, removed, i = [], 0, 0
    while i < len(original):
        r = original[i]
        if r[0] == 1 and r[1] == 0 and r[3] >= 1280:
            if original[i+1] != (3, 2, 0, 0, 0, 0):
                raise ValueError('AD diagnostic store is not independently waited')
            removed += r[5]; i += 2
        else:
            rows.append(r); i += 1
    for name in ('payload.bin', 'input.bin', 'output.bin'):
        (destination / name).write_bytes((source / name).read_bytes())
    (destination / 'commands.bin').write_bytes(pack(rows))
    final = next(line for line in (source / 'checks.txt').read_text().splitlines()
                 if line.split()[1] == 'output.bin')
    (destination / 'checks.txt').write_text(final+'\n')
    save(destination / 'origin.json', {'source': str(source), 'activation_slot_bytes': 640,
                                    'removed_diagnostic_dma_bytes': removed})
    return destination


@dataclass
class Job:
    model: str
    release: int
    deadline: int
    identity: int
    index: int = 0
    completion: int | None = None


def schedule(profiles, releases, policy, word_cycles, fixed_cycles, program_mode,
             slot_cycles=100000, retain_trace=False):
    """EDF changes owner only at quiescent boundaries; overhead blocks service."""
    jobs = [Job(*r) for r in releases]
    jobs.sort(key=lambda j: (j.release, j.identity))
    now, cursor, current, previous = 0, 0, None, None
    ready, events = [], []
    overhead, switches = 0, 0
    max_pending = {m: 0 for m in profiles}
    pending_counts = {m: 0 for m in profiles}
    slot_order = ('kws', 'vww', 'ad')
    # Uniform cost for *all* policies, including cold initial program selection.
    while cursor < len(jobs) or ready:
        while cursor < len(jobs) and jobs[cursor].release <= now:
            job = jobs[cursor]
            ready.append(job); cursor += 1
            pending_counts[job.model] += 1
            max_pending[job.model] = max(max_pending[job.model], pending_counts[job.model])
        if not ready:
            now = jobs[cursor].release
            continue
        if policy == 'nonpreemptive' and current in ready:
            chosen = current
        elif policy == 'static':
            owner = slot_order[(now // slot_cycles) % len(slot_order)]
            matching = [j for j in ready if j.model == owner]
            if not matching:
                now = (now // slot_cycles + 1) * slot_cycles
                continue
            chosen = min(matching, key=lambda j: (j.deadline, j.release, j.identity))
        else:
            chosen = min(ready, key=lambda j: (j.deadline, j.release, j.identity))
        if chosen is not previous:
            cost = fixed_cycles
            saved = restored = 0
            if previous is not None and previous.completion is None and previous.index:
                checkpoint = profiles[previous.model]['segments'][previous.index-1]
                saved = checkpoint['save_dma_bytes']
                cost += math.ceil(saved/8) * word_cycles
            if chosen.index:
                checkpoint = profiles[chosen.model]['segments'][chosen.index-1]
                restored = checkpoint['live_dma_bytes']
                cost += math.ceil(restored/8) * word_cycles
            # 64 bytes of PC/context on each suspended or resumed job.
            if previous is not None and previous.completion is None:
                cost += 8 * word_cycles
            if chosen.index:
                cost += 8 * word_cycles
            if program_mode == 'reload':
                cost += math.ceil(profiles[chosen.model]['program_bytes']/8) * word_cycles
            elif program_mode == 'uart_reload':
                # Selected 750000-baud host: full program writes in <=64-byte
                # packets, 12-byte request framing and a 13-byte ACK. This is
                # only a transport lower bound; polling and pause are extra.
                program_bytes = profiles[chosen.model]['program_bytes']
                wire_bytes = program_bytes + 25*math.ceil(program_bytes/64)
                cost += math.ceil(wire_bytes*10*27000000/UART_BAUD)
                cost += math.ceil(26*10*27000000/UART_BAUD)
            now += cost; overhead += cost; switches += 1
            if retain_trace:
                events.append({'start': now-cost, 'end': now, 'kind': 'switch',
                               'job': chosen.identity, 'save_bytes': saved,
                               'reload_live_bytes': restored})
        current = previous = chosen
        start = now
        segment = profiles[chosen.model]['segments'][chosen.index]
        now += segment['cycles']; chosen.index += 1
        if retain_trace:
            events.append({'start': start, 'end': now, 'kind': 'service',
                           'job': chosen.identity, 'model': chosen.model,
                           'segment': chosen.index-1})
        if chosen.index == len(profiles[chosen.model]['segments']):
            chosen.completion = now; ready.remove(chosen); current = None
            pending_counts[chosen.model] -= 1
    misses = sum(j.completion > j.deadline for j in jobs)
    service = sum(sum(s['cycles'] for s in profiles[j.model]['segments']) for j in jobs)
    responses = {m: [j.completion-j.release for j in jobs if j.model == m]
                 for m in profiles}
    return {'misses': misses, 'jobs': len(jobs), 'switches': switches,
            'overhead_cycles': overhead, 'service_cycles': service,
            'finish_cycle': now, 'max_lateness_cycles': max(j.completion-j.deadline for j in jobs),
            'max_response_cycles': {m: max(v) if v else 0 for m, v in responses.items()},
            'max_pending_instances': max_pending,
            'resident_program_capacity_upper_bytes': sum(max(1, max_pending[m])*
                profiles[m]['program_bytes'] for m in profiles),
            'completed': len(jobs), 'trace': events if retain_trace else None}


def release_grid(baseline):
    """Equal job inputs for all schedules; periodic and urgent-arrival scenarios."""
    costs = {m: sum(s['cycles'] for s in p['segments']) for m, p in baseline.items()}
    cases = []
    for utilization in (0.35, 0.55, 0.70):
        # Fixed workload proportions; periods scale together with utilization.
        periods = {m: int(costs[m]*3/utilization) for m in costs}
        horizon = min(periods.values()) * 18
        for phase in (0, 0.37, 0.73):
            for urgency in (1.10, 1.40, 2.00):
                rows, identity = [], 0
                for m in ('kws', 'vww', 'ad'):
                    offset = int(phase*costs['vww']) if m != 'vww' else 0
                    for release in range(offset, horizon, periods[m]):
                        relative = max(int(costs[m]*urgency), int(periods[m]*0.25))
                        rows.append((m, release, release+relative, identity)); identity += 1
                cases.append({'name': f'periodic-u{utilization}-p{phase}-d{urgency}',
                              'releases': rows, 'utilization': utilization})
    # Explicit blocking experiment: VWW begins alone; urgent AD arrives at
    # fixed fractions of its isolated cost; KWS arrives later with loose deadline.
    for fraction in (0.05, 0.25, 0.5, 0.8):
        for slack in (1.05, 1.2, 1.5, 2.0):
            release = int(costs['vww']*fraction)
            rows = [('vww', 0, costs['vww']*3, 0),
                    ('ad', release, release+int(costs['ad']*slack), 1),
                    ('kws', release+costs['ad']//2, costs['vww']*4, 2)]
            cases.append({'name': f'urgent-f{fraction}-s{slack}', 'releases': rows,
                          'utilization': None})
    return cases


class Invariants(unittest.TestCase):
    def test_sparse_dma_granules(self):
        self.assertEqual(intervals(mask(3, 2) | mask(10, 3)), [(3, 2), (10, 3)])
        self.assertEqual(dma_extent(mask(3, 2) | mask(7, 4)), 16)

    def test_equal_deadline_no_gratuitous_switch(self):
        p = {m: {'program_bytes': 16, 'segments': [dict(cycles=10,
              live_dma_bytes=0, save_dma_bytes=0)]*3} for m in ('kws', 'vww', 'ad')}
        releases = [('kws', 0, 100, 0), ('ad', 5, 100, 1)]
        a = schedule(p, releases, 'edf', 4, 0, 'resident', retain_trace=True)
        self.assertEqual(a['service_cycles'], 60)
        self.assertEqual(a['switches'], 2)
        self.assertEqual(a['misses'], 0)

    def test_suspension_transport_charged(self):
        p = {m: {'program_bytes': 16, 'segments': [dict(cycles=10,
              live_dma_bytes=80, save_dma_bytes=16)]*2} for m in ('kws', 'vww', 'ad')}
        a = schedule(p, [('vww', 0, 1000, 0), ('ad', 5, 500, 1)],
                     'edf', 4, 2, 'reload')
        # Three dispatches: program/context/checkpoint transport all charged.
        self.assertEqual(a['switches'], 3)
        self.assertEqual(a['overhead_cycles'], 3*2+3*2*4+2*4+8*4+10*4+8*4)
        self.assertEqual(a['service_cycles'], 40)

    def test_liveness_kill_before_read(self):
        effects = [(mask(20, 4), mask(0, 8)), (mask(0, 8), mask(20, 4))]
        live = 0
        for read, write in reversed(effects):
            live = read | (live & ~write)
        self.assertEqual(live, mask(20, 4))

    def test_uncommitted_outputs_require_save(self):
        with tempfile.TemporaryDirectory(dir=BASE) as temp:
            path = Path(temp)
            d = Descriptor(1, input=128, output=136, weight=144, params=208,
                           count=8, outputs=8, row_stride=8, next_pc=64)
            payload = bytearray(472)
            payload[64:192] = d.encode()+Descriptor(0).encode()
            (path/'payload.bin').write_bytes(payload)
            (path/'input.bin').write_bytes(b'\0'*8)
            (path/'commands.bin').write_bytes(pack([
                (1, 1, 0, 64, 0, 336), (3, 2, 0, 0, 0, 0),
                (2, 0, 0, 0, 336<<16, 0), (3, 3, 0, 0, 0, 0),
                (1, 0, 0, 464, 136, 8), (3, 2, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0)]))
            states = analyze_fixture(path)['boundaries']
            self.assertEqual(states[0]['dirty_bytes'], 0)
            self.assertEqual(states[1]['live_bytes'], 8)
            self.assertEqual(states[1]['dirty_bytes'], 8)
            self.assertEqual(states[1]['save_dma_bytes'], 8)
            self.assertEqual(states[2]['live_bytes'], 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('run', 'test', 'schedule'))
    parser.add_argument('--only-program-mode', choices=('resident', 'reload', 'uart_reload'))
    args = parser.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    if args.stage == 'test':
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Invariants)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    started = time.monotonic()
    if args.stage == 'run':
        exe, parent = build_trace()
        fixtures = {}
        for m in ('kws', 'vww'):
            r = next(r for r in parent['results'] if r['model'] == m and
                     r['sample'] == 'pinned' and r['stall_seed'] == 0)
            fixtures[m] = REFERENCE / r['candidate']['fixture_directory']
            for name, digest in r['candidate']['fixture_files'].items():
                if sha(fixtures[m] / name) != digest:
                    raise ValueError('selected fixture changed')
        fixtures['ad'] = ad_timed_fixture(ROOT / 'work/phase6/parallel-ad-v1/native/frame-0',
                                         BASE / 'fixtures/ad-baseline')
        profiles, evidence = {}, {}
        for m, folder in fixtures.items():
            analysis = analyze_fixture(folder)
            baseline_result, baseline_trace = run_native(exe, folder, m+'-baseline', 0)
            profiles[m] = {}
            evidence[m] = {'source_directory': str(folder), 'files_sha256': {
                n: sha(folder / n) for n in ('commands.bin', 'payload.bin', 'input.bin', 'checks.txt')},
                'variants': {}}
            variants = {'baseline': folder}
            for target in (25000, 50000):
                label = f'split{target}'
                destination = BASE / 'fixtures' / f'{m}-{label}'
                split_fixture(folder, destination, analysis, baseline_trace, target)
                variants[label] = destination
            for label, directory in variants.items():
                pa = analyze_fixture(directory)
                profiles[m][label] = {}
                evidence[m]['variants'][label] = {'program_bytes': pa['program_bytes'],
                    'dma_payload_bytes': pa['dma_payload_bytes'],
                    'command_sha256': sha(directory / 'commands.bin'),
                    'payload_sha256': sha(directory / 'payload.bin'), 'native': {}}
                for seed in (0, 6063):
                    result, trace = ((baseline_result, baseline_trace) if
                                    label == 'baseline' and seed == 0 else
                                    run_native(exe, directory, m+'-'+label, seed))
                    ss = segments(pa, trace)
                    profiles[m][label][str(seed)] = {'segments': ss,
                        'program_bytes': pa['program_bytes'], 'native_cycles': result['elapsed_cycles']}
                    evidence[m]['variants'][label]['native'][str(seed)] = dict(result,
                        quanta=len(ss), max_quantum_cycles=max(s['cycles'] for s in ss),
                        max_live_bytes=max(s['live_bytes'] for s in ss),
                        max_unbacked_dirty_bytes=max(s['dirty_bytes'] for s in ss))
                    print(m, label, seed, result['elapsed_cycles'], 'cycles', len(ss), 'quanta', flush=True)
        save(BASE / 'profiles.json', profiles)
        save(BASE / 'evidence.json', evidence)
    profiles = json.loads((BASE / 'profiles.json').read_text())
    baseline = {m: profiles[m]['baseline']['0'] for m in profiles}
    cases = release_grid(baseline)
    modes = (args.only_program_mode,) if args.only_program_mode else ('resident', 'reload', 'uart_reload')
    results = ([] if not args.only_program_mode else [r for r in json.loads(
        (BASE / 'results.json').read_text()) if r['program_mode'] != args.only_program_mode])
    for seed in ('0', '6063'):
        for word_cycles in (4, 8, 16, 32):
            for fixed_cycles in (0, 256, 4096, 50000):
                for program_mode in modes:
                    for case in cases:
                        for label in ('baseline', 'split25000', 'split50000'):
                            ps = {m: profiles[m][label][seed] for m in profiles}
                            policies = ('nonpreemptive', 'edf', 'static') if label == 'baseline' else ('edf',)
                            for policy in policies:
                                if policy == 'static':
                                    choices = [(slot, schedule(ps, case['releases'], policy,
                                        word_cycles, fixed_cycles, program_mode, slot_cycles=slot))
                                        for slot in (25000, 100000, 500000, 2000000)]
                                    slot, result = min(choices, key=lambda pair: (
                                        pair[1]['misses'], pair[1]['max_lateness_cycles'],
                                        pair[1]['overhead_cycles']))
                                    result['best_static_slot_cycles'] = slot
                                else:
                                    result = schedule(ps, case['releases'], policy,
                                                      word_cycles, fixed_cycles, program_mode)
                                result.pop('trace')
                                results.append(dict(result, case=case['name'], variant=label,
                                    policy=policy, ram_seed=int(seed), word_cycles=word_cycles,
                                    fixed_cycles=fixed_cycles, program_mode=program_mode))
                    print('screen', seed, word_cycles, fixed_cycles, program_mode,
                          len(results), 'schedules', flush=True)
    save(BASE / 'results.json', results)
    paired = {}
    for r in results:
        key = (r['case'], r['ram_seed'], r['word_cycles'], r['fixed_cycles'], r['program_mode'])
        paired.setdefault(key, {})[(r['variant'], r['policy'])] = r
    counts = {}
    for label in ('split25000', 'split50000'):
        useful = [k for k, group in paired.items() if
                  group[(label, 'edf')]['misses'] < group[('baseline', 'edf')]['misses']]
        worse = [k for k, group in paired.items() if
                 group[(label, 'edf')]['misses'] > group[('baseline', 'edf')]['misses']]
        rescued = [k for k, group in paired.items() if
                   group[(label, 'edf')]['misses'] == 0 and group[('baseline', 'edf')]['misses'] > 0]
        counts[label] = {'fewer_misses_than_descriptor_edf_cases': len(useful),
                         'more_misses_than_descriptor_edf_cases': len(worse),
                         'rescued_zero_miss_cases': len(rescued),
                         'paired_cases': len(paired)}
    descriptor_benefit = sum(g[('baseline', 'edf')]['misses'] <
                             g[('baseline', 'nonpreemptive')]['misses'] for g in paired.values())
    # One reviewable witness per split, plus its exact input sequence and costs.
    witnesses = []
    for label in ('split25000', 'split50000'):
        candidates = [(k, g) for k, g in paired.items() if
                      g[(label, 'edf')]['misses'] == 0 and g[('baseline', 'edf')]['misses'] > 0
                      and k[2] >= 8 and k[3] >= 256 and k[4] == 'reload']
        if candidates:
            k, group = min(candidates, key=lambda p: (p[0][3], p[0][2], p[0][0]))
            case = next(c for c in cases if c['name'] == k[0])
            witness = dict(case=case, ram_seed=k[1], word_cycles=k[2],
                           fixed_cycles=k[3], program_mode=k[4], outcomes=[])
            for variant, policy in (('baseline', 'nonpreemptive'), ('baseline', 'edf'),
                                    ('baseline', 'static'), (label, 'edf')):
                ps = {m: profiles[m][variant][str(k[1])] for m in profiles}
                slot = group[(variant, policy)].get('best_static_slot_cycles', 100000)
                witness['outcomes'].append(dict(variant=variant, policy=policy,
                    **schedule(ps, case['releases'], policy, k[2], k[3], k[4],
                               slot_cycles=slot, retain_trace=True)))
            witnesses.append(witness)
    save(BASE / 'witnesses.json', witnesses)
    break_even = []
    for witness in witnesses:
        variant = witness['outcomes'][-1]['variant']
        seed = str(witness['ram_seed'])
        ps = {m: profiles[m][variant][seed] for m in profiles}
        samples = []
        fixed_grid = sorted(set(range(0, 100001, 1000)) |
                            set(range(0, 5001, 128)) | {witness['fixed_cycles']})
        for fixed in fixed_grid:
            r = schedule(ps, witness['case']['releases'], 'edf', witness['word_cycles'],
                         fixed, witness['program_mode'])
            samples.append({'fixed_switch_cycles': fixed, 'misses': r['misses'],
                            'ad_response_cycles': r['max_response_cycles']['ad']})
        break_even.append({'case': witness['case']['name'], 'variant': variant,
            'word_cycles': witness['word_cycles'], 'program_mode': witness['program_mode'],
            'samples': samples,
            'largest_sampled_fixed_cost_with_zero_misses': max((r['fixed_switch_cycles']
                for r in samples if r['misses'] == 0), default=None),
            'claim': 'sampled sensitivity, not a monotonic theorem or physical bound'})
    save(BASE / 'break-even.json', break_even)
    report = {'status': 'passed-screen', 'scope': 'actual unchanged selected native RTL; discrete-job scheduling and context transport proxy',
              'physical_board': False, 'production_preemption_implemented': False,
              'native_engine_sha256': PARENT_SHA, 'arrival_cases': len(cases),
              'paired_sensitivity_cases': len(paired), 'schedule_runs': len(results),
              'descriptor_edf_fewer_misses_than_nonpreemptive_cases': descriptor_benefit,
              'split_comparisons': counts, 'elapsed_seconds': time.monotonic()-started,
              'driver_sha256': sha(Path(__file__)),
              'combined_program_bytes': {label: sum(profiles[m][label]['0']['program_bytes']
                  for m in profiles) for label in ('baseline', 'split25000', 'split50000')},
              'all_quiescent_software_halt_program_bytes': {label: sum(
                  profiles[m][label]['0']['program_bytes']+16*len(profiles[m][label]['0']['segments'])
                  for m in profiles) for label in ('baseline', 'split25000', 'split50000')},
              'selected_uart_baud': UART_BAUD,
              'uart_reload_zero_miss_schedules': sum(r['misses'] == 0 for r in results
                  if r['program_mode'] == 'uart_reload'),
              'python_max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'limitations': ['Native external RAM is abstract, not physical SDRAM timing.',
                 'No pause signal, scheduler, or live context transport has been implemented on board. The EDF comparisons grant both controls the same hypothetical low-cost pause interface; an all-boundary software HALT implementation also pays host dispatch at every visited HALT, even without changing jobs.',
                 'The resident-program case assumes private job external address namespaces and resident/relocated command images; multiple pending instances need additional program copies or address remapping.',
                 'Program reload uses the same word transport proxy as context. The physical host UART may be much slower.',
                 'Stalled native traces are standalone seeded timing examples, not worst-case response bounds.',
                 'Byte liveness proves future operand-version preservation, not interleaved RTL resume correctness.',
                 'Job arrival/deadline cases are synthetic workloads; observed miss counts are screen results, not measured schedulability.']}
    save(BASE / 'report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
