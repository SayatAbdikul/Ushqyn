#!/usr/bin/env python3
"""Matched conventional/liveness baseline tuning on the selected pooled engine.

This runner does not program a device. All candidates share exact channel
compaction, constant-filter elimination and the quantized activation epilogue.
It measures complete command programs with one frozen native RTL binary.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'compiler'), str(ROOT/'tools/phase6')]

from channel_compaction import compact_channels, check_oracles, command_stats
from constant_filter import compile_constant_chain
from followup_graph import group_channels, trim_constant_only_loads
from fused_activation import lower_fixture, check_table_lifetimes
from scheduler.matched_b1b2_tiling import compile_tiled
from prefetch_tail import analyze_fixture, commands, reorder
from run_boardless import load_model
from scheduler.fused_verify import replay_fused

BASE = ROOT/'work/phase6/matched-baselines-v1/b1b2'
ENGINE = ROOT/'work/phase6/pool-timing-v1/engine.sv'
NATIVE = ROOT/'work/phase6/pool-timing-v1/native/Vv2_tiled_host_bridge'
BUDGETS = (32768, 32512, 24576, 24320, 16384, 16128, 12288, 12032, 8192, 7936)
COMMON = ['exact channel permutation and dead-channel compaction',
          'exact VWW final constant propagation into classifier bias',
          'all-zero Conv filter evaluation and unused-load elimination',
          'exact INT8 activation lookup epilogue',
          'fixed eight-lane pooled engine and arithmetic',
          'dependency-checked descriptor/parameter/input DMA prefetch option']


def sha(value):
    return hashlib.sha256(value.read_bytes() if isinstance(value, Path) else value).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def matched_model(name):
    """Return frozen original, pinned input, common graph/maps and provenance."""
    original, pinned, _, sources = load_model(name)
    grouped, group_maps, _ = group_channels(original)
    certificate = None
    if name == 'vww':
        from novelty_constants import fold_final_constants
        grouped, certificate = fold_final_constants(grouped)
    program, kept, changes, aligned = compact_channels(grouped)
    maps = {name: tuple(group_maps[name][j] for j in selected)
            for name, selected in kept.items()}
    return original, pinned, program, maps, dict(sources=sources,
        compaction=changes, alignment_retained=aligned,
        final_constant_certificate=certificate)


def prefetch_candidates(code):
    """Move only loads whose source and destination survive the whole move.

    Besides parameters this permits already materialized activation inputs in
    the spare SRAM half. SRAM stores are readers, so they also block a move;
    external stores block a move when they produce any source byte.
    """
    decoded = commands(code)
    runs = [i for i,c in enumerate(decoded) if c[0] == 2]
    selected = []
    overlap = lambda a,n,b,m: a < b+m and b < a+n
    for current,nxt in zip(runs,runs[1:]):
        live = decoded[current][4]
        low,high = live&65535, live>>16
        for i in range(current+2,nxt):
            op,flags,_,ext,sram,size = decoded[i]
            if op != 1 or flags != 1 or decoded[i+1][:2] != (3,2):
                continue
            if overlap(sram,size,low,high-low):
                continue
            intervening = decoded[current+1:i]
            # Retain original load ordering. Any SRAM reader/writer before
            # this load can need old contents; a prior SDRAM store can create
            # its source. Later commands execute in their original order.
            if any(c[0] == 1 and (overlap(sram,size,c[4],c[5]) or
                    (c[1] == 0 and overlap(ext,size,c[3],c[5])))
                   for c in intervening):
                continue
            selected.append(dict(current_run_command=current,next_run_command=nxt,
                dma_command=i,ext=ext,sram=sram,bytes=size,current_live=[low,high]))
    return selected


def compile_candidate(program, *, policy, tile_budget, prefetch, snapshots, directory):
    """Compile an executable candidate, including common epilogue optimizations.

    A preferred tile budget is a tuning option, not a reduced hardware budget.
    Indivisible operators retain the same 32KiB fallback in every policy.
    B1 alternates legal half-size stages between two fixed SRAM halves.
    B2 applies the existing first-fit liveness allocator after choosing tiles.
    """
    if policy not in ('B1', 'B2'):
        raise ValueError('unknown matched baseline')
    directory = Path(directory).resolve()
    seed = directory/'unfused'
    seed.mkdir(parents=True, exist_ok=True)
    plan = compile_tiled(program, preferred_capacity=tile_budget)
    code, payload, schedule = compile_constant_chain(program, snapshots,
        reuse_sibling_inputs=policy == 'B2', chain_options=dict(
            prepared_plans={False: plan, True: plan}, retain_tensors=policy == 'B2'))
    code, schedule = trim_constant_only_loads(code, schedule)
    (seed/'commands.bin').write_bytes(code)
    (seed/'payload.bin').write_bytes(payload)
    save(seed/'schedule.json', schedule)
    code, payload, schedule = lower_fixture(seed)
    (directory/'commands.bin').write_bytes(code)
    (directory/'payload.bin').write_bytes(payload)
    save(directory/'schedule.json', schedule)
    selected = prefetch_candidates(code) if prefetch else []
    reordered, mapping = reorder(commands(code), selected)
    code = b''.join(struct.pack('<BBHIII', *c) for c in reordered)
    for key in ('run_contracts', 'constant_contracts', 'pack_contracts'):
        schedule[key] = {str(mapping[int(k)]): v for k,v in schedule.get(key, {}).items()}
    schedule.update(program_sha256=sha(code), image_sha256=sha(payload),
        command_count=len(code)//16, program_bytes=len(code),
        matched_baseline=dict(policy=policy, preferred_tile_bytes=tile_budget,
            actual_sram_budget=32768, indivisible_tile_fallback_bytes=32768,
            allocation='fixed alternating SRAM halves' if policy=='B1' else
                'first-fit live tensor and sibling-input allocation after tiling',
            common_optimizations=COMMON),
        tail_prefetch=dict(enabled=prefetch,
            bytes=sum(c['bytes'] for c in selected), transfers=len(selected)))
    schedule['table_lifetimes'] = check_table_lifetimes(code, payload, schedule)
    (directory/'commands.bin').write_bytes(code)
    (directory/'payload.bin').write_bytes(payload)
    save(directory/'schedule.json', schedule)
    return code, payload, schedule


def write_fixture(directory, program, value, oracle, code, payload, schedule):
    replay = replay_fused(program, code, payload, {program.inputs[0]:value},
        run_contracts=schedule['run_contracts'],
        constant_contracts=schedule.get('constant_contracts'),
        final_output=schedule['final_output'],
        snapshot_regions=schedule['snapshot_regions'], oracle=oracle)
    first = program.layers[0]
    initial = oracle[first.output] if first.op in ('Transpose', 'Reshape') else value
    files = {'commands.bin':code, 'payload.bin':payload, 'input.bin':initial.tobytes(),
             'output.bin':oracle[program.outputs[0]].tobytes()}
    checks = [f'{schedule["final_output"]["ext"]} output.bin']
    for i, region in schedule['snapshot_regions'].items():
        name = f'layer-{i}.bin'
        files[name] = oracle[program.layers[int(i)].output].tobytes()
        checks.append(f'{region["ext"]} {name}')
    files['checks.txt'] = ('\n'.join(checks)+'\n').encode()
    for name, data in files.items():
        (directory/name).write_bytes(data)
    save(directory/'schedule.json', schedule)
    return dict(replay=replay, commands=command_stats(code),
        fusion=schedule['fusion'], tail_prefetch=schedule['tail_prefetch'],
        max_live_address=max(s['live'][1] for s in schedule['stages']),
        files={name:sha(directory/name) for name in (*files,'schedule.json')})


def native_identity():
    report_path = NATIVE.parent/'report.json'
    record = json.loads(report_path.read_text())
    if record['status'] != 'passed' or sha(NATIVE) != record['executable_sha256']:
        raise ValueError('selected native engine is missing or changed')
    for relative, expected in record['sources'].items():
        if sha(ROOT/relative) != expected:
            raise ValueError(f'native RTL source changed: {relative}')
    return dict(executable_sha256=sha(NATIVE), engine_sha256=sha(ENGINE),
        sources=record['sources'], nominal_clock_mhz=27,
        timing_boundary='sequencer start to HALT; excludes model/input upload and UART',
        memory_model='native RAM timing; not electrical SDRAM board measurements')


def fixture_identity(directory):
    names = {'commands.bin','payload.bin','input.bin','checks.txt','schedule.json'}
    for line in (directory/'checks.txt').read_text().splitlines():
        fields = line.split()
        if len(fields) != 2 or Path(fields[1]).name != fields[1]:
            raise ValueError('malformed native check path')
        names.add(fields[1])
    return {name:sha(directory/name) for name in sorted(names)}


def measure(directory, seed, output):
    before = fixture_identity(directory)
    executable = sha(NATIVE)
    with output.with_suffix('.log').open('w') as log:
        subprocess.run([str(NATIVE), str(directory), str(seed), str(output)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    record = json.loads(output.read_text())
    if record['status'] != 'passed':
        raise ValueError('native exactness failed')
    if before != fixture_identity(directory) or executable != sha(NATIVE):
        raise ValueError('native executable or fixture changed during measurement')
    record.update(fixture_files=before,executable_sha256=executable,
                  fixture_directory=str(directory.relative_to(ROOT)))
    save(output,record)
    return record


def compiler_identity():
    names = ['tools/phase6/matched_b1b2_generic.py',
        'compiler/scheduler/matched_b1b2_tiling.py',
        'compiler/phase4_tiling.py','compiler/phase4_compile.py',
        'tools/phase6/chain_resident.py','tools/phase6/constant_filter.py',
        'tools/phase6/channel_compaction.py','tools/phase6/followup_graph.py',
        'tools/phase6/fused_activation.py','tools/phase6/prefetch_tail.py',
        'tools/phase6/run_boardless.py','tools/phase6/output_pipeline_fusion.py',
        'tools/phase6/padded_descriptor.py','compiler/scheduler/fused_verify.py',
        'tools/phase6/novelty_constants.py',
        'compiler/scheduler/resident.py','compiler/hardware_v2.py',
        'compiler/integer_reference.py','compiler/static_pipeline.py',
        'compiler/quantization.py']
    return {name:sha(ROOT/name) for name in names}


def run(output=BASE, models=('kws','vww'), budgets=BUDGETS):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    compiler_sources = compiler_identity()
    report = dict(schema=1, status='running', physical_board=False,
        native=native_identity(), compiler_sources=compiler_sources,
        common_optimizations=COMMON,
        tuning=dict(tile_budgets=list(budgets), policies=['B1','B2'],
            prefetch=[False,True], selection='minimum geometric mean of pinned seed0 and seed6063 elapsed cycles',
            tie_break='lexicographic candidate ID', validation_stall_seed=6063,
            validation_input_seed=6157, b2_includes_b1_fallback=True,
            board_frontier_limit=3),
        limitations=['Tuning explores uniform preferred tile capacities with 32KiB fallback.',
            'No rectangular spatial or reduction tile search in B1/B2.',
            'Native scores use a RAM model and do not establish board latency.',
            'B2 first-fit placement follows tiling; no joint search.'], models={})
    save(output/'report.json', report)
    for model in models:
        original, pinned, program, maps, provenance = matched_model(model)
        from phase4_compile import compile_tiled as frozen_compile_tiled
        for half in (False,True):
            if compile_tiled(program,prefer_half=half) != frozen_compile_tiled(program,prefer_half=half):
                raise ValueError('isolated tiler changed the original full/half tile semantics')
        oracle = check_oracles(original, program, maps, pinned)
        from matched_current import graph_identity
        model_report = dict(provenance=provenance, graph_sha256=graph_identity(program),
                            candidates={}, selected={}, frontier={})
        report['models'][model] = model_report
        seen = {}
        for policy in ('B1','B2'):
            for budget in budgets:
                for prefetch in (False,True):
                    name = f'{model}-{policy.lower()}-t{budget}-p{int(prefetch)}'
                    directory = output/'candidates'/name
                    construction = time.monotonic()
                    try:
                        code, payload, schedule = compile_candidate(program,
                            policy=policy, tile_budget=budget, prefetch=prefetch,
                            snapshots=False, directory=directory)
                    except (ValueError, AssertionError) as exc:
                        model_report['candidates'][name] = dict(status='compile-rejected',
                            policy=policy, tile_budget=budget, prefetch=prefetch,
                            error=f'{type(exc).__name__}: {exc}')
                        save(output/'report.json', report)
                        print(name, 'compile-rejected', str(exc), flush=True)
                        continue
                    entry = write_fixture(directory, program, pinned, oracle,
                                          code, payload, schedule)
                    entry.update(status='passed', policy=policy, tile_budget=budget,
                        prefetch=prefetch, directory=str(directory.relative_to(ROOT)),
                        construction_seconds=time.monotonic()-construction)
                    semantic_files={k:v for k,v in entry['files'].items() if k!='schedule.json'}
                    signature = tuple(sorted(semantic_files.items()))
                    if signature in seen:
                        previous = seen[signature]
                        entry['duplicate_of'] = previous
                        for key in ('native','stalled_native'):
                            row = copy.deepcopy(model_report['candidates'][previous][key])
                            if {k:v for k,v in row['fixture_files'].items() if k!='schedule.json'} != semantic_files:
                                raise ValueError('duplicate candidate execution files differ')
                            if fixture_identity(directory) != entry['files']:
                                raise ValueError('duplicate candidate files changed')
                            row.update(fixture_files=entry['files'],
                                fixture_directory=str(directory.relative_to(ROOT)),
                                reused_from_fixture=previous,
                                reuse_reason='all files read by native executable are byte-identical; schedule metadata is not read by RTL')
                            entry[key] = row
                    else:
                        entry['native'] = measure(directory, 0, directory/'native-s0.json')
                        entry['stalled_native'] = measure(directory,6063,directory/'native-s6063.json')
                        seen[signature] = name
                    entry['selection_score_cycles'] = math.sqrt(entry['native']['elapsed_cycles']*
                        entry['stalled_native']['elapsed_cycles'])
                    model_report['candidates'][name] = entry
                    save(output/'report.json', report)
                    print(name, entry['native']['elapsed_cycles'], flush=True)
        validated = {}
        def validate_selection(name, selected):
            if name in validated:
                return validated[name]
            selection = dict(candidate=name, candidate_policy=selected['policy'],
                native=selected['native'], stalled_timed=selected['stalled_native'],
                selection_score_cycles=selected['selection_score_cycles'],
                directory=selected['directory'], files=selected['files'], validation=[])
            for sample,value in (('pinned',pinned),('stress',np.random.default_rng(6157).integers(
                    -128,128,pinned.shape,dtype=np.int8))):
                local_oracle = oracle if sample=='pinned' else check_oracles(original,program,maps,value)
                target = output/'validation'/f'{name}-{sample}'
                code,payload,schedule = compile_candidate(program,
                    policy=selected['policy'], tile_budget=selected['tile_budget'],
                    prefetch=selected['prefetch'], snapshots=True, directory=target)
                validation = write_fixture(target,program,value,local_oracle,code,payload,schedule)
                validation.update(sample=sample, directory=str(target.relative_to(ROOT)),
                    native=measure(target,6063,target/'native-s6063.json'))
                selection['validation'].append(validation)
            validated[name] = selection
            return selection
        for policy in ('B1','B2'):
            eligible = [(row['selection_score_cycles'],name,row)
                for name,row in model_report['candidates'].items()
                if row['status']=='passed' and (policy=='B2' or row['policy']=='B1')]
            if not eligible:
                raise ValueError(f'no executable {model} {policy} candidate')
            frontier,identities = [],set()
            for _,name,selected in sorted(eligible,key=lambda row:(row[0],row[1])):
                identity = (selected['files']['commands.bin'], selected['files']['payload.bin'])
                if identity in identities:
                    continue
                identities.add(identity)
                frontier.append(validate_selection(name,selected))
                if len(frontier)==3:
                    break
            model_report['selected'][policy] = frontier[0]
            model_report['frontier'][policy] = frontier
            save(output/'report.json', report)
        report['elapsed_seconds'] = time.monotonic()-began
        save(output/'report.json', report)
    if compiler_sources != compiler_identity():
        raise ValueError('compiler sources changed during tuning')
    report.update(status='passed', elapsed_seconds=time.monotonic()-began,
        source_identity_unchanged=True)
    save(output/'report.json', report)
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE)
    parser.add_argument('--models',nargs='+',choices=('kws','vww'),default=['kws','vww'])
    parser.add_argument('--budgets',nargs='+',type=int,default=BUDGETS)
    args = parser.parse_args()
    result = run(args.output,args.models,args.budgets)
    print(result['status'],result['elapsed_seconds'])
