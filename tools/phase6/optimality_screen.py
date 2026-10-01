#!/usr/bin/env python3
"""Bounded strip-family optimality certificates and executable heuristic audit.

The strip objective is explicitly an idealized serialized service model, NOT
device cycles. Every Conv and activation INT8 boundary is kept. Region cuts are
external materialization barriers; there is no cache, prefetch, overlap, channel
tiling or cross-region retained state. This makes the boundary state complete.
The separate executable audit uses the existing compiler and serial ABI only.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'compiler'), str(ROOT / 'tools/phase6')]
from integer_reference import evaluate
from quantization import Quantization, quantize_parameters
from run_boardless import load_model
from static_pipeline import Layer, Program, Tensor
from scheduler.spatial import execute_segment

OUT = ROOT / 'work/phase6/optimality-screen-v1'
PORT_BYTES = 8
STAGING = 128 + 256  # descriptor pair and quantization LUT (model assumptions)


def words(n):
    return (n + 7) // 8


def padded(n):
    return words(n) * 8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def rss():
    n = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return n if sys.platform == 'darwin' else n * 1024


@dataclass(frozen=True)
class Macro:
    layer: int
    ci: int
    co: int
    ih: int
    iw: int
    oh: int
    ow: int
    kh: int
    kw: int
    sh: int
    sw: int
    pt: int
    pl: int
    group: int
    quantization_id: str
    activation_id: str

    @property
    def reduction(self):
        return self.ci // self.group * self.kh * self.kw


def macros(program, start, stop):
    result = []
    for i in range(start, stop, 2):
        l, activation = program.layers[i:i+2]
        assert l.op == 'Conv' and activation.op in ('Relu', 'Clip')
        assert activation.inputs == [l.output]
        inp, out = program.tensors[l.inputs[0]], program.tensors[l.output]
        a, w = l.attributes, l.parameters['weight']
        # Hash exact quantizer parameters; a certificate cannot erase barriers.
        qid = hashlib.sha256(b''.join(np.asarray(l.parameters[k]).tobytes()
                            for k in ('corrected_bias', 'multiplier', 'shift'))
                            + repr(asdict(out.quantization)).encode()).hexdigest()
        aid = hashlib.sha256(repr((activation.op, activation.parameters,
                                  asdict(program.tensors[activation.output].quantization))).encode()).hexdigest()
        result.append(Macro(i, inp.shape[1], out.shape[1], *inp.shape[2:],
                            *out.shape[2:], *w.shape[2:],
                            *a.get('strides', [1, 1]), *a.get('pads', [0]*4)[:2],
                            a.get('group', 1), qid, aid))
    assert all(a.co == b.ci and (a.oh, a.ow) == (b.ih, b.iw)
               for a, b in zip(result, result[1:]))
    return result


def rectangle_needs(ms, a, b, y0, y1):
    rects = {b: (y0, y1, 0, ms[b-1].ow)}
    for j in range(b-1, a-1, -1):
        m = ms[j]
        lo, hi, xl, xh = rects[j+1]
        rects[j] = (max(0, lo*m.sh-m.pt),
                    min(m.ih, (hi-1)*m.sh-m.pt+m.kh),
                    max(0, xl*m.sw-m.pl),
                    min(m.iw, (xh-1)*m.sw-m.pl+m.kw))
    return rects


def area(rect):
    y0, y1, x0, x1 = rect
    return max(0, y1-y0) * max(0, x1-x0)


def choice(ms, a, b, height, budget):
    """Reconstruct every strip and its actual model allocation/access counts.

    In this idealized kernel each entire requested activation is read once,
    each original Conv output is written/read by its activation, and the exact
    activation result is written. Weights/params are staged eight output rows
    at a time. Operand reuse inside Conv is idealized; no hardware claim follows.
    Both compute and DMA buffer words serialize through the single port; MAC
    slots and quantization service also serialize in this intentionally simple
    objective. Input/output arrays use disjoint padded intervals; no in-place
    allocation assumption hides transient bytes.
    """
    last = ms[b-1]
    details, totals = [], dict(macs=0, quant_elements=0, port_words=0,
                              external_words=0, weight_parameter_words=0,
                              mac_slots=0, peak_bytes=0)
    for y0 in range(0, last.oh, height):
        y1 = min(last.oh, y0+height)
        needs = rectangle_needs(ms, a, b, y0, y1)
        source = ms[a].ci * area(needs[a])
        output = last.co * area(needs[b])
        # Per-channel NCHW scatter/gather, 8-byte transfer granularity.
        external = ms[a].ci * words(area(needs[a])) + last.co * words(area(needs[b]))
        stage_rows = []
        for j in range(a, b):
            m = ms[j]
            ni, no = m.ci*area(needs[j]), m.co*area(needs[j+1])
            weight = m.co*padded(m.reduction)
            params = 16*m.co+16
            staging = STAGING + min(8, m.co)*(padded(m.reduction)+16) + 16
            # Separate Conv q8 and activation q8 buffers: their boundary is
            # explicit even when a future implementation can alias storage.
            peak = max(staging+padded(ni)+padded(no), staging+2*padded(no))
            macs = no*m.reduction
            # One input read; Conv q8 write + activation read + q8 write.
            port = words(ni)+3*words(no)+words(weight)+words(params)
            row = dict(macro=j, input_rect=list(needs[j]), output_rect=list(needs[j+1]),
                       input_bytes=ni, output_bytes=no, staging_bytes=staging,
                       peak_bytes=peak, macs=macs, quant_elements=2*no,
                       mac_slots=words(macs), port_words=port,
                       weight_parameter_words=words(weight)+words(params),
                       quantization_id=m.quantization_id, activation_id=m.activation_id)
            stage_rows.append(row)
            for key in ('macs', 'quant_elements', 'mac_slots', 'port_words', 'weight_parameter_words'):
                totals[key] += row[key]
            totals['peak_bytes'] = max(totals['peak_bytes'], peak)
        totals['external_words'] += external
        details.append(dict(y0=y0, y1=y1, source_bytes=source, output_bytes=output,
                            external_words=external, stages=stage_rows))
    # One scratchpad access for each external DMA word; parameter DMA accesses
    # additionally charged separately from the Conv's SRAM reads.
    totals['dma_port_words'] = totals['external_words'] + totals['weight_parameter_words']
    totals['service'] = totals['mac_slots'] + totals['quant_elements'] + totals['port_words'] + totals['dma_port_words']
    return dict(id=f'{a}:{b}:h{height}', start=a, stop=b, height=height,
                budget=budget, feasible=totals['peak_bytes'] <= budget,
                totals=totals, strips=details)


def independent_check_choice(ms, row):
    """Scalar stencil-dependency checker; does not call rectangle_needs/choice.

    It enumerates each requested output coordinate and original kernel tap to
    reconstruct clipped ancestors, then recomputes the port/occupancy objective.
    The checker rejects quantizer identity changes and undersized allocations.
    """
    a, b = row['start'], row['stop']
    assert 0 <= a < b <= len(ms) and 1 <= row['height'] <= ms[b-1].oh
    assert [s['y0'] for s in row['strips']] == list(range(0,ms[b-1].oh,row['height'])), 'incomplete/duplicated output strips'
    tally = dict(macs=0, quant_elements=0, mac_slots=0, port_words=0,
                 external_words=0, weight_parameter_words=0, peak_bytes=0)
    for strip in row['strips']:
        assert strip['y1'] == min(ms[b-1].oh,strip['y0']+row['height'])
        assert [s['macro'] for s in strip['stages']] == list(range(a,b)), 'missing/reordered original macro'
        coords = {b: {(y,x) for y in range(strip['y0'], strip['y1'])
                      for x in range(ms[b-1].ow)}}
        # Bounding rectangles are deliberately part of the declared family,
        # even if a stride leaves holes in the exact needed coordinate set.
        for j in range(b-1, a-1, -1):
            m = ms[j]
            hits = {(y*m.sh-m.pt+ky, x*m.sw-m.pl+kx)
                    for y,x in coords[j+1] for ky in range(m.kh) for kx in range(m.kw)
                    if 0 <= y*m.sh-m.pt+ky < m.ih and 0 <= x*m.sw-m.pl+kx < m.iw}
            assert hits, 'empty dependency rectangle outside supported family'
            yl, yh = min(y for y,x in hits), max(y for y,x in hits)+1
            xl, xh = min(x for y,x in hits), max(x for y,x in hits)+1
            coords[j] = {(y,x) for y in range(yl, yh) for x in range(xl, xh)}
        ext = ms[a].ci*((len(coords[a])+7)//8) + ms[b-1].co*((len(coords[b])+7)//8)
        assert ext == strip['external_words'], 'external port count mismatch'
        tally['external_words'] += ext
        for entry in strip['stages']:
            j, m = entry['macro'], ms[entry['macro']]
            assert entry['quantization_id'] == m.quantization_id and entry['activation_id'] == m.activation_id, 'quantization boundary identity mismatch'
            ni, no = m.ci*len(coords[j]), m.co*len(coords[j+1])
            assert entry['input_bytes'] == ni and entry['output_bytes'] == no, 'operand size mismatch'
            r = (m.ci//m.group)*m.kh*m.kw
            p8 = lambda x: ((x+7)//8)*8
            staging = 384 + min(8,m.co)*(p8(r)+16) + 16
            peak = max(staging+p8(ni)+p8(no), staging+2*p8(no))
            assert entry['peak_bytes'] >= peak and entry['staging_bytes'] >= staging, 'undersized memory certificate'
            wp = (m.co*p8(r)+7)//8 + (16*m.co+16+7)//8
            p = (ni+7)//8+3*((no+7)//8)+wp
            assert entry['port_words'] == p, 'single-port access count mismatch'
            assert entry['macs'] == no*r and entry['quant_elements'] == 2*no
            assert entry['mac_slots'] == (no*r+7)//8
            tally['macs'] += no*r
            tally['quant_elements'] += 2*no
            tally['mac_slots'] += (no*r+7)//8
            tally['port_words'] += p
            tally['weight_parameter_words'] += wp
            tally['peak_bytes'] = max(tally['peak_bytes'], peak)
    tally['dma_port_words'] = tally['external_words']+tally['weight_parameter_words']
    tally['service'] = tally['mac_slots']+tally['quant_elements']+tally['port_words']+tally['dma_port_words']
    assert tally == row['totals'], 'objective/traffic certificate mismatch'
    assert row['feasible'] == (tally['peak_bytes'] <= row['budget']), 'capacity decision mismatch'
    return tally


def build_catalogue(ms, budget):
    return [choice(ms,a,b,h,budget) for a in range(len(ms))
            for b in range(a+1,len(ms)+1) for h in range(1,ms[b-1].oh+1)]


def dynamic_optimum(ms, rows):
    n = len(ms)
    potentials, paths = [math.inf]*(n+1), [None]*(n+1)
    potentials[n], paths[n] = 0, []
    for a in range(n-1,-1,-1):
        for c in rows:
            if c['start'] != a or not c['feasible'] or paths[c['stop']] is None:
                continue
            cost = c['totals']['service']+potentials[c['stop']]
            path = [c['id']]+paths[c['stop']]
            if (cost, path) < (potentials[a], paths[a] or ['~']):
                potentials[a], paths[a] = cost, path
    return dict(objective=potentials[0] if math.isfinite(potentials[0]) else None,
                path=paths[0], potentials=[v if math.isfinite(v) else None for v in potentials])


def verify_optimum_certificate(ms, rows, certificate, check_all=False):
    expected={(a,b,h) for a in range(len(ms)) for b in range(a+1,len(ms)+1)
              for h in range(1,ms[b-1].oh+1)}
    observed={(r['start'],r['stop'],r['height']) for r in rows}
    assert observed == expected and len(rows) == len(expected), 'incomplete/duplicated finite catalogue'
    assert all(r['id']==f"{r['start']}:{r['stop']}:h{r['height']}" for r in rows)
    assert len({r['budget'] for r in rows}) == 1, 'inconsistent capacity contract'
    by_id = {r['id']:r for r in rows}
    p = certificate['potentials']
    assert p[-1] == 0 and len(p) == len(ms)+1
    # For every possible outgoing edge: p[a] <= cost(edge)+p[b]. Thus p[0]
    # lower-bounds every complete path by telescoping, without rerunning DP.
    for r in rows:
        if check_all:
            independent_check_choice(ms,r)
        if r['feasible'] and p[r['stop']] is not None:
            assert p[r['start']] is not None
            assert p[r['start']] <= r['totals']['service']+p[r['stop']], 'unsound Bellman lower-bound certificate'
    assert certificate['path'] is not None
    position, cost = 0, 0
    for name in certificate['path']:
        r = by_id[name]
        independent_check_choice(ms,r)
        assert r['start'] == position and r['feasible']
        position, cost = r['stop'], cost+r['totals']['service']
    assert position == len(ms) and cost == certificate['objective'] == p[0]
    return dict(verified_lower_bound=p[0], verified_legal_upper=cost,
                upper_over_lower=cost/p[0], inequalities_checked=len(rows),
                scope='entire declared finite strip family, serialized idealized model only')


def brute_force(ms, rows):
    """Independent unpruned enumeration of every complete feasible path."""
    outgoing = {i:[r for r in rows if r['feasible'] and r['start']==i] for i in range(len(ms))}
    count, best, best_path = 0, None, None
    def walk(position, total, path):
        nonlocal count,best,best_path
        if position == len(ms):
            count += 1
            if best is None or (total,path) < (best,best_path):
                best,best_path = total,list(path)
            return
        for r in outgoing[position]:
            walk(r['stop'],total+r['totals']['service'],path+[r['id']])
    walk(0,0,[])
    return dict(optimum=best,path=best_path,complete_paths=count,pruning=False)


def heuristic(ms, rows, pair_fusion):
    """Strong greedy control: cheapest tile for each single/pair region.

    This is a proxy control; it is NOT claimed to be the production compiler.
    Pair cuts are fixed before cost selection, which is the only weakness DP
    changes. No largest-tile strawman is used.
    """
    position, path, cost = 0, [], 0
    while position < len(ms):
        b = min(len(ms),position+(2 if pair_fusion else 1))
        feasible = [r for r in rows if r['feasible'] and r['start']==position and r['stop']==b]
        if not feasible and b > position+1:
            b = position+1
            feasible = [r for r in rows if r['feasible'] and r['start']==position and r['stop']==b]
        if not feasible:
            return dict(objective=None,path=None)
        r = min(feasible,key=lambda r:(r['totals']['service'],r['id']))
        cost += r['totals']['service'];path.append(r['id']);position=b
    return dict(objective=cost,path=path)


def relaxed_work_bound(ms,rows):
    """Necessary work bound, independent of the Bellman optimality proof.

    Each complete path includes every macro exactly once in a region. Relax
    region/tile compatibility, capacity and external activation transfers, and
    choose each macro's cheapest contribution separately. Contributions still
    charge exact Conv/activation boundaries and the single scratchpad port.
    """
    minima=[]
    for j in range(len(ms)):
        candidates=[]
        for r in rows:
            if r['start'] <= j < r['stop']:
                entries=[s for strip in r['strips'] for s in strip['stages'] if s['macro']==j]
                candidates.append(sum(s['mac_slots']+s['quant_elements']+s['port_words']+
                                      s['weight_parameter_words'] for s in entries))
        minima.append(min(candidates))
    return dict(lower_bound=sum(minima),per_macro_minima=minima,
                relaxation='independent macro choices; capacity and external activation DMA costs omitted')


def synthetic(name,h=6,w=8,c=64,kernels=(3,1,3),stride_first=1):
    q = Quantization(.125,0)
    tensors = {'input':Tensor('input',(1,c,h,w),quantization=q,layout='NCHW')}
    layers, previous = [], 'input'
    rng = np.random.default_rng(7611)
    for j,k in enumerate(kernels):
        depthwise = k == 3
        out = f'conv{j}';act=f'act{j}'
        shape = tensors[previous].shape
        stride = stride_first if j == 0 else 1
        oh,ow = (shape[2]+stride-1)//stride,(shape[3]+stride-1)//stride
        wgt = rng.integers(-2,3,(c,1 if depthwise else c,k,k),dtype=np.int8)
        params = quantize_parameters(wgt.astype(np.float64)*.125,np.zeros(c),q,q)
        tensors[out] = Tensor(out,(1,c,oh,ow),quantization=q,layout='NCHW')
        tensors[act] = Tensor(act,(1,c,oh,ow),quantization=q,layout='NCHW')
        layers += [Layer('Conv',[previous],out,
                         dict(group=c if depthwise else 1,strides=[stride,stride],
                              pads=[k//2]*4),params),
                   Layer('Relu',[out],act,{}, {})]
        previous=act
    p=Program(tensors,layers,['input'],[previous],{}, {'synthetic':name})
    return p,rng.integers(-128,128,tensors['input'].shape,dtype=np.int8)


def semantics(program,x,ms,path,rows):
    by_id={r['id']:r for r in rows};oracle=evaluate(program,{program.inputs[0]:x})
    first=ms[by_id[path[0]]['start']].layer
    value=oracle[program.layers[first].inputs[0]];checks=[]
    for name in path:
        c=by_id[name];start=ms[c['start']].layer;stop=ms[c['stop']-1].layer+2
        value,counts=execute_segment(program,start,stop,value,
                                    tile=(c['height'],ms[c['stop']-1].ow),cache=False)
        expected=oracle[program.layers[stop-1].output]
        assert np.array_equal(value,expected), 'quantized schedule changed output'
        assert sum(v['macs'] for v in counts['layers'].values()) == c['totals']['macs']
        checks.append(dict(region=name,exact=True,output_sha256=hashlib.sha256(value.tobytes()).hexdigest()))
    return checks


def negative_checks(ms,rows,cert):
    selected=next(r for r in rows if r['id']==cert['path'][0])
    results={}
    for name,mutate in (
        ('quantizer_erasure',lambda r:r['strips'][0]['stages'][0].update(quantization_id='erased')),
        ('undersized_memory',lambda r:r['strips'][0]['stages'][0].update(peak_bytes=0)),
        ('missing_port_access',lambda r:r['strips'][0]['stages'][0].update(port_words=0))):
        bad=copy.deepcopy(selected);mutate(bad)
        try: independent_check_choice(ms,bad)
        except AssertionError as error: results[name]=dict(rejected=True,reason=str(error))
        else: raise AssertionError(f'checker accepted {name}')
    bad=copy.deepcopy(cert);bad['potentials'][0]+=1
    try: verify_optimum_certificate(ms,rows,bad)
    except AssertionError as error: results['unsound_bound']=dict(rejected=True,reason=str(error))
    else: raise AssertionError('checker accepted unsound lower bound')
    try: verify_optimum_certificate(ms,rows[:-1],cert)
    except AssertionError as error: results['omitted_candidate']=dict(rejected=True,reason=str(error))
    else: raise AssertionError('checker accepted incomplete finite family')
    return results


def executable_audit(program,start,stop,x):
    """Production coordinate heuristic versus exhaustive serial ABI catalogue.

    Serial timelines preserve the shared port by prohibiting engine/DMA overlap.
    The event cost is a predictor. The catalogue has exactly two source tile
    plans per active head and optional existing producer/activation retention.
    It does not contain spatial strip fusion or the selected physical schedule.
    """
    from scheduler.resident import compile_resident,eligible
    from scheduler.resident_search import optimize_resident
    from scheduler.event_cost import estimate
    from scheduler.resident_verify import replay_resident
    from phase4_compile import compile_tiled
    ls=copy.deepcopy(program.layers[start:stop]);inp=ls[0].inputs[0];out=ls[-1].output
    names={inp}|{l.output for l in ls}
    p=Program({n:copy.deepcopy(program.tensors[n]) for n in names},ls,[inp],[out],{}, {'fragment': [start,stop]})
    oracle=evaluate(p,{inp:x})
    plans={h:compile_tiled(p,prefer_half=h) for h in (False,True)}
    heads=tuple(i for i,l in enumerate(plans[False][0]['layers']) if l['tiles'])
    pairs=eligible(p)
    _,current=optimize_resident(p,None,seconds=30,max_evaluations=256,max_sweeps=2)
    best=None;records=[]
    for half_bits in itertools.product((False,True),repeat=len(heads)):
        halves={i for i,v in zip(heads,half_bits) if v}
        for fuse_bits in itertools.product((False,True),repeat=len(pairs)):
            fused={i for i,v in zip(pairs,fuse_bits) if v}
            code,payload,schedule=compile_resident(p,fused=fused,prefer_half=False,
                tile_choices={i:i in halves for i in heads},overlap=False,prepared_plans=plans)
            cost=estimate(code,payload)['elapsed_cycles']
            key=(cost,tuple(sorted(halves)),tuple(sorted(fused)))
            records.append(dict(halves=sorted(halves),fused=sorted(fused),serial_event_cost=cost,
                                code_sha256=hashlib.sha256(code).hexdigest(),
                                payload_sha256=hashlib.sha256(payload).hexdigest()))
            if best is None or key < best[0]:best=(key,code,payload,schedule)
    best_key,code,payload,schedule=best
    proof=replay_resident(p,code,payload,{inp:x},oracle=oracle,
                          run_contracts=schedule['run_contracts'],final_output=schedule['final_output'])
    # Compare coordinate descent restricted to the identical serial family.
    seed=min((r for r in records if r['halves'] in ([],list(heads)) and r['fused'] in ([],list(pairs))),
             key=lambda r:(r['serial_event_cost'],r['halves'],r['fused']))
    chosen=seed
    for _ in range(2):
        near=[r for r in records if len(set(r['halves'])^set(chosen['halves']))+
                                    len(set(r['fused'])^set(chosen['fused'])) <= 1]
        nxt=min(near,key=lambda r:(r['serial_event_cost'],r['halves'],r['fused']))
        if nxt['serial_event_cost'] >= chosen['serial_event_cost']:break
        chosen=nxt
    # This second scan is independent of enumeration order and tie selection.
    independent_min=min(r['serial_event_cost'] for r in records)
    assert independent_min==best_key[0] and proof['status']=='passed'
    return dict(fragment=[start,stop],catalogue_candidates=len(records),
                exhaustive_serial_optimum=independent_min,
                matched_serial_coordinate_heuristic=chosen,
                matched_heuristic_gap_fraction=chosen['serial_event_cost']/independent_min-1,
                production_optimize_resident=current,
                production_scope='different catalogue includes overlap; its optimality is not certified here',
                winning_replay=proof,records=records,
                scope='existing executable serial channel-tiling/activation-retention family; event predictor, not board latency')


def run(output):
    began=time.perf_counter();output.mkdir(parents=True,exist_ok=True)
    report=dict(schema=1,status='running',scope=__doc__,port_bytes=8,mac_lanes=8,
                model_limitations=['ideal intra-Conv operand reuse; input read once',
                    'no RTL lowering of new strip family','no overlap, channel tiling or cross-region cache',
                    'service scores and bounds apply only to declared model, never device cycles'],
                cases={},executable_audit={})
    cases=[]
    for name,kw in [('tiny_dw_pw_dw',{}),('tiny_stride_two',dict(h=8,stride_first=2)),
                    ('tiny_four_macros',dict(kernels=(3,1,3,1)))]:
        p,x=synthetic(name,**kw);cases.append((name,p,x,0,len(p.layers),True,{}))
    for model,start,stop in [('kws',3,9),('vww',15,21),('vww',23,31)]:
        p,x,_,pins=load_model(model)
        cases.append((f'{model}_layers_{start}_{stop}',p,x,start,stop,False,pins))
    for name,p,x,start,stop,tiny,pins in cases:
        ms=macros(p,start,stop);case=dict(macros=[asdict(m) for m in ms],sources=pins,budgets={})
        for budget in (8192,16384,32768):
            rows=build_catalogue(ms,budget);cert=dynamic_optimum(ms,rows)
            if cert['path'] is None:
                case['budgets'][str(budget)]=dict(status='family_infeasible',choices=len(rows));continue
            proof=verify_optimum_certificate(ms,rows,cert,check_all=True)
            controls={label:heuristic(ms,rows,pair) for label,pair in [('single_macro',False),('fixed_pair_strong',True)]}
            for control in controls.values():
                control['gap_fraction']=None if control['objective'] is None else control['objective']/cert['objective']-1
            result=dict(status='certified_model_optimum',choices=len(rows),feasible_choices=sum(r['feasible'] for r in rows),
                        optimal=cert,certificate=proof,controls=controls)
            brute=brute_force(ms,rows);assert brute['optimum']==cert['objective']
            result['independent_exhaustive']=brute
            relaxed=relaxed_work_bound(ms,rows)
            assert relaxed['lower_bound'] <= brute['optimum']
            relaxed['upper_over_lower']=cert['objective']/relaxed['lower_bound']
            result['relaxed_work_bound']=relaxed
            result['semantics']={'fixture':semantics(p,x,ms,cert['path'],rows),
                'seeded_int8_stress':semantics(p,np.random.default_rng(7681).integers(-128,128,x.shape,dtype=np.int8),
                                             ms,cert['path'],rows)}
            result['negative_checks']=negative_checks(ms,rows,cert)
            case['budgets'][str(budget)]=result
            save(output/f'{name}-{budget}-catalogue.json',dict(macros=case['macros'],choices=rows,certificate=cert))
            print(name,budget,cert['objective'],controls['fixed_pair_strong']['gap_fraction'],flush=True)
            assert rss() < 1024**3,'1 GiB memory guard exceeded'
        report['cases'][name]=case;save(output/'report.partial.json',report)
    for model,start,stop in [('kws',3,7),('vww',15,19)]:
        p,x,_,pins=load_model(model);oracle=evaluate(p,{p.inputs[0]:x})
        report['executable_audit'][model]=executable_audit(p,start,stop,oracle[p.layers[start].inputs[0]])
        print(model,'executable audit',report['executable_audit'][model]['matched_heuristic_gap_fraction'],flush=True)
    report.update(status='passed',wall_seconds=time.perf_counter()-began,peak_rss_bytes=rss(),
                  source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),
                    **{str(path.relative_to(ROOT)):sha(path) for path in (ROOT/'compiler/scheduler/spatial.py',
                       ROOT/'compiler/scheduler/resident.py',ROOT/'compiler/scheduler/resident_search.py',
                       ROOT/'compiler/scheduler/event_cost.py',ROOT/'compiler/integer_reference.py',
                       ROOT/'compiler/scheduler/resident_verify.py',ROOT/'compiler/phase4_compile.py',
                       ROOT/'compiler/phase4_tiling.py',ROOT/'compiler/quantization.py',
                       ROOT/'compiler/static_pipeline.py',ROOT/'tools/phase6/run_boardless.py')}})
    save(output/'report.json',report)
    return report


def check_saved(path):
    document=json.loads(path.read_text());ms=[Macro(**m) for m in document['macros']]
    proof=verify_optimum_certificate(ms,document['choices'],document['certificate'],check_all=True)
    brute=brute_force(ms,document['choices'])
    assert brute['optimum']==proof['verified_lower_bound']
    return dict(status='passed',certificate=proof,independent_exhaustive=brute)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=OUT)
    ap.add_argument('--check',type=Path,help='independently check a saved catalogue without invoking candidate generation or DP')
    args=ap.parse_args()
    if args.check: print(json.dumps(check_saved(args.check),indent=2))
    else: run(args.output)
