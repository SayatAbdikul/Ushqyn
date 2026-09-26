"""Bounded search in the actually executable retained-activation catalogue.

Every candidate uses unchanged INT8 arithmetic, channel tiling from the frozen
compiler, and optional immediate producer->Relu/Clip SRAM retention. The event
objective includes command overhead and overlap; it is a predictive model.
This adapter is not a complete DeFiNES or tile-expanded COSMA reproduction.
"""
import itertools
import time
from hardware_v2 import Descriptor
from phase4_compile import compile_tiled
from .contract import require
from .event_cost import estimate, _engine
from .resident import compile_resident, eligible


def optimize_resident(program, dma_cost, *, cache_entries=256,parameter_cache=True,
                      spatial_pw=False,seconds=60,max_evaluations=512,max_sweeps=2):
    require(seconds>0 and max_evaluations>=8 and max_sweeps>0,'invalid search budget')
    began=time.monotonic();plans={h:compile_tiled(program,prefer_half=h) for h in (False,True)}
    heads=tuple(i for i,l in enumerate(plans[False][0]['layers']) if l['tiles'])
    pairs=eligible(program);evaluated={};history=[];stop=None
    def key(halves,fused,overlap): return (tuple(sorted(halves)),tuple(sorted(fused)),bool(overlap))
    def evaluate(k):
        if k not in evaluated:
            halves,fused,overlap=k
            artifact=compile_resident(program,fused=fused,prefer_half=False,
                tile_choices={i:i in halves for i in heads},overlap=overlap,prepared_plans=plans)
            costs=estimate(artifact[0],artifact[1],cache_entries=cache_entries,
                           parameter_cache=parameter_cache,spatial_pw=spatial_pw,dma_cost=dma_cost)
            evaluated[k]=(costs['elapsed_cycles'],artifact,costs)
            history.append({'half_layers':list(halves),'fused':list(fused),'overlap':overlap,
                            'predicted_cycles':costs['elapsed_cycles']})
        return evaluated[k]
    # Equal hardware and cost model for every baseline. In particular include
    # the current overlapped schedule; a serial winner may never silently
    # replace a faster overlapped fallback.
    baselines={}
    for half,fuse,overlap in itertools.product((False,True),repeat=3):
        k=key(heads if half else (),pairs if fuse else (),overlap)
        label=('half' if half else 'full')+('-retain' if fuse else '-materialize')+('-overlap' if overlap else '-serial')
        baselines[label]={'configuration':k,'predicted_cycles':evaluate(k)[0]}
    best=min(evaluated,key=lambda k:(evaluated[k][0],k))
    for sweep in range(max_sweeps):
        current=best;halves,fused,overlap=best
        candidates=[key(halves,fused,not overlap)]
        candidates += [key(set(halves)^{i},fused,overlap) for i in heads if i-1 not in fused]
        candidates += [key(halves,set(fused)^{i},overlap) for i in pairs]
        for k in sorted(set(candidates)):
            if k not in evaluated and len(evaluated)>=max_evaluations: stop='evaluation_budget';break
            if time.monotonic()-began>=seconds: stop='wall_timeout';break
            if evaluate(k)[0]<evaluate(best)[0]: best=k
        if stop or best==current: break
    objective,artifact,costs=evaluate(best)
    # Admissible bound for this finite catalogue: a single engine must execute
    # all original layers. Independently minimize each layer's uncontended
    # cycles, relaxing tile compatibility and discarding DMA/dispatch costs.
    # Activation retention may change its descriptor count, so include those
    # counts too. This is NOT a universal bound over other hardware/kernels.
    minima=[]
    for i in heads:
        choices=[]
        for half in (False,True):
            tiles=plans[half][0]['layers'][i]['tiles']
            choices.append(sum(_engine(t['descriptor_hex'],cache_entries,parameter_cache,spatial_pw) for t in tiles))
            if i-1 in pairs:
                producer=plans[half][0]['layers'][i-1]['tiles'];total=0
                for t in producer:
                    d=Descriptor.decode(bytes.fromhex(t['descriptor_hex']))
                    r=Descriptor(2 if program.layers[i].op=='Relu' else 8,input=d.output,output=d.output,params=d.params,count=d.outputs,outputs=d.outputs,next_pc=64)
                    total+=_engine(r.encode().hex(),cache_entries,parameter_cache,spatial_pw)
                choices.append(total)
        minima.append(min(choices))
    lower=sum(minima);require(objective>=lower,'invalid relaxed bound')
    report={'scope':'bounded executable channel-tiling + exact activation-retention catalogue; not complete B3/B4',
            'objective_scope':costs['scope'],'baselines':baselines,'evaluations':len(evaluated),'history':history,
            'selected':{'half_layers':list(best[0]),'fused':list(best[1]),'overlap':best[2]},
            'objective':objective,'lower_bound':lower,'bound_gap_fraction':(objective-lower)/max(1,lower),
            'optimal_within_catalogue':objective==lower,'stop_reason':stop,
            'wall_seconds':time.monotonic()-began,'max_evaluations':max_evaluations,'max_sweeps':max_sweeps,
            'deterministic_expansion_budget':stop!='wall_timeout','cache_entries':cache_entries,'parameter_cache':parameter_cache,'spatial_pw':spatial_pw}
    return artifact,report
