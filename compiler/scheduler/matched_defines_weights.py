"""Whole-stack immutable residency variant of the validated DeFiNES backend.

The default delegates byte-for-byte to the generic backend. The experimental
option reserves all live weight rows, parameters and exact activation LUTs for
the entire stack, so their storage cannot be reused by transient activations.
The explicit source transformation fails closed if the base implementation
changes; its generated source hash is recorded by the experiment runner.
"""
import hashlib
import inspect
from functools import lru_cache
from . import matched_defines as generic


@lru_cache(maxsize=1)
def implementation():
    source=inspect.getsource(generic.compile_stack)
    changes=[
        ("    needs=shapes_and_needs(block,tile_h,tile_w)",
         """    resident={}
    for index,row in enumerate(parameters):
        if not row['live']:continue
        for key in ('weight','params','lut'):
            buffer=arena.alloc(len(row[key]),f'resident-{index}-{key}')
            resident[(index,key)]=buffer
            dma('to_sram',row[key+'_ext'],buffer.base,len(row[key]),key)
    resident_bytes=sum(8+align8(b.size) for b in resident.values())
    needs=shapes_and_needs(block,tile_h,tile_w)"""),
        ("""                        for key in ('weight','params','lut'):
                            b=arena.alloc(len(row[key]),key);live.append(b)
                            dma('to_sram',row[key+'_ext'],b.base,len(row[key]),key)
                        wb,pb_,lb=live;depthwise=a.get('group',1)!=1""",
         """                        wb,pb_,lb=(resident[(j,key)] for key in ('weight','params','lut'))
                        depthwise=a.get('group',1)!=1"""),
        ("    if arena.buffers:raise ValueError('leaked-SRAM-buffer')",
         """    for buffer in resident.values():arena.free(buffer)
    if arena.buffers:raise ValueError('leaked-SRAM-buffer')"""),
        ("        tile_h=tile_h,tile_w=tile_w,mode=mode,retain_weights=retain_weights,prefetch=prefetch,",
         "        tile_h=tile_h,tile_w=tile_w,mode=mode,retain_weights=retain_weights,prefetch=prefetch,\n        resident_weights=True,resident_immutable_bytes=resident_bytes,"),
    ]
    for old,new in changes:
        if source.count(old)!=1:raise ValueError('generic backend changed; whole-stack residency patch needs review')
        source=source.replace(old,new)
    namespace=dict(vars(generic))
    exec(compile(source,'<matched_defines_weights.generated>','exec'),namespace)
    return namespace['compile_stack'],source


def compile_stack(program,start,stop,tile_h,tile_w,mode=1,*,retain_weights=True,
                  prefetch=True,resident_weights=False):
    if type(resident_weights) is not bool:raise ValueError('resident_weights must be boolean')
    function=implementation()[0] if resident_weights else generic.compile_stack
    code,payload,record=function(program,start,stop,tile_h,tile_w,mode,
        retain_weights=retain_weights,prefetch=prefetch)
    if resident_weights:
        record['resident_lowering_sha256']=hashlib.sha256(implementation()[1].encode()).hexdigest()
    return code,payload,record
