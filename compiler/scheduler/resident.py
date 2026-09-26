"""Executable, exact producer -> elementwise SRAM retention on the v2 ABI.

This is a bounded fusion catalogue, not general halo fusion or a reproduction
of DeFiNES. The original producer and activation quantizers execute separately.
After the producer finishes, its input/weights/parameters are dead; its output
stays in SRAM while the activation runs in place. No extra scratch bytes or
arithmetic transformation is hidden in the cost model.
"""
import copy
import hashlib
import struct
from hardware_v2 import Descriptor
from phase4_compile import compile_tiled, _parameter_rows
from phase4_tiling import align8, EXT_BYTES
from .contract import require


def eligible(program):
    return tuple(i for i in range(len(program.layers)-1)
                 if program.layers[i].op in ('Conv','Gemm','MaxPool','AveragePool','GlobalAveragePool')
                 and program.layers[i+1].op in ('Relu','Clip')
                 and program.layers[i+1].inputs == [program.layers[i].output]
                 and program.tensors[program.layers[i].output].shape == program.tensors[program.layers[i+1].output].shape)


def compile_resident(program, *, fused=None, prefer_half=True, overlap=True, snapshots=False,
                     tile_choices=None, prepared_plans=None):
    possible=set(eligible(program));selected=possible if fused is None else set(fused)
    require(selected <= possible, 'unsupported retained segment')
    tile_choices={} if tile_choices is None else dict(tile_choices)
    require(all(type(i) is int and 0<=i<len(program.layers) and type(v) is bool for i,v in tile_choices.items()),'invalid tile choice')
    plans=prepared_plans if prepared_plans is not None else {h:compile_tiled(program,prefer_half=h) for h in (False,True)}
    require(plans[False][1]==plans[True][1],'different immutable image across tile plans')
    plan,image=plans[prefer_half]
    slot=plan['activation_slot_bytes'];payload=bytearray(image)
    stages=[];bank=0;input_slot=0;skip=set()
    for original_layer in plan['layers']:
        layer=plans[tile_choices.get(original_layer['index'],prefer_half)][0]['layers'][original_layer['index']]
        i=layer['index']
        if i in skip or not layer['tiles']: continue
        post=i+1 if i in selected else None
        if post is not None: skip.add(post)
        for tile in layer['tiles']:
            base=bank*16384 if tile['scratch_bytes']<=16384 else 0
            bank=1-bank if tile['scratch_bytes']<=16384 else 0
            d=Descriptor.decode(bytes.fromhex(tile['descriptor_hex']))
            for key in ('input','output','weight','params','next_pc'):
                if key in ('input','output','next_pc') or getattr(d,key): setattr(d,key,getattr(d,key)+base)
            d.validate()
            first=tile['transfers'][-1]['ext']-layer['output_slot']*slot
            def descriptor_load(desc):
                ext=len(payload);payload.extend(desc.encode()+Descriptor(0).encode())
                return dict(direction='to_sram',ext=ext,sram=base,bytes=128,role='descriptor')
            loads=[descriptor_load(d)]
            for t in tile['transfers'][:-1]:
                t=copy.deepcopy(t);t['sram']+=base
                if t['sram']==d.input:
                    t['ext']+= (input_slot-layer['input_slot'])*slot;t['role']='input'
                else: t['role']='parameter'
                loads.append(t)
            output=dict(direction='from_sram',ext=(1-input_slot)*slot+first,sram=d.output,bytes=d.outputs)
            stages.append(dict(layer=i,group=i,first_element=first,pc=base,live=[base,base+tile['scratch_bytes']],
                               descriptor_hex=d.encode().hex(),loads=loads,store=None if post is not None else output,
                               output=dict(sram=d.output,bytes=d.outputs),inplace=False))
            if post is not None:
                _,params=_parameter_rows(program,program.layers[post])
                # Reuse the producer's dead parameter record. Original tiling
                # reserves >=16 bytes for every eligible producer.
                require('params' in tile['regions'] and tile['regions']['params']['bytes']>=16,'missing reusable parameter slot')
                rd=Descriptor(2 if program.layers[post].op=='Relu' else 8,input=d.output,output=d.output,
                              params=d.params,count=d.outputs,outputs=d.outputs,next_pc=base+64)
                rd.validate();rload=descriptor_load(rd)
                ext=len(payload);payload.extend(params)
                stages.append(dict(layer=post,group=i,first_element=first,pc=base,live=[base,base+tile['scratch_bytes']],
                                   descriptor_hex=rd.encode().hex(),
                                   loads=[rload,dict(direction='to_sram',ext=ext,sram=d.params,bytes=16,role='parameter')],
                                   store=output,output=dict(sram=d.output,bytes=d.outputs),inplace=True))
        input_slot=1-input_slot
    snapshot_regions={}
    if snapshots:
        import math
        for i in sorted({s['layer'] for s in stages}):
            size=math.prod(program.tensors[program.layers[i].output].shape)
            snapshot_regions[i]={'ext':len(payload),'bytes':size}
            payload.extend(bytes(align8(size)))
    require(len(payload)<=EXT_BYTES,'resident payload/snapshots exceed SDRAM')
    commands=[];contracts={};prefetched=set();overlap_bytes=0
    def emit(op,flags=0,a=0,b=0,c=0): commands.append((op,flags,a,b,c))
    def dma(t):
        require(t['sram']%8==t['ext']%8==0 and 0<t['bytes']<=32768 and t['sram']+t['bytes']<=32768 and t['ext']+t['bytes']<=len(payload),'resident DMA bounds')
        emit(1,int(t['direction']=='to_sram'),t['ext'],t['sram'],t['bytes']);emit(3,2)
    for index,s in enumerate(stages):
        for j,load in enumerate(s['loads']):
            if (index,j) not in prefetched: dma(load)
        contracts[str(len(commands))]={'layer':s['layer'],'first_element':s['first_element']}
        emit(2,0,s['pc'],s['live'][0]|s['live'][1]<<16)
        if overlap and index+1<len(stages):
            nxt=stages[index+1]
            if s['live'][1]<=nxt['live'][0] or nxt['live'][1]<=s['live'][0]:
                for j,load in enumerate(nxt['loads']):
                    if load['role']!='input' or s['group']==nxt['group']:
                        dma(load);prefetched.add((index+1,j));overlap_bytes+=load['bytes']
        emit(3,3)
        if snapshots:
            dma(dict(direction='from_sram',ext=snapshot_regions[s['layer']]['ext']+s['first_element'],**s['output']))
        if s['store'] is not None: dma(s['store'])
    emit(0)
    command_bytes=b''.join(struct.pack('<BBHIII',op,flags,0,a,b,c) for op,flags,a,b,c in commands)
    require(len(command_bytes)<=32768,'resident program exceeds command capacity')
    import math
    schedule={'schema':1,'catalogue':'producer -> exact Relu/Clip retention only','fused':sorted(selected),
              'prefer_half':prefer_half,'overlap_enabled':overlap,'snapshots_enabled':snapshots,
              'tile_choices':tile_choices,
              'stages':stages,'run_contracts':contracts,'snapshot_regions':snapshot_regions,
              'command_count':len(commands),'program_bytes':len(command_bytes),
              'prefetch_payload_bytes':overlap_bytes,
              'program_sha256':hashlib.sha256(command_bytes).hexdigest(),'image_sha256':hashlib.sha256(payload).hexdigest(),
              'final_output':{'ext':input_slot*slot,'bytes':math.prod(program.tensors[program.outputs[0]].shape)}}
    return command_bytes,bytes(payload),schedule
