"""Command-level timing with independent engine/DMA timelines and waits.

Dispatch/fetch latency is derived from tile_sequencer.sv. Concurrent SRAM port
arbitration is not simulated here: durations are uncontended. This is a ranking
estimate, never a placement certificate, physical-latency bound or board result.
Use RTL/physical held-out errors to assess this deliberate approximation.
"""
from functools import lru_cache
import struct
from hardware_v2 import Descriptor
from .engine_cost import engine_cycles
from .contract import require


@lru_cache(maxsize=8192)
def _canonical_engine(encoded,entries,params,spatial):
    return engine_cycles(Descriptor.decode(bytes.fromhex(encoded)),entries,params,spatial)


def _engine(encoded,entries,params,spatial=False):
    d=Descriptor.decode(bytes.fromhex(encoded));d.validate()
    # An aligned base translation rotates direct-mapped cache indices and
    # translates full tags bijectively; hit/miss equality is unchanged.
    # Cache contents are invalidated for every descriptor.
    d.input=d.output=d.weight=d.params=0;d.next_pc=64
    return _canonical_engine(d.encode().hex(),entries,params,spatial)


def estimate(commands,payload,*,cache_entries=256,parameter_cache=True,spatial_pw=False,dma_cost=None):
    """dma_cost=None is the native harness's fixed one-response-cycle RAM."""
    require(len(commands)%16==0 and 0<len(commands)<=32768,'invalid command bytes')
    scratch=bytearray(32768);t=0;eend=dend=0
    engines=[];transfers=[];runs=[]
    for i in range(len(commands)//16):
        op,flags,res,a,b,c=struct.unpack_from('<BBHIII',commands,i*16)
        require(res==0,'reserved bits')
        t+=4 # FETCH0, READ0, FETCH1, READ1 before EXECUTE
        if op==0:
            require(i==len(commands)//16-1 and not any((flags,a,b,c)),'invalid halt')
            t=max(t,eend,dend)+1;break
        if op==1:
            require(flags in (0,1) and a%8==b%8==0 and 0<c<=32768 and b+c<=32768 and a+c<=len(payload),'invalid DMA')
            t=max(t,dend)
            transfer={'direction':'to_sram' if flags else 'from_sram','ext':a,'sram':b,'bytes':c}
            # The bridge presents dma_busy || dma_pending to the sequencer;
            # that pending register adds one counted cycle per transfer.
            duration=1+3*((c+7)//8) if dma_cost is None else dma_cost(transfer)
            require(type(duration) is int and duration>0,'invalid DMA duration')
            transfers.append((t+2,t+2+duration));dend=t+2+duration
            if flags: scratch[b:b+c]=payload[a:a+c]
        elif op==2:
            require(flags==c==0 and a%64==0 and a+128<=32768,'invalid RUN')
            t=max(t,eend,dend)
            d=Descriptor.decode(bytes(scratch[a:a+64]));d.validate()
            require(d.next_pc==a+64 and Descriptor.decode(scratch[a+64:a+128]).opcode==0,'unsupported descriptor chain')
            duration=_engine(d.encode().hex(),cache_entries,parameter_cache,spatial_pw)
            engines.append((t+2,t+2+duration));eend=t+2+duration
            runs.append({'command':i,'cycles':duration,'descriptor_hex':d.encode().hex()})
        elif op==3:
            require(flags in (1,2,3) and not any((a,b,c)),'invalid WAIT')
            t=max(t,eend if flags&1 else 0,dend if flags&2 else 0)
        else: raise ValueError('unsupported command')
        t+=2 # EXECUTE edge and ADVANCE edge
    else: raise ValueError('missing HALT')
    overlap=sum(max(0,min(b,d)-max(a,c)) for a,b in engines for c,d in transfers)
    return {'elapsed_cycles':t,'engine_cycles':sum(b-a for a,b in engines),
            'dma_cycles':sum(b-a for a,b in transfers),'overlap_cycles':overlap,'runs':runs,
            'scope':'uncontended command timelines; SRAM arbitration and changed physical timing uncalibrated'}
