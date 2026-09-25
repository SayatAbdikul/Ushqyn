"""Schedule ownership checks independent of the compiler's command emitter."""
import hashlib
import json
import struct
from pathlib import Path

import pytest

from hardware_v2 import Descriptor
from phase4_sequence import compile_sequence
from phase4_tiling import plan_graph

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name',['kws','vww'])
@pytest.mark.parametrize('overlap',[False,True])
def test_full_inventory_live_regions_and_prefetch_dependencies(name,overlap):
    inventory=json.loads((ROOT/f'benchmarks/manifests/{name}.canonical-inventory.json').read_text())
    plan=plan_graph(inventory,'test',prefer_half=True)
    image=bytes(plan['parameter_end'])
    plan['parameter_image_sha256']=hashlib.sha256(image).hexdigest()
    program,payload,schedule=compile_sequence(plan,image,overlap,True)
    assert len(program)<=32768
    assert len(schedule['snapshot_regions'])==sum(bool(l['tiles']) for l in plan['layers'])
    live=None;dma=None;computes=0;overlapped=0
    for offset in range(0,len(program),16):
        op,flags,reserved,a,b,c=struct.unpack('<BBHIII',program[offset:offset+16])
        assert reserved==0
        if op==1:
            assert dma is None
            assert a%8==b%8==0 and c>0 and b+c<=32768 and a+c<=8388608
            if live:
                assert b+c<=live[0] or b>=live[1]
                overlapped+=c
                tile=schedule['tiles'][computes]
                assert flags==1
                load=next(t for t in tile['loads'] if (t['ext'],t['sram'],t['bytes'])==(a,b,c))
                if load['role']=='input':
                    assert tile['layer']==schedule['tiles'][computes-1]['layer']
            dma=(a,b,c)
        elif op==2:
            assert live is None and dma is None
            live=(b&65535,b>>16)
            assert a==live[0] and live[1]<=32768
            tile=schedule['tiles'][computes];computes+=1
            d=Descriptor.decode(bytes.fromhex(tile['descriptor_hex']));d.validate()
            for r in tile['regions'].values():
                assert live[0]+128<=r['base'] and r['base']+r['bytes']<=live[1]
        elif op==3:
            if flags&1: live=None
            if flags&2: dma=None
        else:
            assert op==0 and offset==len(program)-16 and live is None and dma is None
    assert computes==len(schedule['tiles'])
    assert overlapped==schedule['prefetch_payload_bytes']
    if overlap: assert overlapped>0


def test_reject_wrong_parameter_image():
    with pytest.raises(ValueError,match='hash'):
        compile_sequence({'parameter_image_sha256':'wrong'},b'')
