"""Independent proof checks for executable retained activation schedules."""
import copy
import struct
import numpy as np
import pytest
from hardware_v2 import Descriptor
from scheduler.fixtures import wide_chain
from scheduler.resident import compile_resident, eligible
from scheduler.resident_verify import replay_resident


def check(p,x,c,b,s):
    return replay_resident(p,c,b,{'x':x},run_contracts=s['run_contracts'],final_output=s['final_output'],snapshot_regions=s['snapshot_regions'])

@pytest.mark.parametrize('half',[False,True])
@pytest.mark.parametrize('overlap',[False,True])
@pytest.mark.parametrize('snapshots',[False,True])
@pytest.mark.parametrize('fused',[(),None])
def test_resident_exact_and_bounded(half,overlap,snapshots,fused):
    p,x=wide_chain();c,b,s=compile_resident(p,fused=fused,prefer_half=half,overlap=overlap,snapshots=snapshots)
    assert check(p,x,c,b,s)['status']=='passed'
    assert max(t['live'][1] for t in s['stages'])<=32768
    assert len(c)<=32768 and len(b)<=8*1024*1024
    if fused is None: assert any(t['inplace'] for t in s['stages'])


def test_reject_unsupported_fusion():
    p,x=wide_chain()
    with pytest.raises(ValueError): compile_resident(p,fused=[999])


@pytest.mark.parametrize('damage',['layer','coordinate','duplicate','missing','parameter','extent','input','wait'])
def test_resident_proofs_are_untrusted(damage):
    p,x=wide_chain();c,b,s=compile_resident(p,overlap=False)
    c=bytearray(c);b=bytearray(b)
    keys=list(s['run_contracts']);key=keys[0]
    if damage=='layer': s['run_contracts'][key]['layer']+=1
    if damage=='coordinate': s['run_contracts'][key]['first_element']+=1
    if damage=='duplicate': s['run_contracts'][keys[1]]=copy.deepcopy(s['run_contracts'][key])
    if damage=='missing': del s['run_contracts'][key]
    if damage=='parameter':
        load=next(t for t in s['stages'][0]['loads'] if t['role']=='parameter');b[load['ext']]^=1
    if damage=='extent': struct.pack_into('<I',c,int(key)*16+8,128<<16)
    if damage=='input':
        load=s['stages'][1]['loads'][0];d=Descriptor.decode(b[load['ext']:load['ext']+64]);d.input+=8
        b[load['ext']:load['ext']+64]=d.encode()
    if damage=='wait':
        offset=(int(key)+1)*16
        assert c[offset]==3;c[offset+1]=2
    with pytest.raises(ValueError): check(p,x,bytes(c),bytes(b),s)


def test_retention_removes_roundtrip_without_changing_quantization():
    p,x=wide_chain()
    a=compile_resident(p,fused=(),overlap=False);b=compile_resident(p,overlap=False)
    ra=check(p,x,*a);rb=check(p,x,*b)
    size=p.tensors[p.layers[0].output].shape
    assert sum(ra['dma_bytes'].values())>sum(rb['dma_bytes'].values())
    assert p.tensors[p.layers[0].output].quantization!=p.tensors[p.layers[1].output].quantization
