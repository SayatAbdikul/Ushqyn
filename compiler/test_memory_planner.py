import pytest
from memory_planner import Region,allocate
from memory_verifier import verify


def test_live_reuse_and_port_budget():
    regions=[Region('weight',16,-1,3,'weights',b'1'*16),
             Region('input',17,-1,0,'tensor'),
             Region('a',24,0,1,'tensor'),
             Region('b',24,1,2,'tensor'),
             Region('c',24,2,3,'tensor')]
    layout=allocate(regions,1024,64)
    report=verify(layout,1024,64)
    assert report['allocated_high_watermark']<64+16+24+24+24+24
    assert report['peak_live_bytes']<=report['allocated_high_watermark']-64
    bad=[dict(x) for x in layout]
    bad[-1]['offset']=bad[-2]['offset']
    bad[-1]['first']=bad[-2]['first']
    with pytest.raises(ValueError,match='live overlap'):verify(bad,1024,64)


def test_physical_bounds_and_rejection():
    with pytest.raises(ValueError,match='exceeds'):allocate([Region('large',1000,0,1,'tensor')],1024,64)
    with pytest.raises(ValueError):verify([dict(name='a',offset=1024,size=8,logical_bytes=8,first=0,last=1)],1024,64)
