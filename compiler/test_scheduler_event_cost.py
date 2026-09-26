"""Independent timeline cases, operation bounds and finite-space checks."""
import struct
import pytest
from hardware_v2 import Descriptor
from phase4_cost import engine_cycles as old_cycles
from scheduler.engine_cost import engine_cycles
from scheduler.event_cost import estimate


def command(op,flags=0,a=0,b=0,c=0): return struct.pack('<BBHIII',op,flags,0,a,b,c)

@pytest.mark.parametrize('n',[1,7,8,9,63,64,65,127,128,129,255,256,257])
def test_disabled_cache_preserves_structural_model(n):
    d=Descriptor(4,input=512,output=16000,weight=24000,params=30000,count=n,outputs=27,
                 row_stride=(n+7)//8*8,next_pc=64,input_h=3,input_w=3,input_c=n,output_c=3)
    assert engine_cycles(d,0,False)==old_cycles(d)
    assert engine_cycles(d,256,True)<=old_cycles(d)


def test_sequencer_timeline_counts_fetch_wait_and_unit_start():
    payload=Descriptor(2,input=128,output=136,params=144,count=8,outputs=8,next_pc=64).encode()+Descriptor(0).encode()
    c=command(1,1,0,0,128)+command(3,2)+command(2,0,0,160<<16)+command(3,3)+command(0)
    r=estimate(c,payload)
    assert r['engine_cycles']==97 and r['dma_cycles']==49 and r['overlap_cycles']==0
    assert r['elapsed_cycles']==97+49+2*8+5


def test_overlap_changes_timeline_not_sum_of_components():
    payload=Descriptor(2,input=128,output=256,params=512,count=128,outputs=128,next_pc=64).encode()+Descriptor(0).encode()+bytes(512)
    start=command(1,1,0,0,128)+command(3,2)+command(2,0,0,528<<16)
    transfer=command(1,1,128,16384,512)+command(3,2)
    parallel=estimate(start+transfer+command(3,3)+command(0),payload)
    serial=estimate(start+command(3,3)+transfer+command(0),payload)
    assert parallel['engine_cycles']==serial['engine_cycles']
    assert parallel['dma_cycles']==serial['dma_cycles']
    assert parallel['overlap_cycles']==193
    assert parallel['elapsed_cycles']<serial['elapsed_cycles']


@pytest.mark.parametrize('plane,expected',[(8,165),(9,255),(16,283)])
def test_spatial_kernel_aligned_unaligned_and_tail_cost(plane,expected):
    d=Descriptor(4,input=512,output=16000,weight=24000,params=30000,count=8,outputs=2*plane,
                 row_stride=8,next_pc=64,input_h=1,input_w=plane,input_c=8,output_c=2)
    assert engine_cycles(d,256,True,True)==expected
    # Cache-index translation and arbitrary aligned placement do not alter
    # misses or split-word decisions. This justifies timing-key normalization.
    d.input=520
    assert engine_cycles(d,256,True,True)==expected
