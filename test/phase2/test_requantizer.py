import random
import cocotb
from cocotb.triggers import Timer

async def step(d):
    d.clk.value=0;await Timer(5,units='ns');d.clk.value=1;await Timer(5,units='ns')

def oracle(a,m,s,z):
    v=a*m;q,r=divmod(abs(v),1<<s)
    q+=2*r>=(1<<s)
    return max(-128,min(127,(-q if v<0 else q)+z))

@cocotb.test()
async def rounding_stalls_flush(d):
    d.rst_n.value=0;d.in_valid.value=0;d.out_ready.value=0;d.flush.value=0;await step(d);d.rst_n.value=1
    rng=random.Random(20260910)
    cases=[(a,m,s,z) for a in [-2147483648,-257,-3,-1,0,1,3,255,2147483647] for m in [1,3,1073741824,2147483647] for s in [0,1,7,31,62] for z in [-128,0,127]]
    cases += [(rng.randrange(-2**31,2**31),rng.randrange(1,2**31),rng.randrange(63),rng.randrange(-128,128)) for _ in range(1000)]
    pending=None;i=0;received=0
    while i<len(cases) or pending is not None:
        ready=rng.randrange(3)!=0;valid=i<len(cases) and rng.randrange(4)!=0
        d.out_ready.value=ready;d.in_valid.value=valid
        if valid:
            a,m,s,z=cases[i];d.accumulator.value=a;d.multiplier.value=m;d.shift.value=s;d.zero_point.value=z
        d.clk.value=0;await Timer(5,units='ns')
        if int(d.out_valid.value):
            assert pending is not None
            assert d.result.value.signed_integer==pending
            if ready:pending=None;received+=1
        if valid and int(d.in_ready.value):pending=oracle(*cases[i]);i+=1
        d.clk.value=1;await Timer(5,units='ns')
    assert received==len(cases)
    d.in_valid.value=1;d.out_ready.value=0;await step(d);d.flush.value=1;await step(d);assert not int(d.out_valid.value)
