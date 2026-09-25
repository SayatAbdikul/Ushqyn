"""Prospective geometry/placement holdouts for the analytical cost model.

The seed and shapes are fixed before physical execution. Outputs use centered
scalar arithmetic; prediction never consumes weights, inputs or device counters.
"""
import random
import struct

import numpy as np

from hardware_v2 import Descriptor
from kernel_vectors import rounded


def vectors():
    rng = random.Random(250926)
    def param(bias=0, zx=-13, low=-128, high=127, shift=37):
        return struct.pack('<iiBbbbbb',bias,1<<30,shift,3,zx,low,high,0)+bytes(2)

    shapes = [(4,3,11,2,3,3,1,1), (4,5,13,3,5,1,2,0),
              (4,3,9,17,3,3,1,0), (4,4,17,9,1,1,1,0),
              (4,5,67,3,3,3,2,1), (4,7,9,1,3,5,1,1),
              (6,7,9,3,3,3,2,1), (6,4,17,5,1,3,1,0),
              (6,5,11,7,3,3,1,1), (6,3,65,2,3,3,1,0)]
    for index,(op,ih,iw,ic,kh,kw,step,pad) in enumerate(shapes):
        oc=ic if op==6 else 3+(index%3)
        oh=(ih+2*pad-kh)//step+1;ow=(iw+2*pad-kw)//step+1
        reduction=kh*kw*(ic if op==4 else 1)
        x=np.array([rng.randrange(-128,128) for _ in range(ic*ih*iw)],np.int8).reshape(ic,ih,iw)
        w=np.array([rng.randrange(-32,33) for _ in range(oc*reduction)],np.int8).reshape(oc,reduction)
        biases=[rng.randrange(-100,101) for _ in range(oc)]
        expected=[]
        for c in range(oc):
            for y in range(oh):
                for z in range(ow):
                    acc=biases[c];j=0
                    for k in (range(ic) if op==4 else [c]):
                        for yy in range(kh):
                            for xx in range(kw):
                                a,b=y*step-pad+yy,z*step-pad+xx
                                value=int(x[k,a,b]) if 0<=a<ih and 0<=b<iw else -13
                                acc+=(value+13)*int(w[c,j]);j+=1
                    expected.append(rounded(acc,1<<30,37,3))
        d=Descriptor(op,input=192+8*(index%4),output=8192,weight=16384,params=24576,
                     count=reduction,outputs=len(expected),row_stride=(reduction+7)//8*8,next_pc=64,
                     input_h=ih,input_w=iw,input_c=ic,output_c=oc,kernel_h=kh,kernel_w=kw,
                     stride_h=step,stride_w=step,pad_top=pad,pad_bottom=pad,pad_left=pad,pad_right=pad)
        params=[param(biases[c]+13*int(w[c].astype(np.int64).sum())) for c in range(oc)]
        yield d,x.ravel(),w,params,expected,len(expected)*reduction,f'holdout_spatial_{index}'
    for n in (1,7,9,31,129,257):
        x=np.array([rng.randrange(-128,128) for _ in range(n)],np.int8)
        w=np.array([rng.randrange(-16,17) for _ in range(3*n)],np.int8).reshape(3,n)
        d=Descriptor(1,input=200,output=8192,weight=16384,params=24576,
                     count=n,outputs=3,row_stride=(n+7)//8*8,next_pc=64)
        expected=[rounded(sum((int(v)+13)*int(t) for v,t in zip(x,row)),1<<30,37,3) for row in w]
        yield d,x,w,[param(13*int(row.astype(np.int64).sum())) for row in w],expected,3*n,f'holdout_fc_{n}'
    for op in (2,8):
        for n in (1,9,33):
            x=np.array([rng.randrange(-128,128) for _ in range(n)],np.int8)
            high=127 if op==2 else 19
            expected=[min(high,max(-13,int(v)))+16 for v in x]
            expected=[min(127,v) for v in expected]
            d=Descriptor(op,input=200,output=8192,params=24576,count=n,outputs=n,next_pc=64)
            yield d,x,None,[param(low=-13,high=high,shift=30)],expected,0,f'holdout_clamp_{op}_{n}'
    for op in (5,7):
        for h,w in ((2,3),(3,7),(5,9)):
            channels=3
            x=np.array([rng.randrange(-128,128) for _ in range(channels*h*w)],np.int8)
            expected=[]
            for c in range(channels):
                values=x[c*h*w:(c+1)*h*w]
                acc=sum(int(v)+13 for v in values) if op==7 else int(max(values))+13
                expected.append(rounded(acc,1<<30,37,3))
            d=Descriptor(op,input=200,output=8192,params=24576,count=len(x),outputs=channels,next_pc=64,
                         input_h=h,input_w=w,input_c=channels,output_c=channels,kernel_h=h,kernel_w=w)
            yield d,x,None,[param()],expected,0,f'holdout_pool_{op}_{h}_{w}'
