"""Deterministic scalar-reference cases shared by RTL and physical tests."""
import random
import struct
import numpy as np
from hardware_v2 import Descriptor
from quantization import multiplier_shift


def rounded(acc, multiplier, shift, zero_point):
    n = int(acc) * int(multiplier)
    q, r = divmod(abs(n), 1 << shift)
    q += 2*r >= (1 << shift)
    return min(127, max(-128, (-q if n < 0 else q) + zero_point))


def vectors():
    rng = random.Random(240923)
    def param(bias, m, s, zy, zx, low=-128, high=127):
        return struct.pack('<iiBbbbbb', bias, m, s, zy, zx, low, high, 0) + b'\0\0'

    # KWS first Conv: actual 10x4 kernel, stride 2, asymmetric SAME padding.
    ih, iw, kh, kw, oc = 11, 5, 10, 4, 4
    pt, pl, pb, pr = 4, 1, 5, 1
    oh, ow = (ih+pt+pb-kh)//2+1, (iw+pl+pr-kw)//2+1
    x = np.array([rng.randrange(-128, 128) for _ in range(ih*iw)], np.int8).reshape(1, ih, iw)
    w = np.array([rng.randrange(-127, 128) for _ in range(oc*kh*kw)], np.int8).reshape(oc, kh*kw)
    zx, zy, m, s = -17, 3, 1 << 30, 38
    biases = [rng.randrange(-100, 101) for _ in range(oc)]
    expected = []
    for c in range(oc):
        for y in range(oh):
            for z in range(ow):
                acc = biases[c]
                for ky in range(kh):
                    for kx in range(kw):
                        yy, xx = y*2-pt+ky, z*2-pl+kx
                        q = int(x[0, yy, xx]) if 0 <= yy < ih and 0 <= xx < iw else zx
                        acc += (q-zx)*int(w[c, ky*kw+kx])
                expected.append(rounded(acc, m, s, zy))
    desc = Descriptor(4, input=512, output=4096, weight=8192, params=16384,
                      count=kh*kw, outputs=len(expected), row_stride=40, next_pc=64,
                      kernel_h=kh, kernel_w=kw, stride_h=2, stride_w=2,
                      pad_top=pt, pad_left=pl, pad_bottom=pb, pad_right=pr,
                      input_h=ih, input_w=iw, input_c=1, output_c=oc)
    params = [param(biases[c]-zx*int(w[c].astype(np.int64).sum()),m,s,zy,zx) for c in range(oc)]
    yield (desc, x.ravel(), w, params, expected, len(expected)*kh*kw,'kws_conv_10x4_s2')

    # Pointwise channel reduction uses the ordinary shared Conv lane path.
    ic,oc,ih,iw=4,5,3,4
    x=np.array([rng.randrange(-128,128) for _ in range(ic*ih*iw)],np.int8).reshape(ic,ih,iw)
    w=np.array([rng.randrange(-127,128) for _ in range(oc*ic)],np.int8).reshape(oc,ic)
    zx,zy,m,s=-31,12,1<<30,37
    biases=[rng.randrange(-30,31) for _ in range(oc)]
    expected=[]
    for c in range(oc):
        for y in range(ih):
            for z in range(iw):
                acc=biases[c]+sum((int(x[k,y,z])-zx)*int(w[c,k]) for k in range(ic))
                expected.append(rounded(acc,m,s,zy))
    desc=Descriptor(4,input=512,output=4096,weight=8192,params=16384,
                    count=ic,outputs=len(expected),row_stride=8,next_pc=64,
                    input_h=ih,input_w=iw,input_c=ic,output_c=oc)
    params=[param(biases[c]-zx*int(w[c].astype(np.int64).sum()),m,s,zy,zx) for c in range(oc)]
    yield (desc,x.ravel(),w,params,expected,len(expected)*ic,'pointwise_4_to_5')

    # A 70-byte row wraps the four-row line buffer's eight-word set. The
    # 81-term filters also cross the old 64-weight cache limit and leave a
    # one-lane tail in the shared MAC reduction.
    ih,iw,ic,oc,kh,kw=3,70,9,2,3,3
    x=np.array([rng.randrange(-128,128) for _ in range(ic*ih*iw)],np.int8).reshape(ic,ih,iw)
    w=np.array([rng.randrange(-128,128) for _ in range(oc*ic*kh*kw)],np.int8).reshape(oc,ic*kh*kw)
    x.flat[0]=-128;x.flat[-1]=127;w.flat[0]=127;w.flat[-1]=-128
    zx,zy,m,s=-73,-4,1<<30,41
    biases=[rng.randrange(-100,101) for _ in range(oc)]
    expected=[]
    for c in range(oc):
        for z in range(iw-kw+1):
            acc=biases[c]
            for k in range(ic):
                for yy in range(kh):
                    for xx in range(kw):
                        acc+=(int(x[k,yy,z+xx])-zx)*int(w[c,k*kh*kw+yy*kw+xx])
            expected.append(rounded(acc,m,s,zy))
    desc=Descriptor(4,input=512,output=4096,weight=8192,params=16384,
                    count=ic*kh*kw,outputs=len(expected),row_stride=88,next_pc=64,
                    kernel_h=kh,kernel_w=kw,input_h=ih,input_w=iw,input_c=ic,output_c=oc)
    params=[param(biases[c]-zx*int(w[c].astype(np.int64).sum()),m,s,zy,zx) for c in range(oc)]
    yield (desc,x.ravel(),w,params,expected,len(expected)*ic*kh*kw,'wide_row_conv_81_terms')

    # A 261-term reduction exercises the uncached weight-stream fallback and
    # accumulation across 33 eight-lane tiles.
    ih,iw,ic,oc,kh,kw=3,9,29,2,3,3
    x=np.array([rng.randrange(-128,128) for _ in range(ic*ih*iw)],np.int8).reshape(ic,ih,iw)
    w=np.array([rng.randrange(-128,128) for _ in range(oc*ic*kh*kw)],np.int8).reshape(oc,ic*kh*kw)
    zx,zy,m,s=127,2,1<<30,43
    biases=[rng.randrange(-100,101) for _ in range(oc)]
    expected=[]
    for c in range(oc):
        for z in range(iw-kw+1):
            acc=biases[c]
            for k in range(ic):
                for yy in range(kh):
                    for xx in range(kw):
                        acc+=(int(x[k,yy,z+xx])-zx)*int(w[c,k*kh*kw+yy*kw+xx])
            expected.append(rounded(acc,m,s,zy))
    desc=Descriptor(4,input=512,output=4096,weight=8192,params=16384,
                    count=ic*kh*kw,outputs=len(expected),row_stride=264,next_pc=64,
                    kernel_h=kh,kernel_w=kw,input_h=ih,input_w=iw,input_c=ic,output_c=oc)
    params=[param(biases[c]-zx*int(w[c].astype(np.int64).sum()),m,s,zy,zx) for c in range(oc)]
    yield (desc,x.ravel(),w,params,expected,len(expected)*ic*kh*kw,'conv_261_term_streamed_fallback')

    # VWW depthwise stride-2 boundary: one input channel per filter.
    ih, iw, ic = 5, 7, 8
    x = np.array([rng.randrange(-128, 128) for _ in range(ic*ih*iw)], np.int8).reshape(ic,ih,iw)
    w = np.array([rng.randrange(-127, 128) for _ in range(ic*9)], np.int8).reshape(ic,9)
    oh, ow = 2, 3
    zx, zy, m, s = 127, -11, 1 << 30, 36
    biases = [rng.randrange(-50,51) for _ in range(ic)]
    expected = []
    for c in range(ic):
        for y in range(oh):
            for z in range(ow):
                acc = biases[c]
                for ky in range(3):
                    for kx in range(3):
                        yy, xx = y*2+ky, z*2+kx
                        q = int(x[c,yy,xx]) if yy < ih and xx < iw else zx
                        acc += (q-zx)*int(w[c,ky*3+kx])
                expected.append(rounded(acc,m,s,zy))
    desc = Descriptor(6,input=512,output=4096,weight=8192,params=16384,
                      count=9,outputs=len(expected),row_stride=16,next_pc=64,
                      kernel_h=3,kernel_w=3,stride_h=2,stride_w=2,
                      pad_bottom=1,pad_right=1,input_h=ih,input_w=iw,
                      input_c=ic,output_c=ic)
    params = [param(biases[c]-zx*int(w[c].astype(np.int64).sum()),m,s,zy,zx) for c in range(ic)]
    yield (desc,x.ravel(),w,params,expected,len(expected)*9,'vww_depthwise_s2')

    # The final VWW depthwise stage has 256 channels. Its scalar tile must
    # fit the on-chip image despite the complete model needing external tiles.
    ic=256
    x=np.array([rng.randrange(-128,128) for _ in range(ic*9)],np.int8).reshape(ic,3,3)
    w=np.array([rng.randrange(-127,128) for _ in range(ic*9)],np.int8).reshape(ic,9)
    zx,zy,m,s=-128,4,1<<30,38
    expected=[rounded(sum((int(x[c].ravel()[k])-zx)*int(w[c,k]) for k in range(9)),m,s,zy)
              for c in range(ic)]
    desc=Descriptor(6,input=512,output=4096,weight=8192,params=16384,
                    count=9,outputs=ic,row_stride=16,next_pc=64,
                    kernel_h=3,kernel_w=3,input_h=3,input_w=3,input_c=ic,output_c=ic)
    params=[param(-zx*int(w[c].astype(np.int64).sum()),m,s,zy,zx) for c in range(ic)]
    yield (desc,x.ravel(),w,params,expected,ic*9,'vww_depthwise_256_channels')

    # KWS and VWW full-window averages test non-power-of-two divisors.
    for channels, height, width in ((2,25,5),(3,3,3)):
        x = np.array([rng.randrange(-128,128) for _ in range(channels*height*width)], np.int8)
        zx,zy=-9,5
        m,s=multiplier_shift(1/(height*width))
        expected=[rounded(sum(int(v)-zx for v in x[c*height*width:(c+1)*height*width]),m,s,zy)
                  for c in range(channels)]
        desc=Descriptor(7,input=512,output=4096,params=16384,
                        count=len(x),outputs=channels,next_pc=64,
                        kernel_h=height,kernel_w=width,stride_h=height,stride_w=width,
                        input_h=height,input_w=width,input_c=channels,output_c=channels)
        yield (desc,x,None,[param(0,m,s,zy,zx)],[*expected],0,
                      f'average_{height}x{width}')

    # MaxPool uses a nonzero input zero point and requantizes the maximum.
    x=np.array([rng.randrange(-128,128) for _ in range(2*4*4)],np.int8).reshape(2,4,4)
    x[0,0,0]=-128;x[1,3,3]=127
    zx,zy=11,-7
    expected=[]
    for c in range(2):
        for y in (0,2):
            for z in (0,2):
                maximum=max(int(v) for v in x[c,y:y+2,z:z+2].ravel())
                expected.append(rounded(maximum-zx,1<<30,30,zy))
    desc=Descriptor(5,input=512,output=4096,params=16384,
                    count=x.size,outputs=len(expected),next_pc=64,
                    kernel_h=2,kernel_w=2,stride_h=2,stride_w=2,
                    input_h=4,input_w=4,input_c=2,output_c=2)
    yield (desc,x.ravel(),None,[param(0,1<<30,30,zy,zx)],expected,0,
           'maxpool_nonzero_zero_point')

    # Clip is an input-domain clamp followed by one normal requantization.
    x=np.array([-128,-40,-17,-16,-11,0,22,23,24,127],np.int8)
    zx,zy,lo,hi=-11,7,-17,23
    desc=Descriptor(8,input=512,output=4096,params=16384,
                    count=len(x),outputs=len(x),next_pc=64)
    expected=[max(lo,min(hi,int(v)))-zx+zy for v in x]
    yield (desc,x,None,[param(0,1<<30,30,zy,zx,lo,hi)],expected,0,'clip')
