"""Pinned KWS/VWW kernel geometries against an independent scalar RTL oracle."""
import random
import struct
import json
import os
from pathlib import Path

import cocotb
import numpy as np
from cocotb.triggers import Timer

from hardware_v2 import Descriptor
from quantization import multiplier_shift


def rounded(acc, multiplier, shift, zero_point):
    n = int(acc) * int(multiplier)
    q, r = divmod(abs(n), 1 << shift)
    q += 2*r >= (1 << shift)
    return min(127, max(-128, (-q if n < 0 else q) + zero_point))


@cocotb.test()
async def pinned_audio_vision_kernels_with_stalls(d):
    rng = random.Random(240923)
    memory = bytearray(32768)
    profiles=[]
    pending = None
    held = None
    d.clk.value = 0
    d.rst_n.value = 0
    d.start.value = 0
    d.abort_run.value = 0
    d.clear_counters.value = 0
    d.start_pc.value = 0
    d.mem_ready.value = 0
    d.mem_rvalid.value = 0
    d.mem_rdata.value = 0

    async def step():
        nonlocal pending, held
        d.clk.value = 0
        ready = pending is None and rng.randrange(4) != 0
        d.mem_ready.value = ready
        d.mem_rvalid.value = 0
        if pending is not None:
            delay, data = pending
            if delay == 0:
                d.mem_rvalid.value = 1
                d.mem_rdata.value = data
                pending = None
            else:
                pending = (delay - 1, data)
        await Timer(5, units='ns')
        request = (int(d.mem_addr.value), int(d.mem_wr.value),
                   int(d.mem_wdata.value), int(d.mem_wstrb.value)) if int(d.mem_req.value) else None
        if held is not None:
            assert request == held, 'memory request changed under backpressure'
        held = request if request is not None and not ready else None
        if request is not None and ready:
            address, wr, data, mask = request
            assert address % 8 == 0 and address + 8 <= len(memory)
            if wr:
                for lane in range(8):
                    if (mask >> lane) & 1:
                        memory[address + lane] = (data >> (8*lane)) & 255
            else:
                assert pending is None
                pending = (rng.randrange(1, 5), int.from_bytes(memory[address:address+8], 'little'))
        d.clk.value = 1
        await Timer(5, units='ns')

    async def execute(desc, inputs, weights, params, expected, macs, label):
        nonlocal pending, held
        memory[:] = bytes(len(memory))
        memory[:128] = desc.encode() + Descriptor(0).encode()
        memory[512:512+len(inputs)] = np.asarray(inputs, dtype=np.int8).tobytes()
        if weights is not None:
            for c, row in enumerate(weights):
                data = np.asarray(row, dtype=np.int8).tobytes()
                memory[8192+c*desc.row_stride:8192+c*desc.row_stride+len(data)] = data
        for c, data in enumerate(params):
            memory[16384+c*16:16384+(c+1)*16] = data
        pending = held = None
        d.start.value = 1
        await step()
        d.start.value = 0
        for _ in range(500000):
            await step()
            if not int(d.busy.value):
                break
        else:
            raise AssertionError('kernel timeout')
        assert int(d.error_code.value) == 0
        actual = np.frombuffer(memory[4096:4096+len(expected)], np.int8)
        np.testing.assert_array_equal(actual, np.array(expected, np.int8))
        assert int(d.useful_macs.value) == macs
        assert int(d.elapsed.value) == (int(d.compute_cycles.value) +
                                         int(d.wait_cycles.value) + int(d.control_cycles.value))
        profiles.append(dict(label=label,opcode=desc.opcode,inputs=len(inputs),
                             outputs=len(expected),useful_macs=int(d.useful_macs.value),
                             simulated_core_cycles=int(d.elapsed.value),
                             simulated_compute_cycles=int(d.compute_cycles.value),
                             simulated_wait_cycles=int(d.wait_cycles.value),
                             simulated_control_cycles=int(d.control_cycles.value),
                             physical_sram_read_bytes=int(d.read_bytes.value),
                             physical_sram_write_bytes=int(d.write_bytes.value)))

    def param(bias, m, s, zy, zx, low=-128, high=127):
        return struct.pack('<iiBbbbbb', bias, m, s, zy, zx, low, high, 0) + b'\0\0'

    await step()
    d.rst_n.value = 1
    await step()

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
    await execute(desc, x.ravel(), w, params, expected, len(expected)*kh*kw,'kws_conv_10x4_s2')

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
    await execute(desc,x.ravel(),w,params,expected,len(expected)*ic,'pointwise_4_to_5')

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
    await execute(desc,x.ravel(),w,params,expected,len(expected)*9,'vww_depthwise_s2')

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
    await execute(desc,x.ravel(),w,params,expected,ic*9,'vww_depthwise_256_channels')

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
        await execute(desc,x,None,[param(0,m,s,zy,zx)],[*expected],0,
                      f'average_{height}x{width}')

    # Clip is an input-domain clamp followed by one normal requantization.
    x=np.array([-128,-40,-17,-16,-11,0,22,23,24,127],np.int8)
    zx,zy,lo,hi=-11,7,-17,23
    desc=Descriptor(8,input=512,output=4096,params=16384,
                    count=len(x),outputs=len(x),next_pc=64)
    expected=[max(lo,min(hi,int(v)))-zx+zy for v in x]
    await execute(desc,x,None,[param(0,1<<30,30,zy,zx,lo,hi)],expected,0,'clip')
    root=Path(os.environ['REPO_ROOT'])
    (root/'work/phase4/kernel-profiles.json').write_text(json.dumps(dict(
        evidence_type='randomized-stall RTL simulation; not board latency',
        random_seed=240923,profiles=profiles),indent=2)+'\n')
