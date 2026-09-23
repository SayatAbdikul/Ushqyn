"""Hardware ABI failure modes and software/hardware image agreement."""
import dataclasses,hashlib,struct,sys
from pathlib import Path
import numpy as np
import pytest
from hardware_v2 import Descriptor,lower,TARGET
from test_static_pipeline import model,compile_case
from onnx import helper as h
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/phase2'))
from host import frame,parse_response


def test_descriptor_roundtrip_full_width_geometry():
    d=Descriptor(1,input=0xfffff0,output=0x10000,weight=0x123456,params=0x987654,count=784,outputs=128,row_stride=800,next_pc=0x10000,kernel_h=10,kernel_w=4,stride_h=2,stride_w=3,pad_top=1,pad_bottom=2,pad_left=3,pad_right=4,input_h=49,input_w=10,input_c=64,output_c=128)
    assert Descriptor.decode(d.encode())==d
    with pytest.raises(ValueError):d.validate()
    for key in ('input','output','weight','params','next_pc'):
        with pytest.raises(ValueError):dataclasses.replace(d,**{key:1<<24}).encode()
    with pytest.raises(ValueError):dataclasses.replace(d,count=1<<32).encode()
    with pytest.raises(ValueError):dataclasses.replace(d,stride_h=1<<16).encode()
    with pytest.raises(ValueError):dataclasses.replace(d,input=-1).encode()
    with pytest.raises((ValueError,TypeError)):dataclasses.replace(d,count=1.5).encode()
    for offset in (0,4,6,7):
        b=bytearray(d.encode());b[offset]^=1
        with pytest.raises(ValueError):Descriptor.decode(b)


def test_descriptor_legality_and_image_ownership():
    m=model([h.make_node('Gemm',['x','w','b'],['g'],transB=1),h.make_node('Relu',['g'],['y'])],{'w':np.ones((3,65))*.1,'b':[-1,2,3]},[1,65],{'y':[1,3]})
    p=compile_case(m,np.ones((1,65),np.float32));blob,meta=lower(p)
    assert len(blob)==TARGET['memory_bytes'] and meta['used_bytes']<2048
    segments=meta['segments'];end=(len(p.layers)+1)*64
    for s in segments:assert s['offset']>=end and s['offset']%8==0;end=s['offset']+s['size']
    d=Descriptor.decode(blob[:64]);d.validate();assert d.count==65 and d.row_stride==72
    params=struct.unpack('<iiBbbbbb2x',blob[d.params:d.params+16])
    assert params[0]==int(p.layers[0].parameters['corrected_bias'][0])
    assert params[1]==int(p.layers[0].parameters['multiplier'][0])
    for change in ({'input':513},{'count':0},{'row_stride':64},{'output':32768},{'next_pc':1},{'kernel_w':3}):
        with pytest.raises(ValueError):dataclasses.replace(d,**change).validate()


def test_hardware_rejects_unsupported_and_oversized():
    conv=model([h.make_node('Conv',['x','w'],['y'])],{'w':np.ones((1,1,1,1))},[1,1,2,2],{'y':[1,1,2,2]})
    with pytest.raises(ValueError,match='unsupported hardware'):lower(compile_case(conv,np.ones((1,1,2,2),np.float32)))
    large=model([h.make_node('Gemm',['x','w'],['y'],transB=1)],{'w':np.ones((64,784))*.01},[1,784],{'y':[1,64]})
    with pytest.raises(ValueError,match='exceeds target SRAM'):lower(compile_case(large,np.ones((1,784),np.float32)))


def test_protocol_crc_and_widths():
    import binascii
    f=frame(3,255,0xffffff,b'\0\xa5\x5a\xff');assert f[:2]==b'\xa5\x5a' and f[5:8]==b'\xff'*3
    assert int.from_bytes(f[-2:],'little')==binascii.crc_hqx(f[2:-2],0xffff)
    for addr in (-1,1<<24):
        with pytest.raises(ValueError):frame(1,address=addr)
    body=bytes([2,129,0,0,0,0,1,0,0]);response=b'\xa5\x5a'+body+struct.pack('<H',binascii.crc_hqx(body,0xffff))
    assert parse_response(response)['status']==0
    for bad in (response[:-1],response[:-1]+bytes([response[-1]^1])):
        with pytest.raises(ValueError):parse_response(bad)
