"""Experimental v2 descriptor extension for in-place channel-plane packing.

The production descriptor remains byte-identical. Reserved byte 6 bit 0 opts
an ordinary unpadded 1x1 Conv into an align8(H*W) physical input stride.
Opcode 9 is a separate in-place PACK operation that expands a dense CHW tensor
one channel plane at a time, backward, with zero-filled padding bytes.
"""
import struct

from hardware_v2 import Descriptor, MAGIC, TARGET

PACK_OPCODE=9
PADDED_INPUT_FLAG=1
CAPACITY=TARGET['memory_bytes']


def align8(value):
    return (value+7)&~7


def physical_input_bytes(descriptor):
    return descriptor.input_c*align8(descriptor.input_h*descriptor.input_w)


def validate_padded_pointwise(descriptor):
    descriptor.validate()
    if (descriptor.opcode != 4 or descriptor.kernel_h != 1 or descriptor.kernel_w != 1
            or descriptor.stride_h != 1 or descriptor.stride_w != 1
            or any((descriptor.pad_top,descriptor.pad_bottom,
                    descriptor.pad_left,descriptor.pad_right))
            or descriptor.count != descriptor.input_c):
        raise ValueError('padded input flag is restricted to unpadded 1x1 Conv')
    base, end=descriptor.input,descriptor.input+physical_input_bytes(descriptor)
    if end>CAPACITY:
        raise ValueError('padded pointwise input exceeds SRAM')
    spans=[(descriptor.output,descriptor.output+descriptor.outputs),
           (descriptor.weight,descriptor.weight+descriptor.output_c*descriptor.row_stride),
           (descriptor.params,descriptor.params+descriptor.output_c*16)]
    if any(base<high and low<end for low,high in spans):
        raise ValueError('padded input overlaps live output/weights/parameters')


def encode_pointwise(descriptor):
    validate_padded_pointwise(descriptor)
    raw=bytearray(descriptor.encode())
    raw[6]=PADDED_INPUT_FLAG
    return bytes(raw)


def encode_pack(base,height,width,channels):
    logical=height*width
    physical=align8(logical)
    if (base%8 or min(height,width,channels)<=0 or max(height,width)>255
            or channels>1024 or logical==physical or base+channels*physical>CAPACITY):
        raise ValueError('invalid in-place PACK geometry')
    return struct.pack('<IBBBB8I12H',MAGIC,2,PACK_OPCODE,0,0,
        base,base,0,0,logical,physical,0,64,
        1,1,1,1,0,0,0,0,height,width,channels,channels)


def decode_experimental(raw):
    if len(raw)!=64:raise ValueError('descriptor length')
    magic,version,opcode,flags,reserved=struct.unpack_from('<IBBBB',raw)
    if (magic,version,reserved)!=(MAGIC,2,0):
        raise ValueError('descriptor magic/version/reserved')
    if opcode==PACK_OPCODE:
        if flags:raise ValueError('PACK cannot carry Conv layout flags')
        fields=struct.unpack('<IBBBB8I12H',raw)
        xb,yb,wb,pb,count,outputs,stride,next_pc=fields[5:13]
        kh,kw,sh,sw,pt,pbm,pl,pr,ih,iw,ic,oc=fields[13:]
        if (xb%8 or xb!=yb or wb or pb or stride or next_pc!=64
                or (kh,kw,sh,sw,pt,pbm,pl,pr)!=(1,1,1,1,0,0,0,0)
                or min(ih,iw,ic)<=0 or max(ih,iw)>255 or ic>1024 or oc!=ic
                or count!=ih*iw or outputs!=align8(count) or count==outputs
                or xb+ic*outputs>CAPACITY):
            raise ValueError('invalid PACK descriptor')
        return dict(opcode=opcode,input=xb,output=yb,
            logical_plane=count,physical_plane=outputs,channels=ic,next_pc=next_pc),False
    if flags not in (0,PADDED_INPUT_FLAG):
        raise ValueError('unsupported descriptor flags')
    legacy=bytearray(raw);legacy[6]=0
    descriptor=Descriptor.decode(bytes(legacy))
    if flags:validate_padded_pointwise(descriptor)
    else:descriptor.validate()
    return descriptor,bool(flags)
