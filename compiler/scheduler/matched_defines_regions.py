"""Exact rectangular packing with the frozen aligned-DMA/COPY instruction set.

Tensor bases and temporary bases are 8-byte aligned. Every SRAM tensor has an
8-byte guard immediately before its base and storage through align8(data end).
The caller must allocate these guarded regions and temporary storage disjointly,
and keep descriptor storage disjoint too. Emitters serialize each operation:
emit_dma(direction, external, sram, size, role) and
emit_copy(aligned_source, arbitrary_destination, size, role).

External writes use read/modify/write to preserve every neighboring byte.
No host-side packing, implicit padding bytes, or relaxed quantization is used.
"""
from math import prod

SRAM_BYTES = 32768
EXTERNAL_BYTES = 8 * 1024 * 1024


def align8(value):
    return (value + 7) & ~7


def rectangle_temp_bytes(width):
    if type(width) is not int or width <= 0:
        raise ValueError('positive rectangle width required')
    # A write's shifted COPY may touch up to seven bytes before row staging.
    return 16 + 8 + align8(width + 7)


def _shape(shape):
    if len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError('expected positive C,H,W shape')
    return tuple(shape)


def _rect(shape, rectangle):
    if len(rectangle) != 4 or any(type(n) is not int for n in rectangle):
        raise ValueError('expected integer y0,x0,y1,x1 rectangle')
    y0, x0, y1, x1 = rectangle
    if not (0 <= y0 < y1 <= shape[1] and 0 <= x0 < x1 <= shape[2]):
        raise ValueError('rectangle outside tensor')
    return y0, x0, y1, x1


def _tensor_region(base, shape):
    if type(base) is not int or base < 8 or base % 8:
        raise ValueError('SRAM tensor requires aligned base and preceding 8-byte guard')
    end = align8(base + prod(shape))
    if end > SRAM_BYTES:
        raise ValueError('SRAM tensor exceeds capacity')
    return base - 8, end


def _temporary_region(base, size):
    if type(base) is not int or base < 0 or base % 8 or base + size > SRAM_BYTES:
        raise ValueError('temporary region is unaligned or out of bounds')
    return base, base + size


def _disjoint(*regions):
    for i, left in enumerate(regions):
        for right in regions[i+1:]:
            if max(left[0], right[0]) < min(left[1], right[1]):
                raise ValueError('guarded SRAM regions overlap')


def _external(base, shape):
    if type(base) is not int or base < 0 or base % 8 or align8(base+prod(shape)) > EXTERNAL_BYTES:
        raise ValueError('external tensor is unaligned or out of bounds')


def _copy_row(source, destination, size, temp16, emit_copy, role):
    """Copy an arbitrary source byte range, restoring the destination prefix.

    COPY can only read an aligned source. Include the source prefix, then undo
    exactly the destination bytes overwritten by that prefix. Saving from an
    aligned address needs at most 14 bytes; temp16 therefore always suffices.
    The caller has checked the source/destination/temporary guarded regions.
    """
    prefix = source % 8
    if not prefix:
        emit_copy(source, destination, size, role)
        return
    changed_begin = destination - prefix
    saved_begin = changed_begin & ~7
    saved_bytes = destination - saved_begin
    if not 1 <= saved_bytes <= 14:
        raise AssertionError('invalid prefix preservation geometry')
    emit_copy(saved_begin, temp16, saved_bytes, role + '-save-prefix')
    emit_copy(source-prefix, changed_begin, prefix+size, role)
    emit_copy(temp16, saved_begin, saved_bytes, role + '-restore-prefix')


def _external_pieces(ext_base, shape, rect, packed_base):
    """Prefer a whole tensor or aligned contiguous full-width channel strip."""
    channels,height,width = shape
    y0,x0,y1,x1 = rect
    rh,rw = y1-y0,x1-x0
    if rect == (0,0,height,width):
        yield ext_base,packed_base,prod(shape),True
        return
    for channel in range(channels):
        external = ext_base+channel*height*width+y0*width+x0
        sram = packed_base+channel*rh*rw
        if x0 == 0 and x1 == width and external % 8 == sram % 8 == 0:
            yield external,sram,rh*rw,True
        else:
            for row in range(rh):
                e,s = external+row*width,sram+row*rw
                yield e,s,rw,e % 8 == s % 8 == 0


def emit_read_rectangle(ext_base, full_shape, rect, packed_sram_base, temp_base,
                        emit_dma, emit_copy):
    """Read Cxrectangle from planar external tensor into packed SRAM C,H,W."""
    shape = _shape(full_shape)
    y0, x0, y1, x1 = _rect(shape, rect)
    channels, height, width = shape
    rh, rw = y1-y0, x1-x0
    packed = (channels, rh, rw)
    temporary_bytes = rectangle_temp_bytes(rw)
    _external(ext_base, shape)
    _disjoint(_tensor_region(packed_sram_base, packed),
              _temporary_region(temp_base, temporary_bytes))
    stage = temp_base + 24
    for source,destination,size,direct in _external_pieces(ext_base,shape,
            (y0,x0,y1,x1),packed_sram_base):
        if direct:
            emit_dma('to_sram',source,destination,size,'rectangle-read-direct')
            continue
        prefix = source % 8
        emit_dma('to_sram', source-prefix, stage, prefix+size, 'rectangle-read')
        _copy_row(stage+prefix, destination, size, temp_base, emit_copy,
                  'rectangle-read-pack')
    return dict(rows=channels*rh, logical_bytes=channels*rh*rw,
                temporary_bytes=temporary_bytes, packed_shape=packed)


def emit_write_rectangle(ext_base, full_shape, rect, packed_sram_base, temp_base,
                         emit_dma, emit_copy):
    """Write packed SRAM Cxrectangle, preserving external bytes outside it."""
    shape = _shape(full_shape)
    y0, x0, y1, x1 = _rect(shape, rect)
    channels, height, width = shape
    rh, rw = y1-y0, x1-x0
    packed = (channels, rh, rw)
    temporary_bytes = rectangle_temp_bytes(rw)
    _external(ext_base, shape)
    _disjoint(_tensor_region(packed_sram_base, packed),
              _temporary_region(temp_base, temporary_bytes))
    stage = temp_base + 24
    for destination,source,size,direct in _external_pieces(ext_base,shape,
            (y0,x0,y1,x1),packed_sram_base):
        if direct:
            emit_dma('from_sram',destination,source,size,'rectangle-write-direct')
            continue
        prefix = destination % 8
        envelope = align8(prefix+size)
        emit_dma('to_sram', destination-prefix, stage, envelope,
                 'rectangle-write-read-neighbors')
        _copy_row(source, stage+prefix, size, temp_base, emit_copy,
                  'rectangle-write-pack')
        emit_dma('from_sram', destination-prefix, stage, envelope,
                 'rectangle-write-preserve-neighbors')
    return dict(rows=channels*rh, logical_bytes=channels*rh*rw,
                temporary_bytes=temporary_bytes, packed_shape=packed)


def emit_copy_rectangle(src_base, src_shape, src_rect, dst_base, dst_shape,
                        dst_origin, temp16, emit_copy):
    """Copy a planar SRAM rectangle into disjoint SRAM tensor storage."""
    source_shape, destination_shape = _shape(src_shape), _shape(dst_shape)
    y0, x0, y1, x1 = _rect(source_shape, src_rect)
    if (len(dst_origin) != 2 or any(type(n) is not int for n in dst_origin)
            or source_shape[0] != destination_shape[0]):
        raise ValueError('destination channels/origin mismatch')
    dy, dx = dst_origin
    rh, rw = y1-y0, x1-x0
    _rect(destination_shape, (dy, dx, dy+rh, dx+rw))
    _disjoint(_tensor_region(src_base, source_shape),
              _tensor_region(dst_base, destination_shape),
              _temporary_region(temp16, 16))
    channels, height, width = source_shape
    _, destination_height, destination_width = destination_shape
    if ((y0,x0,y1,x1) == (0,0,height,width) and (dy,dx) == (0,0)
            and source_shape == destination_shape):
        emit_copy(src_base,dst_base,prod(source_shape),'rectangle-sram-copy-direct')
        return dict(rows=channels*rh,logical_bytes=channels*rh*rw,temporary_bytes=16)
    for channel in range(channels):
        if x0 == dx == 0 and rw == width == destination_width:
            _copy_row(src_base+channel*height*width+y0*width,
                      dst_base+channel*destination_height*destination_width+dy*destination_width,
                      rh*rw,temp16,emit_copy,'rectangle-sram-copy-strip')
            continue
        for row in range(rh):
            source = src_base + channel*height*width + (y0+row)*width + x0
            destination = dst_base + channel*destination_height*destination_width + \
                          (dy+row)*destination_width + dx
            _copy_row(source, destination, rw, temp16, emit_copy,
                      'rectangle-sram-copy')
    return dict(rows=channels*rh, logical_bytes=channels*rh*rw, temporary_bytes=16)
