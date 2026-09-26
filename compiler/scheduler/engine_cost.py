"""Structural cycle model for the isolated Phase 6 cache engine.

The address walk models cache tags and channel-pair reuse without reading any
tensor values, physical counters, or fitted per-model coefficients. Arbitration
stalls must be accounted for separately by the execution scheduler.
"""

from hardware_v2 import Descriptor


def engine_cycles(d: Descriptor, cache_entries=256, parameter_cache=True, spatial_pw=False):
    """Predict one descriptor followed by HALT, including fetch/control cycles."""
    d.validate()
    if cache_entries not in (0,32,64,128,256):
        raise ValueError('unsupported cache capacity')
    if d.opcode == 0:
        return 17
    if d.opcode == 1:
        return 36 + d.outputs * (9 + 5 * ((d.count + 7) // 8))
    if d.opcode in (2, 8):
        return 41 + 7 * d.outputs
    if d.opcode == 3:
        return 36 + 5 * d.outputs
    if spatial_pw and d.opcode==4 and d.count<=256 and d.kernel_h==d.kernel_w==d.stride_h==d.stride_w==1 and not any((d.pad_top,d.pad_bottom,d.pad_left,d.pad_right)):
        plane=d.input_h*d.input_w;cycles=39;words=(d.count+7)//8
        for oc in range(d.output_c):
            for position in range(0,plane,8):
                take=min(8,plane-position)
                cycles+=(2 if parameter_cache and position else 5)+4*take
                cycles+=words # one weight-register/cache access per eight ICs
                if position==0: cycles+=words # first word fetch for this OC
                for ic in range(d.input_c):
                    address=d.input+ic*plane+position
                    cycles+=3 # synchronous input request/response, eight MACs
                    if address%8+take>8: cycles+=2 # unaligned second word
        return cycles
    oh = (d.input_h + d.pad_top + d.pad_bottom - d.kernel_h) // d.stride_h + 1
    ow = (d.input_w + d.pad_left + d.pad_right - d.kernel_w) // d.stride_w + 1
    paired = d.opcode == 4 and d.output_c > 1 and d.count <= 128
    channels = ([list(range(c, min(c + 2, d.output_c)))
                 for c in range(0, d.output_c, 2)] if paired else
                [[c] for c in range(d.output_c)])
    tags = [None] * max(32,cache_entries)
    params = [None,None]
    pointwise = cache_entries and d.opcode==4 and d.kernel_h==d.kernel_w==d.stride_h==d.stride_w==1 and not any((d.pad_top,d.pad_bottom,d.pad_left,d.pad_right))
    cycles = 39
    plane = d.input_h * d.input_w
    pool = d.opcode in (5, 7)
    for group in channels:
        weights = set()
        for oy in range(oh):
            for ox in range(ow):
                for channel in group:
                    cycles += 10 if pool else 9
                    parameter_channel=0 if pool else channel
                    if parameter_cache:
                        if params[parameter_channel%2]==parameter_channel:
                            cycles-=3 # four SRAM request/response states -> one cache state
                        params[parameter_channel%2]=parameter_channel
                    if paired and channel % 2:
                        cycles += 3 * ((d.count + 7) // 8)
                        # The second channel's first window also fills weights.
                        if (channel, 0) not in weights:
                            cycles += (d.count + 7) // 8
                            weights.update((channel, c) for c in range(0, d.count, 8))
                        continue
                    reduction_channels = d.input_c if d.opcode == 4 else 1
                    consumed = 0
                    for ic in range(reduction_channels):
                        source_channel = ic if d.opcode == 4 else channel
                        for ky in range(d.kernel_h):
                            y = oy * d.stride_h - d.pad_top + ky
                            kx = 0
                            while kx < d.kernel_w:
                                x = ox * d.stride_w - d.pad_left + kx
                                inside = 0 <= y < d.input_h and 0 <= x < d.input_w
                                take = 1
                                cycles += 1  # window/pool preparation
                                if inside:
                                    address = d.input + source_channel * plane + y * d.input_w + x
                                    index = ic % cache_entries if pointwise else (y & 3) * 8 + ((address >> 3) & 7)
                                    tag = address >> 3
                                    if tags[index] != tag:
                                        cycles += 2  # synchronous request/response
                                        tags[index] = tag
                                    take = min(d.kernel_w - kx, 8 - address % 8,
                                               d.input_w - x,
                                               4 if pool else 8 - consumed % 8)
                                consumed += take
                                kx += take
                                if not pool and (consumed % 8 == 0 or consumed == d.count):
                                    block = (consumed - 1) // 8 * 8
                                    key = (channel, block)
                                    cycles += 2  # weight request/cache hit + MAC
                                    if d.count > 128 or key not in weights:
                                        cycles += 1  # SRAM response
                                        weights.add(key)
    return cycles
