"""Software semantics for spatial fusion/recomputation, NOT hardware lowering.

Demand-driven NCHW Conv/Relu/Clip segments preserve every INT8 quantization
boundary. The optional one-rectangle cache is a deliberately restricted policy,
not a DeFiNES reproduction. Counts are semantic work/cache bytes, never cycles
or a proof that all physical buffers fit SRAM.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from quantization import multiplier_shift, requantize
from .contract import integer, require


@dataclass(frozen=True)
class Rect:
    y0: int
    x0: int
    y1: int
    x1: int

    def __post_init__(self):
        require(all(type(v) is int for v in asdict(self).values()) and self.y1 >= self.y0 and self.x1 >= self.x0,
                'invalid rectangle')

    @property
    def shape(self):
        return self.y1-self.y0, self.x1-self.x0

    @property
    def area(self):
        return math.prod(self.shape)

    def intersect(self, other):
        y, x = max(self.y0, other.y0), max(self.x0, other.x0)
        return Rect(y, x, max(y, min(self.y1, other.y1)), max(x, min(self.x1, other.x1)))

    def subtract(self, other):
        overlap = self.intersect(other)
        if not overlap.area:
            return [self] if self.area else []
        pieces = [Rect(self.y0, self.x0, overlap.y0, self.x1),
                  Rect(overlap.y1, self.x0, self.y1, self.x1),
                  Rect(overlap.y0, self.x0, overlap.y1, overlap.x0),
                  Rect(overlap.y0, overlap.x1, overlap.y1, self.x1)]
        return [r for r in pieces if r.area]

    def slices_in(self, outer):
        require(self.intersect(outer) == self and self.area > 0, 'rectangle outside parent')
        return (slice(None), slice(None), slice(self.y0-outer.y0, self.y1-outer.y0),
                slice(self.x0-outer.x0, self.x1-outer.x0))


def input_halo(output, kernel, strides=(1, 1), pads=(0, 0, 0, 0), dilations=(1, 1)):
    require(output.area > 0 and len(kernel) == len(strides) == len(dilations) == 2 and len(pads) == 4,
            'invalid halo geometry')
    require(all(integer(v, 1) for v in (*kernel, *strides, *dilations)) and
            all(integer(v) for v in pads), 'invalid halo geometry')
    kh, kw = kernel; sh, sw = strides; dh, dw = dilations
    return Rect(output.y0*sh-pads[0], output.x0*sw-pads[1],
                (output.y1-1)*sh-pads[0]+(kh-1)*dh+1,
                (output.x1-1)*sw-pads[1]+(kw-1)*dw+1)


@dataclass(frozen=True)
class Reduction:
    begin: int
    end: int
    add_bias: bool = False


def validate_reduction(parts, count):
    require(integer(count, 1) and bool(parts), 'empty reduction')
    cursor = 0
    for i, part in enumerate(parts):
        require(integer(part.begin) and integer(part.end, 1) and
                part.begin == cursor and part.begin < part.end <= count, 'reduction gap/overlap')
        require(type(part.add_bias) is bool and part.add_bias == (i == 0), 'bias must be added exactly once at start')
        cursor = part.end
    require(cursor == count, 'incomplete reduction')


def reduction_parts(count, chunk):
    require(integer(chunk, 1), 'invalid reduction chunk')
    parts = tuple(Reduction(i, min(i+chunk, count), i == 0) for i in range(0, count, chunk))
    validate_reduction(parts, count)
    return parts


def supported_segments(program, max_layers=4):
    """Partition supported runs; unsupported operations remain explicit barriers."""
    require(integer(max_layers, 1), 'invalid segment length')
    start = None
    for i, layer in enumerate(program.layers):
        spatial = layer.op in ('Conv', 'Relu', 'Clip') and len(program.tensors[layer.output].shape) == 4
        if start is not None and (not spatial or i-start == max_layers):
            yield start, i
            start = None
        if spatial and start is None:
            start = i
    if start is not None:
        yield start, len(program.layers)


def execute_segment(program, start, stop, input_value, *, tile=(4, 4), cache=False, reduction_chunk=32):
    require(integer(start) and integer(stop, 1) and start < stop <= len(program.layers), 'invalid segment')
    require(len(tile) == 2 and all(integer(v, 1) for v in tile) and type(cache) is bool,
            'invalid tile/cache policy')
    require(integer(reduction_chunk, 1), 'invalid reduction chunk')
    layers = program.layers[start:stop]
    source = program.tensors[layers[0].inputs[0]]
    require(input_value.dtype == np.int8 and input_value.shape == source.shape, 'invalid segment input')
    previous = source.name
    for layer in layers:
        require(layer.op in ('Conv', 'Relu', 'Clip') and layer.inputs == [previous], 'unsupported segment graph')
        inp, out = program.tensors[previous], program.tensors[layer.output]
        require(inp.layout == out.layout == 'NCHW' and len(inp.shape) == len(out.shape) == 4 and
                inp.shape[0] == out.shape[0] == 1, 'batch-one NCHW segment required')
        if layer.op == 'Conv':
            p, a = layer.parameters, layer.attributes
            w = p['weight']; groups = a.get('group', 1)
            require(w.dtype == np.int8 and w.ndim == 4 and integer(groups, 1) and
                    inp.shape[1] == groups*w.shape[1] and len(w) % groups == 0 and out.shape[1] == len(w),
                    'invalid grouped convolution')
            require(a.get('auto_pad', 'NOTSET') in ('NOTSET', b'NOTSET', ''), 'auto padding unsupported')
            sh, sw = a.get('strides', [1, 1]); dh, dw = a.get('dilations', [1, 1])
            pt, pl, pb, pr = a.get('pads', [0, 0, 0, 0])
            input_halo(Rect(0, 0, 1, 1), tuple(w.shape[2:]), (sh, sw), (pt, pl, pb, pr), (dh, dw))
            require(out.shape[2:] == ((inp.shape[2]+pt+pb-(w.shape[2]-1)*dh-1)//sh+1,
                                     (inp.shape[3]+pl+pr-(w.shape[3]-1)*dw-1)//sw+1), 'output geometry mismatch')
            flat = w.astype(np.int64).reshape(len(w), -1)
            bias = np.asarray(p['corrected_bias'], dtype=np.int64)
            require(np.array_equal(bias, p['bias'].astype(np.int64)-inp.quantization.zero_point*flat.sum(axis=1)),
                    'bias correction mismatch')
            # Conservative proof for EVERY raw partial sum and any input INT8,
            # not merely the values observed in this software run.
            require(np.all(np.abs(bias)+128*np.abs(flat).sum(axis=1) <= 2**31-1), 'INT32 partial sum bound exceeded')
        else:
            require(inp.shape == out.shape, 'elementwise shape mismatch')
        previous = layer.output
    counters = {str(start+i): {'computed_elements': 0, 'macs': 0, 'reused_elements': 0,
                             'max_int32_output_tile_bytes': 0, 'max_input_rectangle_bytes': 0}
                for i in range(len(layers))}
    caches, peak_cache = {}, 0
    source_bytes = 0

    def request(index, rect):
        nonlocal peak_cache, source_bytes
        if index < 0:
            source_bytes += source.shape[1]*rect.area
            return input_value[rect.slices_in(Rect(0, 0, *source.shape[2:]))].copy()
        layer = layers[index]
        channels = program.tensors[layer.output].shape[1]
        output = np.empty((1, channels, *rect.shape), dtype=np.int8)
        missing = [rect]
        if cache and index in caches:
            old_rect, old_data = caches[index]
            intersection = rect.intersect(old_rect)
            if intersection.area:
                output[intersection.slices_in(rect)] = old_data[intersection.slices_in(old_rect)]
                counters[str(start+index)]['reused_elements'] += channels*intersection.area
                missing = rect.subtract(old_rect)
        for piece in missing:
            output[piece.slices_in(rect)] = compute(index, piece)
        # The terminal result is consumed by the caller, not retained as a cache.
        if cache and index < len(layers)-1:
            caches[index] = (rect, output.copy())
            peak_cache = max(peak_cache, sum(data.nbytes for _, data in caches.values()))
        return output

    def compute(index, rect):
        layer = layers[index]; p, a = layer.parameters, layer.attributes
        inp, out = program.tensors[layer.inputs[0]], program.tensors[layer.output]
        iq, oq = inp.quantization, out.quantization
        counter = counters[str(start+index)]
        counter['computed_elements'] += out.shape[1]*rect.area
        if layer.op != 'Conv':
            x = request(index-1, rect)
            lo, hi = (iq.zero_point, 127) if layer.op == 'Relu' else p['clip_bounds']
            m, s = multiplier_shift(iq.scale/oq.scale)
            return requantize(np.clip(x.astype(np.int64), lo, hi)-iq.zero_point, m, s, oq.zero_point)
        w = p['weight']; oc, icg, kh, kw = w.shape
        sh, sw = a.get('strides', [1, 1]); dh, dw = a.get('dilations', [1, 1])
        needed = input_halo(rect, (kh, kw), (sh, sw), tuple(a.get('pads', [0]*4)), (dh, dw))
        clipped = needed.intersect(Rect(0, 0, *inp.shape[2:]))
        raw = np.full((1, inp.shape[1], *needed.shape), iq.zero_point, dtype=np.int8)
        if clipped.area:
            raw[clipped.slices_in(needed)] = request(index-1, clipped)
        counter['max_input_rectangle_bytes'] = max(counter['max_input_rectangle_bytes'], raw.nbytes)
        counter['max_int32_output_tile_bytes'] = max(counter['max_int32_output_tile_bytes'], 4*oc*rect.area)
        windows = np.lib.stride_tricks.sliding_window_view(raw, ((kh-1)*dh+1, (kw-1)*dw+1), axis=(2, 3))
        windows = windows[:, :, ::sh, ::sw, ::dh, ::dw]
        result = np.empty((1, oc, *rect.shape), np.int8)
        groups = a.get('group', 1); ocg = oc//groups
        count = icg*kh*kw
        parts = reduction_parts(count, reduction_chunk)
        for g in range(groups):
            patches = windows[0, g*icg:(g+1)*icg].transpose(1, 2, 0, 3, 4).reshape(rect.area, count).astype(np.int64)
            weights = w[g*ocg:(g+1)*ocg].reshape(ocg, count).astype(np.int64)
            acc = np.zeros((rect.area, ocg), np.int64)
            for part in parts:
                if part.add_bias:
                    acc += p['corrected_bias'][g*ocg:(g+1)*ocg]
                acc += patches[:, part.begin:part.end] @ weights[:, part.begin:part.end].T
                require(np.all(acc >= -2**31) and np.all(acc < 2**31), 'INT32 partial sum overflow')
            for c in range(ocg):
                channel = g*ocg+c
                result[0, channel] = requantize(acc[:, c], p['multiplier'][channel], p['shift'][channel], oq.zero_point).reshape(rect.shape)
        counter['macs'] += rect.area*oc*count
        return result

    shape = program.tensors[layers[-1].output].shape
    assembled = np.empty(shape, np.int8)
    full = Rect(0, 0, *shape[2:])
    output_tiles = 0
    for y in range(0, shape[2], tile[0]):
        for x in range(0, shape[3], tile[1]):
            rect = Rect(y, x, min(y+tile[0], shape[2]), min(x+tile[1], shape[3]))
            assembled[rect.slices_in(full)] = request(len(layers)-1, rect)
            output_tiles += 1
    return assembled, {'layers': counters, 'output_tiles': output_tiles,
                       'source_requested_bytes': source_bytes, 'peak_retained_cache_bytes': peak_cache,
                       'cache_policy': 'retain-last-region' if cache else 'recompute',
                       'tile': list(tile), 'reduction_chunk': reduction_chunk,
                       'hardware_executable': False,
                       'scope': 'software semantic counts; cache bytes exclude transient buffers, weights and descriptors'}
