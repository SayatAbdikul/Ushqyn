"""Independent replay of retained-tensor command bytes on oracle tensors.

Does not import the tiler/sequence generator or trust their schedule manifest.
Untrusted run contracts declare graph layer/output coordinate only; every
claim is checked against descriptors, arithmetic, operand bytes and provenance.
Accepts exact in-place Relu/Clip, not arbitrary overlapping live operands.
Engine results are supplied by the independent integer oracle: this verifies
data movement/semantics, not RTL arithmetic, timing or physical port arbitration.
"""
import math
import struct

import numpy as np

from hardware_v2 import Descriptor
from integer_reference import coefficients, evaluate
from .contract import require


def replay_resident(program, commands, payload, inputs, *, run_contracts, final_output,
                    snapshot_regions=None, oracle=None):
    require(len(commands) % 16 == 0 and 0 < len(commands) <= 32768 and len(payload) <= 8*1024*1024,
            'command/payload capacity')
    require(len(program.inputs) == len(program.outputs) == 1 and not program.constants, 'unsupported graph boundary')
    require(set(inputs) == set(program.inputs) and all(v.dtype == np.int8 and v.shape == program.tensors[n].shape
                                                      for n, v in inputs.items()), 'invalid input tensors')
    oracle = evaluate(program, inputs) if oracle is None else oracle
    slot_bytes = (max(math.prod(t.shape) for t in program.tensors.values())+7)//8*8
    ext = bytearray(payload)
    ext.extend(bytes(max(0, 2*slot_bytes-len(ext))))
    versions = np.full(len(ext), -2, np.int32)  # immutable payload
    positions = np.full(len(ext), -1, np.int32)
    versions[:2*slot_bytes] = -1               # no activation has been produced
    source = inputs[program.inputs[0]]
    if program.layers[0].op == 'Transpose':
        require(program.layers[0].attributes.get('perm') == [0, 3, 1, 2], 'unsupported host layout')
        source = oracle[program.layers[0].output]
    ext[:source.nbytes] = source.tobytes(); versions[:source.nbytes] = 0
    positions[:source.nbytes] = np.arange(source.nbytes)
    sram = bytearray(32768); ready = np.full(32768, -1, np.int32)
    origins = np.full(32768, -1, np.int32)
    tensor_versions={program.inputs[0]:0}
    coverage={}
    aliases=('Identity','Reshape','Flatten','Transpose')
    for i,l in enumerate(program.layers):
        require(len(l.inputs)==1 and l.inputs[0] in tensor_versions,'unsupported graph order')
        if l.op in aliases:
            require(l.op!='Transpose' or i==0,'unsupported layout')
            tensor_versions[l.output]=tensor_versions[l.inputs[0]]
        else:
            tensor_versions[l.output]=i+1
            coverage[i]=np.zeros(oracle[l.output].size,dtype=bool)
    snapshot_regions={} if snapshot_regions is None else {int(k):v for k,v in snapshot_regions.items()}
    for i,r in snapshot_regions.items():
        require(i in coverage and r['ext']>=2*slot_bytes and r['bytes']==coverage[i].size and
                r['ext']+r['bytes']<=len(ext),'invalid snapshot extent')
    require(all(max(a['ext'],b['ext'])>=min(a['ext']+a['bytes'],b['ext']+b['bytes'])
                for i,a in snapshot_regions.items() for j,b in snapshot_regions.items() if i<j),'overlapping snapshots')
    seen_contracts=set()
    pending_engine, pending_dma = None, None
    runs, dma_bytes = 0, {'to_sram': 0, 'from_sram': 0}

    def read(base, length, expected, version=None):
        require(0 <= base and base+length <= 32768 and len(expected) == length, 'invalid operand extent')
        require(np.all(ready[base:base+length] != -1) and sram[base:base+length] == expected,
                'unready or incorrect operand bytes')
        if version is not None:
            require(np.all(ready[base:base+length] == version), 'stale tensor version')

    def finish_dma():
        nonlocal pending_dma
        if pending_dma is not None:
            direction, a, b, c = pending_dma
            if direction:
                sram[b:b+c] = ext[a:a+c]; ready[b:b+c] = versions[a:a+c]
                origins[b:b+c] = positions[a:a+c]
            else:
                require(a+c <= 2*slot_bytes or any(r['ext']<=a and a+c<=r['ext']+r['bytes']
                        for r in snapshot_regions.values()), 'store overwrites immutable payload')
                require(np.all(ready[b:b+c] >= 0), 'store reads unproduced activation')
                ext[a:a+c] = sram[b:b+c]; versions[a:a+c] = ready[b:b+c]
                positions[a:a+c] = origins[b:b+c]
            pending_dma = None

    def finish_engine():
        nonlocal pending_engine
        if pending_engine is not None:
            low, high, address, output, version, first_element = pending_engine
            sram[address:address+len(output)] = output
            ready[address:address+len(output)] = version
            origins[address:address+len(output)] = np.arange(first_element, first_element+len(output))
            pending_engine = None

    for command_index in range(len(commands)//16):
        op, flags, reserved, a, b, c = struct.unpack_from('<BBHIII', commands, command_index*16)
        require(reserved == 0, 'reserved command bits')
        if op == 0:
            require(not any((flags, a, b, c)) and command_index == len(commands)//16-1 and
                    pending_engine is None and pending_dma is None, 'invalid/premature HALT')
            require(set(run_contracts)==seen_contracts and all(np.all(v) for v in coverage.values()),'missing compute/output or unused contracts')
            for name,version,address,size in [(program.outputs[0],tensor_versions[program.outputs[0]],final_output['ext'],final_output['bytes'])]+[
                    (program.layers[i].output,i+1,r['ext'],r['bytes']) for i,r in snapshot_regions.items()]:
                expected=oracle[name].tobytes()
                require(0<=address and address+size<=len(ext) and size==len(expected) and
                        ext[address:address+size]==expected and np.all(versions[address:address+size]==version) and
                        np.array_equal(positions[address:address+size],np.arange(size)),'missing or stale final/snapshot tensor')
            return {'status': 'passed', 'engine_runs': runs, 'dma_bytes': dma_bytes,
                    'final_output_bytes': oracle[program.outputs[0]].size,
                    'scope': 'symbolic command replay with independent integer oracle; no RTL timing proof'}
        if op == 3:
            require(flags in (1, 2, 3) and not any((a, b, c)), 'invalid wait')
            if flags & 1:
                finish_engine()
            if flags & 2:
                finish_dma()
            continue
        if op == 1:
            require(flags in (0, 1) and pending_dma is None and a % 8 == b % 8 == 0 and
                    0 < c <= 32768 and a+c <= len(ext) and b+c <= 32768, 'illegal DMA')
            if pending_engine:
                require(b+c <= pending_engine[0] or b >= pending_engine[1], 'DMA intersects live engine region')
            pending_dma = flags, a, b, c
            dma_bytes['to_sram' if flags else 'from_sram'] += c
            continue
        require(op == 2 and flags == c == 0 and pending_engine is None, 'invalid engine dispatch')
        key=str(command_index);contract=run_contracts.get(key)
        require(isinstance(contract,dict) and set(contract)=={'layer','first_element'},'missing/invalid run contract')
        layer_index,consumed=contract['layer'],contract['first_element']
        require(type(layer_index) is int and layer_index in coverage and type(consumed) is int and consumed>=0,'invalid logical operation')
        seen_contracts.add(key)
        low, high = b & 65535, b >> 16
        require(0 <= low < high <= 32768 and a % 64 == 0, 'invalid live region/PC')
        if pending_dma:
            _, _, base, length = pending_dma
            require(base+length <= low or base >= high, 'engine intersects pending DMA')
        require(low <= a and a+128 <= high and np.all(ready[a:a+128] == -2), 'unready descriptor')
        d = Descriptor.decode(bytes(sram[a:a+64])); d.validate()
        require(d.next_pc == a+64 and Descriptor.decode(bytes(sram[a+64:a+128])).opcode == 0, 'invalid descriptor chain')
        layer = program.layers[layer_index]; attrs, p = layer.attributes, layer.parameters
        input_version=tensor_versions[layer.inputs[0]]
        x, y = oracle[layer.inputs[0]], oracle[layer.output]
        iq = program.tensors[layer.inputs[0]].quantization; oq = program.tensors[layer.output].quantization
        depthwise = layer.op == 'Conv' and attrs.get('group', 1) != 1
        opcode = {'Conv': 6 if depthwise else 4, 'Gemm': 1, 'Relu': 2, 'Clip': 8,
                  'MaxPool': 5, 'AveragePool': 7, 'GlobalAveragePool': 7}.get(layer.op)
        require(d.opcode == opcode and consumed+d.outputs <= y.size, 'wrong operation/output coverage')
        require(not np.any(coverage[layer_index][consumed:consumed+d.outputs]),'duplicate output coverage')
        coverage[layer_index][consumed:consumed+d.outputs]=True
        spatial = opcode in (4, 5, 6, 7)
        first, count = consumed, d.outputs
        if spatial:
            plane = math.prod(y.shape[2:])
            require(consumed % plane == d.outputs % plane == 0, 'split spatial plane')
            first, count = consumed//plane, d.outputs//plane
            kh, kw = p['weight'].shape[2:] if layer.op == 'Conv' else attrs.get('kernel_shape', x.shape[2:])
            sh, sw = attrs.get('strides', [1, 1]); pt, pl, pb, pr = attrs.get('pads', [0]*4)
            require(attrs.get('dilations', [1, 1]) == [1, 1] and not attrs.get('ceil_mode', 0) and
                    (not depthwise or attrs['group'] == x.shape[1] == y.shape[1]), 'unsupported geometry')
            expected_geometry = (kh, kw, sh, sw, pt, pb, pl, pr, *x.shape[2:],
                                 x.shape[1] if opcode == 4 else count, count)
            actual_geometry = (d.kernel_h, d.kernel_w, d.stride_h, d.stride_w, d.pad_top, d.pad_bottom,
                               d.pad_left, d.pad_right, d.input_h, d.input_w, d.input_c, d.output_c)
            require(actual_geometry == expected_geometry, 'descriptor geometry differs from graph')
            operand = x if opcode == 4 else x[:, first:first+count]
        else:
            operand = x if opcode == 1 else x.reshape(-1)[consumed:consumed+d.outputs]
        read(d.input, operand.size, operand.tobytes(), input_version)
        input_first = first*math.prod(x.shape[2:]) if spatial and opcode != 4 else consumed if opcode in (2, 8) else 0
        require(np.array_equal(origins[d.input:d.input+operand.size], np.arange(input_first, input_first+operand.size)),
                'incorrect tensor coordinate provenance')
        regions = [(a, a+128), (d.input, d.input+operand.size), (d.output, d.output+d.outputs)]
        if opcode in (1, 4, 6):
            weight = p['weight'][first:first+count].reshape(count, -1)
            require(d.count == weight.shape[1] and d.row_stride >= weight.shape[1], 'wrong reduction')
            rows = np.zeros((count, d.row_stride), np.int8); rows[:, :weight.shape[1]] = weight
            read(d.weight, rows.size, rows.tobytes(), -2)
            params = b''.join(struct.pack('<iiBbbbbb', int(p['corrected_bias'][ch]), int(p['multiplier'][ch]),
                                         int(p['shift'][ch]), oq.zero_point, iq.zero_point, -128, 127, 0)+b'\0\0'
                              for ch in range(first, first+count))
            regions.append((d.weight, d.weight+rows.size))
        else:
            require(d.count == operand.size, 'wrong input count')
            divisor = d.kernel_h*d.kernel_w if opcode == 7 else 1
            require(opcode != 7 or not any((d.pad_top, d.pad_bottom, d.pad_left, d.pad_right)), 'unsupported padded average')
            m, shift = coefficients(iq.scale/(oq.scale*divisor))
            lo, hi = p['clip_bounds'] if opcode == 8 else (iq.zero_point, 127) if opcode == 2 else (-128, 127)
            params = struct.pack('<iiBbbbbb', 0, m, shift, oq.zero_point, iq.zero_point, int(lo), int(hi), 0)+b'\0\0'
        read(d.params, len(params), params, -2); regions.append((d.params, d.params+len(params)))
        require(all(low <= lo < hi <= high for lo, hi in regions), 'operand outside live region')
        require(all(max(lo,l2)>=min(hi,h2) or (i==1 and j==2 and opcode in (2,8) and lo==l2 and hi==h2)
                    for i,(lo,hi) in enumerate(regions) for j,(l2,h2) in enumerate(regions) if i<j),'overlapping live operands')
        output = y.reshape(-1)[consumed:consumed+d.outputs].tobytes()
        pending_engine = low, high, d.output, output, layer_index+1, consumed
        ready[d.output:d.output+d.outputs] = -1
        runs += 1
    require(False, 'missing HALT')
