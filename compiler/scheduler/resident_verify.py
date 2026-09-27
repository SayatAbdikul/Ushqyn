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
from integer_reference import coefficients, evaluate, rounded
from .contract import require


def replay_resident(program, commands, payload, inputs, *, run_contracts, final_output,
                    snapshot_regions=None, oracle=None, constant_contracts=None,
                    pack_contracts=None):
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
    constant_contracts={} if constant_contracts is None else constant_contracts
    seen_constant_contracts=set()
    pack_contracts={} if pack_contracts is None else pack_contracts
    seen_pack_contracts=set()
    pack_runs=0
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
            direction, a, b, c, constant = pending_dma
            if direction:
                sram[b:b+c] = ext[a:a+c]; ready[b:b+c] = versions[a:a+c]
                origins[b:b+c] = positions[a:a+c]
                if constant is not None:
                    layer_index=constant['layer']; layer=program.layers[layer_index]
                    first=constant['first_element']; size=constant['bytes']
                    tile_first=constant['tile_first_element']; tile_sram=constant['tile_output_sram']
                    plane=math.prod(program.tensors[layer.output].shape[2:])
                    weights=layer.parameters['weight']
                    params=layer.parameters
                    zero=np.all(weights == 0,axis=tuple(range(1,weights.ndim)))
                    output_zp=program.tensors[layer.output].quantization.zero_point
                    offset=first-tile_first-constant['dma_first']
                    require(0<=offset and offset+size<=c and
                            b+offset==tile_sram+first-tile_first and
                            first%plane==size%plane==0 and
                            constant['channel_first']==first//plane and
                            constant['channels']==size//plane and
                            np.all(zero[first//plane:(first+size)//plane]),
                            'invalid constant-filter coordinate')
                    require(not np.any(coverage[layer_index][first:first+size]),
                            'duplicate constant-filter coverage')
                    coverage[layer_index][first:first+size]=True
                    # Aligned DMA may also touch a neighboring zero channel.
                    # Restore its provenance as well, while a touched live
                    # channel remains unproduced until its Conv descriptor.
                    for index in range(c):
                        position=tile_first+constant['dma_first']+index
                        channel=position//plane
                        if zero[channel]:
                            value=rounded(np.asarray([int(params['bias'][channel]) *
                                int(params['multiplier'][channel])],dtype=np.int64),
                                params['shift'][channel],output_zp)[0]
                            require(sram[b+index]==(int(value)&255),
                                    'incorrect bias-derived constant byte')
                            ready[b+index]=layer_index+1; origins[b+index]=position
                    require(sram[b+offset:b+offset+size] ==
                            oracle[layer.output].reshape(-1)[first:first+size].tobytes(),
                            'constant output differs from independent oracle')
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
            if first_element == -1:
                logical, physical, channels = version
                ready[address:address+len(output)] = -3
                origins[address:address+len(output)] = -1
                for ch in range(channels):
                    base=address+ch*physical
                    ready[base:base+logical]=pack_input_version
                    origins[base:base+logical]=np.arange(ch*logical,(ch+1)*logical)
            else:
                ready[address:address+len(output)] = version
                origins[address:address+len(output)] = np.arange(first_element, first_element+len(output))
            pending_engine = None

    for command_index in range(len(commands)//16):
        op, flags, reserved, a, b, c = struct.unpack_from('<BBHIII', commands, command_index*16)
        require(reserved == 0, 'reserved command bits')
        if op == 0:
            require(not any((flags, a, b, c)) and command_index == len(commands)//16-1 and
                    pending_engine is None and pending_dma is None, 'invalid/premature HALT')
            require(set(run_contracts)==seen_contracts and
                    set(constant_contracts)==seen_constant_contracts and
                    set(pack_contracts)==seen_pack_contracts and
                    all(np.all(v) for v in coverage.values()),
                    'missing compute/output or unused contracts')
            for name,version,address,size in [(program.outputs[0],tensor_versions[program.outputs[0]],final_output['ext'],final_output['bytes'])]+[
                    (program.layers[i].output,i+1,r['ext'],r['bytes']) for i,r in snapshot_regions.items()]:
                expected=oracle[name].tobytes()
                require(0<=address and address+size<=len(ext) and size==len(expected) and
                        ext[address:address+size]==expected and np.all(versions[address:address+size]==version) and
                        np.array_equal(positions[address:address+size],np.arange(size)),'missing or stale final/snapshot tensor')
            result={'status': 'passed', 'engine_runs': runs, 'dma_bytes': dma_bytes,
                    'final_output_bytes': oracle[program.outputs[0]].size,
                    'scope': 'symbolic command replay with independent integer oracle; no RTL timing proof'}
            if pack_contracts: result['pack_runs']=pack_runs
            return result
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
            constant=constant_contracts.get(str(command_index))
            if constant is not None:
                require(flags==1 and isinstance(constant,dict) and
                        set(constant)=={'layer','first_element','bytes','dma_first','dma_bytes',
                                        'channel_first','channels','tile_first_element',
                                        'tile_output_sram','tile_output_bytes'} and
                        c==constant['dma_bytes'] and b==constant['tile_output_sram']+constant['dma_first'] and
                        0<=constant['dma_first'] and
                        constant['dma_first']+c<=constant['tile_output_bytes'] and
                        np.all(versions[a:a+c]==-2) and
                        constant['layer'] in coverage and
                        program.layers[constant['layer']].op=='Conv' and
                        program.layers[constant['layer']].attributes.get('group',1)==1,
                        'invalid constant-filter DMA contract')
                seen_constant_contracts.add(str(command_index))
            pending_dma = flags, a, b, c, constant
            dma_bytes['to_sram' if flags else 'from_sram'] += c
            continue
        require(op == 2 and flags == c == 0 and pending_engine is None, 'invalid engine dispatch')
        key=str(command_index)
        low, high = b & 65535, b >> 16
        require(0 <= low < high <= 32768 and a % 64 == 0, 'invalid live region/PC')
        if pending_dma:
            _, _, base, length, _ = pending_dma
            require(base+length <= low or base >= high, 'engine intersects pending DMA')
        require(low <= a and a+128 <= high and np.all(ready[a:a+128] == -2), 'unready descriptor')
        raw_descriptor=bytes(sram[a:a+64])
        if key in pack_contracts:
            contract=pack_contracts[key]
            require(isinstance(contract,dict) and set(contract)=={'layer','first_element'} and
                    type(contract['layer']) is int and contract['layer'] in coverage and
                    contract['first_element']>=0, 'invalid PACK contract')
            fields=struct.unpack('<IBBBB8I12H',raw_descriptor)
            magic,version,opcode,flags,reserved=fields[:5]
            input_base,output_base,weight,params,count,physical,row_stride,next_pc=fields[5:13]
            kh,kw,sh,sw,pt,pb,pl,pr,ih,iw,ic,oc=fields[13:]
            require((magic,version,opcode,flags,reserved)==(0x32445355,2,9,0,0) and
                    input_base==output_base and input_base%8==0 and
                    (weight,params,row_stride,next_pc)==(0,0,0,a+64) and
                    (kh,kw,sh,sw,pt,pb,pl,pr)==(1,1,1,1,0,0,0,0) and
                    min(ih,iw,ic)>0 and max(ih,iw)<=255 and ic<=1024 and oc==ic and
                    count==ih*iw and physical==(count+7)//8*8 and count!=physical and
                    input_base+ic*physical<=high and input_base>=low and
                    Descriptor.decode(bytes(sram[a+64:a+128])).opcode==0,
                    'invalid PACK descriptor')
            layer=program.layers[contract['layer']]
            x=oracle[layer.inputs[0]]
            output_plane=math.prod(program.tensors[layer.output].shape[2:])
            require(tuple(x.shape[1:])==(ic,ih,iw) and
                    contract['first_element']%output_plane==0 and
                    contract['first_element']<oracle[layer.output].size,
                    'PACK graph geometry mismatch')
            input_version=tensor_versions[layer.inputs[0]]
            read(input_base,x.size,x.tobytes(),input_version)
            require(np.array_equal(origins[input_base:input_base+x.size],np.arange(x.size)),
                    'PACK input provenance mismatch')
            packed=b''.join(x.reshape(ic,count)[ch].tobytes()+bytes(physical-count)
                            for ch in range(ic))
            pack_input_version=input_version
            pending_engine=low,high,input_base,packed,(count,physical,ic),-1
            ready[input_base:input_base+len(packed)]=-1
            seen_pack_contracts.add(key);pack_runs+=1;runs+=1
            continue
        contract=run_contracts.get(key)
        require(isinstance(contract,dict) and set(contract)=={'layer','first_element'},'missing/invalid run contract')
        layer_index,consumed=contract['layer'],contract['first_element']
        require(type(layer_index) is int and layer_index in coverage and type(consumed) is int and consumed>=0,'invalid logical operation')
        seen_contracts.add(key)
        flags=raw_descriptor[6]
        require(flags in (0,1), 'invalid descriptor layout flag')
        if flags:
            raw_descriptor=raw_descriptor[:6]+b'\0'+raw_descriptor[7:]
        d = Descriptor.decode(raw_descriptor); d.validate()
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
        if flags:
            require(opcode==4 and (d.kernel_h,d.kernel_w,d.stride_h,d.stride_w)==(1,1,1,1) and
                    not any((d.pad_top,d.pad_bottom,d.pad_left,d.pad_right)) and
                    d.count==d.input_c and operand.size==d.input_c*d.input_h*d.input_w,
                    'invalid padded pointwise layout')
            logical=d.input_h*d.input_w;physical=(logical+7)//8*8
            require(logical!=physical and d.input+d.input_c*physical<=high,
                    'invalid padded input extent')
            for ch in range(d.input_c):
                read(d.input+ch*physical,logical,
                     operand.reshape(d.input_c,logical)[ch].tobytes(),input_version)
                require(np.array_equal(origins[d.input+ch*physical:d.input+ch*physical+logical],
                                       np.arange(ch*logical,(ch+1)*logical)),
                        'incorrect padded channel provenance')
                require(sram[d.input+ch*physical+logical:d.input+(ch+1)*physical]==
                        bytes(physical-logical), 'nonzero padded channel tail')
        else:
            read(d.input, operand.size, operand.tobytes(), input_version)
        input_first = first*math.prod(x.shape[2:]) if spatial and opcode != 4 else consumed if opcode in (2, 8) else 0
        if not flags:
            require(np.array_equal(origins[d.input:d.input+operand.size], np.arange(input_first, input_first+operand.size)),
                    'incorrect tensor coordinate provenance')
        regions = [(a, a+128), (d.input, d.input+(d.input_c*physical if flags else operand.size)),
                   (d.output, d.output+d.outputs)]
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
