"""Execute packed quantized tiles through the real engine, DMA and SRAM RTL."""

import random
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cocotb
import numpy as np
from cocotb.triggers import Timer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'compiler'))
from hardware_v2 import Descriptor
from integer_reference import evaluate
from phase4_compile import compile_tiled
from quantization import Quantization, quantize_parameters


def small_program():
    rng = np.random.default_rng(4344)
    qx, qc, qd, qp, qa, qy = (
        Quantization(.05, -7), Quantization(.04, 3),
        Quantization(.06, -4), Quantization(.04, 6),
        Quantization(.06, -2), Quantization(.08, 1))
    shapes = {'x': (1, 2, 4, 4), 'c': (1, 3, 4, 4),
              'd': (1, 3, 4, 4), 'p': (1, 4, 4, 4),
              'r': (1, 4, 4, 4), 'm': (1, 4, 2, 2),
              'a': (1, 4, 1, 1), 'k': (1, 4, 1, 1),
              'f': (1, 4), 'y': (1, 3)}
    quant = {'x': qx, 'c': qc, 'd': qd, 'p': qp, 'r': qp,
             'm': qp, 'a': qa, 'k': qa, 'f': qa, 'y': qy}
    tensors = {name: SimpleNamespace(shape=shape, quantization=quant[name])
               for name, shape in shapes.items()}
    conv_weight = rng.normal(size=(3, 2, 1, 1))
    depthwise_weight = rng.normal(size=(3, 1, 3, 3))
    pointwise_weight = rng.normal(size=(4, 3, 1, 1))
    gemm_weight = rng.normal(size=(3, 4))
    def layer(op, source, output, attrs=None, params=None):
        return SimpleNamespace(op=op, inputs=[source], output=output,
                               attributes=attrs or {}, parameters=params or {})
    layers = [
        layer('Conv', 'x', 'c',
              params=quantize_parameters(conv_weight, rng.normal(size=3), qx, qc)),
        layer('Conv', 'c', 'd', attrs={'group': 3, 'pads': [1, 1, 1, 1]},
              params=quantize_parameters(depthwise_weight,
                                         rng.normal(size=3), qc, qd)),
        layer('Conv', 'd', 'p',
              params=quantize_parameters(pointwise_weight,
                                         rng.normal(size=4), qd, qp)),
        layer('Relu', 'p', 'r'),
        layer('MaxPool', 'r', 'm',
              attrs={'kernel_shape': [2, 2], 'strides': [2, 2]}),
        layer('AveragePool', 'm', 'a',
              attrs={'kernel_shape': [2, 2], 'strides': [2, 2]}),
        layer('Clip', 'a', 'k', params={'clip_bounds': np.array([-20, 20],
                                                              dtype=np.int8)}),
        layer('Flatten', 'k', 'f'),
        layer('Gemm', 'f', 'y',
              params=quantize_parameters(gemm_weight, rng.normal(size=3), qa, qy)),
    ]
    return SimpleNamespace(inputs=['x'], outputs=['y'], constants={},
                           tensors=tensors, layers=layers)


@cocotb.test()
async def packed_conv_pool_fc_chain_matches_independent_oracle(d):
    rng = random.Random(4412)
    external = bytearray(8 * 1024 * 1024)
    pending = None
    held = None
    cycle_count = 0
    records = []
    d.clk.value = 0; d.rst_n.value = 0
    d.engine_start.value = 0; d.engine_abort.value = 0
    d.engine_start_pc.value = 0
    d.dma_start.value = 0; d.dma_abort.value = 0
    d.dma_to_sram.value = 0; d.dma_sram_base.value = 0
    d.dma_ext_base.value = 0; d.dma_length.value = 0
    d.host_req.value = 0; d.host_wr.value = 0
    d.host_addr.value = 0; d.host_wdata.value = 0; d.host_wstrb.value = 0
    d.ext_ready.value = 0; d.ext_rvalid.value = 0; d.ext_rdata.value = 0

    async def step():
        nonlocal pending, held, cycle_count
        d.clk.value = 0
        ready = pending is None and rng.randrange(5) != 0
        d.ext_ready.value = ready
        d.ext_rvalid.value = 0
        if pending is not None:
            delay, data = pending
            if delay == 0:
                d.ext_rvalid.value = 1
                d.ext_rdata.value = data
                pending = None
            else:
                pending = (delay-1, data)
        await Timer(5, units='ns')
        req = (int(d.ext_addr.value), int(d.ext_wr.value),
               int(d.ext_wdata.value), int(d.ext_wstrb.value)) \
            if int(d.ext_req.value) else None
        if held is not None:
            assert req == held, 'external request changed while stalled'
        held = req if req is not None and not ready else None
        if req is not None and ready:
            address, wr, data, mask = req
            assert address % 8 == 0 and address + 8 <= len(external)
            if wr:
                for lane in range(8):
                    if mask & (1 << lane):
                        external[address+lane] = (data >> (8*lane)) & 255
            else:
                assert pending is None
                pending = (rng.randrange(1, 4),
                           int.from_bytes(external[address:address+8], 'little'))
        d.clk.value = 1
        await Timer(5, units='ns')
        cycle_count += 1

    async def host_write(addr, payload):
        for offset in range(0, len(payload), 8):
            chunk = payload[offset:offset+8]
            d.host_req.value = 1; d.host_wr.value = 1
            d.host_addr.value = addr + offset
            d.host_wdata.value = int.from_bytes(chunk.ljust(8, b'\0'), 'little')
            d.host_wstrb.value = (1 << len(chunk)) - 1
            await step()
            assert int(d.host_ready.value)
        d.host_req.value = 0

    async def dma(transfer):
        d.dma_to_sram.value = int(transfer['direction'] == 'to_sram')
        d.dma_sram_base.value = transfer['sram']
        d.dma_ext_base.value = transfer['ext']
        d.dma_length.value = transfer['bytes']
        d.dma_start.value = 1
        await step()
        d.dma_start.value = 0
        for _ in range(50000):
            await step()
            if int(d.dma_done.value):
                break
        else:
            raise AssertionError('tile DMA timed out')
        assert int(d.dma_error.value) == 0
        assert int(d.dma_bytes_copied.value) == transfer['bytes']

    async def engine(descriptor, label):
        await host_write(0, bytes.fromhex(descriptor) + Descriptor(0).encode())
        d.engine_start.value = 1
        await step()
        d.engine_start.value = 0
        for _ in range(2000000):
            await step()
            if int(d.engine_done.value):
                break
        else:
            raise AssertionError('tile engine timed out')
        assert int(d.engine_error.value) == 0, f'{label}: error {int(d.engine_error.value)}'

    async def execute_plan(plan, image, input_bytes, expected_layers, name):
        start_cycle = cycle_count
        external[:] = bytes(len(external))
        external[:len(image)] = image
        external[:len(input_bytes)] = input_bytes
        slot = 0
        for index, (scheduled, expected) in enumerate(zip(plan['layers'],
                                                             expected_layers)):
            if scheduled['kind'] in ('alias', 'host_layout'):
                assert external[slot*plan['activation_slot_bytes']:
                                slot*plan['activation_slot_bytes']+
                                len(expected)] == expected
                continue
            for tile in scheduled['tiles']:
                for transfer in tile['transfers'][:-1]:
                    await dma(transfer)
                await engine(tile['descriptor_hex'], f'{name} node {index}')
                await dma(tile['transfers'][-1])
            slot = scheduled['output_slot']
            start = slot * plan['activation_slot_bytes']
            assert external[start:start+len(expected)] == expected, \
                f'{name} node {index} produced wrong quantized tensor'
            if name in ('kws', 'vww'):
                d._log.info('%s node %d/%d passed at cycle %d', name, index+1,
                            len(plan['layers']), cycle_count)
        records.append({'name': name,
                        'compute_nodes': sum(bool(l['tiles']) for l in plan['layers']),
                        'tiles': sum(len(l['tiles']) for l in plan['layers']),
                        'dma_payload_bytes': sum(t['bytes'] for l in plan['layers']
                                                 for tile in l['tiles']
                                                 for t in tile['transfers']),
                        'parameter_image_sha256': plan['parameter_image_sha256'],
                        'output_sha256': hashlib.sha256(expected).hexdigest(),
                        'simulated_clocks_including_host': cycle_count-start_cycle})
        return plan

    async def execute(program, qx, name):
        plan, image = compile_tiled(program)
        reference = evaluate(program, {'x': qx})
        expected_layers = [reference[layer.output].tobytes()
                           for layer in program.layers]
        return await execute_plan(plan, image, qx.tobytes(), expected_layers,
                                  name)

    await step(); d.rst_n.value = 1; await step()
    small = small_program()
    qx = np.random.default_rng(4242).integers(
        -128, 128, size=small.tensors['x'].shape, dtype=np.int8)
    await execute(small, qx, 'conv-depthwise-pointwise-relu-maxpool-avgpool-clip-fc')

    # This crosses a real tile boundary: input plus output no longer fit the
    # 32-KiB scratchpad at once, so the same engine/DMA path runs two tiles.
    q = Quantization(.05, -7)
    shape = (1, 16320)
    large = SimpleNamespace(inputs=['x'], outputs=['y'], constants={},
                            tensors={name: SimpleNamespace(shape=shape,
                                                           quantization=q)
                                     for name in ('x', 'y')},
                            layers=[SimpleNamespace(op='Relu', inputs=['x'],
                                                    output='y', attributes={},
                                                    parameters={})])
    qx = np.random.default_rng(4243).integers(-128, 128, size=shape,
                                               dtype=np.int8)
    plan = await execute(large, qx, 'two-tile-relu')
    assert len(plan['layers'][0]['tiles']) == 2

    # The weight matrix alone occupies 32 KiB. It must be divided across
    # output-channel tiles while the input vector is loaded for each tile.
    fq_in, fq_out = Quantization(.03, -11), Quantization(.07, 5)
    frng = np.random.default_rng(4244)
    weight = frng.normal(size=(64, 512))
    fc = SimpleNamespace(inputs=['x'], outputs=['y'], constants={},
                         tensors={'x': SimpleNamespace(shape=(1, 512),
                                                       quantization=fq_in),
                                  'y': SimpleNamespace(shape=(1, 64),
                                                       quantization=fq_out)},
                         layers=[SimpleNamespace(
                             op='Gemm', inputs=['x'], output='y', attributes={},
                             parameters=quantize_parameters(
                                 weight, frng.normal(size=64), fq_in, fq_out))])
    qx = frng.integers(-128, 128, size=(1, 512), dtype=np.int8)
    plan = await execute(fc, qx, 'multi-tile-fc-32kib-weights')
    assert len(plan['layers'][0]['tiles']) >= 2
    fixture = os.environ.get('PHASE4_TILED_FIXTURE')
    if fixture:
        folder = Path(fixture)
        name = folder.name.removeprefix('rtl-')
        plan = json.loads((folder.parent/f'{name}-tiled-plan.json').read_text())
        image = (folder.parent/f'{name}-parameter-image.bin').read_bytes()
        with np.load(folder/'expected.npz') as expected:
            outputs = [expected[f'layer_{index}'].tobytes()
                       for index in range(len(plan['layers']))]
        await execute_plan(plan, image, (folder/'input.bin').read_bytes(),
                           outputs, name)
    sources = ('compiler/phase4_compile.py', 'compiler/phase4_tiling.py',
               'rtl/v2/engine.sv', 'rtl/v2/tile_dma.sv',
               'rtl/v2/tiled_core.sv', 'rtl/v2/scratchpad.sv',
               'test/phase4/test_tiled_program.py')
    report = {'evidence_type': 'randomized-stall RTL simulation; no physical SDRAM',
              'random_seed': 4412, 'status': 'passed', 'cases': records,
              'source_sha256': {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
                                for path in sources}}
    (ROOT/'work/phase4/tiled-program-report.json').write_text(
        json.dumps(report, indent=2, sort_keys=True)+'\n')
