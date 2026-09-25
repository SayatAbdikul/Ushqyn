"""Drive the tiled accelerator entirely through its framed host interface."""

import hashlib
import json
import os
import random
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
from quantization import Quantization


def crc16(data):
    value = 0xffff
    for byte in data:
        value ^= byte << 8
        for _ in range(8):
            value = ((value << 1) ^ (0x1021 if value & 0x8000 else 0)) & 0xffff
    return value


@cocotb.test()
async def uart_packet_tile_program_uses_sdram_window_and_dma(d):
    memory = bytearray(8 * 1024 * 1024)
    rng = random.Random(9141)
    pending = None
    held = None
    response = []
    d.clk.value = 0
    d.rst_n.value = 0
    d.rx_valid.value = 0
    d.rx_data.value = 0
    d.tx_ready.value = 1
    d.ext_ready.value = 0
    d.ext_rvalid.value = 0
    d.ext_rdata.value = 0
    d.memory_initialized.value = 0

    async def step(byte=None):
        nonlocal pending, held
        d.clk.value = 0
        d.rx_valid.value = byte is not None
        d.rx_data.value = 0 if byte is None else byte
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
                pending = (delay - 1, data)
        await Timer(5, units='ns')
        if int(d.tx_valid.value):
            response.append(int(d.tx_data.value))
        request = (int(d.ext_addr.value), int(d.ext_wr.value),
                   int(d.ext_wdata.value), int(d.ext_wstrb.value)) \
            if int(d.ext_req.value) else None
        if held is not None:
            assert request == held, 'external request changed while stalled'
        held = request if request is not None and not ready else None
        if request is not None and ready:
            address, write, data, mask = request
            assert address % 8 == 0 and address + 8 <= len(memory)
            if write:
                for lane in range(8):
                    if mask & (1 << lane):
                        memory[address + lane] = (data >> (8 * lane)) & 255
            else:
                assert pending is None
                pending = (rng.randrange(1, 4),
                           int.from_bytes(memory[address:address + 8], 'little'))
        d.clk.value = 1
        await Timer(5, units='ns')

    sequence = 0

    async def command(kind, address=0, length=0, payload=b'', check=0):
        nonlocal sequence
        assert not response
        sequence = (sequence + 1) & 255
        header = bytes([2, kind, sequence]) + address.to_bytes(3, 'little') + \
                 length.to_bytes(2, 'little')
        contents = header + payload
        frame = b'\xa5\x5a' + contents + crc16(contents).to_bytes(2, 'little')
        for byte in frame:
            await step(byte)
        for _ in range(30000):
            if len(response) >= 10:
                total = 10 + int.from_bytes(bytes(response[8:10]), 'little') + 2
                if len(response) == total:
                    break
            await step()
        else:
            raise AssertionError(f'command {kind} response timed out')
        packet = bytes(response)
        response.clear()
        assert packet[:3] == b'\xa5\x5a\x02'
        assert packet[3] == kind | 0x80 and packet[4] == sequence
        assert crc16(packet[2:-2]) == int.from_bytes(packet[-2:], 'little')
        assert packet[10] == check, (kind, address, packet[10])
        await step()  # leave TX_GAP before sending another frame
        return packet[11:-2]

    async def write(address, data):
        for offset in range(0, len(data), 64):
            chunk = data[offset:offset + 64]
            await command(3, address + offset, len(chunk), chunk)

    async def read(address, length):
        result = bytearray()
        for offset in range(0, length, 64):
            n = min(64, length - offset)
            result.extend(await command(2, address + offset, n))
        return bytes(result)

    async def wait_idle():
        for _ in range(4000000):
            if not int(d.engine_busy.value) and not int(d.dma_busy.value):
                break
            await step()
        else:
            raise AssertionError('engine or DMA did not complete')
        status = await command(5)
        assert status[0] == 0 and status[1] == 0

    async def dma(transfer):
        config = transfer['ext'].to_bytes(3, 'little') + b'\0' + \
                 transfer['sram'].to_bytes(3, 'little') + b'\0' + \
                 transfer['bytes'].to_bytes(4, 'little') + \
                 bytes([int(transfer['direction'] == 'to_sram')])
        await write(0x400000, config)
        await write(0x40000d, b'\x01')
        await wait_idle()
        reg = await read(0x400010, 6)
        assert reg[1] == 0
        assert int.from_bytes(reg[2:6], 'little') == transfer['bytes']

    await step()
    d.rst_n.value = 1
    await step()
    caps = await command(1)
    assert len(caps) == 14 and caps[10] == 1
    assert int.from_bytes(caps[11:14], 'little') == len(memory)
    await command(3, 0x800000, 1, b'\x12', check=3)
    d.memory_initialized.value = 1
    await step()
    await command(3, 0x7fff, 2, b'\x11\x22', check=4)
    await command(2, 0x40001f, 2, check=4)
    # Exercise partial words at both physical address limits through the
    # packet parser, not only through the lower-level DMA regression.
    await write(0x007fff, b'\xa5')
    assert await read(0x007fff, 1) == b'\xa5'
    await write(0xfffffc, b'\x13\x37\x5a\xc0')
    assert await read(0xfffffc, 4) == b'\x13\x37\x5a\xc0'
    assert memory[-4:] == b'\x13\x37\x5a\xc0'
    tail = bytes(range(1, 17))
    await write(0xfffff0, tail)
    await dma({'ext': 0x7ffff0, 'sram': 0x1000, 'bytes': len(tail),
               'direction': 'to_sram'})
    assert await read(0x1000, len(tail)) == tail
    transformed = bytes(byte ^ 0x5a for byte in tail)
    await write(0x1000, transformed)
    await dma({'ext': 0x7fffd0, 'sram': 0x1000, 'bytes': len(tail),
               'direction': 'from_sram'})
    assert await read(0xffffd0, len(tail)) == transformed

    q = Quantization(.05, -7)
    x = np.array([[-128, -20, -7, -1, 0, 1, 20, 127,
                   -9, -8, -6, 3, 4, 5, 7, 8]], dtype=np.int8)
    tensors = {name: SimpleNamespace(shape=(1, 16), quantization=q)
               for name in ('x', 'y', 'z')}
    layers = [SimpleNamespace(op='Relu', inputs=['x'], output='y',
                              attributes={}, parameters={}),
              SimpleNamespace(op='Clip', inputs=['y'], output='z',
                              attributes={},
                              parameters={'clip_bounds': np.array([-7, 10], np.int8)})]
    program = SimpleNamespace(inputs=['x'], outputs=['z'], constants={},
                              tensors=tensors, layers=layers)
    plan, image = compile_tiled(program)
    oracle = evaluate(program, {'x': x})['z'].tobytes()
    await write(0x800000, image)
    await write(0x800000, x.tobytes())
    for layer in plan['layers']:
        for tile in layer['tiles']:
            for transfer in tile['transfers'][:-1]:
                await dma(transfer)
            descriptor = bytes.fromhex(tile['descriptor_hex']) + Descriptor(0).encode()
            await write(0, descriptor)
            await command(4, 0)
            await wait_idle()
            await dma(tile['transfers'][-1])
    address = 0x800000 + plan['final_output_slot'] * plan['activation_slot_bytes']
    assert await read(address, len(oracle)) == oracle
    assert memory[address - 0x800000:address - 0x800000 + len(oracle)] == oracle

    fixture = os.environ.get('PHASE4_TILED_HOST_FIXTURE')
    if fixture:
        folder = Path(fixture)
        name = folder.name.removeprefix('rtl-')
        plan_path = folder.parent/f'{name}-tiled-plan.json'
        image_path = folder.parent/f'{name}-parameter-image.bin'
        model_plan = json.loads(plan_path.read_text())
        model_image = image_path.read_bytes()
        manifest = json.loads((folder/'manifest.json').read_text())
        qinput = (folder/'input.bin').read_bytes()
        assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == \
               manifest['plan_sha256']
        assert hashlib.sha256(model_image).hexdigest() == \
               manifest['image_sha256'] == model_plan['parameter_image_sha256']
        assert hashlib.sha256(qinput).hexdigest() == manifest['input_sha256']
        assert hashlib.sha256((folder/'expected.npz').read_bytes()).hexdigest() == \
               manifest['expected_npz_sha256']
        assert len(model_plan['layers']) == manifest['nodes']
        with np.load(folder/'expected.npz') as saved:
            outputs = [saved[f'layer_{i}'].tobytes()
                       for i in range(manifest['nodes'])]
        await write(0x800000, model_image)
        await write(0x800000, qinput)
        slot = 0
        for index, (layer, expected) in enumerate(zip(model_plan['layers'], outputs)):
            for tile in layer['tiles']:
                for move in tile['transfers'][:-1]:
                    await dma(move)
                descriptor = bytes.fromhex(tile['descriptor_hex']) + \
                             Descriptor(0).encode()
                await write(0, descriptor)
                await command(4, 0)
                await wait_idle()
                await dma(tile['transfers'][-1])
            if layer['tiles']:
                slot = layer['output_slot']
            output_addr = 0x800000 + slot * model_plan['activation_slot_bytes']
            actual = await read(output_addr, len(expected))
            assert actual == expected, f'{name} node {index} mismatch'
            d._log.info('%s framed host node %d/%d passed', name, index+1,
                        len(model_plan['layers']))
        assert hashlib.sha256(outputs[-1]).hexdigest() == manifest['output_sha256']
