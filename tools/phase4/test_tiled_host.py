"""Host runner fails closed on incorrect tensor bytes and stale images."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'compiler'))
from integer_reference import evaluate
from phase4_compile import compile_tiled
from quantization import Quantization
from tools.phase4.tiled_host import EXT_BYTES, execute_plan


class FakeLink:
    def __init__(self, tile, expected, corrupt=False):
        self.external = bytearray(256)
        self.sram = bytearray(32768)
        self.tile = tile
        self.expected = expected
        self.corrupt = corrupt
        self.moves = []

    def capabilities(self):
        return EXT_BYTES

    def exchange(self, command):
        assert command == 7  # RESET
        return b''

    def wait_idle(self, timeout):
        assert timeout > 0
        return {'busy': False, 'error': 0}

    def write_external(self, offset, data):
        self.external[offset:offset+len(data)] = data

    def read_external(self, offset, length):
        result = bytes(self.external[offset:offset+length])
        if self.corrupt and offset == 16 and result:
            return bytes([result[0] ^ 1]) + result[1:]
        return result

    def write(self, address, data):
        assert address == 0 and len(data) == 128
        self.sram[address:address+len(data)] = data

    def run(self, pc, timeout):
        assert pc == 0 and timeout > 0
        output = self.tile['regions']['output']['base']
        self.sram[output:output+len(self.expected)] = self.expected
        return {'busy': False, 'error': 0}

    def transfer(self, move, timeout):
        assert timeout > 0
        ext, sram, n = move['ext'], move['sram'], move['bytes']
        if move['direction'] == 'to_sram':
            self.sram[sram:sram+n] = self.external[ext:ext+n]
        else:
            self.external[ext:ext+n] = self.sram[sram:sram+n]
        self.moves.append(move)


def fixture():
    q = Quantization(.05, -7)
    shape = (1, 16)
    tensors = {name: SimpleNamespace(shape=shape, quantization=q)
               for name in ('x', 'y')}
    layer = SimpleNamespace(op='Relu', inputs=['x'], output='y',
                            attributes={}, parameters={})
    program = SimpleNamespace(inputs=['x'], outputs=['y'], constants={},
                              tensors=tensors, layers=[layer])
    qx = np.arange(-12, 4, dtype=np.int8).reshape(shape)
    plan, image = compile_tiled(program)
    expected = evaluate(program, {'x': qx})['y'].tobytes()
    return plan, image, qx.tobytes(), expected


def test_host_runner_replays_transfers_and_checks_exact_output():
    plan, image, qx, expected = fixture()
    tile = plan['layers'][0]['tiles'][0]
    link = FakeLink(tile, expected)
    report = execute_plan(link, plan, image, qx, [expected])
    assert report['status'] == 'passed'
    assert len(link.moves) == len(tile['transfers'])
    assert report['nodes'][0]['dma_payload_bytes'] == sum(
        move['bytes'] for move in tile['transfers'])
    assert bytes(link.external[plan['activation_slot_bytes']:
                               plan['activation_slot_bytes']+len(expected)]) == expected


def test_host_runner_rejects_bad_output_and_stale_parameter_image():
    plan, image, qx, expected = fixture()
    tile = plan['layers'][0]['tiles'][0]
    with pytest.raises(AssertionError, match='node 0 output mismatch'):
        execute_plan(FakeLink(tile, expected, corrupt=True),
                     plan, image, qx, [expected])
    with pytest.raises(ValueError, match='parameter image and plan mismatch'):
        execute_plan(FakeLink(tile, expected), plan, image + b'wrong',
                     qx, [expected])
