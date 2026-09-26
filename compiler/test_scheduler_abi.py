import struct

import pytest

from phase4_compile import compile_tiled
from phase4_sequence import compile_sequence
from scheduler.abi_verify import replay
from scheduler.contract import IllegalSchedule
from test_scheduler_spatial import spatial_fixture


def artifact(overlap=True):
    p, x = spatial_fixture()
    plan, image = compile_tiled(p, prefer_half=True)
    commands, payload, record = compile_sequence(plan, image, overlap=overlap)
    return p, x, commands, payload, record


@pytest.mark.parametrize('overlap', [True, False])
def test_replay_checks_generated_commands_without_schedule_manifest(overlap):
    p, x, commands, payload, _ = artifact(overlap)
    result = replay(p, commands, payload, {'input': x})
    assert result['status'] == 'passed' and result['engine_runs'] == 4


def test_corrupt_parameter_payload_rejected():
    p, x, commands, payload, record = artifact()
    payload = bytearray(payload)
    parameter = next(t for t in record['tiles'][0]['loads'] if t['role'] == 'parameter')
    payload[parameter['ext']] ^= 1
    with pytest.raises(IllegalSchedule, match='operand bytes'):
        replay(p, commands, payload, {'input': x})


def test_missing_engine_wait_rejected():
    p, x, commands, payload, _ = artifact(False)
    words = [commands[i:i+16] for i in range(0, len(commands), 16)]
    index = next(i for i, w in enumerate(words) if w[0] == 3 and w[1] == 3)
    del words[index]
    with pytest.raises(IllegalSchedule, match='live engine region'):
        replay(p, b''.join(words), payload, {'input': x})


def test_missing_dma_wait_rejected():
    p, x, commands, payload, _ = artifact()
    with pytest.raises(IllegalSchedule, match='illegal DMA'):
        replay(p, commands[:16]+commands[32:], payload, {'input': x})


def test_wrong_output_address_and_missing_tile_rejected():
    p, x, commands, payload, _ = artifact()
    words = [commands[i:i+16] for i in range(0, len(commands), 16)]
    index = next(i for i, w in enumerate(words) if w[0:2] == b'\x01\x00')
    op, flags, reserved, a, b, c = struct.unpack('<BBHIII', words[index])
    words[index] = struct.pack('<BBHIII', op, flags, reserved, 0, b, c)
    with pytest.raises(IllegalSchedule, match='not fully committed'):
        replay(p, b''.join(words), payload, {'input': x})
    with pytest.raises(IllegalSchedule, match='missing compute/output'):
        replay(p, bytes(16), payload, {'input': x})


def test_halt_and_reserved_bits_rejected():
    p, x, commands, payload, _ = artifact()
    with pytest.raises(IllegalSchedule, match='missing HALT'):
        replay(p, commands[:-16], payload, {'input': x})
    corrupt = bytearray(commands); corrupt[2] = 1
    with pytest.raises(IllegalSchedule, match='reserved'):
        replay(p, bytes(corrupt), payload, {'input': x})


def test_stale_equal_valued_input_requires_correct_version():
    # Initial activation and the zeroed unused activation slot contain equal
    # bytes. Data-only checking would miss this dependency violation.
    p, x, commands, payload, record = artifact()
    x[:] = 0
    words = [commands[i:i+16] for i in range(0, len(commands), 16)]
    first_input = record['tiles'][0]['loads'][1]
    slot = (max(t.shape[1]*t.shape[2]*t.shape[3] for t in p.tensors.values())+7)//8*8
    for i, w in enumerate(words):
        op, flags, reserved, a, b, c = struct.unpack('<BBHIII', w)
        if op == flags == 1 and a == first_input['ext'] and b == first_input['sram']:
            words[i] = struct.pack('<BBHIII', op, flags, reserved, slot, b, c)
            break
    with pytest.raises(IllegalSchedule, match='unready|stale'):
        replay(p, b''.join(words), payload, {'input': x})
