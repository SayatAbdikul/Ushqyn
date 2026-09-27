"""Exact compiler replacement of eligible all-zero standard Conv filters."""
import math
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'compiler'))
sys.path.insert(0, str(ROOT / 'tools/phase6'))

from chain_resident import compile_chain
from constant_filter import compile_constant_chain, filter_constants
from integer_reference import evaluate
from run_boardless import load_model
from scheduler.resident_verify import replay_resident
from variants import check_frozen


def test_vww_constant_filter_replay_and_command_capacity():
    check_frozen()
    program, pinned, _, _ = load_model('vww')
    code, payload, schedule = compile_constant_chain(program, snapshots=True)
    stats = schedule['constant_filter']
    assert stats['zero_channels'] == 971
    assert stats['skipped_dense_macs'] == 3_131_136
    assert stats['constant_output_bytes'] == 23_022
    assert stats['split_tiles'] == 12
    assert schedule['command_count'] == len(code)//16 == 1849 < 2048
    assert len(payload) <= 8*1024*1024
    assert all(contract['bytes'] == contract['channels'] *
               math.prod(program.tensors[program.layers[contract['layer']].output].shape[2:])
               for contract in schedule['constant_contracts'].values())

    for value in (pinned, np.random.default_rng(6073).integers(-128, 128, pinned.shape, dtype=np.int8)):
        oracle = evaluate(program, {program.inputs[0]: value})
        result = replay_resident(program, code, payload, {program.inputs[0]: value},
            run_contracts=schedule['run_contracts'],
            constant_contracts=schedule['constant_contracts'],
            final_output=schedule['final_output'],
            snapshot_regions=schedule['snapshot_regions'], oracle=oracle)
        assert result['status'] == 'passed'
    check_frozen()


def test_kws_without_constant_filters_is_byte_identical_to_chain():
    program, _, _, _ = load_model('kws')
    baseline = compile_chain(program, snapshots=True)
    candidate = compile_constant_chain(program, snapshots=True)
    assert candidate[:2] == baseline[:2]
    assert candidate[2]['constant_filter']['enabled'] is False
    assert not any(code is not None for i in range(len(program.layers))
                   for code in (filter_constants(program, i) or ()))


def test_replay_rejects_corrupt_constant_payload_and_contract():
    program, pinned, _, _ = load_model('vww')
    commands, payload, schedule = compile_constant_chain(program, snapshots=False)
    oracle = evaluate(program, {program.inputs[0]: pinned})
    contracts = schedule['constant_contracts']
    first_key = next(iter(contracts))
    command_index = int(first_key)
    op, flags, _, ext, _, count = struct.unpack_from('<BBHIII', commands, command_index*16)
    assert op == flags == 1 and count > 0
    altered = bytearray(payload)
    altered[ext] ^= 1
    with pytest.raises(ValueError, match='constant'):
        replay_resident(program, commands, bytes(altered), {program.inputs[0]: pinned},
            run_contracts=schedule['run_contracts'], constant_contracts=contracts,
            final_output=schedule['final_output'], oracle=oracle)
    omitted = dict(contracts)
    del omitted[first_key]
    with pytest.raises(ValueError, match='unproduced|missing|stale tensor version'):
        replay_resident(program, commands, payload, {program.inputs[0]: pinned},
            run_contracts=schedule['run_contracts'], constant_contracts=omitted,
            final_output=schedule['final_output'], oracle=oracle)
