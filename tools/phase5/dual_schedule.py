"""Place audio and vision models in disjoint SDRAM and command-memory regions."""

import struct

COMMAND_BYTES = 16
PROGRAM_BYTES = 32768
EXTERNAL_BYTES = 8388608
VWW_BASE = 1048576


def relocate_commands(program, external_base):
    if len(program) % COMMAND_BYTES:
        raise ValueError('truncated command program')
    result = bytearray(program)
    for offset in range(0, len(program), COMMAND_BYTES):
        op, flags, reserved, arg0, arg1, arg2 = struct.unpack_from(
            '<BBHIII', result, offset)
        if reserved or op not in (0, 1, 2, 3):
            raise ValueError('unsupported or noncanonical command')
        if op == 1:
            if arg0 + external_base + arg2 > EXTERNAL_BYTES:
                raise ValueError('relocated DMA exceeds SDRAM')
            struct.pack_into('<I', result, offset + 4, arg0 + external_base)
    return bytes(result)


def combine(kws_commands, kws_payload, vww_commands, vww_payload):
    if len(kws_payload) > VWW_BASE or VWW_BASE + len(vww_payload) > EXTERNAL_BYTES:
        raise ValueError('resident model regions overlap or exceed SDRAM')
    if len(kws_commands) + len(vww_commands) > PROGRAM_BYTES:
        raise ValueError('resident schedules exceed command BSRAM')
    vww_entry = len(kws_commands) // COMMAND_BYTES
    program = relocate_commands(kws_commands, 0) + relocate_commands(
        vww_commands, VWW_BASE)
    return program, {'kws': {'entry': 0, 'external_base': 0},
                     'vww': {'entry': vww_entry, 'external_base': VWW_BASE}}
