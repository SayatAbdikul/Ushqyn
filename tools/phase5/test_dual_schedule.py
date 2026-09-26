"""Check the two-model placement contract before any physical run."""

import struct

import pytest

from dual_schedule import combine, relocate_commands, VWW_BASE


def cmd(op, address=0, length=0):
    return struct.pack('<BBHIII', op, 1 if op == 1 else 0,
                       0, address, 0, length)


def test_disjoint_resident_programs_and_dma_addresses():
    kws = cmd(1, 4096, 8) + cmd(0)
    vww = cmd(1, 8192, 64) + cmd(0)
    combined, locations = combine(kws, bytes(65536), vww, bytes(262144))
    assert locations['vww']['entry'] == 2
    assert combined[:len(kws)] == kws
    assert combined[len(kws):] == cmd(1, VWW_BASE + 8192, 64) + cmd(0)


def test_relocation_rejects_sdram_overrun_and_noncanonical_command():
    with pytest.raises(ValueError, match='exceeds SDRAM'):
        relocate_commands(cmd(1, 8388600, 16), 0)
    with pytest.raises(ValueError, match='noncanonical'):
        relocate_commands(cmd(4), 0)
