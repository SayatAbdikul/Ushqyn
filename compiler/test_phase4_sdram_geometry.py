"""Vendor-IP configuration guard; no SDRAM hardware access is implied."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/phase4'))
from audit_sdram_ip import GEOMETRY, audit_ipc


def test_board_embedded_sdram_geometry_and_config_guard(tmp_path):
    g = GEOMETRY
    assert (1 << (g['bank_address_bits']+g['row_address_bits']+
                  g['column_address_bits'])) * g['data_width_bits']//8 == 8*1024*1024
    good = tmp_path/'good.ipc'
    good.write_text('[Config]\nData_Width=32\nBank_Width=2\nAddr_Row_Width=11\nAddr_Column_Width=8\n')
    assert audit_ipc(good)['capacity_bytes'] == 8*1024*1024
    wrong = tmp_path/'wrong.ipc'
    wrong.write_text('[Config]\nData_Width=16\nBank_Width=2\nAddr_Row_Width=13\nAddr_Column_Width=9\n')
    with pytest.raises(ValueError, match='wrong Tang Nano 20K SDRAM geometry'):
        audit_ipc(wrong)
