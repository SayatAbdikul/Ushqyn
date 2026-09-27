"""The geometry patch must be exact and independent of feature parameters."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/phase6'))

from geometry_variant import apply_geometry_narrowing, materialize


def test_reproduces_routed_all_exact_snapshot():
    source = (ROOT/'work/phase6/experiments-v1/all-exact/engine.sv').read_text()
    expected = (ROOT/'rtl/phase6/geometry_engine.sv').read_text()
    assert apply_geometry_narrowing(source) == expected


def test_parameter_values_do_not_enter_pattern_matching():
    source = (ROOT/'work/phase6/experiments-v1/all-exact/engine.sv').read_text()
    altered = source.replace('parameter integer PW_CACHE_ENTRIES = 256',
                             'parameter integer PW_CACHE_ENTRIES = 128')
    assert altered != source
    output = apply_geometry_narrowing(altered)
    assert 'parameter integer PW_CACHE_ENTRIES = 128' in output
    assert 'wire [15:0] plane_calc=ih[7:0]*iw[7:0];' in output


def test_missing_guard_or_changed_arithmetic_refuses_patch():
    source = (ROOT/'work/phase6/experiments-v1/all-exact/engine.sv').read_text()
    with pytest.raises(ValueError, match='guard changed'):
        apply_geometry_narrowing(source.replace('ih>255||iw>255', 'ih>256||iw>255'))
    with pytest.raises(ValueError, match='arithmetic site changed'):
        apply_geometry_narrowing(source.replace("wire [31:0] plane_calc=32'(ih)*32'(iw);",
                                                "wire [31:0] plane_calc=32'(iw)*32'(ih);"))


def test_materialize_never_overwrites_distinct_snapshot(tmp_path):
    source = ROOT/'work/phase6/experiments-v1/all-exact/engine.sv'
    output = tmp_path/'engine.sv'
    evidence = materialize(source, output)
    assert evidence['exact_replacements'] == 6
    assert output.read_text() == (ROOT/'rtl/phase6/geometry_engine.sv').read_text()
    assert materialize(source, output)['output_sha256'] == evidence['output_sha256']
    output.write_text('preserved different snapshot')
    with pytest.raises(FileExistsError):
        materialize(source, output)
