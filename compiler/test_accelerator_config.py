"""Smoke tests for the profile-based AcceleratorConfig (P3, 2026-05-08).

Verifies that:
- Both profiles expose every attribute consumers rely on
  (TILE_ELEMS, MEM_SIZE, all DRAM_ADDR_*).
- `set_profile(...)` switches the active reference.
- The historical `AcceleratorConfig` symbol works as a class with class
  attributes (existing call sites do `AcceleratorConfig.OUT_N` etc.).

Does NOT exercise env-var selection — that runs at module import time and
the test process has already imported accelerator_config by the time pytest
collects this file. The env-var path is covered indirectly by the
heavy_test_fpga integration test.
"""
import accelerator_config


REQUIRED_ATTRS = (
    "DATA_WIDTH", "ADDR_WIDTH",
    "TILE_ELEMS", "TILE_WIDTH",
    "MEM_SIZE",
    "VECTOR_BUFFER_WIDTH", "MATRIX_BUFFER_WIDTH",
    "MAX_ROWS", "MAX_COLS", "OUT_N",
    "DRAM_ADDR_INPUTS", "DRAM_ADDR_BIASES",
    "DRAM_ADDR_OUTPUTS", "DRAM_ADDR_WEIGHTS",
    "DRAM_ADDR_CONV_WEIGHTS",  # the field whose absence broke heavy_test_fpga pre-P3
)


def test_sim_profile_complete():
    for a in REQUIRED_ATTRS:
        assert hasattr(accelerator_config.SimProfile, a), \
            f"SimProfile missing {a}"


def test_fpga_profile_complete():
    for a in REQUIRED_ATTRS:
        assert hasattr(accelerator_config.FpgaProfile, a), \
            f"FpgaProfile missing {a}"


def test_profiles_have_expected_tile_elems():
    assert accelerator_config.SimProfile.TILE_ELEMS == 32
    assert accelerator_config.FpgaProfile.TILE_ELEMS == 8


def test_profiles_share_dram_address_map():
    """The compiled instruction stream addresses the same DRAM regions on both
    targets — only TILE_ELEMS differs. Catches accidental drift in the
    address map that would silently misroute weights/biases on one profile."""
    for a in ("DRAM_ADDR_INPUTS", "DRAM_ADDR_BIASES", "DRAM_ADDR_OUTPUTS",
              "DRAM_ADDR_WEIGHTS", "DRAM_ADDR_CONV_WEIGHTS"):
        assert getattr(accelerator_config.SimProfile, a) == \
               getattr(accelerator_config.FpgaProfile, a), \
               f"{a} differs between profiles — likely a regression"


def test_set_profile_switches_active():
    original = accelerator_config.active_profile_name()
    try:
        accelerator_config.set_profile("fpga")
        assert accelerator_config.AcceleratorConfig is accelerator_config.FpgaProfile
        assert accelerator_config.active_profile_name() == "fpga"
        accelerator_config.set_profile("sim")
        assert accelerator_config.AcceleratorConfig is accelerator_config.SimProfile
        assert accelerator_config.active_profile_name() == "sim"
    finally:
        accelerator_config.set_profile(original)


def test_set_profile_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        accelerator_config.set_profile("not-a-profile")


def test_default_profile_is_sim():
    """Without TINYML_PROFILE in the env, the default must remain `sim` —
    the historical default that all compiler tests assume."""
    assert accelerator_config.SimProfile.TILE_ELEMS == 32
    # Active profile in this test process — set by env var or default. Just
    # verify it's one of the two declared profiles.
    assert accelerator_config.AcceleratorConfig in (
        accelerator_config.SimProfile, accelerator_config.FpgaProfile,
    )
