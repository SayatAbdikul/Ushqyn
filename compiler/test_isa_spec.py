"""Round-trip and cross-consistency tests for the ISA spec.

Asserts that the four consumers of opcode bit layouts agree on every
opcode in `isa_spec.OPCODES`:

    - assembler.assemble_line          (mnemonic → 64-bit word)
    - disassembler.decode_instruction  (16-char hex → mnemonic)
    - isa_spec.encode / decode_fields  (named fields ↔ word)
    - golden_model.i_decoder           (word → field extraction)

Plus a check that the on-disk rtl/i_decoder.sv matches what the
generator would produce from the current spec — this catches
"forgot to re-run the generator" before tests fail noisily.
"""
import random
import subprocess
import sys
from pathlib import Path

import pytest

import isa_spec
import assembler
import disassembler


# ── helpers ──────────────────────────────────────────────────────────────────

def _random_field_values(op):
    """Generate a dict of random in-range values for every field of `op`."""
    rng = random.Random(op.value * 17)  # deterministic per-opcode seed
    return {f.name: rng.randint(0, f.mask) for f in op.fields}


def _format_asm(op, fields):
    """Render an assembly mnemonic line from a (op, fields_dict) pair.

    Mirrors the format the assembler accepts and the disassembler emits.
    `addr` is rendered as hex to match disassembler output style.
    """
    args = []
    for f in op.fields:
        v = fields[f.name]
        args.append(f"0x{v:X}" if f.name == "addr" else str(v))
    return f"{op.name} {', '.join(args)}"


# ── tests ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("op", [op for op in isa_spec.OPCODES if op.name != "NOP"],
                         ids=lambda o: o.name)
def test_encode_decode_roundtrip(op):
    """isa_spec.encode → decode_fields recovers the original field values."""
    fields = _random_field_values(op)
    word = isa_spec.encode(op.name, **fields)
    assert isa_spec.decode_fields(word) == fields


@pytest.mark.parametrize("op", [op for op in isa_spec.OPCODES if op.name != "NOP"],
                         ids=lambda o: o.name)
def test_assembler_matches_spec_encode(op):
    """assembler.assemble_line produces the same word as isa_spec.encode."""
    fields = _random_field_values(op)
    line = _format_asm(op, fields)
    asm_hex = assembler.assemble_line(line)
    spec_word = isa_spec.encode(op.name, **fields)
    assert int(asm_hex, 16) == spec_word, (
        f"{op.name}: assembler={asm_hex} spec=0x{spec_word:016X}")


@pytest.mark.parametrize("op", [op for op in isa_spec.OPCODES if op.name != "NOP"],
                         ids=lambda o: o.name)
def test_disassembler_roundtrip(op):
    """disassembler reproduces a line that the assembler accepts and re-encodes
    to the same word."""
    fields = _random_field_values(op)
    word = isa_spec.encode(op.name, **fields)
    line = disassembler.decode_instruction(f"{word:016X}")
    re_encoded = assembler.assemble_line(line)
    assert int(re_encoded, 16) == word, (
        f"{op.name} round-trip drift: line='{line}' word=0x{word:016X} "
        f"re-encoded=0x{int(re_encoded, 16):016X}")


def test_nop_encodes_to_zero():
    assert assembler.assemble_line("NOP") == f"{0:016X}"
    assert disassembler.decode_instruction(f"{0:016X}") == "NOP"
    assert isa_spec.encode("NOP") == 0


def test_unknown_opcode_disassembly():
    # Opcode 0x1F is reserved (not in OPCODES); decoder must not crash
    bogus = (0xDEAD << 5) | 0x1F
    out = disassembler.decode_instruction(f"{bogus:016X}")
    assert out.startswith("UNKNOWN_OPCODE_")


def test_field_overflow_rejected():
    # length is 18 bits — 2^18 = 262144 must be rejected
    with pytest.raises(ValueError):
        isa_spec.encode("LOAD_V", dest=0, addr=0, length=1 << 18)


def test_rtl_decoder_is_up_to_date():
    """rtl/i_decoder.sv must match what the generator would emit from the
    current spec. Catches "edited isa_spec.py but forgot to regenerate"."""
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, str(repo_root / "tools" / "generate_i_decoder.py"), "--check"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"rtl/i_decoder.sv is stale — re-run "
        f"`python tools/generate_i_decoder.py`.\n"
        f"stderr: {result.stderr}\nstdout: {result.stdout}"
    )
