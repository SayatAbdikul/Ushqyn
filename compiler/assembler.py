"""Assembler for the tinyML accelerator's 64-bit ISA.

Instructions are little-endian 64-bit words; opcode is in bits [4:0].
All bit layouts come from `compiler/isa_spec.py` — that file is the
single source of truth shared with disassembler.py, golden_model.py
and (via tools/generate_i_decoder.py) rtl/i_decoder.sv.

To add or change an opcode: edit isa_spec.OPCODES and re-run the RTL
decoder generator. Do NOT add per-opcode bit-shifting branches here.
"""
import re
import numpy as np
from dram import write_to_dram
from isa_spec import OPCODE_BY_NAME, OPCODES

# Re-exported for callers that historically did `from assembler import OPCODES`
# (a few cocotb tests imported the dict for opcode value lookups). The new
# canonical access is `isa_spec.OPCODE_BY_NAME[name].value`, but this preserves
# the old API.
OPCODES_DICT = {op.name: op.value for op in OPCODES}
OPNAMES = {op.value: op.name for op in OPCODES}
# Keep the original attribute name `OPCODES` for backwards compat. The dict
# spec object lives at module level — anything wanting the new spec imports
# from isa_spec directly.
OPCODES = OPCODES_DICT  # noqa: F811 — intentional shadow for back-compat


def assemble_line(line):
    """Encode one assembly line into a 16-char hex string (8 bytes, MSB-first).

    Returns None for blank/comment lines so callers can filter them out.
    Raises ValueError on unknown mnemonics or out-of-range field values.
    """
    parts = re.split(r'[,\s]+', line.strip())
    if not parts or parts[0].startswith(";") or parts[0] == "":
        return None

    name = parts[0]
    op = OPCODE_BY_NAME.get(name)
    if op is None:
        raise ValueError(f"Unknown instruction: {name}")

    if op.name == "NOP":
        return f"{0:016X}"

    args = [int(x, 0) for x in parts[1:]]
    if len(args) != len(op.fields):
        raise ValueError(
            f"{name} expects {len(op.fields)} args "
            f"({', '.join(f.name for f in op.fields)}); got {len(args)}: {parts[1:]}"
        )

    word = op.value & 0x1F
    for f, v in zip(op.fields, args):
        if v < 0 or v > f.mask:
            raise ValueError(
                f"{name}.{f.name}={v} out of range [0, {f.mask}] (width {f.width})"
            )
        word |= (v & f.mask) << f.lo
    return f"{word:016X}"


def assemble_file(asm_file, output_file=None):
    with open(asm_file) as f:
        lines = f.readlines()

    machine_code = []
    hex_lines = []
    for line in lines:
        encoded = assemble_line(line)
        if encoded:
            machine_code.extend(int(encoded[i:i + 2], 16) for i in range(0, len(encoded), 2))
            hex_lines.append(encoded)

    machine_code_np = np.array(machine_code, dtype=np.uint8).view(np.int8)
    write_to_dram(machine_code_np, 0)  # Write at the starting point 0

    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(hex_lines) + '\n')


if __name__ == "__main__":
    asm_file = "model_assembly.asm"
    assemble_file(asm_file)
