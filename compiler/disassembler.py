"""Disassembler for Ushqyn's 64-bit ISA.

Bit layouts come from `compiler/isa_spec.py` — the same source the
assembler and golden model read. Field values render as decimal except
`addr`, which is rendered as hex (matches the assembly mnemonic style
that compile.py emits).
"""
from isa_spec import OPCODE_BY_VALUE


def decode_instruction(word):
    """Decode a hex-string instruction word back into its assembly mnemonic.

    The output round-trips through `assembler.assemble_line(...)` for any
    instruction that was originally produced by the assembler — round-trip
    coverage lives in `compiler/test_isa_spec.py`.
    """
    instr = int(word, 16)
    opcode = instr & 0x1F
    op = OPCODE_BY_VALUE.get(opcode)
    if op is None:
        return f"UNKNOWN_OPCODE_{opcode:02X}"
    if op.name == "NOP":
        return "NOP"

    args = []
    for f in op.fields:
        v = (instr >> f.lo) & f.mask
        # `addr` reads more naturally as hex in disassembled output; everything
        # else (sizes, buffer IDs, geometry) reads as decimal.
        args.append(f"0x{v:X}" if f.name == "addr" else str(v))
    return f"{op.name} {', '.join(args)}"


def disassemble_file(hex_file, out_file="disassembled.asm"):
    with open(hex_file) as f:
        lines = [line.strip() for line in f if line.strip()]

    with open(out_file, "w") as out:
        out.write("; Disassembled code\n\n")
        for i, line in enumerate(lines):
            decoded = decode_instruction(line)
            out.write(f"{decoded}\n")
            print(f"{i:02}: {line} -> {decoded}")

    print(f"\n✅ Disassembly complete: {out_file}")


if __name__ == "__main__":
    disassemble_file("program.hex")
