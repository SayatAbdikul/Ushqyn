""" Disassembler for a custom architecture based on a simplified instruction set.

Bit layouts mirror compiler/assembler.py and compiler/golden_model.py::i_decoder.
"""


def decode_instruction(word):
    instr = int(word, 16)
    opcode = instr & 0x1F  # bits [4:0]

    if opcode == 0x00:
        return "NOP"

    elif opcode == 0x01 or opcode == 0x03:  # LOAD_V or STORE
        # length: 18 bits at [27:10] — matches golden_model.i_decoder.
        dest   = (instr >>  5) & 0x1F
        length = (instr >> 10) & 0x3FFFF
        addr   = (instr >> 40) & 0xFFFFFF
        name   = "LOAD_V" if opcode == 0x01 else "STORE"
        return f"{name} {dest}, 0x{addr:X}, {length}"

    elif opcode == 0x02:  # LOAD_M
        dest = (instr >>  5) & 0x1F
        cols = (instr >> 10) & 0x3FF
        rows = (instr >> 20) & 0x3FF
        addr = (instr >> 40) & 0xFFFFFF
        return f"LOAD_M {dest}, 0x{addr:X}, {rows}, {cols}"

    elif opcode == 0x04:  # GEMV
        dest = (instr >>  5) & 0x1F
        cols = (instr >> 10) & 0x3FF
        rows = (instr >> 20) & 0x3FF
        b    = (instr >> 30) & 0x1F
        x    = (instr >> 35) & 0x1F
        w    = (instr >> 40) & 0x1F
        return f"GEMV {dest}, {w}, {x}, {b}, {rows}, {cols}"

    elif opcode == 0x05:  # RELU
        dest   = (instr >>  5) & 0x1F
        x      = (instr >> 10) & 0x1F
        length = (instr >> 20) & 0x3FF
        return f"RELU {dest}, {x}, {length}"

    elif opcode == 0x06:  # CONV2D_CFG
        # Bit layout (matches assembler.py / i_decoder):
        #   [ 4: 0] opcode    [ 9: 5] dest
        #   [15:10] fmap_h(6) [21:16] fmap_w(6)
        #   [27:22] in_c(6)   [33:28] out_c(6)
        #   [37:34] kh(4)     [41:38] kw(4)
        #   [44:42] stride(3) [47:45] pad(3)
        dest   = (instr >>  5) & 0x1F
        fmap_h = (instr >> 10) & 0x3F
        fmap_w = (instr >> 16) & 0x3F
        in_c   = (instr >> 22) & 0x3F
        out_c  = (instr >> 28) & 0x3F
        kh     = (instr >> 34) & 0x0F
        kw     = (instr >> 38) & 0x0F
        stride = (instr >> 42) & 0x07
        pad    = (instr >> 45) & 0x07
        return (f"CONV2D_CFG {dest}, {fmap_h}, {fmap_w}, {in_c}, {out_c}, "
                f"{kh}, {kw}, {stride}, {pad}")

    elif opcode == 0x07:  # CONV2D_RUN
        # Bit layout:
        #   [ 4: 0] opcode  [ 9: 5] dest
        #   [14:10] x_id    [19:15] w_id
        #   [24:20] b_id    [25] relu_flag
        dest      = (instr >>  5) & 0x1F
        x_id      = (instr >> 10) & 0x1F
        w_id      = (instr >> 15) & 0x1F
        b_id      = (instr >> 20) & 0x1F
        relu_flag = (instr >> 25) & 0x01
        return f"CONV2D_RUN {dest}, {x_id}, {w_id}, {b_id}, {relu_flag}"

    elif opcode == 0x08:  # MAXPOOL
        # Bit layout:
        #   [ 4: 0] opcode    [ 9: 5] dest    [14:10] x_id
        #   [17:15] pool_size(3)  [20:18] stride(3)
        #   [26:21] fmap_h(6) [32:27] fmap_w(6) [37:33] channels(5)
        dest      = (instr >>  5) & 0x1F
        x_id      = (instr >> 10) & 0x1F
        pool_size = (instr >> 15) & 0x07
        stride    = (instr >> 18) & 0x07
        fmap_h    = (instr >> 21) & 0x3F
        fmap_w    = (instr >> 27) & 0x3F
        channels  = (instr >> 33) & 0x1F
        return (f"MAXPOOL {dest}, {x_id}, {fmap_h}, {fmap_w}, "
                f"{channels}, {pool_size}, {stride}")

    else:
        return f"UNKNOWN_OPCODE_{opcode:02X}"


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


# Example usage
if __name__ == "__main__":
    disassemble_file("program.hex")
