"""Assembler for the tinyML accelerator's 64-bit ISA.

Instructions are little-endian 64-bit words; opcode is in bits [4:0].
Per-opcode bit layouts are documented inline next to each `assemble_line`
branch below, and decoders for the same layouts live in
golden_model.py::i_decoder and disassembler.py::decode_instruction
— keep all three in sync when the ISA changes.
"""
import re
import numpy as np
from dram import write_to_dram
# === Define opcode mapping ===
OPCODES = {
    "LOAD_V":     0x01,
    "LOAD_M":     0x02,
    "STORE":      0x03,
    "GEMV":       0x04,
    "RELU":       0x05,
    "CONV2D_CFG": 0x06,  # Configures conv geometry (paired with CONV2D_RUN)
    "CONV2D_RUN": 0x07,  # Executes conv using last CONV2D_CFG configuration
    "MAXPOOL":    0x08,  # Max-pooling over a feature map buffer
    "NOP":        0x00
}

# Reverse map for disassembly
OPNAMES = {v: k for k, v in OPCODES.items()}

def assemble_line(line):
    parts = re.split(r'[,\s]+', line.strip()) # Split by commas or whitespace
    if not parts or parts[0].startswith(";") or parts[0] == "": # Ignore empty lines or comments
        return None

    instr = parts[0]
    opcode = OPCODES.get(instr)
    if opcode is None:
        raise ValueError(f"Unknown instruction: {instr}")

    if instr == "NOP":
        return f"{opcode:016X}000000"

    # Basic format: OPCODE, ARG1, ARG2, ...
    args = [int(x, 0) for x in parts[1:]]  # auto handles hex/dec

    if instr == "LOAD_V" or instr == "STORE":
        dest, addr, length = args
        word = (addr << 40) | (length << 10)  | (dest << 5) | opcode
        return f"{word:016X}"

    elif instr == "LOAD_M":
        dest, addr, rows, cols = args
        word = (addr << 40) | (rows << 20) | (cols << 10) | (dest << 5) | opcode
        return f"{word:016X}"

    elif instr == "GEMV":
        dest, w, x, b, rows, cols = args
        word = (w << 40) | (x << 35) | (b << 30) | (rows << 20) | (cols << 10) | (dest << 5) | opcode
        return f"{word:016X}"

    elif instr == "RELU":
        dest, x, length = args
        word = (length << 20) | (x << 10) | (dest << 5) | opcode
        return f"{word:016X}"

    elif instr == "CONV2D_CFG":
        # CONV2D_CFG dest, fmap_h, fmap_w, in_c, out_c, kh, kw, stride, pad
        # Bit layout:
        #   [ 4: 0] opcode   (5)
        #   [ 9: 5] dest     (5)
        #   [15:10] fmap_h   (6)  max 63
        #   [21:16] fmap_w   (6)  max 63
        #   [27:22] in_c     (6)  max 63
        #   [33:28] out_c    (6)  max 63
        #   [37:34] kh       (4)  max 15
        #   [41:38] kw       (4)  max 15
        #   [44:42] stride   (3)  max 7
        #   [47:45] pad      (3)  max 7
        #   [63:48] reserved
        dest, fmap_h, fmap_w, in_c, out_c, kh, kw, stride, pad = args
        word = (pad    << 45) | (stride << 42) | (kw    << 38) | (kh   << 34) | \
               (out_c  << 28) | (in_c   << 22) | (fmap_w<< 16) | (fmap_h << 10) | \
               (dest   <<  5) | opcode
        return f"{word:016X}"

    elif instr == "CONV2D_RUN":
        # CONV2D_RUN dest, x_id, w_id, b_id, relu_flag
        # Bit layout:
        #   [ 4: 0] opcode     (5)
        #   [ 9: 5] dest       (5)
        #   [14:10] x_id       (5)  input feature-map buffer
        #   [19:15] w_id       (5)  weight buffer (loaded via LOAD_M)
        #   [24:20] b_id       (5)  bias buffer  (loaded via LOAD_V)
        #   [25]    relu_flag  (1)  apply ReLU to output
        #   [63:26] reserved
        dest, x_id, w_id, b_id, relu_flag = args
        word = (relu_flag << 25) | (b_id << 20) | (w_id << 15) | \
               (x_id << 10) | (dest << 5) | opcode
        return f"{word:016X}"

    elif instr == "MAXPOOL":
        # MAXPOOL dest, x_id, fmap_h, fmap_w, channels, pool_size, stride
        # Bit layout:
        #   [ 4: 0] opcode    (5)
        #   [ 9: 5] dest      (5)
        #   [14:10] x_id      (5)  input feature-map buffer
        #   [17:15] pool_size (3)  kernel size (2 or 3)
        #   [20:18] stride    (3)  pooling stride
        #   [26:21] fmap_h    (6)  input spatial height
        #   [32:27] fmap_w    (6)  input spatial width
        #   [37:33] channels  (5)  number of feature-map channels
        #   [63:38] reserved
        dest, x_id, fmap_h, fmap_w, channels, pool_size, stride = args
        word = (channels  << 33) | (fmap_w   << 27) | (fmap_h    << 21) | \
               (stride    << 18) | (pool_size << 15) | (x_id      << 10) | \
               (dest      <<  5) | opcode
        return f"{word:016X}"

    else:
        raise NotImplementedError(f"Unsupported instruction: {instr}")

def assemble_file(asm_file, output_file=None):
    with open(asm_file) as f:
        lines = f.readlines()

    machine_code = []
    hex_lines = []
    for line in lines:
        encoded = assemble_line(line)
        # print(f"Encoding line: '{line.strip()}' -> '{encoded}'")
        if encoded:
            machine_code.extend(int(encoded[i:i+2], 16) for i in range(0, len(encoded), 2))  # Split into 8-char chunks
            hex_lines.append(encoded)

    machine_code_np = np.array(machine_code, dtype=np.uint8).view(np.int8)
    write_to_dram(machine_code_np, 0) # Write at the starting point 0

    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(hex_lines) + '\n')

    # print(f"✅ Assembled {len(machine_code)} instructions to DRAM")

if __name__ == "__main__":
    # Example usage
    asm_file = "model_assembly.asm"
    assemble_file(asm_file)
