"""Single source of truth for Ushqyn's 64-bit ISA.

Every consumer of opcode/field/bit-layout information reads from this file:

    compiler/assembler.py        — encodes mnemonic → 64-bit word
    compiler/disassembler.py     — decodes word → mnemonic
    compiler/golden_model.py     — extracts fields in i_decoder()
    tools/generate_i_decoder.py  — emits rtl/i_decoder.sv

Add or modify an opcode by editing the OPCODES list below, then re-run
`python tools/generate_i_decoder.py` to refresh the RTL decoder. The
round-trip test in compiler/test_isa_spec.py asserts that all three
Python paths agree on every opcode.

For each Field:
    name:  used as the assembly mnemonic argument name and as a Python
           dict key in golden_model.i_decoder.
    lo:    bit position (LSB) within the 64-bit instruction word.
    width: field width in bits.
    port:  RTL i_decoder output port driven when this opcode is active.
           Multiple opcodes can target the same port (muxed by case).

Field tuple order is the assembly mnemonic argument order:
    LOAD_V dest, addr, length     (matches Field("dest"), Field("addr"), Field("length"))

Instructions are 64-bit little-endian words; opcode lives in bits [4:0].
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Field:
    name: str
    lo: int
    width: int
    port: Optional[str] = None  # RTL i_decoder output port (None = no port)

    @property
    def hi(self) -> int:
        return self.lo + self.width - 1

    @property
    def mask(self) -> int:
        return (1 << self.width) - 1


@dataclass(frozen=True)
class Op:
    name: str
    value: int
    fields: tuple = ()
    description: str = ""


# RTL i_decoder output ports and their widths.
# Multiple Field.port entries across different opcodes drive the same port,
# muxed by the opcode case statement in i_decoder.sv.
RTL_PORTS = {
    "dest":           5,
    "length_or_cols": 18,
    "rows":           10,
    "addr":           24,
    "b":              5,
    "x":              5,
    "w":              5,
    "fmap_h":         6,
    "fmap_w":         6,
    "in_channels":    6,
    "out_channels":   6,
    "kernel_h":       4,
    "kernel_w":       4,
    "stride_val":     3,
    "pad_val":        3,
    "pool_size":      3,
    "relu_flag":      1,
}


OPCODES = (
    Op("NOP", 0x00, description="No-op (zero word terminates fetch)"),

    Op("LOAD_V", 0x01,
       description="Load vector from DRAM into a vector buffer",
       fields=(
           Field("dest",    5,  5, "dest"),
           Field("addr",   40, 24, "addr"),
           Field("length", 10, 18, "length_or_cols"),
       )),

    Op("LOAD_M", 0x02,
       description="Load matrix from DRAM into a matrix buffer (cols padded to TILE_ELEMS)",
       fields=(
           Field("dest",  5,  5, "dest"),
           Field("addr", 40, 24, "addr"),
           Field("rows", 20, 10, "rows"),
           Field("cols", 10, 10, "length_or_cols"),
       )),

    Op("STORE", 0x03,
       description="Write a vector buffer back to DRAM",
       fields=(
           Field("dest",    5,  5, "dest"),
           Field("addr",   40, 24, "addr"),
           Field("length", 10, 18, "length_or_cols"),
       )),

    Op("GEMV", 0x04,
       description="dest = quant(W*x + b), int32 accumulator, per-tensor max-abs scale",
       fields=(
           Field("dest",  5,  5, "dest"),
           Field("w",    40,  5, "w"),
           Field("x",    35,  5, "x"),
           Field("b",    30,  5, "b"),
           Field("rows", 20, 10, "rows"),
           Field("cols", 10, 10, "length_or_cols"),
       )),

    Op("RELU", 0x05,
       description="Element-wise max(0, x); FC outputs only (10-bit length)",
       fields=(
           Field("dest",    5,  5, "dest"),
           Field("x",      10,  5, "x"),
           Field("length", 20, 10, "length_or_cols"),
       )),

    Op("CONV2D_CFG", 0x06,
       description="Latch convolution geometry; precedes CONV2D_RUN",
       fields=(
           Field("dest",    5, 5, "dest"),
           Field("fmap_h", 10, 6, "fmap_h"),
           Field("fmap_w", 16, 6, "fmap_w"),
           Field("in_c",   22, 6, "in_channels"),
           Field("out_c",  28, 6, "out_channels"),
           Field("kh",     34, 4, "kernel_h"),
           Field("kw",     38, 4, "kernel_w"),
           Field("stride", 42, 3, "stride_val"),
           Field("pad",    45, 3, "pad_val"),
       )),

    Op("CONV2D_RUN", 0x07,
       description="Execute conv2d using last CONV2D_CFG; relu_flag fuses ReLU",
       fields=(
           Field("dest",       5, 5, "dest"),
           Field("x_id",      10, 5, "x"),
           Field("w_id",      15, 5, "w"),
           Field("b_id",      20, 5, "b"),
           Field("relu_flag", 25, 1, "relu_flag"),
       )),

    Op("MAXPOOL", 0x08,
       description="Sliding-window NCHW max-pool; operates on int8 directly",
       fields=(
           Field("dest",      5, 5, "dest"),
           Field("x_id",     10, 5, "x"),
           # NB: pool_size and stride at [17:15]/[20:18] precede the spatial fields
           # so the bit layout matches the historical encoding. Field tuple order
           # below is the assembly arg order, NOT the bit-layout order.
           Field("fmap_h",   21, 6, "fmap_h"),
           Field("fmap_w",   27, 6, "fmap_w"),
           Field("channels", 33, 5, "in_channels"),
           Field("pool_size",15, 3, "pool_size"),
           Field("stride",   18, 3, "stride_val"),
       )),
)


OPCODE_BY_NAME = {op.name: op for op in OPCODES}
OPCODE_BY_VALUE = {op.value: op for op in OPCODES}


def _validate():
    """Sanity-check the spec at module load: no bit overlaps, fields in range,
    ports declared, widths fit. Catches editing mistakes immediately rather
    than on the next round-trip test failure."""
    seen_values = set()
    for op in OPCODES:
        if op.value in seen_values:
            raise ValueError(f"Duplicate opcode value 0x{op.value:02X} ({op.name})")
        seen_values.add(op.value)

        # Reserve bits [4:0] for the opcode itself
        used = (1 << 5) - 1  # bits 0..4
        for f in op.fields:
            if f.lo < 5:
                raise ValueError(f"{op.name}.{f.name} overlaps opcode bits [4:0]")
            if f.hi >= 64:
                raise ValueError(f"{op.name}.{f.name} extends past bit 63")
            field_mask = ((1 << f.width) - 1) << f.lo
            if used & field_mask:
                raise ValueError(
                    f"{op.name}.{f.name} bits [{f.hi}:{f.lo}] overlap an earlier field"
                )
            used |= field_mask

            if f.port is not None:
                if f.port not in RTL_PORTS:
                    raise ValueError(
                        f"{op.name}.{f.name}: unknown RTL port '{f.port}'"
                    )
                if f.width > RTL_PORTS[f.port]:
                    raise ValueError(
                        f"{op.name}.{f.name} (width {f.width}) wider than "
                        f"port '{f.port}' (width {RTL_PORTS[f.port]})"
                    )


_validate()


# ── Encoding helpers (kept here so all paths share one implementation) ────────

def encode(op_name: str, **field_values) -> int:
    """Encode an instruction by opcode name and named field values.

    Use from anywhere that needs to manually build an instruction word
    (e.g. cocotb stimulus, tests, the in-process LOAD_V prolog in
    train_and_eval_cnn.py). Field names match the Field.name in OPCODES.

    >>> hex(encode("LOAD_V", dest=9, addr=0xC0, length=784))
    '0xc0000000c4ff31'
    """
    op = OPCODE_BY_NAME.get(op_name)
    if op is None:
        raise ValueError(f"Unknown opcode: {op_name}")
    if op.name == "NOP":
        return 0
    expected = {f.name for f in op.fields}
    given = set(field_values)
    if expected != given:
        missing = expected - given
        extra = given - expected
        raise ValueError(
            f"{op_name} expects fields {sorted(expected)}; "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )
    word = op.value & 0x1F
    for f in op.fields:
        v = int(field_values[f.name])
        if v < 0 or v > f.mask:
            raise ValueError(
                f"{op_name}.{f.name}={v} out of range [0, {f.mask}] (width {f.width})"
            )
        word |= (v & f.mask) << f.lo
    return word


def decode_fields(word: int) -> dict:
    """Extract every field of `word` as a {name: value} dict.

    Returns {} for NOP and unknown opcodes (caller checks opcode separately).
    """
    op = OPCODE_BY_VALUE.get(word & 0x1F)
    if op is None or op.name == "NOP":
        return {}
    return {f.name: (word >> f.lo) & f.mask for f in op.fields}
