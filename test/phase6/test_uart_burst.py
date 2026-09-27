"""Command-only simulation for the isolated 256-byte UART transfer extension."""

import binascii
import random
import struct

import cocotb
from cocotb.triggers import Timer


def frame(command, sequence=0, address=0, data=b"", length=0):
    if command in (3, 9):
        length = len(data)
    body = bytes([2, command, sequence]) + address.to_bytes(3, "little") + \
        struct.pack("<H", length) + data
    return b"\xa5\x5a" + body + struct.pack("<H", binascii.crc_hqx(body, 0xffff))


class Link:
    def __init__(self, dut):
        self.dut = dut
        self.memory = bytearray(8 * 1024 * 1024)
        self.random = random.Random(613)
        self.pending = None
        self.response = bytearray()

    async def step(self, byte=None, ready=True):
        dut = self.dut
        dut.clk.value = 0
        dut.rx_valid.value = byte is not None
        dut.rx_data.value = 0 if byte is None else byte
        dut.tx_ready.value = ready
        dut.mem_ready.value = self.random.randrange(5) != 0
        dut.mem_rvalid.value = self.pending is not None
        dut.mem_rdata.value = self.pending or 0
        self.pending = None
        await Timer(5, units="ns")
        if ready and int(dut.tx_valid.value):
            self.response.append(int(dut.tx_data.value))
        request = int(dut.mem_req.value) and int(dut.mem_ready.value)
        if request:
            base = int(dut.mem_addr.value)
            assert base & 7 == 0
            if int(dut.mem_wr.value):
                mask = int(dut.mem_wstrb.value)
                word = int(dut.mem_wdata.value)
                for lane in range(8):
                    if mask & (1 << lane):
                        address = base + lane
                        assert 0x800000 <= address < 0x1000000
                        self.memory[address - 0x800000] = (word >> (8 * lane)) & 255
            else:
                assert self.pending is None
                self.pending = int.from_bytes(
                    self.memory[base - 0x800000:base - 0x800000 + 8], "little")
        dut.clk.value = 1
        await Timer(5, units="ns")

    async def packet(self, contents, before_crc=None):
        assert not self.response
        for index, byte in enumerate(contents):
            if index == len(contents) - 2 and before_crc is not None:
                before_crc()
            await self.step(byte)
            if self.random.randrange(3) == 0:
                await self.step()
        for _ in range(2500):
            if len(self.response) >= 10:
                total = 12 + int.from_bytes(self.response[8:10], "little")
                if len(self.response) == total:
                    result = bytes(self.response)
                    self.response.clear()
                    assert binascii.crc_hqx(result[2:-2], 0xffff) == \
                        int.from_bytes(result[-2:], "little")
                    assert result[:3] == b"\xa5\x5a\x02"
                    await self.step()  # finish TX_GAP before the next frame
                    return result
            await self.step(ready=self.random.randrange(4) != 0)
        raise AssertionError("response timeout")


@cocotb.test()
async def crc_bounds_and_legacy_compatibility(dut):
    dut.clk.value = 0
    dut.rst_n.value = 0
    dut.rx_valid.value = 0
    dut.rx_data.value = 0
    dut.tx_ready.value = 0
    dut.busy.value = 0
    dut.core_error.value = 0
    dut.mmio_while_busy.value = 0
    for name in ("elapsed", "compute_cycles", "wait_cycles", "control_cycles",
                 "useful_macs", "read_bytes", "write_bytes", "layer_count"):
        getattr(dut, name).value = 0
    dut.mem_ready.value = 0
    dut.mem_rvalid.value = 0
    dut.mem_rdata.value = 0
    link = Link(dut)
    for _ in range(3):
        await link.step()
    dut.rst_n.value = 1
    await link.step()

    features = await link.packet(frame(8, 250))
    assert features[3:5] == bytes([0x88, 250])
    assert features[10:-2] == b"\x00BW\x00\x01"
    caps = await link.packet(frame(1, 251))
    assert caps[10] == 0 and caps[14] == 64  # legacy max transfer unchanged

    data = bytes((i * 37 + 13) & 255 for i in range(256))
    base = 0x800121
    assert not any(link.memory[base - 0x800000:base - 0x800000 + len(data)])
    reply = await link.packet(frame(9, 252, base, data), before_crc=lambda: (
        None if not any(link.memory[base - 0x800000:base - 0x800000 + len(data)])
        else (_ for _ in ()).throw(AssertionError("mutation before CRC"))))
    assert reply[10] == 0 and reply[3:5] == bytes([0x89, 252])
    assert link.memory[base - 0x800000:base - 0x800000 + len(data)] == data

    for sequence, address, payload in (
        (253, 0xffff00, bytes(range(256))),
        (254, 0xfffffd, b"\x13\x37\x99"),
        (255, 0x800000, b"\xa5"),
    ):
        response = await link.packet(frame(9, sequence, address, payload))
        assert response[10] == 0
        assert link.memory[address - 0x800000:address - 0x800000 + len(payload)] == payload

    # The bad CRC, invalid window, and oversized frame cannot change memory.
    snapshot = bytes(link.memory[base - 0x800000:base - 0x800000 + 256])
    corrupt = bytearray(frame(9, 0, base, bytes(256)))
    corrupt[-1] ^= 1
    assert (await link.packet(corrupt))[10] == 1
    assert link.memory[base - 0x800000:base - 0x800000 + 256] == snapshot
    assert (await link.packet(frame(9, 1, 0x7fffff, b"x")))[10] == 4
    assert (await link.packet(frame(9, 2, 0xffff01, bytes(256))))[10] == 4
    nested = frame(9, 3, base, b"evil")
    oversize = bytes(257 - len(nested)) + nested
    assert (await link.packet(frame(9, 4, base, oversize)))[10] == 4
    assert link.memory[base - 0x800000:base - 0x800000 + 256] == snapshot

    dut.busy.value = 1
    assert (await link.packet(frame(9, 5, base, b"busy")))[10] == 3
    dut.busy.value = 0
    assert (await link.packet(frame(3, 6, base, b"legacy")))[10] == 0
    readback = await link.packet(frame(2, 7, base, length=6))
    assert readback[10:-2] == b"\x00legacy"
    assert (await link.packet(frame(3, 8, base, bytes(65))))[10] == 4
