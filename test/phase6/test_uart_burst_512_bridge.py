"""Burst WRITE through the complete tiled bridge and stalled SDRAM port."""

import binascii
import random

import cocotb
from cocotb.triggers import Timer


def packet(command, sequence, address=0, data=b"", length=0):
    if command == 9:
        length = len(data)
    body = bytes([2, command, sequence]) + address.to_bytes(3, "little") + \
        length.to_bytes(2, "little") + data
    return b"\xa5\x5a" + body + binascii.crc_hqx(body, 0xffff).to_bytes(2, "little")


@cocotb.test()
async def burst_crosses_bridge_to_external_memory(d):
    memory = bytearray(8 * 1024 * 1024)
    rng = random.Random(630)
    response = bytearray()
    pending = None
    held = None
    d.clk.value = 0
    d.rst_n.value = 0
    d.rx_valid.value = 0
    d.rx_data.value = 0
    d.tx_ready.value = 1
    d.ext_ready.value = 0
    d.ext_rvalid.value = 0
    d.ext_rdata.value = 0
    d.memory_initialized.value = 1
    d.memory_port_busy.value = 0

    async def step(byte=None):
        nonlocal pending, held
        d.clk.value = 0
        d.rx_valid.value = byte is not None
        d.rx_data.value = 0 if byte is None else byte
        ready = pending is None and rng.randrange(5) != 0
        d.ext_ready.value = ready
        d.ext_rvalid.value = pending is not None
        d.ext_rdata.value = pending or 0
        pending = None
        await Timer(5, units="ns")
        if int(d.tx_valid.value):
            response.append(int(d.tx_data.value))
        req = (int(d.ext_addr.value), int(d.ext_wr.value),
               int(d.ext_wdata.value), int(d.ext_wstrb.value)) \
            if int(d.ext_req.value) else None
        if held is not None:
            assert req == held, "external request changed during a stall"
        held = req if req is not None and not ready else None
        if req is not None and ready:
            address, write, data, mask = req
            assert address % 8 == 0 and address + 8 <= len(memory)
            if write:
                for lane in range(8):
                    if mask & (1 << lane):
                        memory[address + lane] = (data >> (lane * 8)) & 255
            else:
                pending = int.from_bytes(memory[address:address + 8], "little")
        d.clk.value = 1
        await Timer(5, units="ns")

    async def exchange(command, sequence, address=0, data=b"", length=0):
        assert not response
        for byte in packet(command, sequence, address, data, length):
            await step(byte)
        for _ in range(10000):
            if len(response) >= 10:
                size = int.from_bytes(response[8:10], "little")
                if len(response) == size + 12:
                    result = bytes(response)
                    response.clear()
                    assert result[:3] == b"\xa5\x5a\x02"
                    assert result[3:5] == bytes([command | 0x80, sequence])
                    assert binascii.crc_hqx(result[2:-2], 0xffff) == \
                        int.from_bytes(result[-2:], "little")
                    await step()
                    return result[10:-2]
            await step()
        raise AssertionError("tiled bridge response timeout")

    await step()
    d.rst_n.value = 1
    await step()
    assert await exchange(8, 1) == b"\x00BW\x00\x02"
    base = 0x800121
    data = bytes((i * 71 + 7) & 255 for i in range(512))
    assert await exchange(9, 2, base, data) == b"\x00"
    assert memory[0x121:0x121 + 512] == data
    for offset in range(0, 512, 64):
        result = await exchange(2, 3 + offset // 64, base + offset, length=64)
        assert result == b"\x00" + data[offset:offset + 64]
    tail = bytes(i & 255 for i in range(512))
    assert await exchange(9, 17, 0xfffe00, tail) == b"\x00"
    assert memory[-512:] == tail
    assert await exchange(9, 18, 0xfffe01, tail) == b"\x04"
    assert memory[-512:] == tail
