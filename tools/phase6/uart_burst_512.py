#!/usr/bin/env python3
"""Opt-in 512-byte UART burst WRITE client for a matching experimental image.

The legacy protocol remains available. BURST_WRITE is used only for idle
external SDRAM writes after the board answers the read-only FEATURES probe.
There is no automatic retry after an ambiguous response.
"""

import binascii
import struct

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools/phase2"), str(ROOT / "tools/phase4")]
from host import TARGET, parse_response
from tiled_host import TiledClient, EXT_BASE, EXT_BYTES

FEATURES = 8
BURST_WRITE = 9
BURST_BYTES = 512


def burst_frame(sequence: int, address: int, data: bytes) -> bytes:
    """Construct the same CRC16 frame as v2 WRITE, with the burst opcode."""
    if not 0 <= sequence <= 255 or not EXT_BASE <= address < 1 << 24:
        raise ValueError("invalid burst sequence or SDRAM address")
    if not 1 <= len(data) <= BURST_BYTES or address + len(data) > 1 << 24:
        raise ValueError("invalid burst length or SDRAM boundary")
    body = bytes([TARGET["protocol_version"], BURST_WRITE, sequence]) + \
        address.to_bytes(3, "little") + struct.pack("<H", len(data)) + data
    return b"\xa5\x5a" + body + struct.pack("<H", binascii.crc_hqx(body, 0xffff))


class BurstTiledClient(TiledClient):
    def __init__(self, serial):
        super().__init__(serial)
        self._burst_probed = False

    def capabilities(self):
        capacity = super().capabilities()
        self.probe_burst()
        return capacity

    def probe_burst(self):
        """Read-only compatibility check; a legacy image will return status 5."""
        data = self.exchange(FEATURES)
        if data != b"BW" + BURST_BYTES.to_bytes(2, "little"):
            raise ValueError("incompatible UART burst feature response")
        self._burst_probed = True
        return BURST_BYTES

    def write_external(self, offset, data):
        if not self._burst_probed:
            raise ValueError("probe_burst must validate the experimental image first")
        if offset < 0 or offset + len(data) > EXT_BYTES:
            raise ValueError("external write exceeds SDRAM")
        for start in range(0, len(data), BURST_BYTES):
            chunk = data[start:start + BURST_BYTES]
            sequence = self.sequence
            address = EXT_BASE + offset + start
            packet = burst_frame(sequence, address, chunk)
            self.sequence = (sequence + 1) & 255
            if self.serial.write(packet) != len(packet):
                raise IOError("partial burst write; inspect device before retry")
            header = self.serial.read(10)
            if len(header) != 10:
                raise TimeoutError("truncated burst response; inspect device before retry")
            size = int.from_bytes(header[8:10], "little")
            if size != 1:
                raise ValueError("unexpected burst response length")
            tail = self.serial.read(size + 2)
            if len(tail) != size + 2:
                raise TimeoutError("truncated burst response; inspect device before retry")
            result = parse_response(header + tail)
            if (result["command"], result["sequence"], result["address"]) != \
                    (BURST_WRITE, sequence, address):
                raise ValueError("burst response correlation")
            if result["status"]:
                raise RuntimeError(f"device burst status {result['status']}")
