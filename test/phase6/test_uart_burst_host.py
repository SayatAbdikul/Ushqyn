"""Host burst framing, sequence, bounds, and ambiguous-response checks."""

import binascii
import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/phase6"))
from uart_burst import BurstTiledClient, BURST_WRITE, FEATURES, burst_frame


class FakeSerial:
    def __init__(self):
        self.writes = []
        self.response = bytearray()
        self.truncate = False

    def write(self, packet):
        self.writes.append(packet)
        assert packet[:2] == b"\xa5\x5a"
        assert binascii.crc_hqx(packet[2:-2], 0xffff) == \
            int.from_bytes(packet[-2:], "little")
        command, sequence = packet[3:5]
        data = b"BW\x00\x01" if command == FEATURES else b""
        body = bytes([packet[2], command | 0x80, sequence]) + packet[5:8] + \
            bytes([len(data) + 1, 0, 0]) + data
        self.response.extend(b"\xa5\x5a" + body +
                             struct.pack("<H", binascii.crc_hqx(body, 0xffff)))
        if self.truncate:
            self.response.clear()
        return len(packet)

    def read(self, count):
        result = self.response[:count]
        del self.response[:count]
        return bytes(result)


class BurstHostTest(unittest.TestCase):
    def test_chunking_sequence_and_probe(self):
        serial = FakeSerial()
        client = BurstTiledClient(serial)
        with self.assertRaisesRegex(ValueError, "probe_burst"):
            client.write_external(0, b"x")
        self.assertEqual(client.probe_burst(), 256)
        data = bytes(i % 256 for i in range(513))
        client.write_external(123, data)
        frames = serial.writes[1:]
        self.assertEqual([packet[3] for packet in frames], [BURST_WRITE] * 3)
        self.assertEqual([packet[4] for packet in frames], [1, 2, 3])
        self.assertEqual([int.from_bytes(packet[5:8], "little") for packet in frames],
                         [0x80007b, 0x80017b, 0x80027b])
        self.assertEqual([int.from_bytes(packet[8:10], "little") for packet in frames],
                         [256, 256, 1])
        self.assertEqual(b"".join(packet[10:-2] for packet in frames), data)
        with self.assertRaisesRegex(ValueError, "exceeds SDRAM"):
            client.write_external(8 * 1024 * 1024 - 1, b"xx")

    def test_ambiguous_response_is_not_retried(self):
        serial = FakeSerial()
        client = BurstTiledClient(serial)
        client.probe_burst()
        serial.truncate = True
        with self.assertRaises(TimeoutError):
            client.write_external(0, b"payload")
        self.assertEqual(len(serial.writes), 2)

    def test_frame_rejects_crossing_24bit_boundary(self):
        with self.assertRaises(ValueError):
            burst_frame(0, 0xffffff, b"xx")


if __name__ == "__main__":
    unittest.main()
