"""Fail-closed checks for short-screen accounting and guarded transport."""
import binascii
import copy
import struct
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/phase6')]
from quick_experiments import summarize
from host_batch_experiment import BatchedClient


class FakeUART:
    def __init__(self, corrupt=False):
        self.responses=bytearray();self.memory={};self.corrupt=corrupt
    def write(self,data):
        offset=0
        while offset<len(data):
            if data[offset]==0:offset+=1;continue
            assert data[offset:offset+2]==b'\xa5\x5a'
            size=int.from_bytes(data[offset+8:offset+10],'little')
            packet=data[offset:offset+12+size]
            assert binascii.crc_hqx(packet[2:-2],0xffff)==int.from_bytes(packet[-2:],'little')
            address=int.from_bytes(packet[5:8],'little')
            self.memory[address]=packet[10:-2]
            body=bytes([packet[2],packet[3]|128,packet[4]^(1 if self.corrupt else 0)])+packet[5:8]+struct.pack('<H',1)+b'\0'
            self.responses.extend(b'\xa5\x5a'+body+struct.pack('<H',binascii.crc_hqx(body,0xffff)))
            offset+=12+size
        return len(data)
    def read(self,size):
        data=bytes(self.responses[:size]);del self.responses[:size];return data


class ExperimentRunners(unittest.TestCase):
    def rows(self):
        return [dict(variant='trial',model=model,kind=kind,repeat=repeat,
            elapsed_cycles=100+repeat,device_latency_ms=(100+repeat)/22500,
            output_hex='13',expected_hex='13')
            for model in ('kws','vww') for kind,repeat in
            [('stress',0),('warmup',0),('timed',1),('timed',2),('timed',3)]]
    def test_clock_specific_median(self):
        result=summarize(self.rows(),['trial'])['trial']['kws']
        self.assertEqual(result['median_cycles'],102)
        self.assertEqual(result['median_ms'],102/22500)
    def test_short_or_duplicate_or_mismatch_rejected(self):
        for mutation in ('short','duplicate','mismatch'):
            rows=self.rows()
            if mutation=='short':rows.pop()
            elif mutation=='duplicate':rows[-1]['repeat']=2
            else:rows[0]['output_hex']='12'
            with self.assertRaises(ValueError):summarize(rows,['trial'])
    def test_guard_batch_tails_and_sequence_wrap(self):
        uart=FakeUART();client=BatchedClient(uart);client.sequence=254
        data=bytes(i%256 for i in range(1027))
        client.write_guarded(0xf00000,data)
        self.assertEqual(b''.join(uart.memory[a] for a in sorted(uart.memory)),data)
        self.assertEqual(client.sequence,(254+17)&255)
        self.assertFalse(uart.responses)
    def test_bad_response_and_control_writes_rejected(self):
        with self.assertRaises(ValueError):BatchedClient(FakeUART(True)).write_guarded(0xf00000,b'abc')
        for address in (0,0x410000,0xffffff):
            with self.assertRaises(ValueError):BatchedClient(FakeUART()).write_guarded(address,b'abc')


if __name__=='__main__':unittest.main()
