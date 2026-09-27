"""Compiler schedule screen preflight and sample transfer checks."""
import struct
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/phase6'))
from screen_schedule import execute_verified_sample, prepare


class FakeClient:
    def __init__(self, corrupt_readback=False):
        self.external=bytearray(256)
        self.external[100:102]=b'\x05\x02'
        self.corrupt_readback=corrupt_readback
        self.writes=[]
    def write(self,address,data):
        self.writes.append((address,data))
    def write_external(self,address,data):
        self.external[address:address+len(data)]=data
    def read_external(self,address,length):
        result=bytes(self.external[address:address+length])
        if self.corrupt_readback and address==0:return b'\x00'*length
        return result
    def exchange(self,command):
        return b'\x00\x00'+struct.pack('<9I',*[0]*9)
    def read(self,address,length):
        counters=bytearray(32)
        counters[4:8]=b'SEQ4'
        struct.pack_into('<6I',counters,8,120,80,40,0,7,0)
        return bytes(counters)


class ScheduleScreen(unittest.TestCase):
    def test_preflight_selected_image_and_sibling_manifest(self):
        root=ROOT/'work/phase6/experiments-v1/sibling-input-retention'
        plan=prepare(root,variant='sibling')
        self.assertEqual(plan['planned'],10)
        self.assertEqual(plan['fixture_names']['vww']['pinned'],'vww-pinned-sibling-timed')
        self.assertEqual(plan['fixture_names']['kws']['stress'],'kws-stress-sibling-check')

    def test_input_readback_and_counters(self):
        schedule={'command_count':8,'final_output':{'ext':100,'bytes':2}}
        client=FakeClient()
        row=execute_verified_sample(client,schedule,b'\x01\x02',b'\x05\x02')
        self.assertTrue(row['input_readback_verified'])
        self.assertEqual(row['elapsed_cycles'],120)
        self.assertEqual(row['output_hex'],'0502')
        with self.assertRaisesRegex(AssertionError,'input SDRAM readback mismatch'):
            execute_verified_sample(FakeClient(True),schedule,b'\x01\x02',b'\x05\x02')


if __name__=='__main__':unittest.main()
