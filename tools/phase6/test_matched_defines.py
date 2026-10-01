"""Focused differential and malformed-cache tests for the DeFiNES backend."""
import copy
import hashlib
from pathlib import Path
import struct
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'compiler'),str(ROOT/'tools/phase6')]
import numpy as np
from hardware_v2 import Descriptor
from integer_reference import evaluate
from scheduler.matched_defines import compile_stack,replay_stack
from test_scheduler_spatial import spatial_fixture


class CacheBackendTests(unittest.TestCase):
    def test_substack_layer_selection_preserves_original_quantization(self):
        program,x=spatial_fixture(seed=13,groups=2,zp=-37)
        oracle=evaluate(program,{program.inputs[0]:x})
        for start,stop in ((0,2),(2,4),(0,4)):
            inp=program.layers[start].inputs[0]
            source=x if inp==program.inputs[0] else oracle[inp]
            shape=program.tensors[program.layers[stop-1].output].shape
            for mode in (1,2,3):
                code,payload,record=compile_stack(program,start,stop,3,shape[3],mode)
                got=replay_stack(program,start,stop,source,code,payload,record)
                self.assertEqual(got['output_sha256'],hashlib.sha256(
                    oracle[program.layers[stop-1].output].tobytes()).hexdigest())

    def test_same_compiled_cache_schedule_handles_new_input_values(self):
        program,pinned=spatial_fixture(seed=6,groups=1,zp=-128)
        code,payload,record=compile_stack(program,0,4,3,4,3)
        self.assertGreater(record['counters']['cache_read_bytes'],0)
        # Include saturated extremes and a new random sample. Every invocation
        # replays the full input/cache program against a fresh independent oracle.
        values=(pinned,np.full_like(pinned,-128),np.full_like(pinned,127),
                np.random.default_rng(9983).integers(-128,128,pinned.shape,dtype=np.int8))
        for source in values:
            self.assertEqual(replay_stack(program,0,4,source,code,payload,record)['status'],'passed')

    def test_prefetch_and_parameter_retention_preserve_outputs(self):
        program,source=spatial_fixture(seed=8,groups=1,zp=-37)
        signatures=[]
        for retain in (False,True):
            for prefetch in (False,True):
                code,payload,record=compile_stack(program,0,4,3,8,3,
                    retain_weights=retain,prefetch=prefetch)
                signatures.append(replay_stack(program,0,4,source,code,payload,record)['output_sha256'])
        self.assertEqual(len(set(signatures)),1)

    def test_dropped_horizontal_cache_write_is_detected(self):
        program,source=spatial_fixture(seed=5,groups=1,zp=-128)
        code,payload,record=compile_stack(program,0,4,5,4,2,prefetch=False)
        changed=bytearray(code);removed=0
        for row in record['runs']:
            if row['kind']!='copy':continue
            d=Descriptor.decode(bytes.fromhex(row['descriptor_hex']))
            if any(tag=='h-next--1' and lo <= d.output < hi
                   for lo,hi,tag in row['live_ranges']):
                struct.pack_into('<BBHIII',changed,row['command']*16,3,3,0,0,0,0)
                removed+=1
        self.assertGreater(removed,0)
        damaged=copy.deepcopy(record)
        damaged['code_sha256']=hashlib.sha256(changed).hexdigest()
        with self.assertRaisesRegex(ValueError,'input mismatch|final rectangle result mismatch'):
            replay_stack(program,0,4,source,bytes(changed),payload,damaged)

    def test_copy_reads_are_aligned_disjoint_and_outside_descriptor_storage(self):
        program,_=spatial_fixture(seed=5,groups=1,zp=-37)
        _,_,record=compile_stack(program,0,4,3,4,3)
        copies=0
        for row in record['runs']:
            ranges=sorted((lo,hi) for lo,hi,_ in row['live_ranges'])
            self.assertTrue(all(128 <= lo < hi <= 32768 for lo,hi in ranges))
            self.assertTrue(all(left[1] <= right[0] for left,right in zip(ranges,ranges[1:])))
            if row['kind']!='copy':continue
            copies+=1;d=Descriptor.decode(bytes.fromhex(row['descriptor_hex']))
            self.assertEqual(d.input % 8,0)
            self.assertEqual(d.outputs,d.count)
            self.assertGreaterEqual(min(d.input,d.output),128)
            self.assertTrue(d.input+d.count <= d.output or d.output+d.outputs <= d.input)
            self.assertLessEqual(max(d.input+d.count,d.output+d.outputs),32768)
        self.assertGreater(copies,100)

    def test_unsupported_dilation_is_explicit(self):
        program,_=spatial_fixture(seed=5,dilation=2)
        with self.assertRaisesRegex(ValueError,'dilation-backend-gap'):
            compile_stack(program,0,4,1,1,3)


if __name__=='__main__':unittest.main()
