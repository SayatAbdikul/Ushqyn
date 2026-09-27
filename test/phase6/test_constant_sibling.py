"""Constant filter lowering composes with sibling input retention."""
import hashlib
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/phase6'),str(ROOT/'compiler')]
from constant_filter import compile_constant_chain
from integer_reference import evaluate
from run_boardless import load_model
from scheduler.resident_verify import replay_resident


class ConstantSibling(unittest.TestCase):
    def test_existing_constant_fixture_identity(self):
        for name in ('kws','vww'):
            program,_,_,_=load_model(name)
            code,payload,_=compile_constant_chain(program,False)
            fixture=ROOT/f'work/phase6/constant-filter-v1/fixtures/{name}-pinned-constant-timed'
            self.assertEqual(hashlib.sha256(code).digest(),
                             hashlib.sha256((fixture/'commands.bin').read_bytes()).digest())
            self.assertEqual(hashlib.sha256(payload).digest(),
                             hashlib.sha256((fixture/'payload.bin').read_bytes()).digest())

    def test_composed_vww_replay_and_savings(self):
        program,value,_,_=load_model('vww')
        old_code,_,old=compile_constant_chain(program,False)
        code,payload,schedule=compile_constant_chain(program,False,reuse_sibling_inputs=True)
        self.assertEqual(schedule['removed_sibling_input_bytes'],144000)
        self.assertEqual(schedule['constant_filter']['skipped_dense_macs'],3131136)
        self.assertEqual(old['constant_filter']['skipped_dense_macs'],3131136)
        self.assertEqual(len(old_code)-len(code),9*32)
        oracle=evaluate(program,{program.inputs[0]:value})
        verified=replay_resident(program,code,payload,{program.inputs[0]:value},
            run_contracts=schedule['run_contracts'],
            constant_contracts=schedule['constant_contracts'],
            final_output=schedule['final_output'],oracle=oracle)
        self.assertEqual(verified['status'],'passed')


if __name__=='__main__':unittest.main()
