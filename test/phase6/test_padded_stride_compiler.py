"""Opt-in PACK schedule and independent replay regression."""
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/phase6'),str(ROOT/'compiler')]

from constant_filter import compile_constant_chain
from integer_reference import evaluate
from run_boardless import load_model
from scheduler.contract import IllegalSchedule
from scheduler.resident_verify import replay_resident


class PaddedStrideCompiler(unittest.TestCase):
    def _compile(self,name):
        program,value,_,_=load_model(name)
        commands,payload,schedule=compile_constant_chain(program,False,
            reuse_sibling_inputs=True,padded_pointwise=True)
        return program,value,commands,payload,schedule

    def test_selected_layers_replay_and_fallback(self):
        for name,expected in [('kws',[5,9,13,17]),('vww',[25,29])]:
            with self.subTest(model=name):
                program,value,commands,payload,schedule=self._compile(name)
                self.assertEqual([stage['layer'] for stage in schedule['stages']
                                  if 'padded_pack' in stage],expected)
                self.assertEqual(len(schedule['pack_contracts']),len(expected))
                oracle=evaluate(program,{program.inputs[0]:value})
                result=replay_resident(program,commands,payload,{program.inputs[0]:value},
                    run_contracts=schedule['run_contracts'],
                    constant_contracts=schedule.get('constant_contracts'),
                    pack_contracts=schedule['pack_contracts'],
                    final_output=schedule['final_output'],oracle=oracle)
                self.assertEqual(result['pack_runs'],len(expected))
                self.assertEqual(result['status'],'passed')

    def test_tampered_pack_geometry_rejected(self):
        program,value,commands,payload,schedule=self._compile('kws')
        first=next(stage for stage in schedule['stages'] if 'padded_pack' in stage)
        offset=first['padded_pack']['descriptor_load']['ext']
        bad=bytearray(payload)
        bad[offset+24]^=1  # PACK count field; no corresponding geometry change.
        with self.assertRaises((IllegalSchedule,ValueError)):
            replay_resident(program,commands,bytes(bad),{program.inputs[0]:value},
                run_contracts=schedule['run_contracts'],
                constant_contracts=schedule.get('constant_contracts'),
                pack_contracts=schedule['pack_contracts'],
                final_output=schedule['final_output'])

    def test_selected_baseline_bytes_unchanged_without_option(self):
        for name in ('kws','vww'):
            with self.subTest(model=name):
                program,_,_,_,_=self._compile(name)
                code,payload,_=compile_constant_chain(program,False,
                    reuse_sibling_inputs=True)
                fixture=(ROOT/'work/phase6/constant-sibling-v1/fixtures'/
                         f'{name}-pinned-constant-sibling-timed')
                self.assertEqual(code,(fixture/'commands.bin').read_bytes())
                self.assertEqual(payload,(fixture/'payload.bin').read_bytes())


if __name__=='__main__': unittest.main()
