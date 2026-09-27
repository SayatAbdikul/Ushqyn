"""Sibling input retention preserves bytes and is checked by independent replay."""
import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'tools/phase6'), str(ROOT/'compiler')]

from chain_resident import compile_chain, _input_load
from hardware_v2 import Descriptor
from integer_reference import evaluate
from run_boardless import load_model
from scheduler.contract import IllegalSchedule
from scheduler.resident_verify import replay_resident


class SiblingInputRetention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vww, cls.vww_input, _, _ = load_model('vww')
        cls.old_vww = compile_chain(cls.vww, False)
        cls.new_vww = compile_chain(cls.vww, False, reuse_sibling_inputs=True)

    def test_exact_transfer_reduction_and_kws_identity(self):
        old, new = self.old_vww[2], self.new_vww[2]
        removed = [
            (a['layer'], _input_load(a)['bytes'])
            for a, b in zip(old['stages'], new['stages'])
            if _input_load(a) is not None and _input_load(b) is None and not b['inplace']
        ]
        self.assertEqual(len(removed), 9)
        self.assertEqual(sum(n for _, n in removed), 144000)
        self.assertEqual(new['removed_sibling_input_bytes'], 144000)
        self.assertEqual(new['removed_transfer_bytes']-old['removed_transfer_bytes'], 144000)
        self.assertEqual(len(self.old_vww[0])-len(self.new_vww[0]), 9*32)
        kws, _, _, _ = load_model('kws')
        old_kws = compile_chain(kws, False)
        new_kws = compile_chain(kws, False, reuse_sibling_inputs=True)
        self.assertEqual((old_kws[0], old_kws[1]), (new_kws[0], new_kws[1]))
        self.assertEqual(new_kws[2]['removed_sibling_input_bytes'], 0)

    def test_replay_rejects_parameter_overwrite_of_retained_input(self):
        commands, payload, schedule = self.new_vww
        stages = schedule['stages']
        producer, activation, sibling = stages[:3]
        self.assertIsNone(_input_load(sibling))
        source = _input_load(producer)
        self.assertIsNotNone(source)
        activation_param = next(t for t in activation['loads'] if t['role'] == 'parameter')
        self.assertFalse(source['sram'] <= activation_param['sram'] < source['sram']+source['bytes'])

        altered_commands = bytearray(commands)
        changed = 0
        for i in range(len(commands)//16):
            op, flags, _, ext, sram, count = struct.unpack_from('<BBHIII', commands, i*16)
            if (op, flags, ext, sram, count) == (1, 1, activation_param['ext'],
                                                  activation_param['sram'], activation_param['bytes']):
                struct.pack_into('<I', altered_commands, i*16+8, source['sram'])
                changed += 1
        self.assertEqual(changed, 1)
        altered_payload = bytearray(payload)
        descriptor_load = next(t for t in activation['loads'] if t['role'] == 'descriptor')
        descriptor = Descriptor.decode(bytes(altered_payload[descriptor_load['ext']:descriptor_load['ext']+64]))
        descriptor.params = source['sram']
        descriptor.validate()
        altered_payload[descriptor_load['ext']:descriptor_load['ext']+64] = descriptor.encode()
        oracle = evaluate(self.vww, {self.vww.inputs[0]: self.vww_input})
        with self.assertRaisesRegex(IllegalSchedule, 'incorrect operand bytes|stale tensor version'):
            replay_resident(self.vww, bytes(altered_commands), bytes(altered_payload),
                {self.vww.inputs[0]: self.vww_input},
                run_contracts=schedule['run_contracts'], final_output=schedule['final_output'],
                oracle=oracle)


if __name__ == '__main__': unittest.main()
