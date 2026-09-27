#!/usr/bin/env python3
"""Concrete compiler/BRAM fusion feasibility probe; no executable fused image.

The proposed opt-in descriptor tag uses bits63:49 as an aligned SRAM LUT
pointer and bit48 as enable. It keeps the descriptor64bytes long. The baseline
ABI rejects this tag; an explicit engine capability bit is required before
emitting it in an executable schedule. This probe emits tagged descriptors,
the exact256-byte maps, allocation proofs, and standalone epilogue RTL tests.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np

from variants import ROOT, check_frozen, sha
from run_screening import save_json
sys.path.insert(0, str(ROOT/'compiler'))
from hardware_v2 import Descriptor
from quantization import requantize

BASE = ROOT/'work/phase6/experiments-v1/output-fusion-prototype-v1'
RTL = ROOT/'rtl/phase6/output_pipeline_fused_epilogue.sv'


def encode_fused(descriptor, table_address):
    if descriptor.opcode not in (4, 6):
        raise ValueError('only Conv/DW producers are supported by prototype')
    if table_address % 8 or not 0 <= table_address <= 32768-256:
        raise ValueError('table must be aligned and fit the SRAM')
    data = bytearray(descriptor.encode())
    struct.pack_into('<H', data, 6, ((table_address//8) << 1) | 1)
    return bytes(data)


def decode_fused(data):
    raw = bytearray(data)
    tag = struct.unpack_from('<H', raw, 6)[0]
    if not tag & 1:
        raise ValueError('fusion tag missing')
    address = (tag >> 1)*8
    raw[6:8] = bytes(2)
    desc = Descriptor.decode(bytes(raw))
    if encode_fused(desc, address) != data:
        raise ValueError('invalid fusion encoding')
    return desc, address


def activation_table(op, params):
    bias, multiplier, shift, zy, zx, lower, upper, reserved = struct.unpack(
        '<iiBbbbbb', params[:14])
    assert bias == 0 and reserved == 0 and params[14:] == bytes(2)
    assert multiplier > 0 and 0 <= shift <= 62
    if op == 2:
        assert lower == zx and upper == 127
    elif op != 8:
        raise ValueError('unsupported activation')
    result = []
    for code in range(256):
        x = code if code < 128 else code-256
        centered = max(lower, min(upper, x))-zx
        product = centered*multiplier
        quotient = abs(product) >> shift
        if shift and (abs(product) & ((1 << shift)-1)) >= 1 << (shift-1):
            quotient += 1
        value = (-quotient if product < 0 else quotient)+zy
        result.append(max(-128, min(127, value)) & 255)
    # Independent production reference checks every possible producer INT8.
    values = np.arange(256, dtype=np.uint8).view(np.int8).astype(np.int64)
    expected = requantize(np.clip(values, lower, upper)-zx,
                          multiplier, shift, zy).astype(np.int8).tobytes()
    assert bytes(result) == expected
    return bytes(result)


def compile_probe():
    check_frozen()
    BASE.mkdir(parents=True, exist_ok=True)
    report = dict(status='passed-feasibility', physical_board=False,
        executable_full_model=False, descriptor_bytes=64,
        required_engine_capability='exact_activation_epilogue_v1',
        table_bytes=256, table_bram_copies=1,
        integration_requirement='reuse existing activation LUT; add tagged descriptor validation and table loading',
        limitations=['No full-engine fused implementation or routed resource proof.',
            'VWW constant fills must be transformed through the same map before removing activation.',
            'Cycle savings require integrated full-schedule measurement.'], models={}, cases=[])
    for model in ('kws', 'vww'):
        directory = ROOT/f'work/phase6/constant-sibling-v1/fixtures/{model}-pinned-constant-sibling-timed'
        schedule = json.loads((directory/'schedule.json').read_text())
        payload = (directory/'payload.bin').read_bytes()
        stages = schedule['stages']
        rows = []
        for index, activation in enumerate(stages):
            if not index or 'descriptor_hex' not in activation:
                continue
            ad = Descriptor.decode(bytes.fromhex(activation['descriptor_hex']))
            if ad.opcode not in (2, 8):
                continue
            producer = stages[index-1]
            pd = Descriptor.decode(bytes.fromhex(producer['descriptor_hex']))
            assert pd.opcode in (4, 6) and pd.output == ad.input == ad.output
            assert pd.outputs == ad.outputs
            load = next(load for load in activation['loads'] if load['sram'] == ad.params
                        and load['bytes'] == 16 and load['role'] == 'parameter')
            params = payload[load['ext']:load['ext']+16]
            table = activation_table(ad.opcode, params)
            # Conservative proof uses only the tail outside the complete live
            # interval; it never assumes dead holes inside retained tensors.
            address = (producer['live'][1]+7)//8*8
            fits = address+256 <= 32768
            key = f'{model}-stage-{index}'
            table_path = BASE/f'{key}.lut.bin'
            table_path.write_bytes(table)
            row = dict(key=key, producer_layer=producer['layer'],
                activation_layer=activation['layer'], outputs=ad.outputs,
                table_sram=address, live_interval=producer['live'],
                fits_existing_tile=fits, extra_sram_bytes=256,
                table=str(table_path.relative_to(ROOT)), table_sha256=sha(table_path),
                params_hex=params.hex(), activation_opcode=ad.opcode)
            # Keep the activation's final store after the fused producer.
            # Remove its descriptor/parameter DMA pairs and RUN/WAIT pair;
            # add one LUT DMA/WAIT pair before the producer's live runs.
            row['projected_command_reduction'] = 2*len(activation['loads']) if fits else 0
            if fits:
                encoded = encode_fused(pd, address)
                decoded, pointer = decode_fused(encoded)
                assert decoded == pd and pointer == address and len(encoded) == 64
                try:
                    Descriptor.decode(encoded)
                except ValueError:
                    pass
                else:
                    raise AssertionError('baseline must reject the new optional ABI')
                row['prototype_descriptor_hex'] = encoded.hex()
            rows.append(row)
        report['models'][model] = dict(pairs=len(rows), fit_without_retiling=sum(
            row['fits_existing_tile'] for row in rows), activation_output_bytes=sum(
            row['outputs'] for row in rows), table_upload_bytes=256*len(rows),
            min_tail_free_bytes=min(32768-row['table_sram'] for row in rows),
            current_commands=schedule['command_count'],
            projected_fused_commands=schedule['command_count']-sum(
                row['projected_command_reduction'] for row in rows),
            command_projection_is_executable=False,
            schedule_sha256=sha(directory/'schedule.json'), payload_sha256=sha(directory/'payload.bin'))
        report['cases'].extend(rows)
    report['sources'] = {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), RTL)}
    save_json(BASE/'compiler-report.json', report)
    return report


def rtl_test():
    from cocotb.runner import get_runner
    compile_probe()
    build = BASE/'rtl'
    runner = get_runner('verilator')
    runner.build(verilog_sources=[RTL], hdl_toplevel='phase6_fused_epilogue',
        build_dir=build, build_args=['--timing', '-Wno-fatal'], timescale=('1ns', '1ps'))
    paths = [str(ROOT/'test/phase6')]+sys.path
    runner.test(hdl_toplevel='phase6_fused_epilogue', test_module='test_output_fusion',
        test_dir=ROOT/'test/phase6', build_dir=build,
        results_xml=str(build/'results.xml'), extra_env={
            'PYTHONPATH': os.pathsep.join(paths), 'FUSION_PROBE': str(BASE/'compiler-report.json')})
    tests = ET.parse(build/'results.xml').findall('.//testcase')
    assert len(tests) == 1 and not any(t.find('failure') is not None or
        t.find('error') is not None for t in tests)
    result = dict(status='passed', physical_board=False, tests=1,
        compiler_report_sha256=sha(BASE/'compiler-report.json'),
        sources={str(p.relative_to(ROOT)): sha(p) for p in (RTL, ROOT/'test/phase6/test_output_fusion.py')},
        results_sha256=sha(build/'results.xml'))
    save_json(build/'report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('compiler', 'rtl'))
    args = parser.parse_args()
    print(json.dumps(compile_probe() if args.stage == 'compiler' else rtl_test(),
                     sort_keys=True), flush=True)
