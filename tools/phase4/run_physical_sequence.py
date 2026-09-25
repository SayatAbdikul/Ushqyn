#!/usr/bin/env python3
"""Validate every model node, then compare autonomous sequential/overlap runs."""

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import serial

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'compiler'))
from phase4_compile import compile_tiled
from phase4_sequence import compile_sequence
from static_pipeline import compile_static
from tiled_host import TiledClient
from host import RESET, STATUS, decode_status


class RecoverableUploadClient(TiledClient):
    """Recover idempotent setup/readback only; launch commands never retry.

    Some macOS FTDI sessions become invalid while the board remains powered.
    Reopening is allowed only while the accelerator is idle and the sequencer
    capability is still present. A lost/reconfigured FPGA fails this check.
    """
    recoveries = 0

    def recover(self):
        self.serial.close()
        for attempt in range(8):
            time.sleep(0.5)
            try:
                self.serial.open()
                break
            except serial.SerialException:
                if attempt==7:raise
        self.serial.reset_input_buffer()
        self.serial.timeout=0.25
        self.serial.read(4096)
        self.serial.timeout=5
        self.capabilities()
        status=decode_status(self.exchange(STATUS))
        if status['busy'] or super().read(0x410004,4)!=b'SEQ4':
            raise RuntimeError('USB recovery requires the compatible idle sequencer image')
        self.recoveries+=1
        print(f'reopened interrupted USB session ({self.recoveries})',flush=True)

    def write(self,address,data):
        for offset in range(0,len(data),64):
            part=data[offset:offset+64]
            for attempt in range(3):
                try:
                    super().write(address+offset,part)
                    break
                except serial.SerialException:
                    if address<0x500000 or attempt==2:
                        raise
                    self.recover()

    def read(self,address,length):
        result=bytearray()
        for offset in range(0,length,64):
            for attempt in range(3):
                try:
                    result.extend(super().read(address+offset,min(64,length-offset)))
                    break
                except serial.SerialException:
                    if attempt==2:
                        raise
                    self.recover()
        return bytes(result)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def execute(client, commands, input_data, final, timeout=60):
    client.exchange(RESET)
    client.write(0x500000, commands)
    if client.read(0x500000, len(commands)) != commands:
        raise AssertionError('command BSRAM upload mismatch')
    wall_start = time.monotonic()
    client.write_external(0, input_data)
    client.write(0x410000, b'\x01')
    deadline = time.monotonic()+timeout
    while True:
        status = decode_status(client.exchange(STATUS))
        if not status['busy']:
            break
        if time.monotonic()>deadline:
            raise TimeoutError('autonomous model execution')
    registers = client.read(0x410000, 32)
    if status['error'] or registers[1] or registers[4:8]!=b'SEQ4':
        raise AssertionError(f'sequencer failure: {status}, {registers.hex()}')
    elapsed, engine, dma, overlap, index, _ = struct.unpack('<6I', registers[8:])
    actual = client.read_external(final['ext'], final['bytes'])
    return {'elapsed_cycles': elapsed, 'engine_busy_cycles': engine,
            'dma_busy_cycles': dma, 'overlap_cycles': overlap,
            'last_command_index': index,
            'input_run_output_wall_seconds': time.monotonic()-wall_start,
            'output_sha256': sha(actual), 'output_hex': actual.hex()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='/dev/cu.usbserial-20250303171')
    parser.add_argument('--model', required=True, choices=('kws','vww'))
    parser.add_argument('--bitstream', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--resume', type=Path,
                        help='reuse completed node checks; reverify resident model and rerun timings')
    args = parser.parse_args()
    if args.repeats<1:
        parser.error('repeats must be positive')
    name = args.model
    folder = ROOT/'work/phase4'
    fixture = folder/f'rtl-{name}'
    manifest = json.loads((fixture/'manifest.json').read_text())
    input_data = (fixture/'input.bin').read_bytes()
    model_data = (folder/f'{name}-logits.onnx').read_bytes()
    calibration_data = (folder/f'{name}-calibration-rebased.json').read_bytes()
    for data, key in ((input_data,'input_sha256'),(model_data,'source_onnx_sha256'),
                      (calibration_data,'calibration_sha256'),
                      ((fixture/'expected.npz').read_bytes(),'expected_npz_sha256')):
        if sha(data)!=manifest[key]:
            raise ValueError(f'fixture hash mismatch: {key}')
    program = compile_static(onnx.load_from_string(model_data), json.loads(calibration_data))
    plan, image = compile_tiled(program, prefer_half=True)
    commands, payload, schedule = compile_sequence(plan,image,True,True)
    report = {'status':'running', 'physical_board':True, 'model':name,
              'bitstream_sha256':sha(args.bitstream.read_bytes()),
              'fixture_manifest_sha256':sha((fixture/'manifest.json').read_bytes()),
              'core_clock_hz':20250000, 'correctness_nodes':[], 'modes':{},
              'limits':['one pinned deterministic input; no full-set model accuracy',
                        'no measured power; timings use nominal PLL frequency',
                        'command/weight loading excluded from input-run-output wall time']}
    if args.resume:
        prior=json.loads(args.resume.read_text())
        if prior['bitstream_sha256']!=report['bitstream_sha256'] or \
           prior['fixture_manifest_sha256']!=report['fixture_manifest_sha256'] or \
           len(prior['correctness_nodes'])!=len(plan['layers']):
            raise ValueError('resume requires complete correctness checks for this exact model and bitstream')
        report['correctness_nodes']=prior['correctness_nodes']
        report['snapshot_run']=prior['snapshot_run']
        report['resumed_from_report_sha256']=sha(args.resume.read_bytes())
    args.report.parent.mkdir(parents=True,exist_ok=True)

    def save():
        args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')

    save()
    try:
        with serial.Serial(args.port,115200,timeout=5,write_timeout=5) as uart:
            uart.reset_input_buffer()
            client=RecoverableUploadClient(uart)
            client.capabilities();client.wait_idle(60);client.exchange(RESET)
            load_start=time.monotonic()
            if not args.resume:
                client.write_external(0,payload)
                if client.read_external(0,len(payload))!=payload:
                    raise AssertionError('model/descriptor SDRAM upload mismatch')
            else:
                # Activation slots are mutable. Verify every immutable model
                # and descriptor byte before trusting a resumed measurement.
                start=2*plan['activation_slot_bytes']
                if client.read_external(start,len(payload)-start)!=payload[start:]:
                    raise AssertionError('resident model changed; require a fresh upload')
            report['model_load_and_verify_seconds']=time.monotonic()-load_start
            print(f'{name}: uploaded and verified {len(payload)} bytes',flush=True)
            if not args.resume:
                check=execute(client,commands,input_data,schedule['final_output'])
                if check['output_sha256']!=manifest['output_sha256']:
                    raise AssertionError('snapshot run final output mismatch')
                with np.load(fixture/'expected.npz') as expected:
                    previous=input_data
                    for index,layer in enumerate(plan['layers']):
                        if index in schedule['snapshot_regions']:
                            r=schedule['snapshot_regions'][index]
                            previous=client.read_external(r['ext'],r['bytes'])
                        wanted=expected[f'layer_{index}'].tobytes()
                        if previous!=wanted:
                            raise AssertionError(f'node {index} snapshot mismatch')
                        report['correctness_nodes'].append({'index':index,'bytes':len(wanted),
                                                           'output_sha256':sha(previous)})
                        save()
                report['snapshot_run']=check
            print(f'{name}: all {len(plan["layers"])} node snapshots exact',flush=True)
            for overlap in (False,True):
                commands,same_payload,schedule=compile_sequence(plan,image,overlap,False)
                if same_payload!=payload:
                    raise AssertionError('benchmark modes changed model placement')
                label='overlap' if overlap else 'sequential'
                record={'schedule':schedule,'runs':[]}
                report['modes'][label]=record
                for repeat in range(args.repeats):
                    result=execute(client,commands,input_data,schedule['final_output'])
                    if result['output_sha256']!=manifest['output_sha256']:
                        raise AssertionError(f'{label} output mismatch')
                    record['runs'].append(result);save()
                    print(f'{name} {label} {repeat}: {result["elapsed_cycles"]} cycles, '
                          f'{result["overlap_cycles"]} overlap cycles',flush=True)
            report['status']='passed'
            report['usb_reopen_count']=client.recoveries
            save()
    except Exception as error:
        report['status']='failed';report['failure']=repr(error);save();raise


if __name__=='__main__':
    main()
