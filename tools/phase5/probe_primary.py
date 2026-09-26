#!/usr/bin/env python3
"""Exercise pinned Phase 5 inputs against the autonomous physical release."""

import argparse
import hashlib
import json
import shutil
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import onnx
import serial

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'compiler'))
sys.path.insert(0, str(ROOT / 'tools/phase4'))
from phase4_compile import compile_tiled
from phase4_sequence import compile_sequence
from static_pipeline import compile_static, execute_layer, run
from tiled_host import TiledClient
from host import STATUS, decode_status


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def arrays(model, input_name, data_manifest):
    archive = ROOT / f'work/quality/{model}.accuracy.npz'
    if sha(archive) != data_manifest['splits']['accuracy']['npz_sha256']:
        raise ValueError('accuracy payload differs from frozen split')
    extracted = ROOT / f'work/phase5/{model}.features.npy'
    if not extracted.is_file():
        extracted.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as source, source.open(input_name + '.npy') as item, \
                extracted.open('wb') as target:
            shutil.copyfileobj(item, target)
    with np.load(archive, allow_pickle=False) as source:
        ids = source['sample_ids'].tolist()
        labels = source['labels'].tolist()
    records = data_manifest['splits']['accuracy']['records']
    if len(ids) != len(records) or any(
            ids[i] != r['id'] or labels[i] != r['label']
            for i, r in enumerate(records)):
        raise ValueError('accuracy IDs or labels differ from frozen manifest')
    features = np.load(extracted, mmap_mode='r', allow_pickle=False)
    if len(features) != len(records):
        raise ValueError('accuracy tensor length mismatch')
    return features, records, sha(archive)


def prepared(model):
    folder = ROOT / 'work/phase4'
    fixture = json.loads((folder / f'rtl-{model}/manifest.json').read_text())
    model_path = folder / f'{model}-logits.onnx'
    calibration_path = folder / f'{model}-calibration-rebased.json'
    if (sha(model_path) != fixture['source_onnx_sha256'] or
            sha(calibration_path) != fixture['calibration_sha256']):
        raise ValueError('model/calibration differs from Phase 4 verified source')
    program = compile_static(onnx.load(model_path),
                             json.loads(calibration_path.read_text()))
    plan, image = compile_tiled(program, prefer_half=True)
    commands, payload, schedule = compile_sequence(plan, image, True, False)
    data_manifest = json.loads((ROOT / f'benchmarks/manifests/{model}.data.json').read_text())
    input_name = program.inputs[0]
    features, records, split_sha = arrays(model, input_name, data_manifest)
    return program, commands, payload, schedule, features, records, split_sha


def run_one(client, program, commands, payload, schedule, feature, record):
    input_name = program.inputs[0]
    sample = np.asarray(feature)
    if hashlib.sha256(sample.tobytes()).hexdigest() != record['feature_sha256']:
        raise ValueError('feature content differs from manifest')
    quantized = program.tensors[input_name].quantization.encode(sample)
    if program.layers[0].op in ('Transpose', 'Reshape'):
        board_input = execute_layer(program, program.layers[0],
                                    {input_name: quantized})
    else:
        board_input = quantized
    expected = run(program, {input_name: sample})[program.outputs[0]]
    final = schedule['final_output']
    if board_input.nbytes > 8 * 1024 * 1024:
        raise AssertionError('invalid input size')
    wall_start = time.monotonic()
    client.write_external(0, board_input.tobytes())
    client.write(0x410000, b'\x01')
    deadline = time.monotonic() + 20
    while True:
        status = decode_status(client.exchange(STATUS))
        if not status['busy']:
            break
        if time.monotonic() > deadline:
            raise TimeoutError('ambiguous board inference; no launch retry')
    regs = client.read(0x410000, 32)
    if status['error'] or regs[1] or regs[4:8] != b'SEQ4':
        raise RuntimeError(f'board status error: {status}, {regs.hex()}')
    elapsed, engine, dma, overlap, index, _ = struct.unpack('<6I', regs[8:])
    if index != schedule['command_count'] - 1:
        raise AssertionError('sequencer halted at an unexpected command')
    actual = client.read_external(final['ext'], final['bytes'])
    if actual != expected.tobytes():
        raise AssertionError(f'raw INT8 output mismatch: {record["id"]}; '
                             f'{actual.hex()} != {expected.tobytes().hex()}')
    return {'sample_id': record['id'], 'label': record['label'],
            'predicted_class': int(np.argmax(expected)),
            'output_hex': actual.hex(), 'elapsed_cycles': elapsed,
            'engine_cycles': engine, 'dma_cycles': dma,
            'overlap_cycles': overlap, 'command_index': index,
            'wall_seconds': time.monotonic() - wall_start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=('kws', 'vww'))
    parser.add_argument('--bitstream', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--limit', type=int, default=1)
    parser.add_argument('--port', default='/dev/cu.usbserial-20250303171')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    program, commands, payload, schedule, features, records, split_sha = prepared(args.model)
    bit_sha = sha(args.bitstream)
    report = {'status': 'running', 'model': args.model,
              'bitstream_sha256': bit_sha, 'split_sha256': split_sha,
              'model_image_sha256': hashlib.sha256(payload).hexdigest(),
              'command_sha256': hashlib.sha256(commands).hexdigest(),
              'records': []}
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.report.write_text(json.dumps(report, indent=2) + '\n')

    save()
    try:
        with serial.Serial(args.port, 115200, timeout=5, write_timeout=5) as port:
            port.reset_input_buffer()
            client = TiledClient(port)
            client.capabilities()
            if client.read(0x410004, 4) != b'SEQ4':
                raise ValueError('connected board lacks autonomous sequencer')
            client.wait_idle(20)
            client.write_external(0, payload)
            if client.read_external(0, len(payload)) != payload:
                raise AssertionError('resident model readback mismatch')
            client.write(0x500000, commands)
            if client.read(0x500000, len(commands)) != commands:
                raise AssertionError('sequencer program readback mismatch')
            for index in range(min(args.limit, len(records))):
                row = run_one(client, program, commands, payload, schedule,
                              features[index], records[index])
                report['records'].append(row)
                save()
                print(args.model, index + 1, row['elapsed_cycles'], row['predicted_class'], flush=True)
        report['status'] = 'passed'
        report['correct'] = sum(r['predicted_class'] == r['label'] for r in report['records'])
        save()
    except Exception as error:
        report['status'] = 'failed'
        report['failure'] = repr(error)
        save()
        raise


if __name__ == '__main__':
    main()
