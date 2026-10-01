#!/usr/bin/env python3
"""Run two real held-out AD windows through the existing candidate native RTL.

The executable is an external-RAM Verilator model, not a physical FPGA timing
measurement. Every layer snapshot is checked against the independent oracle.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'compiler'))
from integer_reference import evaluate
from phase4_compile import compile_tiled
from phase4_sequence import compile_sequence
from program_image import load_image


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--image', type=Path, default=ROOT / 'work/phase6/parallel-ad-v1/ad.uq2')
    parser.add_argument('--accuracy', type=Path, default=ROOT / 'work/phase0-ad/features/ad.accuracy.npz')
    parser.add_argument('--output', type=Path, default=ROOT / 'work/phase6/parallel-ad-v1/native')
    args = parser.parse_args()
    native_sha = sha(args.native.read_bytes())
    build_report_path = args.native.parent / 'report.json'
    if not build_report_path.is_file():
        raise ValueError('native executable requires its source-pinned build report')
    build_report = json.loads(build_report_path.read_text())
    if build_report.get('status') != 'passed' or build_report.get('executable_sha256') != native_sha:
        raise ValueError('native executable differs from its passed build report')
    program = load_image(args.image.read_bytes())
    plan, image = compile_tiled(program)
    commands, payload, schedule = compile_sequence(plan, image, overlap=True, snapshots=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.accuracy, allow_pickle=False) as data:
        features = data[program.inputs[0]]
        ids = data['sample_ids']
    result = {'status': 'running', 'scope': 'candidate native RTL with abstract external RAM; no physical board',
              'native_executable_sha256': native_sha,
              'native_build_report_sha256': sha(build_report_path.read_bytes()),
              'native_build_source_sha256': build_report.get('source_sha256', build_report.get('sources')),
              'native_engine_sha256': build_report.get('engine_sha256'),
              'image_sha256': sha(args.image.read_bytes()),
              'accuracy_npz_sha256': sha(args.accuracy.read_bytes()),
              'command_sha256': sha(commands), 'payload_sha256': sha(payload),
              'command_count': schedule['command_count'], 'program_bytes': len(commands),
              'sample_runs': []}
    report_path = args.output / 'report.json'
    report_path.write_text(json.dumps(result, indent=2) + '\n')
    for index in (0, len(features) - 1):
        sample = features[index]
        qinput = program.tensors[program.inputs[0]].quantization.encode(sample)
        oracle = evaluate(program, {program.inputs[0]: qinput})
        folder = args.output / f'frame-{index}'
        folder.mkdir(exist_ok=True)
        (folder / 'payload.bin').write_bytes(payload)
        (folder / 'commands.bin').write_bytes(commands)
        (folder / 'input.bin').write_bytes(qinput.tobytes())
        checks = []
        output = oracle[program.outputs[0]].tobytes()
        (folder / 'output.bin').write_bytes(output)
        checks.append(f"{schedule['final_output']['ext']} output.bin")
        for layer_index, region in schedule['snapshot_regions'].items():
            name = f'layer-{layer_index}.bin'
            (folder / name).write_bytes(oracle[program.layers[layer_index].output].tobytes())
            checks.append(f"{region['ext']} {name}")
        (folder / 'checks.txt').write_text('\n'.join(checks) + '\n')
        for seed in (0, 6063):
            seed_report = folder / f'native-{seed}.json'
            invocation = subprocess.run([str(args.native), str(folder), str(seed), str(seed_report)],
                                        cwd=ROOT, capture_output=True, text=True, timeout=300)
            if invocation.returncode:
                result['status'] = 'failed'
                result['failure'] = {'index': index, 'seed': seed, 'returncode': invocation.returncode,
                                     'stdout': invocation.stdout[-1000:], 'stderr': invocation.stderr[-1000:]}
                report_path.write_text(json.dumps(result, indent=2) + '\n')
                raise RuntimeError(f"native AD frame {index} seed {seed} failed: {invocation.stderr}")
            native_record = json.loads(seed_report.read_text())
            if native_record['status'] != 'passed' or native_record['tensor_checks'] != len(checks):
                raise AssertionError('native result did not check every AD layer')
            result['sample_runs'].append({'index': index, 'sample_id': str(ids[index]),
                                          'seed': seed, 'output_sha256': sha(output),
                                          'checks': len(checks), 'native': native_record})
            report_path.write_text(json.dumps(result, indent=2) + '\n')
            print(index, seed, native_record['elapsed_cycles'], 'cycles', flush=True)
    result['status'] = 'passed'
    report_path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
