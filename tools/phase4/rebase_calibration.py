#!/usr/bin/env python3
"""Rebase pinned ranges after a structure-preserving ONNX conversion.

This is intentionally strict: it accepts only auto-renamed constant tensor
inputs. It does not relabel the new model as the old pinned binary. The source
conversion parity report and upstream artifact hashes must also match.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_conversion(name, source, report):
    manifest = json.loads((ROOT/f'benchmarks/manifests/{name}.json').read_text())
    if report.get('onnx_sha256') != digest(source) or not report.get('status', '').startswith('passed-'):
        raise ValueError('conversion parity report does not match source ONNX')
    if name == 'vww':
        entry = next(x for x in manifest['files'] if
                     x['path'].endswith('/vww_96_float.tflite'))
        if report.get('source_sha256') != entry['sha256']:
            raise ValueError('VWW source artifact hash mismatch')
    else:
        prefix = 'benchmark/training/keyword_spotting/trained_models/kws_ref_model/'
        expected = {x['path'][len(prefix):]: x['sha256']
                    for x in manifest['files'] if x['path'].startswith(prefix)}
        if report.get('source_files_sha256') != expected:
            raise ValueError('KWS SavedModel source hashes mismatch')


def check_inventory(old, new):
    if len(old['operators']) != len(new['operators']):
        raise ValueError('operator count changed')
    old_tensors = {x['name']: x for x in old['tensors']}
    new_tensors = {x['name']: x for x in new['tensors']}
    old_data = [x for x in old['tensors'] if not x['constant']]
    new_data = [x for x in new['tensors'] if not x['constant']]
    if old_data != new_data:
        raise ValueError('activation tensor inventory changed')
    old_constants = [x for x in old['tensors'] if x['constant']]
    new_constants = [x for x in new['tensors'] if x['constant']]
    if len(old_constants) != len(new_constants) or any(
        {k: v for k, v in x.items() if k != 'name'} !=
        {k: v for k, v in y.items() if k != 'name'}
        for x, y in zip(old_constants, new_constants)):
        raise ValueError('constant tensor geometry changed')
    renames = {}
    for index, (before, after) in enumerate(zip(old['operators'], new['operators'])):
        if {k: v for k, v in before.items() if k != 'inputs'} != \
           {k: v for k, v in after.items() if k != 'inputs'} or \
           len(before['inputs']) != len(after['inputs']):
            raise ValueError(f'operator {index} changed')
        for old_name, new_name in zip(before['inputs'], after['inputs']):
            if old_name == new_name:
                continue
            a, b = old_tensors[old_name], new_tensors[new_name]
            if not a['constant'] or not b['constant'] or \
               {k: v for k, v in a.items() if k != 'name'} != \
               {k: v for k, v in b.items() if k != 'name'}:
                raise ValueError(f'operator {index} changed a nonconstant input')
            if old_name in renames and renames[old_name] != new_name:
                raise ValueError('inconsistent constant rename')
            renames[old_name] = new_name
    return renames


def rebase(name, source, inventory_path, conversion_path):
    before = json.loads((ROOT/f'benchmarks/manifests/{name}.canonical-inventory.json').read_text())
    after = json.loads(inventory_path.read_text())
    conversion = json.loads(conversion_path.read_text())
    check_conversion(name, source, conversion)
    if after['source_onnx_sha256'] != digest(source):
        raise ValueError('new inventory source hash mismatch')
    renames = check_inventory(before, after)
    calibration = json.loads((ROOT/f'benchmarks/manifests/{name}.calibration.json').read_text())
    if calibration['model_sha256'] != before['canonical_onnx_sha256']:
        raise ValueError('pinned calibration does not match pinned inventory')
    if set(calibration['ranges']) != {x['name'] for x in before['tensors']
                                      if not x['constant']}:
        raise ValueError('calibration range tensor set changed')
    calibration['rebased_from_model_sha256'] = calibration['model_sha256']
    calibration['model_sha256'] = after['canonical_onnx_sha256']
    calibration['rebase_scope'] = 'same pinned source; constant-name-only inventory drift'
    report = {'status': 'passed-structure-checked-calibration-rebase',
              'workload': name, 'source_onnx_sha256': digest(source),
              'source_parity_report_sha256': digest(conversion_path),
              'old_canonical_sha256': before['canonical_onnx_sha256'],
              'new_canonical_sha256': after['canonical_onnx_sha256'],
              'operators': len(after['operators']),
              'constant_input_renames': renames,
              'calibration_sample_count': len(calibration['sample_ids']),
              'scope': 'boardless development; not byte-identical to frozen conversion'}
    return calibration, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workload', choices=('kws', 'vww'))
    parser.add_argument('source_onnx', type=Path)
    parser.add_argument('new_inventory', type=Path)
    parser.add_argument('conversion_report', type=Path)
    parser.add_argument('output_calibration', type=Path)
    parser.add_argument('output_report', type=Path)
    args = parser.parse_args()
    calibration, report = rebase(args.workload, args.source_onnx,
                                 args.new_inventory, args.conversion_report)
    for path, content in ((args.output_calibration, calibration),
                          (args.output_report, report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'workload': args.workload, 'status': report['status'],
                      'renames': len(report['constant_input_renames'])}))


if __name__ == '__main__':
    main()
