#!/usr/bin/env python3
"""Rebuild a pinned accuracy payload from its verified evaluation archive.

The calibration split was prepared earlier and is pinned separately. This
runner needs only evaluation files, and verifies every raw and feature hash
before accepting a payload with the frozen NPZ hash.
"""

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def kws_preprocessor(upstream):
    import tensorflow as tf

    sys.path.insert(0, str(ROOT / 'tools/research'))
    from prepare_quality_data import upstream_function

    source = upstream / 'benchmark/training/keyword_spotting'
    scope = {'tf': tf, 'np': np}
    settings_fn = upstream_function(source / 'keras_model.py',
                                    'prepare_model_settings', scope)
    settings = settings_fn(12, SimpleNamespace(
        sample_rate=16000, clip_duration_ms=1000, window_size_ms=30.,
        window_stride_ms=20., feature_type='mfcc',
        dct_coefficient_count=10))
    original = upstream_function(source / 'get_dataset.py',
                                 'get_preprocess_audio_func', scope)(
                                     settings, is_training=False)

    def prepare_copy(example):
        return original(dict(example))

    fn = tf.function(prepare_copy, input_signature=[{
        'audio': tf.TensorSpec([None], tf.int16),
        'label': tf.TensorSpec([], tf.int64)}])

    def prepare(path, label):
        with wave.open(str(path), 'rb') as stream:
            if (stream.getframerate(), stream.getnchannels(),
                    stream.getsampwidth()) != (16000, 1, 2):
                raise ValueError(f'unexpected WAV format: {path}')
            audio = np.frombuffer(stream.readframes(stream.getnframes()),
                                  dtype='<i2').copy()
        return fn({'audio': tf.convert_to_tensor(audio),
                   'label': tf.constant(label, tf.int64)})['audio'].numpy()[None]

    return prepare


def vww_preprocessor():
    from PIL import Image

    def prepare(path, _label):
        with Image.open(path) as image:
            if image.size != (96, 96):
                raise ValueError(f'unexpected image dimensions: {path}')
            return np.asarray(image.convert('RGB'), np.float32)[None] / 255.

    return prepare


def materialize(model, data, output, upstream):
    manifest_path = ROOT / f'benchmarks/manifests/{model}.data.json'
    manifest = json.loads(manifest_path.read_text())
    records = manifest['splits']['accuracy']['records']
    expected = manifest['splits']['accuracy']['npz_sha256']
    if len(records) != manifest['splits']['accuracy']['count']:
        raise ValueError('record count mismatch')
    prepare = (kws_preprocessor(upstream) if model == 'kws'
               else vww_preprocessor())
    shape = (len(records), 1, 49, 10, 1) if model == 'kws' else (
        len(records), 1, 96, 96, 3)
    features = np.empty(shape, dtype=np.float32)
    labels = []
    ids = []
    for index, record in enumerate(records):
        path = (data / record['path']).resolve()
        if not path.is_relative_to(data.resolve()) or not path.is_file():
            raise ValueError(f'missing or unsafe sample: {record["path"]}')
        if digest(path) != record['raw_sha256']:
            raise ValueError(f'raw sample hash mismatch: {record["id"]}')
        value = prepare(path, int(record['label']))
        if (not np.isfinite(value).all() or
                hashlib.sha256(value.tobytes()).hexdigest() !=
                record['feature_sha256']):
            raise ValueError(f'preprocessed feature mismatch: {record["id"]}')
        features[index] = value
        labels.append(int(record['label']))
        ids.append(record['id'])
        if (index + 1) % 500 == 0:
            print(f'{model}: verified {index + 1}/{len(records)}', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{
        manifest['input_name']: features,
        'sample_ids': np.asarray(ids),
        'labels': np.asarray(labels)})
    actual = digest(output)
    if actual != expected:
        raise ValueError(f'payload hash differs from frozen manifest: {actual}')
    return {'status': 'passed', 'model': model, 'samples': len(records),
            'accuracy_payload_sha256': actual,
            'data_manifest_sha256': digest(manifest_path)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=('kws', 'vww'))
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--upstream', type=Path, default=ROOT / 'work/upstream')
    args = parser.parse_args()
    print(json.dumps(materialize(args.model, args.data, args.output,
                                 args.upstream), indent=2))
