"""The calibration CLI must not retain one decompressed NPZ copy per sample."""

import numpy as np
from onnx import TensorProto, helper
import pytest

from static_cli import load_calibration_samples


def _model():
    graph = helper.make_graph([], 'calibration-inputs',
                              [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 4])],
                              [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 4])])
    return helper.make_model(graph)


def test_calibration_samples_share_one_loaded_array(tmp_path):
    path = tmp_path / 'calibration.npz'
    np.savez_compressed(path, sample_ids=np.array(['a', 'b', 'c']),
                        x=np.arange(12, dtype=np.float32).reshape(3, 1, 4))
    ids, samples = load_calibration_samples(_model(), path)
    values = [item['x'] for item in samples]
    assert ids == ['a', 'b', 'c']
    assert all(values[0].base is item.base for item in values[1:])
    np.testing.assert_array_equal(values[2], [[8, 9, 10, 11]])


def test_calibration_input_count_is_checked(tmp_path):
    path = tmp_path / 'short.npz'
    np.savez_compressed(path, sample_ids=np.array(['a', 'b']),
                        x=np.zeros((1, 1, 4), dtype=np.float32))
    with pytest.raises(ValueError, match='calibration input count mismatch'):
        load_calibration_samples(_model(), path)
