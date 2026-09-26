"""AD provenance tests: corrupt inputs must fail before evidence is published."""
import copy
import io
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/research"))
from fetch_ad_data import RangeFile, decode_member, selection
from prepare_ad_data import windows_from_histogram


@pytest.mark.parametrize("method", [zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED])
def test_remote_zip_member_integrity(method):
    buffer = io.BytesIO()
    content = bytes(range(256)) * 17
    with zipfile.ZipFile(buffer, "w", compression=method) as archive:
        archive.writestr("ToyCar/test/normal_id_01_00000003.wav", content)
    with zipfile.ZipFile(buffer) as archive:
        info = archive.infolist()[0]
        member = buffer.getvalue()[:archive.start_dir]
    assert decode_member(info, member) == content
    corrupted = copy.copy(info)
    corrupted.CRC ^= 1
    with pytest.raises(ValueError, match="integrity"):
        decode_member(corrupted, member)
    with pytest.raises(ValueError, match="header"):
        decode_member(info, b"broken")
    with pytest.raises(ValueError):
        decode_member(info, member[:30])


def test_histogram_uses_all_196_overlapping_windows():
    histogram = np.arange(200 * 128, dtype="<f4").reshape(200, 128)
    actual = windows_from_histogram(histogram.tobytes())
    assert actual.shape == (196, 640)
    np.testing.assert_array_equal(actual[0], histogram[:5].reshape(-1))
    np.testing.assert_array_equal(actual[1], histogram[1:6].reshape(-1))
    np.testing.assert_array_equal(actual[-1], histogram[-5:].reshape(-1))


@pytest.mark.parametrize("value", [np.zeros(640, "<f4"), np.full(200 * 128, np.nan, "<f4")])
def test_histogram_rejects_truncation_and_nonfinite(value):
    with pytest.raises(ValueError, match="histogram"):
        windows_from_histogram(value.tobytes())


def test_changed_split_is_rejected(tmp_path):
    base = tmp_path / "benchmark/training/anomaly_detection"
    base.mkdir(parents=True)
    (base / "calibration.txt").write_text("tampered\n")
    index = tmp_path / "benchmark/evaluation/datasets/ad01"
    index.mkdir(parents=True)
    (index / "y_labels.csv").write_text("tampered\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        selection(tmp_path)


@pytest.mark.parametrize("status,header,payload", [
    (200, "bytes 2-4/10", b"abc"),
    (206, "bytes 1-3/10", b"abc"),
    (206, "bytes 2-4/10", b"ab"),
])
def test_range_reader_rejects_wrong_response(monkeypatch, status, header, payload):
    response = io.BytesIO(payload)
    response.status = status
    response.headers = {"Content-Range": header}
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: response)
    with pytest.raises(ValueError):
        RangeFile("https://example.invalid/archive", 10).range(2, 5)
