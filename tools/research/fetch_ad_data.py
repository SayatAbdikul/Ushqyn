#!/usr/bin/env python3
"""Fetch the frozen AD research split from public ToyCar ZIPs using HTTP ranges.

Only requested members are downloaded. ZIP CRC/size and local SHA256 are checked;
the publisher's whole-archive MD5 is recorded, NOT claimed to have been verified.
Research calibration: 16 evenly spaced upstream calibration recordings per ID.
The full upstream calibration list is retained as provenance, not silently used.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib

ARCHIVES = [
    ("3678171", "dev_data_ToyCar.zip", 1816443231,
     "4dec75ca8d9f666aa9e4c1894a740501"),
    ("3727685", "eval_data_train_ToyCar.zip", 861354584,
     "a5d4ccd498af70b04dd74b644b820006"),
]
INDEX_SHA = "4ecd91868e197ee0a6739d4bd7abde73eac2fa31e9a88d6bc6aedefa136ff2a4"
CAL_SHA = "f6b2a371295165c1b977aa6289d93bba7b50f2c1161267886426f0280a58dd3f"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def selection(upstream):
    base = upstream / "benchmark/training/anomaly_detection"
    cal = (base / "calibration.txt").read_bytes()
    index = (upstream / "benchmark/evaluation/datasets/ad01/y_labels.csv").read_bytes()
    if sha(cal) != CAL_SHA or sha(index) != INDEX_SHA:
        raise ValueError("upstream split hash mismatch")
    groups = {}
    for path in cal.decode().splitlines():
        match = re.fullmatch(r"dev_data/ToyCar/train/normal_id_(\d\d)_\d{8}\.wav", path)
        if not match:
            raise ValueError("unexpected calibration path")
        groups.setdefault(match[1], []).append(path)
    if len(groups) != 7 or any(len(v) != 200 for v in groups.values()):
        raise ValueError("expected seven IDs with 200 upstream calibration clips each")
    chosen = [sorted(paths)[i * 199 // 15] for _, paths in sorted(groups.items()) for i in range(16)]
    records = [{"split": "calibration", "path": p, "label": 0} for p in chosen]
    rows = list(csv.reader(io.StringIO(index.decode())))
    for name, classes, label, count, stride in rows:
        if (classes, count, stride) != ("2", "2560", "512") or label not in ("0", "1"):
            raise ValueError("unsupported evaluation framing")
        if not re.fullmatch(r"(?:normal|anomaly)_id_0[1-4]_\d{8}_hist_librosa.bin", name):
            raise ValueError("unexpected evaluation name")
        wav = name.replace("_hist_librosa.bin", ".wav")
        if int(label) != int(wav.startswith("anomaly_")):
            raise ValueError("filename/label disagreement")
        records.append({"split": "accuracy", "path": "dev_data/ToyCar/test/" + wav,
                        "label": int(label), "upstream_bin": name})
    if len(rows) != 248 or len({r["path"] for r in records}) != len(records):
        raise ValueError("unexpected sample count or duplicate path")
    return records


class RangeFile(io.RawIOBase):
    """Small seekable HTTP reader for ZIP directory lookup; no whole-file fallback."""
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.last_request = 0.0

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (self.pos if whence == 1 else self.size if whence == 2 else 0)
        if self.pos < 0:
            raise ValueError("negative seek")
        return self.pos

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.size, self.pos + n)
        if end <= self.pos:
            return b""
        data = self.range(self.pos, end)
        self.pos = end
        return data

    def range(self, start, end):
        if not 0 <= start < end <= self.size or end - start > 8 * 1024 * 1024:
            raise ValueError("invalid or unexpectedly large range")
        for attempt in range(6):
            time.sleep(max(0, 1.1 - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end-1}"})
            try:
                with urllib.request.urlopen(req, timeout=60) as response:
                    expected = f"bytes {start}-{end-1}/{self.size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("server did not honor exact byte range")
                    data = response.read(end - start + 1)
                    if len(data) != end - start:
                        raise ValueError("truncated or oversized range")
                    return data
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 5:
                    raise
                time.sleep(max(2 ** attempt, float(error.headers.get("Retry-After", 10))))
            except (urllib.error.URLError, TimeoutError):
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)


def decode_member(info, data):
    if data[:4] != b"PK\x03\x04" or len(data) < 30:
        raise ValueError("bad ZIP local header")
    fields = struct.unpack("<4s5H3I2H", data[:30])
    flags, method, name_len, extra_len = fields[2], fields[3], fields[-2], fields[-1]
    if flags & 1 or method != info.compress_type:
        raise ValueError("encrypted or inconsistent ZIP member")
    name = data[30:30 + name_len].decode("utf-8" if flags & 0x800 else "cp437")
    if name != info.filename or info.file_size > 2 * 1024 * 1024:
        raise ValueError("unexpected member name or size")
    start = 30 + name_len + extra_len
    payload = data[start:start + info.compress_size]
    if method == zipfile.ZIP_DEFLATED:
        decoder = zlib.decompressobj(-15)
        raw = decoder.decompress(payload, info.file_size + 1)
        if not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
            raise ValueError("invalid compressed member")
    elif method == zipfile.ZIP_STORED:
        raw = payload
    else:
        raise ValueError("unsupported ZIP compression")
    if len(raw) != info.file_size or zlib.crc32(raw) != info.CRC:
        raise ValueError("ZIP member integrity mismatch")
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = selection(args.upstream)
    remaining = {r["path"]: r for r in records}
    archives = []
    for record, filename, size, md5 in ARCHIVES:
        url = f"https://zenodo.org/api/records/{record}/files/{filename}/content"
        remote = RangeFile(url, size)
        with zipfile.ZipFile(remote) as archive:
            members = sorted(archive.infolist(), key=lambda x: x.header_offset)
            directory_offset = archive.start_dir
        archives.append({"doi": f"10.5281/zenodo.{record}", "url": url, "bytes": size,
                         "publisher_md5": md5, "whole_archive_digest_verified": False,
                         "license": "CC-BY-NC-SA-4.0"})
        for i, info in enumerate(members):
            # Both archives use ToyCar/train or ToyCar/test; additional IDs 05-07
            # are merged under dev_data as in upstream get_dataset.sh.
            path = "dev_data/" + info.filename
            if path not in remaining:
                continue
            item = remaining.pop(path)
            target = args.output / path
            raw = target.read_bytes() if target.exists() else b""
            if len(raw) != info.file_size or zlib.crc32(raw) != info.CRC:
                end = members[i + 1].header_offset if i + 1 < len(members) else directory_offset
                raw = decode_member(info, remote.range(info.header_offset, end))
                target.parent.mkdir(parents=True, exist_ok=True)
                temp = target.with_suffix(".partial")
                temp.write_bytes(raw)
                temp.replace(target)
            item.update(raw_sha256=sha(raw), raw_bytes=len(raw), zip_crc32=f"{info.CRC:08x}",
                        source_record=record, archive_member=info.filename)
            done = len(records) - len(remaining)
            if done % 20 == 0:
                print(f"verified {done}/{len(records)} WAVs", flush=True)
    if remaining:
        raise ValueError(f"archive missing {len(remaining)} samples: {list(remaining)[:3]}")
    if {r["raw_sha256"] for r in records if r["split"] == "calibration"} & {
            r["raw_sha256"] for r in records if r["split"] == "accuracy"}:
        raise ValueError("calibration/evaluation raw content overlap")
    result = {"schema": 1, "archives": archives, "records": records,
              "upstream_index_sha256": INDEX_SHA, "upstream_calibration_list_sha256": CAL_SHA,
              "calibration_policy": "16 evenly spaced indices floor(i*199/15) per machine ID, i=0..15, from sorted upstream 200 clips per ID; 112 clips total",
              "official_preprocessed_binary_parity_verified": False,
              "script_sha256": sha(Path(__file__).read_bytes())}
    (args.output / "raw-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print("AD raw provenance complete", flush=True)


if __name__ == "__main__":
    main()
