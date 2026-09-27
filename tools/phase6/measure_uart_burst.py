#!/usr/bin/env python3
"""Short physical transfer comparison on a preprogrammed UART burst image.

This runner never programs the FPGA. It requires the read-only burst feature
probe, tests both legacy 64-byte writes and 256-byte burst writes on the same
image, checks every byte by readback, and records UART wall time separately
from accelerator execution time.
"""

import argparse
import fcntl
import hashlib
import json
import statistics
import time
from pathlib import Path

import serial

from uart_burst import BurstTiledClient, ROOT
from tiled_host import TiledClient
from host import STATUS, RESET, decode_status
from run_screening import require_board_free, save_json
from variants import check_frozen


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(port, output, expected_bitstream):
    check_frozen()
    require_board_free(port)
    if output.exists():
        raise FileExistsError("preserve previous physical evidence")
    if not expected_bitstream.is_file():
        raise FileNotFoundError(expected_bitstream)
    lock = ROOT / "work/phase6/physical-board.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.parent.mkdir(parents=True, exist_ok=True)
        report = dict(status="running", physical_board=True, baud=750000,
                      tested_image_path=str(expected_bitstream.relative_to(ROOT)),
                      expected_image_sha256=digest(expected_bitstream.read_bytes()),
                      address=0x700000, transfer_bytes=8192, records=[])

        def save():
            save_json(output, report)

        save()
        try:
            with serial.Serial(port, 750000, timeout=5, write_timeout=5) as uart:
                uart.reset_input_buffer()
                client = BurstTiledClient(uart)
                client.capabilities()  # checks both legacy target and burst feature
                status = decode_status(client.exchange(STATUS))
                if status["busy"] or status["error"]:
                    raise ValueError("FPGA is not idle and error-free")
                client.exchange(RESET)
                # Alternating order limits thermal/USB drift in the comparison.
                for run_index, mode in enumerate(("legacy", "burst", "burst", "legacy")):
                    payload = bytes((i * 73 + run_index * 29 + 11) & 255
                                    for i in range(8192))
                    start = time.monotonic()
                    if mode == "legacy":
                        TiledClient.write_external(client, 0x700000, payload)
                    else:
                        client.write_external(0x700000, payload)
                    upload_seconds = time.monotonic() - start
                    readback = client.read_external(0x700000, len(payload))
                    if readback != payload:
                        raise ValueError("physical SDRAM readback mismatch")
                    status = decode_status(client.exchange(STATUS))
                    if status["busy"] or status["error"] or status["protocol_errors"]:
                        raise ValueError("physical protocol status failure")
                    report["records"].append(dict(mode=mode, repeat=run_index,
                        payload_sha256=digest(payload), upload_seconds=upload_seconds,
                        upload_bytes_per_second=len(payload) / upload_seconds,
                        readback_matched=True))
                    save()
            medians = {mode: statistics.median(row["upload_seconds"]
                       for row in report["records"] if row["mode"] == mode)
                       for mode in ("legacy", "burst")}
            if digest(expected_bitstream.read_bytes()) != report["expected_image_sha256"]:
                raise ValueError("bitstream file changed during physical transfer screen")
            check_frozen()
            report.update(status="passed-short-transfer-screen", medians=medians,
                          speedup=medians["legacy"] / medians["burst"])
            save()
            print(json.dumps(dict(medians=medians, speedup=report["speedup"])))
        except BaseException as error:
            report.update(status="failed", failure=repr(error))
            save()
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--port", default="/dev/cu.usbserial-20250303171")
    parser.add_argument("--bitstream", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("Use --run after programming the matching experimental image")
    run(args.port, args.report, args.bitstream.resolve())
