#!/usr/bin/env python3
"""Reject generated Gowin SDRAM IP with the wrong embedded memory geometry."""

import argparse
import configparser
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEOMETRY = json.loads((ROOT/'hardware/sdram_tang_nano_20k.json').read_text())
FIELDS = {'Data_Width': 'data_width_bits', 'Bank_Width': 'bank_address_bits',
          'Addr_Row_Width': 'row_address_bits', 'Addr_Column_Width': 'column_address_bits'}


def audit_ipc(path):
    config = configparser.ConfigParser()
    if not config.read(path):
        raise ValueError(f'cannot read IP configuration: {path}')
    if 'Config' not in config:
        raise ValueError('missing Config section')
    actual = {}
    for key in FIELDS:
        if key not in config['Config']:
            raise ValueError(f'missing {key}')
        actual[key] = config['Config'].getint(key)
    expected = {key: GEOMETRY[name] for key, name in FIELDS.items()}
    capacity = (1 << (actual['Bank_Width']+actual['Addr_Row_Width']+
                      actual['Addr_Column_Width'])) * actual['Data_Width']//8
    if actual != expected or capacity != GEOMETRY['capacity_bytes']:
        raise ValueError(f'wrong Tang Nano 20K SDRAM geometry: actual={actual}, '
                         f'capacity={capacity}; expected={expected}, '
                         f'capacity={GEOMETRY["capacity_bytes"]}')
    return {'configuration': actual, 'capacity_bytes': capacity}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ipc', type=Path, help='generated Gowin SDRAM controller .ipc file')
    args = parser.parse_args()
    try:
        result = audit_ipc(args.ipc)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
