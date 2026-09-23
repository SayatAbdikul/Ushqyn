#!/usr/bin/env python3
import json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
target=json.loads((ROOT/'hardware/targets/tang_nano_20k_v2.json').read_text())
subprocess.run(['verilator','--lint-only','--timing','--top-module','tang_nano_v2']+[str(ROOT/p) for p in target['sources']],check=True)
