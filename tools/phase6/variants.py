"""Source selection for isolated experiments; never changes production inputs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {'baseline': None, 'c0p0': (0, 0), 'c0p1': (0, 1),
            'c64p0': (64, 0), 'c128p0': (128, 0), 'c256p0': (256, 0),
            'c256p1': (256, 1), 'spatial': (256, 1)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_frozen():
    record = json.loads((ROOT/'work/phase6/phase5-frozen-files.json').read_text())
    # The milestone manifest is a path -> SHA256 map.
    for name, digest in record.items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'Phase 5 frozen source changed: {name}')
    return len(record)


def configured_engine_text(name):
    choice = VARIANTS[name]
    if choice is None:
        return (ROOT/'rtl/v2/engine.sv').read_text()
    text = (ROOT/'rtl/phase6'/('spatial_engine.sv' if name=='spatial' else 'engine.sv')).read_text()
    text = text.replace('PW_CACHE_ENTRIES = 256', f'PW_CACHE_ENTRIES = {choice[0]}')
    text = text.replace('PARAM_CACHE = 1', f'PARAM_CACHE = {choice[1]}')
    return text


def engine_source(name, build):
    if VARIANTS[name] is None:
        return ROOT/'rtl/v2/engine.sv'
    build.mkdir(parents=True, exist_ok=True)
    path = build/'engine.sv'
    path.write_text(configured_engine_text(name))
    return path


def sources(name, build, engine_only=False):
    paths = [ROOT/'rtl/v2/target_pkg.sv', ROOT/'rtl/v2/requantizer.sv', engine_source(name, build)]
    if not engine_only:
        paths += [ROOT/'rtl/v2'/n for n in ('scratchpad.sv', 'tile_dma.sv', 'tiled_core.sv',
                   'command.sv', 'tile_sequencer.sv', 'tiled_host_bridge.sv')]
    return paths
