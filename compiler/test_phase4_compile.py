"""Executable tile-image packing checks against the existing on-chip ABI."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from onnx import helper as h

from hardware_v2 import Descriptor, lower
from phase4_compile import compile_tiled
from phase4_tiling import plan_inventory
from quantization import Quantization
from static_pipeline import Layer, Program, Tensor
from static_pipeline import calibrate
from test_static_pipeline import compile_case, model


def _small_program():
    rng = np.random.default_rng(4344)
    x = rng.normal(size=(1, 2, 4, 4)).astype(np.float32)
    nodes = [
        h.make_node('Conv', ['x', 'cw', 'cb'], ['c']),
        h.make_node('Relu', ['c'], ['r']),
        h.make_node('AveragePool', ['r'], ['a'], kernel_shape=[2, 2], strides=[2, 2]),
        h.make_node('Flatten', ['a'], ['f'], axis=1),
        h.make_node('Gemm', ['f', 'gw', 'gb'], ['y'], transB=1),
    ]
    weights = {'cw': rng.normal(size=(3, 2, 1, 1)), 'cb': rng.normal(size=3),
               'gw': rng.normal(size=(4, 12)), 'gb': rng.normal(size=4)}
    graph = model(nodes, weights, x.shape, {'y': [1, 4]})
    return compile_case(graph, x)


def test_tiled_image_matches_existing_descriptor_parameter_abi():
    program = _small_program()
    plan, image = compile_tiled(program)
    on_chip, metadata = lower(program)
    assert len(plan['layers']) == len(program.layers) == 5
    assert hashlib.sha256(image).hexdigest() == plan['parameter_image_sha256']
    source = {segment['name']: segment['offset'] for segment in metadata['segments']}
    for index, layer in enumerate(plan['layers']):
        if layer['kind'] == 'alias':
            assert not layer['tiles']
            continue
        assert len(layer['tiles']) == 1
        tile = layer['tiles'][0]
        descriptor = Descriptor.decode(bytes.fromhex(tile['descriptor_hex']))
        assert descriptor.next_pc == 64
        for name, segment in tile['regions'].items():
            if name not in ('weight', 'params'):
                continue
            transfer = next(t for t in tile['transfers']
                            if t['direction'] == 'to_sram' and
                            t['sram'] == segment['base'])
            original_name = f'{index}/weights' if name == 'weight' else f'{index}/params'
            start = source[original_name]
            assert image[transfer['ext']:transfer['ext']+transfer['bytes']] == \
                   on_chip[start:start+transfer['bytes']]


def test_large_elementwise_tile_image_is_split_and_disjoint():
    from quantization import Quantization
    q = Quantization(.05, -7)
    shape = (1, 40000)
    program = Program({'x': Tensor('x', shape, q, 'row-major'),
                       'y': Tensor('y', shape, q, 'row-major')},
                      [Layer('Relu', ['x'], 'y', {}, {})], ['x'], ['y'], {}, {})
    plan, image = compile_tiled(program)
    tiles = plan['layers'][0]['tiles']
    assert len(tiles) > 2
    assert sum(t['count'] for t in tiles) == 40000
    assert all(t['scratch_bytes'] <= 32768 for t in tiles)
    assert len(image) == plan['parameter_end']
    for tile in tiles:
        descriptor = Descriptor.decode(bytes.fromhex(tile['descriptor_hex']))
        assert descriptor.outputs == tile['count']
        assert tile['transfers'][-1]['ext'] >= plan['activation_slot_bytes']


def test_non_linear_or_runtime_constant_graph_fails_closed():
    program = _small_program()
    branched = copy.deepcopy(program)
    branched.layers[2].inputs = ['c']
    with pytest.raises(ValueError, match='linear chain'):
        compile_tiled(branched)
    with_constants = copy.deepcopy(program)
    with_constants.constants = {'hidden': np.zeros(1, np.int8)}
    with pytest.raises(ValueError, match='runtime constants'):
        compile_tiled(with_constants)


@pytest.mark.parametrize('name', ['kws', 'vww'])
def test_all_pinned_geometries_materialize_with_synthetic_parameters(name):
    """Geometry coverage only; the actual model weights are not in Git."""
    path = Path(__file__).resolve().parents[1] / \
        f'benchmarks/manifests/{name}.canonical-inventory.json'
    inventory = json.loads(path.read_text())
    shape = {item['name']: tuple(item['shape']) for item in inventory['tensors']}
    q = Quantization(.05, -7)
    activation_names = {inventory['operators'][0]['inputs'][0]} | {
        node['outputs'][0] for node in inventory['operators']}
    tensors = {key: SimpleNamespace(shape=shape[key], quantization=q)
               for key in activation_names}
    layers = []
    for node in inventory['operators']:
        params = {}
        if node['op'] in ('Conv', 'Gemm'):
            w = np.zeros(shape[node['inputs'][1]], np.int8)
            rows = w.shape[0]
            params = {'weight': w, 'corrected_bias': np.zeros(rows, np.int32),
                      'multiplier': np.full(rows, 1 << 30, np.int32),
                      'shift': np.full(rows, 30, np.uint8)}
        elif node['op'] == 'Clip':
            params = {'clip_bounds': np.array([-20, 20], np.int8)}
        layers.append(Layer(node['op'], [node['inputs'][0]], node['outputs'][0],
                            node['attributes'], params))
    program = Program(tensors, layers,
                      [inventory['operators'][0]['inputs'][0]],
                      [inventory['operators'][-1]['outputs'][0]], {}, {})
    materialized, image = compile_tiled(program)
    planned = plan_inventory(path)
    assert materialized['layers'] == planned['layers']
    assert len(image) == planned['parameter_end']
    assert hashlib.sha256(image).hexdigest() == \
           materialized['parameter_image_sha256']


def test_tiled_cli_checks_calibration_before_replacing_outputs(tmp_path):
    import onnx
    graph = model([h.make_node('Relu', ['x'], ['y'])], {}, [1, 8],
                  {'y': [1, 8]})
    sample = np.arange(-4, 4, dtype=np.float32).reshape(1, 8)
    cal = calibrate(graph, [{'x': sample}], ['test-input'])
    model_path, cal_path = tmp_path/'model.onnx', tmp_path/'cal.json'
    image_path, plan_path = tmp_path/'image.bin', tmp_path/'plan.json'
    onnx.save(graph, model_path)
    cal_path.write_text(json.dumps(cal))
    command = [sys.executable,
               str(Path(__file__).resolve().parents[1]/'tools/phase4/compile_tiled.py'),
               str(model_path), str(cal_path), str(image_path), str(plan_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(plan_path.read_text())
    assert len(plan['layers']) == 1
    assert hashlib.sha256(image_path.read_bytes()).hexdigest() == \
           plan['parameter_image_sha256']
    old_image, old_plan = image_path.read_bytes(), plan_path.read_bytes()
    cal['model_sha256'] = 'invalid'
    cal_path.write_text(json.dumps(cal))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert image_path.read_bytes() == old_image
    assert plan_path.read_bytes() == old_plan
