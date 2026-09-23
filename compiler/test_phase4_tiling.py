"""Independent schedule checks for all frozen KWS and VWW node geometries."""

import json
import math
from pathlib import Path

import pytest

from hardware_v2 import Descriptor
from phase4_tiling import EXT_BYTES, SRAM_BYTES, plan_inventory

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('model,nodes', [('kws', 22), ('vww', 58)])
def test_pinned_inventory_schedules_and_data_movement(model, nodes):
    path = ROOT / f'benchmarks/manifests/{model}.canonical-inventory.json'
    inventory = json.loads(path.read_text())
    tensors = {t['name']: t for t in inventory['tensors']}
    schedule = plan_inventory(path)
    assert len(schedule['layers']) == nodes
    assert schedule['parameter_end'] <= EXT_BYTES
    external = bytearray(EXT_BYTES)
    sram = bytearray(SRAM_BYTES)
    slot_size = schedule['activation_slot_bytes']
    assert 2*slot_size < schedule['parameter_end']
    expected_slot = 0
    for layer, node in zip(schedule['layers'], inventory['operators']):
        assert layer['index'] == inventory['operators'].index(node)
        if layer['kind'] in ('alias', 'host_layout'):
            assert not layer['tiles']
            assert math.prod(tensors[node['inputs'][0]]['shape']) == math.prod(tensors[node['outputs'][0]]['shape'])
            continue
        assert layer['input_slot'] == expected_slot
        assert layer['output_slot'] == 1-expected_slot
        n_input = math.prod(tensors[node['inputs'][0]]['shape'])
        n_output = math.prod(tensors[node['outputs'][0]]['shape'])
        assert layer['input_bytes'] == n_input and layer['output_bytes'] == n_output
        src = expected_slot * slot_size
        dst = (1-expected_slot) * slot_size
        source_pattern = bytes((i*31 + layer['index']*17) & 255 for i in range(n_input))
        external[src:src+n_input] = source_pattern
        external[dst:dst+n_output] = b'\xa5' * n_output
        output_coverage = bytearray(n_output)
        first = 0
        if layer['kind'] in ('conv', 'depthwise', 'avgpool', 'maxpool'):
            output_plane = math.prod(tensors[node['outputs'][0]]['shape'][2:])
        else:
            output_plane = 1
        for tile in layer['tiles']:
            assert tile['first'] == first
            first += tile['count']
            regions = sorted((r['base'], r['base']+r['bytes']) for r in tile['regions'].values())
            assert regions[0][0] >= 128 and regions[-1][1] <= SRAM_BYTES
            assert all(left[1] <= right[0] for left, right in zip(regions, regions[1:]))
            desc = Descriptor.decode(bytes.fromhex(tile['descriptor_hex']))
            desc.validate()
            assert desc.input == tile['regions']['input']['base']
            assert desc.output == tile['regions']['output']['base']
            assert desc.outputs == tile['regions']['output']['bytes']
            for transfer in tile['transfers'][:-1]:
                assert transfer['direction'] == 'to_sram'
                assert transfer['ext'] % 8 == transfer['sram'] % 8 == 0
                start, end = transfer['ext'], transfer['ext']+transfer['bytes']
                sram[transfer['sram']:transfer['sram']+transfer['bytes']] = external[start:end]
                assert sram[transfer['sram']:transfer['sram']+transfer['bytes']] == external[start:end]
            writeback = tile['transfers'][-1]
            assert writeback['direction'] == 'from_sram'
            output_offset = tile['first']*output_plane
            assert writeback['ext'] == dst+output_offset
            assert writeback['bytes'] == tile['count']*output_plane
            output_coverage[output_offset:output_offset+writeback['bytes']] = b'\x01'*writeback['bytes']
            result = bytes((output_offset+i+layer['index']) & 255 for i in range(writeback['bytes']))
            sram[writeback['sram']:writeback['sram']+writeback['bytes']] = result
            external[writeback['ext']:writeback['ext']+writeback['bytes']] = result
            assert external[src:src+n_input] == source_pattern, 'live source was overwritten'
        assert all(output_coverage), f'{model} node {layer["index"]} has an output hole'
        assert first == (tensors[node['outputs'][0]]['shape'][1] if layer['kind'] in
                         ('conv', 'depthwise', 'avgpool', 'maxpool') else n_output)
        expected = bytes((i+layer['index']) & 255 for i in range(n_output))
        assert external[dst:dst+n_output] == expected
        expected_slot = 1-expected_slot
    assert schedule['final_output_slot'] == expected_slot


def test_unfittable_or_unaligned_tiling_fails_closed(tmp_path):
    original = json.loads((ROOT/'benchmarks/manifests/kws.canonical-inventory.json').read_text())
    # A single required input exceeds SRAM and ordinary convolution cannot
    # divide the reduction dimension with the current descriptor ABI.
    tensor = next(t for t in original['tensors'] if t['name'] == original['operators'][1]['inputs'][0])
    tensor['shape'] = [1, 1, 255, 255]
    source = next(t for t in original['tensors'] if t['name'] == original['operators'][0]['inputs'][0])
    source['shape'] = [1, 255, 255, 1]
    path = tmp_path/'oversized.json'
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match='no aligned tile fits'):
        plan_inventory(path)
