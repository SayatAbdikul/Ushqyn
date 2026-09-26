"""Restricted per-layer tuning on the frozen, fully materializing command ABI.

The candidate space is full-32KiB versus preferred-16KiB channel/element tiles,
with serialized DMA. It is an engineering baseline, not the proposed spatial
fusion scheduler or a B3 reproduction. Costs are engine + fitted DMA components,
not total latency, confidence bounds or measured candidate performance.
"""
import copy
from dataclasses import replace
from functools import lru_cache
import hashlib
import math
from pathlib import Path

from hardware_v2 import Descriptor
from phase4_compile import compile_tiled
from phase4_sequence import compile_sequence
from phase4_cost import engine_cycles
from .catalogue import Catalogue, Choice, search
from .contract import require


@lru_cache(maxsize=4096)
def _engine(encoded):
    return engine_cycles(Descriptor.decode(bytes.fromhex(encoded)))


class CostModel:
    def __init__(self, record):
        source = Path(__file__).resolve().parents[1] / 'phase4_cost.py'
        require(record['status'] == 'passed' and hashlib.sha256(source.read_bytes()).hexdigest() == record['predictor_sha256'],
                'unvalidated or changed engine cost source')
        self.record = record

    def dma(self, transfer):
        n, a = transfer['bytes'], transfer['ext']
        features = [1, (n+7)//8, (a % 64+n+63)//64, int((a+n-1) % 64 >= 56), int(a % 64 != 0)]
        return max(1, math.ceil(sum(x*y for x, y in zip(features, self.record['dma'][transfer['direction']]['coefficients']))))

    def layer(self, tiles):
        engine = sum(_engine(t['descriptor_hex']) for t in tiles)
        transfers = [load for t in tiles for load in (*t['loads'], t['transfers'][-1])]
        return {'engine_cycles': engine, 'dma_fit_cycles': sum(self.dma(t) for t in transfers),
                'dma_bytes': sum(t['bytes'] for t in transfers), 'tiles': len(tiles)}


def optimize(program, cost_model, *, beam_width=8, seconds=60, max_expansions=100000):
    plans, images, schedules, costs = {}, {}, {}, {}
    for label, half in (('full32', False), ('prefer16', True)):
        plans[label], images[label] = compile_tiled(program, prefer_half=half)
        _, _, schedules[label] = compile_sequence(plans[label], images[label], overlap=False)
        costs[label] = {i: cost_model.layer([t for t in schedules[label]['tiles'] if t['layer'] == i])
                        for i in range(len(program.layers))}
    require(images['full32'] == images['prefer16'] and
            plans['full32']['activation_slot_bytes'] == plans['prefer16']['activation_slot_bytes'],
            'candidate boundary/immutable image differs')
    # All combinations must fit global budgets; otherwise a scalar frontier is
    # insufficient and command/payload occupancy must enter the search state.
    max_tiles = sum(max(costs[label][i]['tiles'] for label in costs) for i in range(len(program.layers)))
    max_commands = 1 + sum(max(sum(2*(len(t['loads'])+1)+2 for t in schedules[label]['tiles'] if t['layer'] == i)
                               for label in schedules) for i in range(len(program.layers)))
    require(max_commands*16 <= 32768 and len(images['full32'])+128*max_tiles <= 8*1024*1024,
            'mixed catalogue needs capacity state')
    choices, selection = [], {}
    fallback = []
    for i in range(len(program.layers)):
        seen = set()
        for label in ('full32', 'prefer16'):
            # Geometry and immutable offsets determine the legal local choice.
            signature = repr(plans[label]['layers'][i])
            if signature in seen:
                continue
            seen.add(signature)
            name = f'{i:03d}:{label}'
            local = costs[label][i]
            choices.append(Choice(name, i, i+1, 'materialized', 'materialized',
                                  local['engine_cycles']+local['dma_fit_cycles']))
            selection[name] = label
            if label == 'full32':
                fallback.append(name)
    catalogue = Catalogue(len(program.layers), tuple(choices))
    exact = search(catalogue, seconds=seconds, max_expansions=max_expansions, fallback=fallback)
    beam = search(catalogue, beam_width=beam_width, seconds=seconds, max_expansions=max_expansions, fallback=fallback)
    # Causal cost ablation: same legal space and checks, only the DMA term removed.
    blind_catalogue = replace(catalogue, choices=tuple(replace(c, cost=costs[selection[c.name]][c.start]['engine_cycles'])
                                                      for c in catalogue.choices))
    blind = search(blind_catalogue, seconds=seconds, max_expansions=max_expansions, fallback=fallback)
    artifacts = {}
    paths = {'full32-serial': fallback,
             'prefer16-serial': [next(c.name for c in choices if c.start == i and
                                     (selection[c.name] == 'prefer16' or len([k for k in choices if k.start == i]) == 1))
                                 for i in range(len(program.layers))],
             'mixed-serial': beam['path'], 'dma-blind-serial': blind['path']}
    for label, path in paths.items():
        mixed = copy.deepcopy(plans['full32'])
        for i, name in enumerate(path):
            mixed['layers'][i] = copy.deepcopy(plans[selection[name]]['layers'][i])
        commands, payload, schedule = compile_sequence(mixed, images['full32'], overlap=False)
        actual = cost_model.layer(schedule['tiles'])
        predicted = sum(next(c.cost for c in choices if c.name == n) for n in path)
        require(actual['engine_cycles']+actual['dma_fit_cycles'] == predicted, 'non-additive candidate cost')
        artifacts[label] = {'commands': commands, 'payload': payload, 'schedule': schedule,
                            'path': path, 'components': actual}
    # An executable paired configuration for a future physical overlap ablation.
    # No additive latency score is assigned: arbitration changes engine cycles.
    for label, path in (('mixed-prefetch', beam['path']), ('prefer16-prefetch', paths['prefer16-serial'])):
        mixed = copy.deepcopy(plans['full32'])
        for i, name in enumerate(path):
            mixed['layers'][i] = copy.deepcopy(plans[selection[name]]['layers'][i])
        commands, payload, schedule = compile_sequence(mixed, images['full32'], overlap=True)
        artifacts[label] = {'commands': commands, 'payload': payload, 'schedule': schedule,
                            'path': path, 'components': None}
    return artifacts, {'exact': exact, 'beam': beam, 'dma_blind': blind,
                       'choices': [dict(name=c.name, layer=c.start, cost=c.cost, **costs[selection[c.name]][c.start])
                                   for c in choices],
                       'worst_catalogue_command_bytes': max_commands*16,
                       'cost_scope': 'serialized engine plus fitted DMA components; excludes dispatch/host uncertainty; not a latency bound',
                       'candidate_scope': 'materialized layer boundaries, full32/prefer16 only; no spatial fusion or B3'}
