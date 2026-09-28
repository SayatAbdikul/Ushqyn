"""Conservative same-layer input DMA prefetch for matched B1/B2 baselines.

An input tile may start loading during the preceding run only when both runs
execute the same graph layer, the source tensor was complete before that layer,
the destination is outside the active engine live range, and no intervening
SRAM transfer or external store conflicts with the move. Command replay and
native execution provide the final correctness checks.
"""
import json
from pathlib import Path

from prefetch_tail import commands


def next_input_candidates(fixture):
    fixture = Path(fixture)
    schedule = json.loads((fixture / 'schedule.json').read_text())
    code = commands((fixture / 'commands.bin').read_bytes())
    inputs = {(load['ext'], load['sram'], load['bytes'])
              for stage in schedule['stages'] for load in stage['loads']
              if load['role'] == 'input' and load['direction'] == 'to_sram'}
    runs = [i for i, command in enumerate(code) if command[0] == 2]
    same_layer_adjacencies = 0
    candidates = []
    rejected = {'live_overlap': 0, 'sram_transfer': 0, 'external_write': 0}
    for here, following in zip(runs, runs[1:]):
        current = schedule['run_contracts'].get(str(here))
        next_run = schedule['run_contracts'].get(str(following))
        if (current is None or next_run is None or
                current['layer'] != next_run['layer']):
            continue
        same_layer_adjacencies += 1
        low, high = code[here][4] & 65535, code[here][4] >> 16
        if code[here + 1][:2] != (3, 3):
            raise ValueError('engine RUN lacks immediate WAIT_ENGINE')
        for index in range(here + 2, following):
            op, flags, _, ext, sram, size = code[index]
            if op != 1 or flags != 1 or (ext, sram, size) not in inputs:
                continue
            if index + 1 >= following or code[index + 1][:2] != (3, 2):
                raise ValueError('input DMA lacks immediate WAIT_DMA')
            if not (sram + size <= low or sram >= high):
                rejected['live_overlap'] += 1
                continue
            if any(j != index and code[j][0] == 1 and
                   sram < code[j][4] + code[j][5] and code[j][4] < sram + size
                   for j in range(here + 1, following)):
                rejected['sram_transfer'] += 1
                continue
            if any(code[j][0] == 1 and code[j][1] == 0 and
                   ext < code[j][3] + code[j][5] and code[j][3] < ext + size
                   for j in range(here + 1, index)):
                rejected['external_write'] += 1
                continue
            candidates.append(dict(current_run_command=here,
                next_run_command=following, dma_command=index, role='input',
                ext=ext, sram=sram, bytes=size, current_live=[low, high],
                safety='same graph layer, disjoint live SRAM, no intervening SRAM transfer or external overwrite'))
    return dict(same_layer_adjacencies=same_layer_adjacencies,
                candidates=candidates, eligible_bytes=sum(x['bytes'] for x in candidates),
                rejected=rejected)
