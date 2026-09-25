#!/usr/bin/env python3
"""Pin the routed, physically exercised host image and its Gowin reports."""

import gzip
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/'docs/research/evidence/phase4'
PNR = ROOT/'work/phase4/tiled-host-route-v4/phase4_tiled_host/impl/pnr'
PREFIX = 'phase4_tiled_host'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def number(pattern, content):
    match = re.search(pattern, content)
    if not match:
        raise ValueError(f'Gowin report changed format: {pattern}')
    return float(match.group(1))


def main():
    prior = json.loads((EVIDENCE/'physical-tiled-host-route.json').read_text())
    files = {
        'route_report_sha256': (PNR/f'{PREFIX}.rpt.txt',
                                'physical-tiled-host-overlap-route.txt.gz'),
        'timing_report_sha256': (PNR/f'{PREFIX}_tr_content.html',
                                 'physical-tiled-host-overlap-timing.html.gz'),
        'power_report_sha256': (PNR/f'{PREFIX}.power.html',
                                'physical-tiled-host-overlap-power.html.gz'),
    }
    report = dict(prior)
    for field, (source, archive) in files.items():
        data = source.read_bytes()
        (EVIDENCE/archive).write_bytes(gzip.compress(data, mtime=0))
        report[field] = sha(data)
    text = files['route_report_sha256'][0].read_text()
    timing = files['timing_report_sha256'][0].read_text()
    power = files['power_report_sha256'][0].read_text()
    for key, label in (('logic', 'Logic'), ('register', 'Register'),
                       ('bsram', 'BSRAM'), ('dsp', 'DSP')):
        match = re.search(r'^\s*'+label+r'\s*\|\s*([\d.]+)/([\d.]+)',
                          text, re.MULTILINE)
        if not match:
            raise ValueError(f'missing {label} resource row')
        report['resources'][key] = {'used': float(match.group(1)),
                                    'available': int(match.group(2))}
    report['routed_core_fmax_mhz'] = number(
        r'<td>20\.250\(MHz\)</td>\s*<td>([\d.]+)\(MHz\)</td>', timing)
    report['setup_tns_ns'] = number(
        r'pll/pll_s2/CLKOUT\.default_gen_clk</td>\s*<td>Setup</td>\s*'
        r'<td>([\d.]+)</td>', timing)
    for key, label in (('total_mw', 'Total Power'),
                       ('quiescent_mw', 'Quiescent Power'),
                       ('dynamic_mw', 'Dynamic Power')):
        report['power_estimate'][key] = number(
            rf'<td class="label">{label} \(mW\)</td>\s*<td>([\d.]+)</td>',
            power)
    report['bitstream_sha256'] = sha((ROOT/'hardware/releases/phase4-sdram/'
                                      'hs_tiled_host_overlap.fs').read_bytes())
    for filename in report['source_sha256']:
        report['source_sha256'][filename] = sha((ROOT/filename).read_bytes())
    report['status'] = 'passed-post-route-and-physically-tested'
    report['scope'] = ('UART host, tiled engine/DMA/scratchpad, Gowin HS SDRAM, '
                       'guarded concurrent DMA and overlap counter')
    report['physical_validation'] = ('KWS/VWW all-node exactness and a directed '
                                     'compute/DMA overlap with live-region rejection')
    report['limits'] = ['single-outstanding two-word SDRAM commands',
                        'no full-model ping-pong schedule or streaming long bursts',
                        'power estimate is not measured board energy']
    output = EVIDENCE/'physical-tiled-host-overlap-route.json'
    output.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'bitstream_sha256': report['bitstream_sha256'],
                      'core_fmax_mhz': report['routed_core_fmax_mhz'],
                      'resources': report['resources']}))


if __name__ == '__main__':
    main()
