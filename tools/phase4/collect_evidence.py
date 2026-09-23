#!/usr/bin/env python3
"""Freeze routed kernel and abstract tile-DMA evidence, with strict scope labels."""
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/research/evidence/phase4'
BUILD=ROOT/'work/phase4/gowin-kernels-wide'
PNR=BUILD/'tinyml_v2/impl/pnr'
BITSTREAM=ROOT/'hardware/releases/phase4/tinyml_v4_kernels.fs'
OUT.mkdir(parents=True,exist_ok=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest=json.loads((BUILD/'source-manifest.json').read_text())
if manifest['target']['target_id']!=8196:
    raise ValueError('wrong phase-4 kernel target')
if manifest['manifest_sha256']!=sha(ROOT/'hardware/targets/tang_nano_20k_v2.json'):
    raise ValueError('target changed since route')
for name,digest in manifest['sources_sha256'].items():
    if sha(ROOT/name)!=digest:
        raise ValueError(f'board source changed since route: {name}')
if sha(BITSTREAM)!=sha(PNR/'tinyml_v2.fs'):
    raise ValueError('release candidate differs from route')

native_path=ROOT/'work/phase3/smallcnn/native-rtl-results.json'
native=json.loads(native_path.read_text())
if (native['sources_sha256']!=manifest['sources_sha256'] or
        native['jobs']!=1000 or native['integer_mismatches']):
    raise ValueError('SmallCNN regression stale or failed')
audit=json.loads((OUT/'inventory-audit.json').read_text())
if any(audit[m]['unsupported_geometry'] for m in ('kws','vww')):
    raise ValueError('pinned operator geometry not accounted for')
profiles_path=ROOT/'work/phase4/kernel-profiles.json'
profiles=json.loads(profiles_path.read_text())['profiles']
if ({p['label'] for p in profiles}!=
        {'kws_conv_10x4_s2','pointwise_4_to_5','vww_depthwise_s2',
         'vww_depthwise_256_channels','average_25x5','average_3x3','clip'}):
    raise ValueError('kernel microbenchmark coverage changed')
for profile in profiles:
    if profile['simulated_core_cycles']!=sum(profile[f'simulated_{part}_cycles']
                                             for part in ('compute','wait','control')):
        raise ValueError('kernel cycle counters do not reconcile')

cases={}
for name,path in (
    ('kernels',ROOT/'work/phase4/sim_kernels/results.xml'),
    ('tile_dma',ROOT/'work/phase4/sim_dma/results.xml'),
    ('engine',ROOT/'work/phase2/sim_engine/results.xml'),
    ('system',ROOT/'work/phase2/sim_system/results.xml'),
    ('board',ROOT/'work/phase2/sim_board/results.xml'),
    ('switch',ROOT/'work/phase3/sim_switch/results.xml'),
):
    elements=ET.parse(path).findall('.//testcase')
    if not elements or any(e.find('failure') is not None or e.find('error') is not None for e in elements):
        raise ValueError(f'failed RTL suite: {name}')
    cases[name]=[e.attrib['name'] for e in elements]
    shutil.copyfile(path,OUT/f'{name}-results.xml')

route=(PNR/'tinyml_v2.rpt.txt').read_text()
timing=(PNR/'tinyml_v2_tr_content.html').read_text()
def get(pattern,data):
    m=re.search(pattern,data,re.M|re.S)
    if not m:raise ValueError(f'missing route metric: {pattern}')
    return float(m.group(1))
resources={name:[get(r'^\s+'+name+r'\s+\|\s+([\d.]+)/[\d.]+',route),
                 get(r'^\s+'+name+r'\s+\|\s+[\d.]+/([\d.]+)',route)]
           for name in ('Logic','Register','BSRAM','DSP')}
fmax=get(r'<td>27\.000\(MHz\)</td>\s*<td>([\d.]+)\(MHz\)</td>',timing)
slack=get(r'<td class="label">Slack</td>\s*<td>([-\d.]+)</td>',timing)
if fmax<27 or slack<0 or resources['BSRAM'][0]!=16 or resources['DSP'][0]>24:
    raise ValueError('kernel candidate failed routed resource/timing gate')

(OUT/'gowin-route.txt').write_text(
    '\n'.join(line.rstrip() for line in route.splitlines()).rstrip()+'\n')
for source,dest in ((BUILD/'source-manifest.json','source-manifest.json'),
                    (ROOT/'work/phase4/ci.txt','ci.txt'),
                    (profiles_path,'kernel-profiles.json'),
                    (native_path,'smallcnn-1000-rtl.json')):
    shutil.copyfile(source,OUT/dest)
sources=['compiler/hardware_v2.py','compiler/memory_planner.py',
         'compiler/test_phase4_kernels.py','rtl/v2/tile_dma.sv',
         'test/phase4/test_kernels.py','test/phase4/test_tile_dma.py',
         'test/phase4/run.py','test/phase4/run_dma.py','test/phase3/test_switch.py',
         'tools/phase4/audit_inventories.py','tools/phase4/collect_evidence.py']
software_hashes={name:sha(ROOT/name) for name in sources}
(OUT/'software-source-manifest.json').write_text(json.dumps(software_hashes,indent=2)+'\n')
summary=dict(date='2026-09-23',scope='kernel-only routed board candidate; tile DMA is standalone abstract-port RTL',
             tool='Gowin Education V1.9.11.03',part='GW2AR-LV18QN88C8/I7',revision='C',
             constraint_mhz=27,routed_fmax_mhz=fmax,worst_setup_slack_ns=slack,
             resources=resources,bitstream_sha256=sha(BITSTREAM),
             raw_route_sha256=sha(PNR/'tinyml_v2.rpt.txt'),
             source_manifest_sha256=sha(BUILD/'source-manifest.json'),
             software_manifest_sha256=sha(OUT/'software-source-manifest.json'),
             pinned_inventory_sha256={m:audit[m]['inventory_sha256'] for m in audit},
             simulated_kernel_profiles_sha256=sha(profiles_path),
             native_smallcnn_jobs=native['jobs'],integer_mismatches=native['integer_mismatches'],
             native_smallcnn_results_sha256=sha(native_path),rtl_cases=cases,
             generic_clock_routing_warning=True,
             sdram_controller_integrated=False,dma_integrated=False,
             complete_kws_vww_rtl=False,physical_board_programmed=False,measured_power=False)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
