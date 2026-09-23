#!/usr/bin/env python3
"""Freeze reproducible post-route and simulated SmallCNN evidence."""
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/research/evidence/phase3'
BUILD=ROOT/'work/phase3/gowin-cache-pipeline'
PNR=BUILD/'tinyml_v2/impl/pnr'
FIXTURE=ROOT/'work/phase3/smallcnn'
FULL=ROOT/'work/phase3/full-quality'
BITSTREAM=ROOT/'hardware/releases/phase3/tinyml_v3.fs'
OUT.mkdir(parents=True,exist_ok=True)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def integer(pattern,data):
    match=re.search(pattern,data,re.M|re.S)
    if not match:raise ValueError(f'missing report field: {pattern}')
    return tuple(float(x) if '.' in x else int(x) for x in match.groups())


manifest=json.loads((BUILD/'source-manifest.json').read_text())
for name,digest in manifest['sources_sha256'].items():
    if sha(ROOT/name)!=digest:raise ValueError(f'RTL changed since source manifest: {name}')
native=json.loads((FIXTURE/'native-rtl-results.json').read_text())
if native['sources_sha256']!=manifest['sources_sha256'] or native['jobs']!=1000 or native['integer_mismatches']:
    raise ValueError('native result/source mismatch')
if native['fixture_sha256']!=sha(FIXTURE/'checks.npz') or native['image_sha256']!=sha(FIXTURE/'board.bin'):
    raise ValueError('native result/fixture mismatch')
if any(c[4]!=61184 or c[0]!=sum(c[1:4]) or c[7]!=8 for c in native['counters']):
    raise ValueError('SmallCNN counter mismatch')
meta=json.loads((FIXTURE/'board.json').read_text())
quality=json.loads((FULL/'board.json').read_text())
if meta['target_manifest_sha256']!=manifest['manifest_sha256'] or quality['target_manifest_sha256']!=manifest['manifest_sha256']:
    raise ValueError('model target mismatch')
if sha(FIXTURE/'board.bin')!=meta['image_sha256']:
    raise ValueError('model image hash mismatch')
if quality['jobs']!=10000 or quality['image_sha256']!=meta['image_sha256']:
    raise ValueError('full-quality model mismatch')
with np.load(FULL/'quality-predictions.npz',allow_pickle=False) as predictions:
    correct=int(np.sum(np.argmax(predictions['outputs'].reshape(10000,-1),axis=1)==predictions['labels']))
    if correct!=quality['integer_correct']:
        raise ValueError('full-quality prediction summary mismatch')
route=(PNR/'tinyml_v2.rpt.txt').read_text()
timing=(PNR/'tinyml_v2_tr_content.html').read_text()
resources={name:integer(r'^\s+'+name+r'\s+\|\s+([\d.]+)/([\d.]+)',route)
           for name in ('Logic','Register','BSRAM','DSP')}
ssram,=integer(r'--SSRAM\(RAM16\)\s+\|\s+(\d+)',route)
resources['SSRAM_RAM16_used']=ssram
fmax,=integer(r'<td>27\.000\(MHz\)</td>\s*<td>([\d.]+)\(MHz\)</td>',timing)
slack,=integer(r'<td class="label">Slack</td>\s*<td>([-\d.]+)</td>',timing)
if resources['BSRAM'][0]!=meta['memory']['nominal_bsram_blocks'] or fmax<27 or slack<0:
    raise ValueError('bank budget/timing gate failed')
if sha(BITSTREAM)!=sha(PNR/'tinyml_v2.fs'):
    raise ValueError('release bitstream differs from routed build')

tests={}
for suite in ('engine','system','board','switch'):
    path=ROOT/f'work/{"phase3" if suite=="switch" else "phase2"}/sim_{suite}/results.xml'
    tree=ET.parse(path)
    cases=tree.findall('.//testcase')
    if not cases or any(c.find('failure') is not None or c.find('error') is not None for c in cases):
        raise ValueError(f'RTL suite failure: {suite}')
    tests[suite]=[c.attrib['name'] for c in cases]
    shutil.copyfile(path,OUT/f'{suite}-results.xml')
for source,dest in ((BUILD/'source-manifest.json','source-manifest.json'),
                    (FIXTURE/'native-rtl-results.json','smallcnn-1000-rtl.json')):
    shutil.copyfile(source,OUT/dest)
(OUT/'gowin-route.txt').write_text(
    '\n'.join(line.rstrip() for line in route.splitlines()).rstrip()+'\n')
(OUT/'memory-plan.json').write_text(json.dumps(dict(image_sha256=meta['image_sha256'],
    memory=meta['memory'],segments=meta['segments']),indent=2)+'\n')
(OUT/'quality-summary.json').write_text(json.dumps(dict(jobs=quality['jobs'],
    float_correct=quality['float_correct'],integer_correct=quality['integer_correct'],
    model_weights_sha256=quality['weights_sha256'],model_onnx_sha256=quality['source_sha256'],
    image_sha256=quality['image_sha256'],dataset_sha256=quality['dataset_sha256'],
    calibration_ids=quality['calibration_ids']),indent=2)+'\n')
software_sources=['compiler/hardware_v2.py','compiler/memory_planner.py',
                  'compiler/memory_verifier.py','compiler/static_pipeline.py',
                  'compiler/integer_reference.py','compiler/smallcnn.py',
                  'tools/phase3/prepare_smallcnn.py','tools/phase3/run_native.py',
                  'tools/phase3/collect_evidence.py','test/phase3/cnn_native.cpp',
                  'test/phase3/run.py','test/phase3/test_switch.py']
software_hashes={name:sha(ROOT/name) for name in software_sources}
(OUT/'software-source-manifest.json').write_text(json.dumps(software_hashes,indent=2)+'\n')
summary=dict(date='2026-09-23',tool='Gowin Education V1.9.11.03',
             part='GW2AR-LV18QN88C8/I7',revision='C',constraint_mhz=27,
             routed_fmax_mhz=fmax,worst_setup_slack_ns=slack,resources=resources,
             bitstream_sha256=sha(BITSTREAM),bitstream_bytes=BITSTREAM.stat().st_size,
             raw_route_sha256=sha(PNR/'tinyml_v2.rpt.txt'),
             source_manifest_sha256=sha(BUILD/'source-manifest.json'),
             software_source_manifest_sha256=sha(OUT/'software-source-manifest.json'),
             board_image_sha256=sha(FIXTURE/'board.bin'),
             fixture_sha256=sha(FIXTURE/'checks.npz'),
             full_quality_predictions_sha256=sha(FULL/'quality-predictions.npz'),
             simulated_jobs=native['jobs'],integer_mismatches=native['integer_mismatches'],
             all_layer_jobs=native['all_layer_jobs'],stage_checks=native['stage_checks'],
             core_cycles_per_job=native['counters'][0][0],
             useful_macs_per_job=native['counters'][0][4],
             full_mnist_float_correct=quality['float_correct'],
             full_mnist_integer_correct=quality['integer_correct'],
             directed_rtl_tests=tests,generic_clock_routing_warning=True,
             physical_board_programmed=False,measured_power=False)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
