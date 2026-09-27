#!/usr/bin/env python3
"""Immutable fused engine plus proven512-byte UART transport.

Preparation, integration, native, route-inputs and preflight are boardless.
Only the explicit route stage invokes Gowin; no stage programs the FPGA.
"""
import argparse
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import combined_stream_uart512 as transport
from run_screening import save_json, verify_files
from variants import ROOT, check_frozen, sha

BASE=ROOT/'work/phase6/experiments-v1/fused-activation-uart512-v1'
CORE=ROOT/'work/phase6/experiments-v1/fused-activation-v1'
PROVEN_UART=ROOT/'work/phase6/experiments-v1/stream-mask-uart512-v1'
ENGINE=BASE/'engine.sv'


def checked_report(path, status='passed'):
    report=json.loads(path.read_text())
    if report.get('status')!=status:
        raise ValueError(f'{path}: expected {status}')
    return report


def prepare():
    check_frozen()
    core_identity=json.loads((CORE/'identity.json').read_text())
    core_sha=sha(CORE/'engine.sv')
    if core_sha!=core_identity['engine_sha256']:
        raise ValueError('fused parent engine changed')
    uart=json.loads((PROVEN_UART/'identity.json').read_text())
    for field,path in (('parser_sha256',transport.PARSER),('bridge_sha256',transport.BRIDGE),
            ('top_sha256',transport.TOP),('build_template_sha256',transport.BUILD),
            ('host_client_sha256',transport.HOST_CLIENT)):
        if sha(path)!=uart[field]:
            raise ValueError(f'proven512 transport source changed: {path}')
    uart_test=checked_report(PROVEN_UART/'integration/report.json')
    verify_files(ROOT,uart_test['sources'])
    if uart_test['tests']!=2 or sha(PROVEN_UART/'integration/results.xml')!=uart_test['results_sha256']:
        raise ValueError('proven512 integration changed')
    BASE.mkdir(parents=True,exist_ok=True)
    transport.immutable_write(ENGINE,(CORE/'engine.sv').read_text())
    record=dict(label=BASE.name,parent=str((CORE/'engine.sv').relative_to(ROOT)),
        parent_sha256=core_sha,engine_sha256=sha(ENGINE),
        parameters=core_identity['parameters'],
        parser_sha256=sha(transport.PARSER),bridge_sha256=sha(transport.BRIDGE),
        top_sha256=sha(transport.TOP),build_template_sha256=sha(transport.BUILD),
        host_client_sha256=sha(transport.HOST_CLIENT),
        proven_uart_identity_sha256=sha(PROVEN_UART/'identity.json'),
        proven_uart_integration_sha256=sha(PROVEN_UART/'integration/report.json'),
        change='byte-identical exact fused-activation core with proven512-byte CRC-committed UART bridge',
        physical_board=False)
    path=BASE/'identity.json'
    if path.exists() and json.loads(path.read_text())!=record:
        raise ValueError('immutable fused512 identity changed')
    if not path.exists():save_json(path,record)
    transport.BASE=BASE;transport.PARENT=CORE;transport.ENGINE=ENGINE
    transport.prepare=prepare
    transport.route_inputs=route_inputs
    return record


def native():
    identity=prepare()
    reference=checked_report(CORE/'native/report.json')
    binary=CORE/'native/Vv2_tiled_host_bridge'
    if reference['executable_sha256']!=sha(binary):
        raise ValueError('fused parent executable changed')
    verify_files(ROOT,reference['sources'])
    if reference['sources'].get(str((CORE/'engine.sv').relative_to(ROOT)))!=identity['engine_sha256']:
        raise ValueError('parent native executable used a different engine')
    manifest=checked_report(CORE/'fixtures.json','passed-replay')
    verify_files(ROOT,manifest['sources'])
    if reference['fixture_manifest_sha256']!=sha(CORE/'fixtures.json'):
        raise ValueError('parent fused fixture proof changed')
    target=BASE/'native-grouped-tail-prefetch';target.mkdir(exist_ok=True)
    report=dict(status='running',physical_board=False,label=BASE.name,
        engine_sha256=identity['engine_sha256'],executable_sha256=sha(binary),
        fixture_manifest_sha256=sha(CORE/'fixtures.json'),
        reused_executable=str(binary.relative_to(ROOT)),
        parent_native_report_sha256=sha(CORE/'native/report.json'),
        parent_sources=reference['sources'],
        scope='exact fused full-model core through byte-identical parent executable;512 host integration checked separately',
        results=[])
    save_json(target/'report.json',report)
    for item in manifest['fixtures']:
        name=item['name'];directory=CORE/'fixtures'/name;verify_files(directory,item['files'])
        for seed in (0,6063):
            path=target/f'{name}-s{seed}.json'
            subprocess.run([str(binary),str(directory),str(seed),str(path)],check=True)
            result=json.loads(path.read_text())
            if result['status']!='passed':raise ValueError('fused native output mismatch')
            report['results'].append(dict(fixture=name,seed=seed,result_sha256=sha(path),
                elapsed_cycles=result['elapsed_cycles'],engine_cycles=result['engine_cycles'],
                dma_cycles=result['dma_cycles'],tensor_checks=result['tensor_checks']))
            save_json(target/'report.json',report)
    if len(report['results'])!=12:raise ValueError('expected12 fused native cases')
    report['status']='passed';save_json(target/'report.json',report)
    return report


def route_inputs(mhz=24):
    identity=prepare()
    if mhz!=24:raise ValueError('this isolated composition targets24MHz')
    integration=checked_report(BASE/'integration/report.json')
    native_report=checked_report(BASE/'native-grouped-tail-prefetch/report.json')
    if any(r['engine_sha256']!=identity['engine_sha256'] for r in (integration,native_report)):
        raise ValueError('fused512 boardless engine identity mismatch')
    verify_files(ROOT,integration['sources'])
    name=transport.clock_label(mhz)
    parent_pll=(ROOT/'work/phase6/experiments-v1/combined-spec-scalar-uart-v1/pll.v').read_text()
    if parent_pll.count('.FBDIV_SEL(4), .IDIV_SEL(5)')!=1:
        raise ValueError('PLL source changed')
    pll=parent_pll.replace('.FBDIV_SEL(4), .IDIV_SEL(5)',
        '.FBDIV_SEL(7), .IDIV_SEL(8)').replace('// Experimental22.5 MHz from27 MHz.',
                                            '// Experimental24 MHz from27 MHz.')
    # Keep the existing template comment spelling while matching the divisors.
    pll=pll.replace('// Experimental 22.5 MHz from 27 MHz.','// Experimental 24 MHz from 27 MHz.')
    host=transport.TOP.read_text()
    if host.count('20250000')!=3:raise ValueError('host clock patch changed')
    host=host.replace('20250000','24000000')
    script=transport.BUILD.read_text()
    old='set root [file normalize [file join [file dirname [info script]] ../..]]'
    if script.count(old)!=1:raise ValueError('Gowin root patch changed')
    script=script.replace(old,f'set root {{{ROOT}}}')
    pll_path=BASE/f'pll{name}.v';host_path=BASE/f'host{name}.sv';script_path=BASE/f'build{name}.tcl'
    script=script.replace('hardware/phase4_sdram/pll_dma_20.v',str(pll_path.relative_to(ROOT)))
    script=script.replace('hardware/phase6/uart_burst_512_tiled_host.sv',str(host_path.relative_to(ROOT)))
    for path,source in ((pll_path,pll),(host_path,host),(script_path,script)):
        transport.immutable_write(path,source)
    record=dict(status='prepared-route-inputs',physical_board=False,core_clock_mhz=mhz,
        engine_sha256=identity['engine_sha256'],
        sources={str(p.relative_to(ROOT)):sha(p) for p in
            (ENGINE,pll_path,host_path,script_path,transport.PARSER,transport.BRIDGE)},
        build_script=str(script_path.relative_to(ROOT)),gowin_project='phase6_uart_burst_512')
    path=BASE/f'route-inputs-{name}.json'
    if path.exists() and json.loads(path.read_text())!=record:
        raise ValueError('immutable fused512 route inputs changed')
    if not path.exists():save_json(path,record)
    return record


def preflight():
    identity=prepare();inputs=route_inputs(24)
    evidence={}
    for label,count in (('engine',3),('edges',1),('stream-edges',1),('scalar-edges',1)):
        path=CORE/label/'report.json';report=checked_report(path)
        verify_files(ROOT,report['sources'])
        if report['sources'].get(str((CORE/'engine.sv').relative_to(ROOT)))!=identity['engine_sha256']:
            raise ValueError('reused RTL test engine differs')
        xml=CORE/label/'results.xml';cases=ET.parse(xml).findall('.//testcase')
        if sha(xml)!=report['results_sha256'] or len(cases)!=count or any(
            c.find('failure') is not None or c.find('error') is not None for c in cases):
            raise ValueError('reused RTL test XML differs')
        evidence[label]=sha(path)
    integration=checked_report(BASE/'integration/report.json')
    xml=BASE/'integration/results.xml';cases=ET.parse(xml).findall('.//testcase')
    if sha(xml)!=integration['results_sha256'] or len(cases)!=2 or any(
        c.find('failure') is not None or c.find('error') is not None for c in cases):
        raise ValueError('new512 integration XML differs')
    native_report=checked_report(BASE/'native-grouped-tail-prefetch/report.json')
    manifest=checked_report(CORE/'fixtures.json','passed-replay')
    verify_files(ROOT,manifest['sources']);verify_files(ROOT,native_report['parent_sources'])
    if (native_report['fixture_manifest_sha256']!=sha(CORE/'fixtures.json') or
        native_report['executable_sha256']!=sha(CORE/'native/Vv2_tiled_host_bridge') or
        len(native_report['results'])!=12):
        raise ValueError('new native proof differs')
    for item in manifest['fixtures']:
        verify_files(CORE/'fixtures'/item['name'],item['files'])
        for seed in (0,6063):
            rows=[r for r in native_report['results'] if r['fixture']==item['name'] and r['seed']==seed]
            if len(rows)!=1:raise ValueError('native case coverage missing')
            path=BASE/'native-grouped-tail-prefetch'/f'{item["name"]}-s{seed}.json'
            if sha(path)!=rows[0]['result_sha256'] or json.loads(path.read_text())['status']!='passed':
                raise ValueError('native result file changed')
    plan=dict(status='ready-for-route',physical_board=False,label=BASE.name,
        engine_sha256=identity['engine_sha256'],identity_sha256=sha(BASE/'identity.json'),
        core_test_reports=evidence,
        integration_report_sha256=sha(BASE/'integration/report.json'),
        native_report_sha256=sha(BASE/'native-grouped-tail-prefetch/report.json'),
        fixture_manifest=str((CORE/'fixtures.json').relative_to(ROOT)),
        fixture_manifest_sha256=sha(CORE/'fixtures.json'),
        route_inputs_sha256=sha(BASE/'route-inputs-24p0.json'),
        route_inputs=inputs,
        route_command='.venv/bin/python3 tools/phase6/combined_fused_uart512.py route',
        dependencies={str(p.relative_to(ROOT)):sha(p) for p in (
            Path(__file__),ROOT/'tools/phase6/combined_stream_uart512.py',
            ROOT/'tools/phase6/run_uart_burst_512_integration.py')})
    save_json(BASE/'preflight-plan.json',plan)
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','integration','native','route-inputs','preflight','route'))
    stage=parser.parse_args().stage;identity=prepare()
    result=(identity if stage=='prepare' else transport.integration() if stage=='integration' else
        native() if stage=='native' else route_inputs() if stage=='route-inputs' else
        preflight() if stage=='preflight' else transport.route(24))
    print(json.dumps(dict(stage=stage,status=result.get('status','prepared'),
        engine_sha256=identity['engine_sha256'])),flush=True)
