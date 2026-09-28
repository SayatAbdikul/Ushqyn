#!/usr/bin/env python3
"""Screen compiler-certified spatial fast-path hints on the existing engine.

This is an isolated ABI experiment. An unmarked descriptor uses the generic
spatial walker. Hints are assigned only after the ordinary descriptor
validator and exact geometry predicate agree; the RTL trusts marked hints.
Nothing here changes the production compiler or programs the board.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'compiler'))
from hardware_v2 import Descriptor

BASE = ROOT / 'work/phase6/novelty-contracts-v1'
SOURCE = ROOT / 'work/phase6/pool-timing-v1/engine.sv'
ENGINE = BASE / 'engine.sv'
FIXTURES = ROOT / 'work/phase6/channel-compaction-v1/fused/fixtures'
BASELINE = ROOT / 'work/phase6/pool-timing-v1/compact-native/report.json'
ROUTE_BASELINE = ROOT / 'work/phase6/pool-timing-v1/route27/report.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('engine patch site changed: ' + old[:90])
    return text.replace(old, new, 1)


def candidate_engine():
    src = SOURCE.read_text()
    src = replace_once(src,
        "    wire [23:0] fused_table_base = {6'b0,d[0][63:49],3'b0};\n"
        "    wire fusion_abi_valid = d[0][63:48]==0 ||\n"
        "        (fused_activation&&(op==OP_CONV||op==OP_DWCONV)&&fused_table_base<=24'd32512);",
        "    // Bits 61/62 carry compiler-proven PW/DW fast-path contracts. The\n"
        "    // fused-table index needs only bits 49:60 for a 32 KiB SRAM.\n"
        "    wire [23:0] fused_table_base = {9'b0,d[0][60:49],3'b0};\n"
        "    wire fusion_abi_valid = !d[0][63] && d[0][62:61]!=2'b11 &&\n"
        "        (fused_activation ? ((op==OP_CONV||op==OP_DWCONV)&&\n"
        "                             fused_table_base<=24'd32512) : d[0][60:49]==0) &&\n"
        "        (!d[0][61] || op==OP_CONV) && (!d[0][62] || op==OP_DWCONV);")
    src = replace_once(src,
        "    wire spatial_pw_decode=SPATIAL_PW&&op==OP_CONV&&kh==1&&kw==1&&sh==1&&sw==1&&\n"
        "        pt==0&&pbm==0&&pl==0&&pr==0&&count<=256;\n", "")
    src = replace_once(src,
        "    wire broadcast_mode=(!spatial_pw&&op==OP_CONV&&oc_total>1&&count<=128);",
        "    // Validate marked descriptors in GEOM_CHECK, before any fast-path\n"
        "    // memory request. Unmarked descriptors use the generic walker.\n"
        "    wire pw_contract_valid = kh==1&&kw==1&&sh==1&&sw==1&&\n"
        "        pt==0&&pbm==0&&pl==0&&pr==0&&count<=256;\n"
        "    wire dw_contract_valid = kh==3&&kw==3&&sh==1&&sw==1;\n"
        "    wire broadcast_mode=(!spatial_pw&&op==OP_CONV&&oc_total>1&&count<=128);")
    src = replace_once(src,
        "                    spatial_pw<=spatial_pw_decode;\n"
        "                    spatial_dw<=SPATIAL_DW&&op==OP_DWCONV&&kh==3&&kw==3&&sh==1&&sw==1;",
        "                    spatial_pw<=SPATIAL_PW&&d[0][61];\n"
        "                    spatial_dw<=SPATIAL_DW&&d[0][62];")
    src = replace_once(src,
        "                        kh==0||kw==0||kh>31||kw>31||ih==0||iw==0||ic==0||oc_total==0||",
        "                        (d[0][61]&&!pw_contract_valid)||(d[0][62]&&!dw_contract_valid)||\n"
        "                        kh==0||kw==0||kh>31||kw>31||ih==0||iw==0||ic==0||oc_total==0||")
    return '// Isolated compiler-certificate fast-path experiment.\n' + src


def normalized_descriptor(blob):
    if len(blob) != 64:
        raise ValueError('descriptor bytes')
    clean = bytearray(blob)
    clean[6:8] = b'\0\0'
    d = Descriptor.decode(bytes(clean))
    d.validate()
    return d


def contract_kind(d):
    pw = (d.opcode == 4 and d.kernel_h == d.kernel_w ==
          d.stride_h == d.stride_w == 1 and
          d.pad_top == d.pad_bottom == d.pad_left == d.pad_right == 0 and
          d.count <= 256)
    dw = (d.opcode == 6 and d.kernel_h == d.kernel_w == 3 and
          d.stride_h == d.stride_w == 1)
    return 'pw' if pw else 'dw' if dw else 'generic'


def patch_fixture(name):
    source = FIXTURES / name
    dest = BASE / 'fixtures' / name
    dest.mkdir(parents=True, exist_ok=True)
    for file in source.iterdir():
        if file.name != 'payload.bin' and file.is_file():
            shutil.copyfile(file, dest / file.name)
    payload = bytearray((source / 'payload.bin').read_bytes())
    schedule = json.loads((source / 'schedule.json').read_text())
    stage_layers = {}
    for stage in schedule['stages']:
        for load in stage['loads']:
            if load['role'] == 'descriptor':
                stage_layers[load['ext']] = stage['layer']
    certs = []
    commands = (source / 'commands.bin').read_bytes()
    if len(commands) % 16:
        raise ValueError('unaligned command stream')
    for command in range(len(commands) // 16):
        op, flags, reserved, off, sram, size = struct.unpack_from('<BBHIII', commands, command*16)
        if op != 1 or flags != 1 or sram != 0 or size != 128:
            continue
        if off+128 > len(payload):
            raise ValueError('descriptor DMA exceeds payload')
        old = bytes(payload[off:off+64])
        if old[:4] != b'USD2':
            raise ValueError('descriptor DMA lacks magic')
        if off in stage_layers:
            stage = next(s for s in schedule['stages'] if any(
                x['role'] == 'descriptor' and x['ext'] == off for x in s['loads']))
            untagged = bytes.fromhex(stage['descriptor_hex'])
            if old[:6] + b'\0\0' + old[8:] != untagged[:6] + b'\0\0' + untagged[8:]:
                raise ValueError('stage descriptor geometry differs from payload')
        d = normalized_descriptor(old)
        kind = contract_kind(d)
        if kind != 'generic':
            bit = 0x20 if kind == 'pw' else 0x40
            if old[7] & 0xe0:
                raise ValueError('reserved contract bits already used')
            payload[off+7] |= bit
        certs.append(dict(command=command, layer=stage_layers.get(off), ext=off,
                          descriptor_sha256=hashlib.sha256(old).hexdigest(),
                          contract=kind, opcode=d.opcode,
                          kernel=[d.kernel_h, d.kernel_w],
                          stride=[d.stride_h, d.stride_w], count=d.count))
    (dest / 'payload.bin').write_bytes(payload)
    return dict(name=name, original_payload_sha256=digest(source / 'payload.bin'),
                certified_payload_sha256=digest(dest / 'payload.bin'),
                contracts=certs)


def prepare():
    BASE.mkdir(parents=True, exist_ok=True)
    source = candidate_engine()
    if ENGINE.exists() and ENGINE.read_text() != source and digest(ENGINE) != \
            '3327fe9ca563b4d953c9977cc0495ef218cb29492e92ebb6cf16fc70472f0340':
        raise ValueError('candidate engine changed externally; use a new label')
    ENGINE.write_text(source)
    selected = [
        'kws-pinned-compacted-fused-check',
        'vww-pinned-compacted-fused-check',
        'kws-stress-compacted-fused-check',
        'vww-stress-compacted-fused-check',
        'kws-pinned-compacted-fused-timed',
        'vww-pinned-compacted-fused-timed',
    ]
    fixtures = [patch_fixture(name) for name in selected]
    report = dict(status='prepared', physical_board=False,
                  source_sha256=digest(SOURCE), engine_sha256=digest(ENGINE),
                  fixture_count=len(fixtures), fixtures=fixtures,
                  trust_boundary='compiler proves hints; RTL rejects contradictory or ineligible hints before memory writes')
    path = BASE / 'report.json'
    if path.exists():
        prior = json.loads(path.read_text())
        if prior.get('status') == 'passed-native' and all(prior.get(k) == report[k]
                for k in ('source_sha256', 'engine_sha256', 'fixtures')):
            return prior
    save(path, report)
    return report


def native():
    report = prepare()
    build = BASE / 'native'
    build.mkdir(exist_ok=True)
    sources = [ROOT / 'rtl/v2' / n for n in (
        'target_pkg.sv', 'requantizer.sv', 'scratchpad.sv',
        'tile_dma.sv', 'tiled_core.sv', 'command.sv',
        'tile_sequencer.sv', 'tiled_host_bridge.sv')]
    sources.insert(2, ENGINE)
    harness = ROOT / 'test/phase6/native.cpp'
    with (build / 'build.log').open('w') as log:
        subprocess.run(['verilator', '--cc', '--exe', '--build', '-j', '2',
                        '--public-flat-rw', '-Wno-fatal', '--top-module',
                        'v2_tiled_host_bridge', '--Mdir', str(build),
                        *map(str, sources), str(harness)], cwd=ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    exe = build / 'Vv2_tiled_host_bridge'
    reference = json.loads(BASELINE.read_text())
    lookup = {(x['fixture'], x['stall_seed']): x for x in reference['results']}
    report.update(status='running', native_executable_sha256=digest(exe),
                  native_results=[])
    save(BASE / 'report.json', report)
    for fixture in report['fixtures']:
        name = fixture['name']
        for seed in ((0, 6063) if '-check' in name else (0,)):
            output = build / (name + f'-s{seed}.json')
            subprocess.run([str(exe), str(BASE / 'fixtures' / name),
                            str(seed), str(output)], check=True)
            row = json.loads(output.read_text())
            baseline = lookup[(name, seed)]
            row.update(fixture=name, baseline_elapsed_cycles=baseline['elapsed_cycles'],
                       cycle_delta=row['elapsed_cycles'] - baseline['elapsed_cycles'])
            report['native_results'].append(row)
            save(BASE / 'report.json', report)
    report['status'] = 'passed-native' if all(x['status'] == 'passed' and
        x['cycle_delta'] == 0 for x in report['native_results']) else 'failed-native'
    save(BASE / 'report.json', report)
    return report


def edges():
    from cocotb.runner import get_runner
    prepare()
    build = BASE / 'edges'
    source = ROOT / 'tools/phase6/novelty_contract_edges.py'
    inputs = [ROOT / 'rtl/v2/target_pkg.sv',
              ROOT / 'rtl/v2/requantizer.sv', ENGINE]
    runner = get_runner('verilator')
    runner.build(verilog_sources=inputs, hdl_toplevel='v2_engine',
                 build_dir=build, build_args=['--timing', '-Wno-fatal'],
                 timescale=('1ns', '1ps'))
    paths = [str(ROOT / 'compiler'), str(ROOT / 'tools/phase6')]
    runner.test(hdl_toplevel='v2_engine', test_module='novelty_contract_edges',
                test_dir=ROOT / 'tools/phase6', build_dir=build,
                results_xml=str(build / 'results.xml'),
                extra_env={'PYTHONPATH': os.pathsep.join(paths + sys.path)})
    cases = ET.parse(build / 'results.xml').findall('.//testcase')
    passed = len(cases) == 1 and all(c.find('failure') is None and
                                     c.find('error') is None for c in cases)
    report = dict(status='passed' if passed else 'failed', tests=len(cases),
                  physical_board=False, engine_sha256=digest(ENGINE),
                  test_sha256=digest(source), results_sha256=digest(build / 'results.xml'))
    save(build / 'report.json', report)
    if not passed:
        raise ValueError('contract edge test failed')
    return report


def route():
    report = json.loads((BASE / 'report.json').read_text())
    edge = json.loads((BASE / 'edges/report.json').read_text())
    if report['status'] != 'passed-native' or edge['status'] != 'passed':
        raise ValueError('native and adversarial ABI gates must pass before route')
    if report['engine_sha256'] != digest(ENGINE) or edge['engine_sha256'] != digest(ENGINE):
        raise ValueError('tested engine identity changed')
    build = BASE / 'route27'
    build.mkdir(exist_ok=True)
    prior = build / 'report.json'
    if prior.exists() and json.loads(prior.read_text()).get('status') == 'failed-build':
        if (build / 'phase6_uart_burst/impl/pnr/phase6_uart_burst.fs').exists():
            raise ValueError('failed-build record unexpectedly has a bitstream')
        for name in ('build.log', 'report.json'):
            path = build / name
            preserved = build / ('sandbox-failed-' + name)
            if path.exists():
                if preserved.exists():
                    raise ValueError('sandbox failure already preserved')
                path.rename(preserved)
    gowin = Path('/Applications/GowinIDE.app/Contents/Resources/Gowin_EDA/IDE')
    script = ROOT / 'work/phase6/pool-timing-v1/build27.tcl'
    env = dict(os.environ, PHASE6_ENGINE=str(ENGINE),
               DYLD_FRAMEWORK_PATH=str(gowin / 'lib'),
               DYLD_LIBRARY_PATH=str(gowin / 'lib'))
    start = time.monotonic()
    with (build / 'build.log').open('w') as log:
        process = subprocess.run([str(gowin / 'bin/gw_sh'), str(script)],
                                 cwd=build, env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
    prefix = build / 'phase6_uart_burst/impl/pnr/phase6_uart_burst'
    route_txt = Path(str(prefix) + '.rpt.txt')
    timing_html = Path(str(prefix) + '_tr_content.html')
    bitstream = Path(str(prefix) + '.fs')
    result = dict(status='failed-build', physical_board=False,
                  engine_sha256=digest(ENGINE),
                  build_tcl_sha256=digest(script),
                  build_seconds=time.monotonic()-start,
                  returncode=process.returncode)
    if process.returncode == 0 and all(p.exists() for p in
                                      (route_txt, timing_html, bitstream)):
        route_text, timing_text = route_txt.read_text(), timing_html.read_text()
        resources = {}
        for kind in ('Logic', 'Register', 'BSRAM', 'DSP', 'CLS'):
            match = re.search(rf'^\s*{kind}\s*\|\s*([\d.]+)/([\d.]+)',
                              route_text, re.M)
            if not match:
                raise ValueError('missing route resource ' + kind)
            resources[kind] = dict(used=float(match[1]), available=float(match[2]))
        clock = re.search(r'<td>27\.000\(MHz\)</td>\s*<td[^>]*>([\d.]+)\(MHz\)</td>',
                          timing_text)
        if not clock:
            raise ValueError('missing routed 27 MHz timing result')
        violations = {}
        for kind in ('Setup', 'Hold'):
            found = re.search(rf'Numbers of {kind} Violated Endpoints</td>\s*'
                              rf'<td[^>]*>(\d+)</td>', timing_text)
            if not found:
                raise ValueError('missing ' + kind + ' violation count')
            violations[kind.lower() + '_violated_endpoints'] = int(found[1])
        baseline = json.loads(ROUTE_BASELINE.read_text())
        result.update(resources=resources,
                      routed_core_fmax_mhz=float(clock[1]),
                      bitstream_sha256=digest(bitstream),
                      route_sha256=digest(route_txt),
                      timing_sha256=digest(timing_html),
                      baseline=dict(cls=baseline['resources']['CLS']['used'],
                                    fmax_mhz=baseline['routed_core_fmax_mhz'],
                                    bsram=baseline['resources']['BSRAM']['used'],
                                    dsp=baseline['resources']['DSP']['used']),
                      **violations)
        result['status'] = ('passed-route' if result['routed_core_fmax_mhz'] >= 27 and
                            not any(violations.values()) else 'failed-timing')
    save(build / 'report.json', result)
    return result


def decision():
    native_report = json.loads((BASE / 'report.json').read_text())
    edge = json.loads((BASE / 'edges/report.json').read_text())
    routed = json.loads((BASE / 'route27/report.json').read_text())
    baseline = json.loads(ROUTE_BASELINE.read_text())
    if (native_report['status'] != 'passed-native' or
            edge['status'] != 'passed' or routed['status'] != 'passed-route' or
            any(r['cycle_delta'] for r in native_report['native_results'])):
        raise ValueError('decision gates incomplete')
    if any(r['engine_sha256'] != digest(ENGINE) for r in
           (native_report, edge, routed)):
        raise ValueError('candidate source identity differs across gates')
    frozen_inputs = {}
    for name in ('build27.tcl', 'host27.sv', 'pll27.v'):
        path = ROOT / 'work/phase6/pool-timing-v1' / name
        rel = str(path.relative_to(ROOT))
        if baseline['sources'][rel] != digest(path):
            raise ValueError('frozen route input changed: ' + rel)
        frozen_inputs[rel] = digest(path)
    parent_hierarchy = (ROOT / 'work/phase6/pool-timing-v1/route27/'
        'phase6_uart_burst/impl/gwsynthesis/phase6_uart_burst_syn_resource.html')
    candidate_hierarchy = (BASE / 'route27/phase6_uart_burst/impl/gwsynthesis/'
                           'phase6_uart_burst_syn_resource.html')
    def hierarchy(path, module):
        html = path.read_text()
        pos = html.find('|--' + module)
        if pos < 0:
            raise ValueError('missing synthesis module ' + module)
        row = html[pos:html.find('</tr>', pos)]
        values = re.findall(r'<td align = "center">([^<]+)</td>', row)
        return dict(registers=int(values[0]), alu=int(values[1]),
                    lut=int(values[2]), dsp=int(values[3]) if values[3] != '-' else 0,
                    bsram=int(values[4]) if values[4] != '-' else 0)
    modules = {name: dict(baseline=hierarchy(parent_hierarchy, name),
                          candidate=hierarchy(candidate_hierarchy, name))
               for name in ('engine', 'command', 'sequencer')}
    result = dict(
        status='no-go', physical_board=False,
        reason='validated hint checks erase area saving; the existing direct PLL has no legal 28 MHz step and candidate Fmax is below the next direct 30 MHz clock',
        scope='single 27 MHz routed candidate plus short native/edge checks; no board test',
        native_cases=len(native_report['native_results']),
        exact_native_cases=sum(r['status'] == 'passed' and r['cycle_delta'] == 0
                               for r in native_report['native_results']),
        adversarial_cases=('generic unmarked PW equals marked PW',
            'PW hint on 3x3 CONV rejects before write',
            'PW hint on DW rejects before write',
            'DW hint on PW rejects before write',
            'both hints reject before write'),
        baseline=dict(cls=baseline['resources']['CLS']['used'],
                      logic=baseline['resources']['Logic']['used'],
                      fmax_mhz=baseline['routed_core_fmax_mhz'],
                      bsram=baseline['resources']['BSRAM']['used'],
                      dsp=baseline['resources']['DSP']['used']),
        candidate=dict(cls=routed['resources']['CLS']['used'],
                       logic=routed['resources']['Logic']['used'],
                       fmax_mhz=routed['routed_core_fmax_mhz'],
                       bsram=routed['resources']['BSRAM']['used'],
                       dsp=routed['resources']['DSP']['used']),
        delta=dict(cls=routed['resources']['CLS']['used']-baseline['resources']['CLS']['used'],
                   logic=routed['resources']['Logic']['used']-baseline['resources']['Logic']['used'],
                   fmax_mhz=routed['routed_core_fmax_mhz']-baseline['routed_core_fmax_mhz']),
        setup_violated_endpoints=routed['setup_violated_endpoints'],
        hold_violated_endpoints=routed['hold_violated_endpoints'],
        synthesized_modules=modules,
        source_audit=dict(candidate_engine_sha256=digest(ENGINE),
                          frozen_inputs=frozen_inputs,
                          baseline_engine_sha256=baseline['engine_sha256'],
                          route_bitstream_sha256=routed['bitstream_sha256']),
        evidence=dict(native='work/phase6/novelty-contracts-v1/report.json',
                      edge='work/phase6/novelty-contracts-v1/edges/report.json',
                      route='work/phase6/novelty-contracts-v1/route27/report.json',
                      clock='docs/research/PHASE_6_FUSED_CLOCK_FEASIBILITY.md'))
    save(BASE / 'decision.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'native', 'edges', 'route', 'decision'))
    args = parser.parse_args()
    result = {'prepare': prepare, 'native': native, 'edges': edges,
              'route': route, 'decision': decision}[args.stage]()
    print(json.dumps({'status': result['status'],
                      'fixture_count': result.get('fixture_count'),
                      'runs': len(result.get('native_results', [])),
                      'engine_sha256': result.get('engine_sha256', digest(ENGINE))}))
