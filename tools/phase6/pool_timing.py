#!/usr/bin/env python3
"""Isolated exact average-pool reduction and route experiment.

The existing compiler fixtures and host protocol are unchanged. No stage
programs the FPGA. Outputs are immutable within this experiment label.
"""
import argparse
import json
import subprocess
import xml.etree.ElementTree as ET

import combined_scalar_uart as integrated
import run_writeback as runner
import tail_mask as route_runner
from run_screening import save_json, verify_files
from variants import ROOT, check_frozen, sha


BASE = ROOT / 'work/phase6/pool-timing-v1'
ENGINE = BASE / 'engine.sv'
PARENT = ROOT / 'work/phase6/experiments-v1/fused-activation-v1'


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('pool patch site changed: ' + old[:90])
    return source.replace(old, new, 1)


def generated_engine():
    source = (PARENT / 'engine.sv').read_text()
    source = once(source, '    logic signed [31:0] pool_sum;',
                  '''    // Each accepted pool contains at most 31*31 samples. One centered
    // INT8 sample is in [-255,255], hence the total is in [-245055,245055]
    // and cannot overflow a signed 19-bit accumulator.
    logic signed [18:0] pool_sum;''')
    source = once(source, '''    logic signed [31:0] pool_chunk_sum;
    logic signed [7:0] pool_chunk_max;''', '''    logic signed [10:0] pool_chunk_sum;
    logic signed [8:0] pool_delta [0:3];
    logic signed [9:0] pool_pair [0:1];
    logic signed [7:0] pool_chunk_max;''')
    source = once(source, '''        pool_chunk_sum=0;pool_chunk_max=-8'sd128;pool_sample=0;
        for(integer k=0;k<4;k=k+1)
            if(k<pool_take)begin
                pool_sample=$signed(pool_word[(32'(win_addr[2:0])+k)*8+:8]);
                pool_chunk_sum=pool_chunk_sum+{{24{pool_sample[7]}},pool_sample}-{{24{zx[7]}},zx};
                if(pool_sample>pool_chunk_max)pool_chunk_max=pool_sample;
            end''', '''        pool_chunk_max=-8'sd128;pool_sample=0;
        for(integer k=0;k<4;k=k+1)begin
            pool_delta[k]=0;
            if(k<pool_take)begin
                pool_sample=$signed(pool_word[(32'(win_addr[2:0])+k)*8+:8]);
                pool_delta[k]=$signed({pool_sample[7],pool_sample})-
                              $signed({zx[7],zx});
                if(pool_sample>pool_chunk_max)pool_chunk_max=pool_sample;
            end
        end
        pool_pair[0]=$signed({pool_delta[0][8],pool_delta[0]})+
                     $signed({pool_delta[1][8],pool_delta[1]});
        pool_pair[1]=$signed({pool_delta[2][8],pool_delta[2]})+
                     $signed({pool_delta[3][8],pool_delta[3]});
        pool_chunk_sum=$signed({pool_pair[0][9],pool_pair[0]})+
                       $signed({pool_pair[1][9],pool_pair[1]});''')
    source = once(source,
                  'acc<=(op==OP_AVGPOOL)?pool_sum:({{24{pool_max[7]}},pool_max}-{{24{zx[7]}},zx});',
                  'acc<=(op==OP_AVGPOOL)?{{13{pool_sum[18]}},pool_sum}:({{24{pool_max[7]}},pool_max}-{{24{zx[7]}},zx});')
    return '// Isolated balanced, bounded average-pool reduction.\n' + source


def prepare():
    check_frozen()
    original = json.loads((PARENT / 'identity.json').read_text())
    if sha(PARENT / 'engine.sv') != original['engine_sha256']:
        raise ValueError('parent engine hash changed')
    BASE.mkdir(parents=True, exist_ok=True)
    source = generated_engine()
    if ENGINE.exists() and ENGINE.read_text() != source:
        raise ValueError('pool engine changed; use a new label')
    if not ENGINE.exists():
        ENGINE.write_text(source)
    record = dict(label=BASE.name, parent=str((PARENT / 'engine.sv').relative_to(ROOT)),
                  parent_sha256=sha(PARENT / 'engine.sv'), engine_sha256=sha(ENGINE),
                  parameters=original['parameters'],
                  bound=dict(kernel_h_max=31, kernel_w_max=31, centered_sample_abs_max=255,
                             signed_pool_bits=19, signed_chunk_bits=11),
                  change='19-bit pool total and balanced 9/10/11-bit four-sample sum',
                  physical_board=False)
    path = BASE / 'identity.json'
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError('pool identity changed')
    if not path.exists():
        save_json(path, record)
    runner.BASE = BASE
    runner.ENGINE = ENGINE
    integrated.BASE = BASE
    integrated.ENGINE = ENGINE
    route_runner.BASE = BASE
    route_runner.ENGINE = ENGINE
    return record


def edges():
    from cocotb.runner import get_runner
    import os
    import sys
    build = BASE / 'edges'
    simulation = get_runner('verilator')
    simulation.build(verilog_sources=runner.sources(True), hdl_toplevel='v2_engine',
                     build_dir=build, build_args=['--timing', '-Wno-fatal'],
                     timescale=('1ns', '1ps'))
    paths = [str(ROOT / p) for p in ('compiler', 'test/phase6')] + sys.path
    sys.path[:0] = paths
    simulation.test(hdl_toplevel='v2_engine', test_module='test_pool_timing',
                    test_dir=ROOT / 'test/phase6', build_dir=build,
                    results_xml=str(build / 'results.xml'),
                    extra_env={'PYTHONPATH': os.pathsep.join(paths)})
    cases = ET.parse(build / 'results.xml').findall('.//testcase')
    if len(cases) != 1 or any(c.find('failure') is not None or c.find('error') is not None for c in cases):
        raise ValueError('pool edge simulation failed')
    report = dict(status='passed', physical_board=False, tests=len(cases),
                  sources={str(p.relative_to(ROOT)): sha(p) for p in runner.sources(True) +
                           [ROOT / 'test/phase6/test_pool_timing.py']},
                  results_sha256=sha(build / 'results.xml'))
    save_json(build / 'report.json', report)
    return report


def native():
    manifest = json.loads((PARENT / 'fixtures.json').read_text())
    assert manifest['status'] == 'passed-replay' and len(manifest['fixtures']) == 6
    build = BASE / 'native'
    build.mkdir(exist_ok=True)
    inputs = runner.sources()
    harness = ROOT / 'test/phase6/native.cpp'
    with (build / 'build.log').open('w') as log:
        subprocess.run(['verilator', '--cc', '--exe', '--build', '-j', '2',
                        '--public-flat-rw', '-Wno-fatal', '--top-module',
                        'v2_tiled_host_bridge', '--Mdir', str(build),
                        *map(str, inputs), str(harness)], cwd=ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    exe = build / 'Vv2_tiled_host_bridge'
    report = dict(status='running', physical_board=False,
                  executable_sha256=sha(exe),
                  sources={str(p.relative_to(ROOT)): sha(p) for p in inputs+[harness]},
                  fixture_manifest_sha256=sha(PARENT / 'fixtures.json'), results=[])
    save_json(build / 'report.json', report)
    for item in manifest['fixtures']:
        directory = PARENT / 'fixtures' / item['name']
        verify_files(directory, item['files'])
        for seed in (0, 6063):
            output = build / f'{item["name"]}-s{seed}.json'
            subprocess.run([str(exe), str(directory), str(seed), str(output)], check=True)
            result = json.loads(output.read_text())
            assert result['status'] == 'passed'
            result.update(fixture=item['name'], fixture_files=item['files'])
            report['results'].append(result)
            save_json(build / 'report.json', report)
    report['status'] = 'passed'
    save_json(build / 'report.json', report)
    return report


def compact_native():
    """Run every channel-compacted fused fixture on this exact engine binary."""
    source = ROOT / 'work/phase6/channel-compaction-v1/fused'
    manifest = json.loads((source / 'report.json').read_text())
    assert manifest['status'] == 'passed' and len(manifest['results']) == 6
    native_report = json.loads((BASE / 'native/report.json').read_text())
    assert native_report['status'] == 'passed'
    exe = BASE / 'native/Vv2_tiled_host_bridge'
    assert sha(exe) == native_report['executable_sha256']
    build = BASE / 'compact-native'
    build.mkdir(exist_ok=True)
    report = dict(status='running', physical_board=False,
                  executable_sha256=sha(exe),
                  compacted_manifest_sha256=sha(source / 'report.json'),
                  engine_sha256=sha(ENGINE), results=[])
    save_json(build / 'report.json', report)
    for item in manifest['results']:
        folder = source / 'fixtures' / item['label']
        verify_files(folder, item['files'])
        for seed in (0, 6063):
            output = build / f'{item["label"]}-s{seed}.json'
            subprocess.run([str(exe), str(folder), str(seed), str(output)], check=True)
            row = json.loads(output.read_text())
            assert row['status'] == 'passed'
            row.update(fixture=item['label'], fixture_files=item['files'])
            report['results'].append(row)
            save_json(build / 'report.json', report)
    report['status'] = 'passed'
    save_json(build / 'report.json', report)
    return report


def uart27():
    """Exercise the exact 27 MHz / 750 kbaud UART divisor in RTL."""
    from cocotb.runner import get_runner
    import os
    import sys
    build = BASE / 'uart27'
    sources = [ROOT / 'rtl/v2/uart_rx.sv', ROOT / 'rtl/v2/uart_tx.sv',
               ROOT / 'test/phase6/uart_loopback27.sv']
    simulation = get_runner('verilator')
    simulation.build(verilog_sources=sources, hdl_toplevel='uart_loopback27',
                     build_dir=build, build_args=['--timing', '-Wno-fatal'],
                     timescale=('1ns', '1ps'))
    paths = [str(ROOT / 'test/phase6')] + sys.path
    sys.path[:0] = paths
    simulation.test(hdl_toplevel='uart_loopback27', test_module='test_uart_27',
                    test_dir=ROOT / 'test/phase6', build_dir=build,
                    results_xml=str(build / 'results.xml'),
                    extra_env={'PYTHONPATH': os.pathsep.join(paths)})
    cases = ET.parse(build / 'results.xml').findall('.//testcase')
    if len(cases) != 1 or any(c.find('failure') is not None or c.find('error') is not None for c in cases):
        raise ValueError('27 MHz UART PHY test failed')
    report = dict(status='passed', physical_board=False, tests=1,
                  clock_hz=27_000_000, baud=750_000, clocks_per_bit=36,
                  sources={str(p.relative_to(ROOT)): sha(p) for p in sources +
                           [ROOT / 'test/phase6/test_uart_27.py']},
                  results_sha256=sha(build / 'results.xml'))
    save_json(build / 'report.json', report)
    return report


def compare():
    result = json.loads((BASE / 'native/report.json').read_text())
    baseline = json.loads((PARENT / 'native/report.json').read_text())
    assert result['status'] == baseline['status'] == 'passed'
    keys = ('fixture', 'stall_seed', 'elapsed_cycles', 'engine_cycles',
            'dma_cycles', 'overlap_cycles', 'tensor_checks')
    rows = []
    for item in result['results']:
        match = next(x for x in baseline['results'] if
                     x['fixture'] == item['fixture'] and x['stall_seed'] == item['stall_seed'])
        rows.append({key: (item[key] if key not in ('elapsed_cycles','engine_cycles',
                     'dma_cycles','overlap_cycles') else
                     dict(candidate=item[key], parent=match[key])) for key in keys})
        for key in ('elapsed_cycles','engine_cycles','dma_cycles','overlap_cycles','tensor_checks'):
            assert item[key] == match[key], (item['fixture'], item['stall_seed'], key)
    report = dict(status='passed', physical_board=False, cases=len(rows),
                  exact_parent_cycles=True, rows=rows)
    save_json(BASE / 'comparison.json', report)
    return report


def route(mhz):
    # Gowin's macOS GUI runtime can exit before synthesis when invoked inside
    # the command sandbox. Preserve that diagnostic and allow an unsandboxed
    # retry without mistaking the failed attempt for a completed route.
    folder = BASE / f'route{mhz}'
    prior = folder / 'report.json'
    if prior.exists() and json.loads(prior.read_text())['status'] == 'failed-build':
        assert not (folder / 'phase6_uart_burst/impl/pnr/phase6_uart_burst.fs').exists()
        for name in ('report.json', 'build.log'):
            path = folder / name
            if path.exists():
                preserved = folder / f'sandbox-failed-{name}'
                if preserved.exists():
                    raise ValueError('already preserved sandbox-failed route attempt')
                path.rename(preserved)
    return route_runner.route_clock(mhz)


def summary():
    paths = {name: BASE / path for name, path in {
        'edges': 'edges/report.json', 'engine': 'engine/report.json',
        'native': 'native/report.json',
        'compact-native': 'compact-native/report.json',
        'uart27': 'uart27/report.json',
        'integration': 'integration/report.json',
        'comparison': 'comparison.json', 'route24': 'route24/report.json',
        'route27': 'route27/report.json'}.items()}
    reports = {name: json.loads(path.read_text()) for name, path in paths.items()}
    if any(report['status'] != 'passed' for name, report in reports.items()
           if name not in ('route24', 'route27')):
        raise ValueError('one or more semantic checks did not pass')
    if any(reports[name]['status'] != 'passed-route' for name in ('route24', 'route27')):
        raise ValueError('one or more routes did not pass')
    if reports['comparison']['cases'] != 12 or not reports['comparison']['exact_parent_cycles']:
        raise ValueError('full-model cycle comparison incomplete')
    if reports['native']['executable_sha256'] != sha(BASE/'native/Vv2_tiled_host_bridge'):
        raise ValueError('native executable changed')
    if (len(reports['compact-native']['results']) != 12 or
            reports['compact-native']['executable_sha256'] != reports['native']['executable_sha256'] or
            reports['compact-native']['engine_sha256'] != sha(ENGINE)):
        raise ValueError('compacted native coverage incomplete')
    if (reports['uart27']['clock_hz'] != 27_000_000 or
            reports['uart27']['baud'] != 750_000 or
            reports['uart27']['clocks_per_bit'] != 36):
        raise ValueError('27 MHz UART test missing')
    for name, report in reports.items():
        for source, digest in report.get('sources', {}).items():
            if sha(ROOT / source) != digest:
                raise ValueError(f'{name} source changed: {source}')
    before = json.loads((PARENT / 'route24/report.json').read_text())
    after24, after27 = reports['route24'], reports['route27']
    result = dict(status='passed-boardless', physical_board=False,
                  semantic_tests=dict(pool_edge_tests=reports['edges']['tests'],
                                      engine_tests=reports['engine']['tests'],
                                      uart_integration_tests=reports['integration']['tests'],
                                      full_model_cases=len(reports['native']['results']),
                                      compacted_full_model_cases=len(reports['compact-native']['results']),
                                      uart_27mhz_tests=reports['uart27']['tests'],
                                      exact_parent_cycles=reports['comparison']['exact_parent_cycles']),
                  proof=dict(max_kernel_area=31*31, max_abs_centered_sample=255,
                             max_abs_pool_sum=31*31*255,
                             signed19_limit=(1<<18)-1,
                             chunk_max_abs=4*255,
                             signed11_limit=(1<<10)-1),
                  route24=dict(report=str(paths['route24'].relative_to(ROOT)),
                               fmax_mhz=after24['routed_core_fmax_mhz'],
                               cls=after24['resources']['CLS']['used']),
                  route27=dict(report=str(paths['route27'].relative_to(ROOT)),
                               fmax_mhz=after27['routed_core_fmax_mhz'],
                               cls=after27['resources']['CLS']['used'],
                               bitstream=after27['bitstream'],
                               bitstream_sha256=after27['bitstream_sha256']),
                  baseline24=dict(report=str((PARENT/'route24/report.json').relative_to(ROOT)),
                                  fmax_mhz=before['routed_core_fmax_mhz'],
                                  cls=before['resources']['CLS']['used']),
                  projection=dict(clock_only_speedup_if_cycles_unchanged=27/24,
                                  board_measurement_required=True,
                                  accuracy_campaign_required_for_new_image=True),
                  report_hashes={name: sha(path) for name, path in paths.items()},
                  engine_sha256=sha(ENGINE))
    save_json(BASE / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'edges', 'engine', 'native',
                                          'compact-native', 'uart27', 'integration',
                                          'compare', 'route24', 'route27', 'summary'))
    args = parser.parse_args()
    identity = prepare()
    action = dict(prepare=lambda: identity, edges=edges, engine=runner.engine,
                  native=native, **{'compact-native': compact_native}, uart27=uart27,
                  integration=integrated.integration,
                  compare=compare, route24=lambda: route(24),
                  route27=lambda: route(27), summary=summary)[args.stage]
    result = action()
    check_frozen()
    print(json.dumps(dict(stage=args.stage, status=result.get('status', 'passed') if result else 'passed',
                          engine_sha256=identity['engine_sha256'])), flush=True)


if __name__ == '__main__':
    main()
