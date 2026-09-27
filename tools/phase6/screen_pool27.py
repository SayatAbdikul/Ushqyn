#!/usr/bin/env python3
"""Source-verified 27 MHz pool/clock screen, with grouped or compacted fixtures.

Read-only preflight is the default. --run alone programs volatile FPGA SRAM
and performs ten exact KWS/VWW inferences through the 256-byte UART bridge.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import screen_channel_compaction as compacted
import screen_fused_uart256_candidate as grouped
import run_screening as screening
from uart_burst import BURST_BYTES
from variants import ROOT, check_frozen, sha


POOL = ROOT / 'work/phase6/pool-timing-v1'
PARENT = ROOT / 'work/phase6/experiments-v1/fused-activation-v1'
PROJECT = POOL / 'route27/phase6_uart_burst/phase6_uart_burst.gprj'
ROUTE = POOL / 'route27/report.json'
BASELINES = {
    'grouped': ROOT / 'work/phase6/fused-activation-uart256-24-screen-v1',
    'compacted': ROOT / 'work/phase6/channel-compaction-v1/physical-screen-v1',
}


def checked_report(name, relative, expected, cases=None):
    path = POOL / relative
    value = json.loads(path.read_text())
    if value.get('status') != expected:
        raise ValueError(f'{name}: {expected} evidence missing')
    if cases is not None:
        xml = path.parent / 'results.xml'
        tests = ET.parse(xml).findall('.//testcase')
        if (value.get('results_sha256') != sha(xml) or len(tests) != cases or
                any(test.find('failure') is not None or test.find('error') is not None
                    for test in tests)):
            raise ValueError(f'{name}: RTL tests failed or XML changed')
    screening.verify_files(ROOT, value.get('sources', {}))
    return value


def baseline(kind, original_plan):
    folder = BASELINES[kind]
    report = json.loads((folder / 'report.json').read_text())
    prior = json.loads((folder / 'plan.json').read_text())
    seal = json.loads((folder / 'seal.json').read_text())
    if (report['status'] != 'passed-short-screen' or report['completed'] != 10 or
            prior['image']['clock_hz'] != 24_000_000 or
            prior['image']['sha256'] != original_plan['image']['sha256'] or
            prior['fixture_names'] != original_plan['fixture_names'] or
            report['plan_sha256'] != sha(folder / 'plan.json') or
            seal['report_sha256'] != sha(folder / 'report.json') or
            seal['plan_sha256'] != sha(folder / 'plan.json') or
            seal['records_sha256'] != sha(folder / 'records.jsonl')):
        raise ValueError(f'{kind}: selected 24 MHz physical baseline changed')
    for model in ('kws', 'vww'):
        if not 0 < report['summary'][model]['median_cycles']:
            raise ValueError(f'{kind}: missing physical baseline {model}')
    return dict(folder=str(folder.relative_to(ROOT)),
                bitstream_sha256=prior['image']['sha256'],
                record_sha256=sha(folder / 'records.jsonl'),
                report_sha256=sha(folder / 'report.json'),
                summary=report['summary'])


def prepare(kind):
    check_frozen()
    if kind not in BASELINES:
        raise ValueError('fixture set must be grouped or compacted')
    original = grouped.prepare() if kind == 'grouped' else compacted.prepare()
    parent_identity = json.loads((PARENT / 'identity.json').read_text())
    identity = json.loads((POOL / 'identity.json').read_text())
    engine = POOL / 'engine.sv'
    engine_sha = sha(engine)
    if (identity['engine_sha256'] != engine_sha or
            identity['parent_sha256'] != parent_identity['engine_sha256'] or
            identity['parameters'] != parent_identity['parameters']):
        raise ValueError('pool candidate ancestry changed')
    checked_report('pool edges', 'edges/report.json', 'passed', 1)
    checked_report('engine', 'engine/report.json', 'passed', 3)
    integration = checked_report('UART bridge', 'integration/report.json', 'passed', 2)
    uart = checked_report('27 MHz UART PHY', 'uart27/report.json', 'passed', 1)
    native = checked_report('grouped native', 'native/report.json', 'passed')
    compact = checked_report('compacted native', 'compact-native/report.json', 'passed')
    comparison = checked_report('parent cycle comparison', 'comparison.json', 'passed')
    summary = checked_report('pool summary', 'summary.json', 'passed-boardless')
    route = checked_report('27 MHz route', 'route27/report.json', 'passed-route')
    if (integration['sources'].get(str(engine.relative_to(ROOT))) != engine_sha or
            native['sources'].get(str(engine.relative_to(ROOT))) != engine_sha or
            len(native['results']) != 12 or
            native['executable_sha256'] != sha(POOL / 'native/Vv2_tiled_host_bridge') or
            len(compact['results']) != 12 or
            compact['executable_sha256'] != native['executable_sha256'] or
            compact['engine_sha256'] != engine_sha or
            compact['compacted_manifest_sha256'] !=
                sha(ROOT / 'work/phase6/channel-compaction-v1/fused/report.json') or
            comparison['cases'] != 12 or not comparison['exact_parent_cycles'] or
            summary['engine_sha256'] != engine_sha or
            summary['semantic_tests']['compacted_full_model_cases'] != 12):
        raise ValueError('selected schedule or native RTL coverage incomplete')
    if (uart['clock_hz'] != 27_000_000 or uart['baud'] != 750_000 or
            uart['clocks_per_bit'] != 36 or BURST_BYTES != 256):
        raise ValueError('27 MHz/750 kbaud UART evidence mismatch')
    if (route['engine_sha256'] != engine_sha or route['core_clock_mhz'] != 27 or
            route['uart_divider'] != 36 or route['pll_vco_mhz'] != 864 or
            route['routed_core_fmax_mhz'] < 27 or
            route['setup_violated_endpoints'] or route['hold_violated_endpoints'] or
            any(item['used'] > item['available'] for item in route['resources'].values())):
        raise ValueError('27 MHz route has no physical timing margin')
    screening.verify_files(ROOT, {route['bitstream']: route['bitstream_sha256']})
    if (summary['route27']['bitstream_sha256'] != route['bitstream_sha256'] or
            summary['route27']['fmax_mhz'] != route['routed_core_fmax_mhz']):
        raise ValueError('summary and route disagree')

    project = ET.parse(PROJECT)
    selected = {Path(node.attrib['path']).resolve() for node in
                project.findall('.//File') if node.attrib.get('enable') == '1'}
    required = {path.resolve() for path in (
        engine, POOL/'host27.sv', POOL/'pll27.v',
        ROOT/'rtl/phase6/uart_burst_command.sv',
        ROOT/'rtl/phase6/uart_burst_bridge.sv')}
    if (len(selected) != 18 or not required <= selected or
            any('uart_burst_512' in str(path) for path in selected)):
        raise ValueError('routed project source list is not UART256 pool27')
    host = (POOL / 'host27.sv').read_text()
    pll = (POOL / 'pll27.v').read_text()
    if (host != (PARENT / 'host24.sv').read_text().replace('24000000', '27000000') or
            host.count('27000000') != 3 or host.count('BAUD_RATE(750000)') != 2 or
            '.FBDIV_SEL(7), .IDIV_SEL(7)' not in pll or
            '.ODIV_SEL(32)' not in pll):
        raise ValueError('generated UART, PLL or SDRAM clock changed')
    script = (POOL / 'build27.tcl').read_text()
    if (script.count('rtl/phase6/uart_burst_command.sv') != 1 or
            script.count('rtl/phase6/uart_burst_bridge.sv') != 1 or
            'uart_burst_512' in script):
        raise ValueError('routed Tcl UART source list changed')
    source_paths = selected | {(POOL / 'build27.tcl').resolve()}
    source_hashes = {str(path.relative_to(ROOT)): sha(path)
                     for path in sorted(source_paths)}
    for name, digest in route['sources'].items():
        if source_hashes.get(name) != digest:
            raise ValueError(f'route source changed: {name}')
    for name in ('rtl/phase6/uart_burst_command.sv',
                 'rtl/phase6/uart_burst_bridge.sv'):
        if source_hashes[name] != integration['sources'][name]:
            raise ValueError(f'bridge integration source changed: {name}')
    if sha(PROJECT) != summary['report_hashes'].get('gowin_project', sha(PROJECT)):
        raise ValueError('project XML changed')
    prior = baseline(kind, original)

    plan = dict(original)
    plan.update(schema=3, variant=f'pool27-{kind}-uart256-v1',
                image=dict(file=route['bitstream'], sha256=route['bitstream_sha256'],
                           clock_hz=27_000_000,
                           evidence={key: sha(POOL / path) for key, path in {
                               'edges': 'edges/report.json',
                               'engine': 'engine/report.json',
                               'native': 'native/report.json',
                               'uart27': 'uart27/report.json',
                               'integration': 'integration/report.json',
                               'route': 'route27/report.json'}.items()}),
                image_label=POOL.name,
                pool27_source_coverage=dict(gowin_project=str(PROJECT.relative_to(ROOT)),
                                            gowin_project_sha256=sha(PROJECT),
                                            project_files=len(selected),
                                            source_sha256=source_hashes),
                protocol_burst_bytes=BURST_BYTES, baud=750000,
                pool27_summary_sha256=sha(POOL / 'summary.json'),
                pool27_route_report_sha256=sha(ROUTE),
                pool27_runner_sha256=sha(Path(__file__)),
                matched_24mhz_baseline=prior,
                scope='one stress, one warmup and three pinned timings per model; 27 MHz candidate on the same fixture set')
    dependencies = dict(plan['physical_dependencies'])
    for path in (Path(__file__), ROOT/'tools/phase6/pool_timing.py',
                 ROOT/'test/phase6/test_pool_timing.py',
                 ROOT/'test/phase6/test_uart_27.py',
                 ROOT/'test/phase6/uart_loopback27.sv'):
        dependencies[str(path.relative_to(ROOT))] = sha(path)
    for name in ('identity.json', 'summary.json', 'comparison.json',
                 'edges/report.json', 'engine/report.json', 'native/report.json',
                 'compact-native/report.json', 'uart27/report.json',
                 'integration/report.json', 'route24/report.json',
                 'route27/report.json'):
        path = POOL / name
        dependencies[str(path.relative_to(ROOT))] = sha(path)
    for name in ('plan.json', 'report.json', 'records.jsonl', 'seal.json'):
        path = BASELINES[kind] / name
        dependencies[str(path.relative_to(ROOT))] = sha(path)
    for path in (ROOT/'work/phase6/channel-compaction-v1/fused/report.json',
                 ROOT/'work/phase6/channel-compaction-v1/report.json'):
        dependencies[str(path.relative_to(ROOT))] = sha(path)
    plan['physical_dependencies'] = dependencies
    return plan


def matched_comparison(kind, output, plan):
    current = json.loads((output / 'report.json').read_text())
    if current['status'] != 'passed-short-screen' or current['completed'] != 10:
        raise ValueError('27 MHz physical screen incomplete')
    earlier = plan['matched_24mhz_baseline']['summary']
    result = dict(status='passed-matched-short-comparison', physical_board=True,
                  fixture_set=kind,
                  baseline_report_sha256=plan['matched_24mhz_baseline']['report_sha256'],
                  candidate_report_sha256=sha(output/'report.json'),
                  candidate_records_sha256=sha(output/'records.jsonl'),
                  models={})
    for model in ('kws', 'vww'):
        base, new = earlier[model], current['summary'][model]
        result['models'][model] = dict(
            baseline_clock_hz=24_000_000, candidate_clock_hz=27_000_000,
            baseline_cycles=base['median_cycles'], candidate_cycles=new['median_cycles'],
            baseline_latency_ms=base['median_device_latency_ms'],
            candidate_latency_ms=new['median_device_latency_ms'],
            latency_speedup=base['median_device_latency_ms']/new['median_device_latency_ms'])
    screening.save_json(output/'matched-comparison.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-set', required=True, choices=tuple(BASELINES))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--port', default='/dev/cu.usbserial-20250303171')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    plan = prepare(args.fixture_set)
    if args.run:
        if args.output is None:
            parser.error('--run requires a fresh --output directory')
        output = args.output.resolve()
        if args.fixture_set == 'grouped':
            grouped.run(plan, output, args.port)
        else:
            compacted.run(plan, output, args.port)
        screening.verify_files(ROOT, plan['pool27_source_coverage']['source_sha256'])
        if sha(PROJECT) != plan['pool27_source_coverage']['gowin_project_sha256']:
            raise ValueError('27 MHz Gowin project changed during board run')
        result = matched_comparison(args.fixture_set, output, plan)
        print(json.dumps(dict(status=result['status'], fixture_set=args.fixture_set,
                              models=result['models']), sort_keys=True), flush=True)
    else:
        print(json.dumps(dict(status='preflight-passed',
                              fixture_set=args.fixture_set, planned=plan['planned'],
                              bitstream_sha256=plan['image']['sha256'],
                              clock_hz=plan['image']['clock_hz'],
                              baseline_report_sha256=plan['matched_24mhz_baseline']['report_sha256']),
                         sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
