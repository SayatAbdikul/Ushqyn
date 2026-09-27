#!/usr/bin/env python3
"""Screen a registered reduction-tail mask on the final scalar UART engine.

This candidate moves the validity comparison out of the MAC combinational
path.  The mask is refreshed in every non-MAC state, including the operand
fetch states immediately before a MAC, and is consumed only in MAC.
"""

import argparse
import json
import os
import re
import subprocess
import time

import combined_scalar_uart as parent
from run_screening import save_json
from variants import ROOT, check_frozen, sha


BASE = ROOT / 'work/phase6/experiments-v2/tail-mask-v2'
ENGINE = BASE / 'engine.sv'
SOURCE = ROOT / 'work/phase6/experiments-v1/combined-spec-scalar-uart-v1/engine.sv'

DECL = '    logic [31:0] row,col;'
MASK_DECL = '''    logic [31:0] row,col;
    logic [7:0] mac_lane_mask;
    // Operand fetch always precedes MAC.  Register tail validity there so
    // the col/count comparison does not feed the multiplier reduction tree.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) mac_lane_mask <= 8'b0;
        else if (state != MAC) begin
            for (integer lane = 0; lane < 8; lane = lane + 1)
                mac_lane_mask[lane] <= (col + 32'(lane) < count);
        end
    end'''


def prepare():
    check_frozen()
    source = SOURCE.read_text()
    if source.count(DECL) != 1 or source.count('(col+k<count)') != 2:
        raise ValueError('frozen mask patch sites changed')
    candidate = source.replace(DECL, MASK_DECL, 1)
    candidate = candidate.replace('(col+k<count)?products[k]', 'mac_lane_mask[k]?products[k]', 1)
    candidate = candidate.replace('if (col+k<count) begin', 'if (mac_lane_mask[k]) begin', 1)
    BASE.mkdir(parents=True, exist_ok=True)
    if ENGINE.exists() and ENGINE.read_text() != candidate:
        raise ValueError('tail-mask candidate source changed')
    if not ENGINE.exists():
        ENGINE.write_text(candidate)
    record = dict(label=BASE.name, parent=str(SOURCE.relative_to(ROOT)),
                  parent_sha256=sha(SOURCE), engine_sha256=sha(ENGINE),
                  parameters=json.loads((SOURCE.parent / 'identity.json').read_text())['parameters'],
                  change='registered eight-lane validity mask before MAC',
                  physical_board=False)
    identity = BASE / 'identity.json'
    if identity.exists():
        old = json.loads(identity.read_text())
        if {k: v for k, v in record.items() if k != 'parameters'} != old and old != record:
            raise ValueError('tail-mask candidate identity changed')
    if not identity.exists() or json.loads(identity.read_text()) != record:
        save_json(identity, record)
    parent.BASE = BASE
    parent.ENGINE = ENGINE
    parent.engine_runner.BASE = BASE
    parent.engine_runner.ENGINE = ENGINE
    parent.engine_runner.runner.BASE = BASE
    parent.engine_runner.runner.ENGINE = ENGINE
    return record


def route_clock(mhz):
    if mhz not in (24, 27):
        raise ValueError('unsupported experimental clock')
    if json.loads((BASE / 'native/report.json').read_text())['status'] != 'passed':
        raise ValueError('complete native RTL regression required before routing')
    if json.loads((BASE / 'integration/report.json').read_text())['status'] != 'passed':
        raise ValueError('integrated UART regression required before routing')
    base_pll = (ROOT / 'work/phase6/experiments-v1/combined-spec-scalar-uart-v1/pll.v').read_text()
    if base_pll.count('.FBDIV_SEL(4), .IDIV_SEL(5)') != 1:
        raise ValueError('PLL patch site changed')
    div = 8 if mhz == 24 else 7
    uart_divider = 32 if mhz == 24 else 36
    pll = base_pll.replace('.FBDIV_SEL(4), .IDIV_SEL(5)',
                           f'.FBDIV_SEL(7), .IDIV_SEL({div})').replace(
                               '22.5 MHz from 27 MHz', f'{mhz} MHz from 27 MHz')
    base_host = (ROOT / 'work/phase6/experiments-v1/combined-spec-scalar-uart-v1/host.sv').read_text()
    if base_host.count('22500000') != 3:
        raise ValueError('clock patch site changed')
    host = base_host.replace('22500000', str(mhz * 1000000))
    base_script = (ROOT / 'work/phase6/experiments-v1/combined-spec-scalar-uart-v1/build.tcl').read_text()
    old = 'work/phase6/experiments-v1/combined-spec-scalar-uart-v1/'
    if base_script.count(old) != 2:
        raise ValueError('build script patch sites changed')
    script = base_script.replace(old + 'pll.v', str((BASE / f'pll{mhz}.v').relative_to(ROOT)))
    script = script.replace(old + 'host.sv', str((BASE / f'host{mhz}.sv').relative_to(ROOT)))
    for path, content in ((BASE / f'pll{mhz}.v', pll), (BASE / f'host{mhz}.sv', host),
                          (BASE / f'build{mhz}.tcl', script)):
        if path.exists() and path.read_text() != content:
            raise ValueError(f'immutable route input changed: {path}')
        if not path.exists():
            path.write_text(content)
    build = BASE / f'route{mhz}'
    build.mkdir(exist_ok=True)
    report = build / 'report.json'
    if report.exists():
        old_report = json.loads(report.read_text())
        if old_report['engine_sha256'] != sha(ENGINE):
            raise ValueError('route report engine identity changed')
        return old_report
    gowin = '/Applications/GowinIDE.app/Contents/Resources/Gowin_EDA/IDE'
    env = dict(os.environ, DYLD_FRAMEWORK_PATH=f'{gowin}/lib',
               DYLD_LIBRARY_PATH=f'{gowin}/lib', PHASE6_ENGINE=str(ENGINE))
    started = time.monotonic()
    with (build / 'build.log').open('w') as log:
        process = subprocess.run([f'{gowin}/bin/gw_sh', str(BASE / f'build{mhz}.tcl')],
                                 cwd=build, env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
    result = dict(status='failed-build', physical_board=False, core_clock_mhz=mhz,
                  engine_sha256=sha(ENGINE), build_seconds=time.monotonic()-started,
                  sources={str(p.relative_to(ROOT)): sha(p) for p in (
                      ENGINE, BASE / f'pll{mhz}.v', BASE / f'host{mhz}.sv',
                      BASE / f'build{mhz}.tcl')},
                  uart_divider=uart_divider, pll_vco_mhz=mhz * 32)
    if process.returncode:
        save_json(report, result)
        raise RuntimeError(f'{mhz} MHz Gowin build failed; inspect route{mhz}/build.log')
    pnr = build / 'phase6_uart_burst/impl/pnr'
    prefix = 'phase6_uart_burst'
    route_file = pnr / f'{prefix}.rpt.txt'
    timing_file = pnr / f'{prefix}_tr_content.html'
    bitstream = pnr / f'{prefix}.fs'
    routed = route_file.read_text()
    timing = timing_file.read_text()
    resources = {}
    for key in ('Logic', 'Register', 'BSRAM', 'DSP', 'CLS'):
        match = re.search(rf'^\s*{key}\s*\|\s*([\d.]+)/([\d.]+)', routed, re.M)
        if not match:
            raise ValueError(f'missing resource {key}')
        resources[key] = dict(used=float(match[1]), available=float(match[2]))
    fmax = re.search(rf'<td>{mhz}\.000\(MHz\)</td>\s*<td[^>]*>([\d.]+)\(MHz\)</td>', timing)
    if not fmax:
        raise ValueError(f'{mhz} MHz timing constraint absent')
    result.update(resources=resources, routed_core_fmax_mhz=float(fmax[1]),
                  bitstream=str(bitstream.relative_to(ROOT)),
                  bitstream_sha256=sha(bitstream), route_sha256=sha(route_file),
                  timing_sha256=sha(timing_file))
    for kind in ('Setup', 'Hold'):
        match = re.search(rf'Numbers of {kind} Violated Endpoints</td>\s*<td[^>]*>(\d+)</td>', timing)
        if not match:
            raise ValueError(f'missing {kind} endpoint count')
        result[kind.lower() + '_violated_endpoints'] = int(match[1])
    if (result['routed_core_fmax_mhz'] >= mhz and
            result['setup_violated_endpoints'] == 0 and
            result['hold_violated_endpoints'] == 0 and
            all(r['used'] <= r['available'] for r in resources.values())):
        result['status'] = 'passed-route'
    else:
        result['status'] = 'failed-timing'
    save_json(report, result)
    return result


def route_24():
    return route_clock(24)


def route_27():
    return route_clock(27)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'engine', 'edges',
                                          'native', 'integration', 'route22', 'route24', 'route27'))
    stage = parser.parse_args().stage
    identity = prepare()
    if stage == 'engine':
        parent.engine_runner.runner.engine()
    elif stage == 'edges':
        parent.engine_runner.cocotb('edges', 'test_experiment_edges')
    elif stage == 'native':
        parent.engine_runner.runner.native()
    elif stage == 'integration':
        parent.integration()
    elif stage == 'route22':
        parent.route_22_5()
    elif stage == 'route24':
        route_24()
    elif stage == 'route27':
        route_27()
    check_frozen()
    print(json.dumps({'stage': stage, 'engine_sha256': identity['engine_sha256']}), flush=True)


if __name__ == '__main__':
    main()
