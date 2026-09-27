#!/usr/bin/env python3
"""Audit the proposed 25.5 MHz direct-rPLL fused/UART256 clock step.

This is deliberately a feasibility gate, not a route driver: the documented
minimum PFD frequency rules out this step before Gowin or board execution.
It never changes the frozen 24 MHz engine, fixtures, or route inputs.
"""
import hashlib
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'work/phase6/experiments-v1/fused-activation-v1'
BASE = ROOT / 'work/phase6/experiments-v1/fused-clock-25p5-feasibility-v1'
INPUT_HZ = 27_000_000
TARGET_HZ = 25_500_000
PFD_MIN_HZ = 3_000_000
VCO_MIN_HZ = 500_000_000
VCO_MAX_HZ = 1_250_000_000
ODIV_VALUES = (2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_files(base, files):
    for name, digest in files.items():
        if sha(base / name) != digest:
            raise ValueError(f'frozen source changed: {base / name}')


def assess():
    identity = json.loads((SOURCE / 'identity.json').read_text())
    engine_sha = sha(SOURCE / 'engine.sv')
    assert engine_sha == identity['engine_sha256']
    route = json.loads((SOURCE / 'route24/report.json').read_text())
    assert route['status'] == 'passed-route'
    assert route['engine_sha256'] == engine_sha
    verify_files(ROOT, route['sources'])
    assert sha(ROOT / route['bitstream']) == route['bitstream_sha256']

    manifest = json.loads((SOURCE / 'fixtures.json').read_text())
    assert manifest['status'] == 'passed-replay'
    verify_files(ROOT, manifest['sources'])
    for fixture in manifest['fixtures']:
        verify_files(SOURCE / 'fixtures' / fixture['name'], fixture['files'])
    native = json.loads((SOURCE / 'native/report.json').read_text())
    integration = json.loads((SOURCE / 'integration/report.json').read_text())
    for report in (native, integration):
        assert report['status'] == 'passed'
        verify_files(ROOT, report['sources'])
    assert native['fixture_manifest_sha256'] == sha(SOURCE / 'fixtures.json')
    assert len(native['results']) == 12 and integration['tests'] == 2

    # Exhaust the static primitive's 0..63 selector range using exact rational
    # arithmetic. Actual FBDIV/IDIV coefficients equal their selectors plus one.
    candidates = []
    legal_between = set()
    for idiv in range(1, 65):
        pfd = Fraction(INPUT_HZ, idiv)
        for fbdiv in range(1, 65):
            output = pfd * fbdiv
            for odiv in ODIV_VALUES:
                vco = output * odiv
                vco_ok = VCO_MIN_HZ <= vco <= VCO_MAX_HZ
                if output == TARGET_HZ and vco_ok:
                    candidates.append(dict(idiv=idiv, fbdiv=fbdiv, odiv=odiv,
                        idiv_sel=idiv-1, fbdiv_sel=fbdiv-1,
                        pfd_hz=float(pfd), vco_hz=int(vco),
                        pfd_in_spec=pfd >= PFD_MIN_HZ))
                if pfd >= PFD_MIN_HZ and vco_ok and 24_000_000 < output < 27_000_000:
                    legal_between.add(str(output))
    assert candidates and not any(row['pfd_in_spec'] for row in candidates)
    assert not legal_between
    ratio = Fraction(TARGET_HZ, INPUT_HZ)
    assert ratio == Fraction(17, 18)
    refresh_cycles = TARGET_HZ // 64_000
    assert Fraction(refresh_cycles, TARGET_HZ) <= Fraction(1, 64_000)
    assert TARGET_HZ % 750_000 == 0

    selected = (SOURCE / 'identity.json', SOURCE / 'fixtures.json',
        SOURCE / 'native/report.json', SOURCE / 'integration/report.json',
        SOURCE / 'route24/report.json', SOURCE / 'engine.sv',
        SOURCE / 'pll24.v', SOURCE / 'host24.sv', SOURCE / 'build24.tcl',
        ROOT / 'rtl/v2/sdram_refresh.sv',
        ROOT / 'hardware/phase4_sdram/sdram_controller_hs.ipc', Path(__file__))
    return dict(status='no-go-pfd-out-of-spec', physical_board=False,
        gowin_executed=False, route_inputs_generated=False,
        architecture='one rPLL: direct CLKOUT core and CLKOUTP SDRAM, internal feedback',
        scope='25.5 MHz fused engine + unchanged 256-byte UART bridge',
        source_label=SOURCE.name, engine_sha256=engine_sha,
        sources={str(p.relative_to(ROOT)): sha(p) for p in selected},
        preserved_validation=dict(native_cases=len(native['results']),
            uart_integration_tests=integration['tests'], fixtures=len(manifest['fixtures']),
            source_and_fixture_hashes='passed'),
        primary_references=[
            dict(url='https://cdn.gowinsemi.com.cn/DS226E.pdf',
                version='DS226-2.7E, 2026-07-31', section='3.4.7, Table 3-20, printed page 38',
                device_grade='GW2AR-18 C8/I7', pfd_min_hz=PFD_MIN_HZ,
                vco_min_hz=VCO_MIN_HZ, vco_max_hz=VCO_MAX_HZ),
            dict(url='https://cdn.gowinsemi.com.cn/UG286E.pdf',
                version='UG286-2.0.2E', section='5.1 rPLL',
                formulas=['CLKOUT = CLKIN * FBDIV / IDIV',
                    'VCO = CLKOUT * ODIV', 'PFD = CLKIN / IDIV'])],
        proposed=dict(input_hz=INPUT_HZ, target_hz=TARGET_HZ,
            reduced_ratio=str(ratio), max_possible_pfd_hz=INPUT_HZ // ratio.denominator,
            uart_baud=750_000, uart_divider=TARGET_HZ // 750_000,
            sdram_clock_hz=TARGET_HZ, refresh_period_cycles=refresh_cycles,
            refresh_period_us=float(Fraction(refresh_cycles * 1_000_000, TARGET_HZ))),
        exact_divider_combinations_with_valid_vco=candidates,
        legal_direct_outputs_strictly_between_24_and_27_mhz=sorted(legal_between),
        next_legal_direct_clock_hz=27_000_000,
        existing_24mhz_routed_fmax_mhz=route['routed_core_fmax_mhz'],
        conclusion='Reject the bounded clock-only change. Exact 17/18 requires IDIV>=18, '
            'so PFD<=1.5 MHz, below the specified 3 MHz. Valid PFD implies IDIV<=9; '
            'the largest fraction below 1 with those denominators is 8/9. '
            'Changing ODIV cannot repair PFD. Alternative clock architecture or a '
            'new 27 MHz timing-closure experiment requires separate validation.')


if __name__ == '__main__':
    result = assess()
    BASE.mkdir(parents=True, exist_ok=True)
    path = BASE / 'report.json'
    content = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if path.exists() and path.read_text() != content:
        raise ValueError('immutable clock feasibility report changed')
    if not path.exists():
        path.write_text(content)
    print(json.dumps(dict(status=result['status'], report=str(path.relative_to(ROOT)),
                         source_checks='passed', physical_board=False)))
