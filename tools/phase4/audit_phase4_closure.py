#!/usr/bin/env python3
"""Fail-closed audit of the archived Phase 4 engineering acceptance evidence."""
import gzip
import hashlib
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'docs/research/evidence/phase4'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    route=json.loads((E/'physical-sequence-route.json').read_text())
    bitstream=route['bitstream_sha256']
    assert sha(ROOT/route['bitstream_path'])==bitstream
    assert route['setup_tns_ns']==0 and route['routed_core_fmax_mhz']>20.25
    for name,digest in route['source_sha256'].items():
        assert sha(ROOT/name)==digest, f'RTL/source drift: {name}'
    assert sha(ROOT/route['generated_ip_path'])==route['generated_ip_sha256']
    for resource in route['resources'].values():assert resource['used']<=resource['available']
    assert route['resources']['bsram']['used']<=40
    assert route['resources']['logic']['used']/route['resources']['logic']['available']<=0.8
    log=gzip.decompress((E/'physical-sequence-program.txt.gz').read_bytes())
    assert hashlib.sha256(log).hexdigest()==route['program_log_sha256']
    assert b'after program sram: displayReadReg 00006020' in log
    names=['physical-sequence-kws.json','physical-sequence-vww.json',
           'physical-sequence-dma.json','physical-sequence-kernel-holdout.json',
           'physical-sequence-checks.json','physical-sequence-costs.json',
           'physical-sequence-legacy.json','physical-sequence-synthetic.json']
    reports={name:json.loads((E/name).read_text()) for name in names}
    for name,report in reports.items():
        assert report['status']=='passed' and report['bitstream_sha256']==bitstream,name
    performance={}
    retained=json.loads((E/'physical-sdram-hs.json').read_text())
    assert retained['status']=='passed' and retained['aggregate_bytes']>=1073741824
    assert retained['expected']['address_bytes_per_sweep']==8388608
    for model,nodes in (('kws',22),('vww',58)):
        report=reports[f'physical-sequence-{model}.json']
        assert report['physical_board'] and len(report['correctness_nodes'])==nodes
        oracle=json.loads((E/f'real-{model}-fixture.json').read_text())
        assert [n['output_sha256'] for n in report['correctness_nodes']]==[
            n['output_sha256'] for n in oracle['layers']]
        if 'resumed_from_report_sha256' in report:
            prior=E/f'physical-sequence-{model}-initial.json'
            assert sha(prior)==report['resumed_from_report_sha256']
            initial=json.loads(prior.read_text())
            assert initial['bitstream_sha256']==bitstream
            assert initial['correctness_nodes']==report['correctness_nodes']
        sequential=report['modes']['sequential'];overlap=report['modes']['overlap']
        assert sequential['schedule']['tiles']==overlap['schedule']['tiles']
        assert sequential['schedule']['image_sha256']==overlap['schedule']['image_sha256']
        for mode in (sequential,overlap):
            assert len(mode['runs'])>=3
            assert len({r['output_sha256'] for r in mode['runs']})==1
            assert all(r['last_command_index']==mode['schedule']['command_count']-1 for r in mode['runs'])
        assert sequential['runs'][0]['output_sha256']==overlap['runs'][0]['output_sha256']
        assert all(r['overlap_cycles']>0 for r in overlap['runs'])
        # Benefit must exceed observed refresh/measurement variation.
        assert max(r['elapsed_cycles'] for r in overlap['runs'])<min(r['elapsed_cycles'] for r in sequential['runs'])
        a=statistics.median(r['elapsed_cycles'] for r in sequential['runs'])
        b=statistics.median(r['elapsed_cycles'] for r in overlap['runs'])
        performance[model]={'sequential_cycles_median':a,'overlap_cycles_median':b,
                            'overlap_speedup':a/b,'execution_ms_median':b/20250,
                            'input_run_output_wall_seconds_median':statistics.median(
                                r['input_run_output_wall_seconds'] for r in overlap['runs'])}
    dma=reports['physical-sequence-dma.json']
    assert len(dma['records'])>=78
    assert {r['direction'] for r in dma['records']}=={'to_sram','from_sram'}
    assert all(r['dma_cycles']>0 for r in dma['records'])
    kernels=reports['physical-sequence-kernel-holdout.json']
    assert kernels['prospective_holdout'] and len(kernels['cases'])==28
    assert all(len(c['runs'])>=3 and c['integer_mismatches']==0 for c in kernels['cases'])
    checks=reports['physical-sequence-checks.json']
    assert checks['live_guard_error']==9 and checks['abort_while_busy'] and checks['reset_and_transfer_recovery']
    legacy=reports['physical-sequence-legacy.json']
    assert [(m['model'],m['nodes']) for m in legacy['models']]==[('mlp',5),('smallcnn',8)]
    synthetic=reports['physical-sequence-synthetic.json']
    ad=next(c for c in synthetic['cases'] if c['case'].startswith('toycar-'))
    assert ad['nodes']==19 and ad['source_inventory_sha256']==sha(ROOT/'benchmarks/manifests/ad.canonical-inventory.json')
    assert any(c['case']=='multi-tile-fc-32kib-weights' and c['tiles']>=2 for c in synthetic['cases'])
    costs=reports['physical-sequence-costs.json']
    assert costs['predictor_sha256']==sha(ROOT/'compiler/phase4_cost.py')
    errors=[costs['engine']['prospective_holdout_error'],costs['engine']['full_model_uncontended_error']]
    errors += [r['validation_error'] for r in costs['dma'].values()]
    assert all(r['median_absolute_percent_error']<=10 and r['p95_absolute_percent_error']<=20 for r in errors)
    suite=gzip.decompress((E/'physical-sequence-suite.txt.gz').read_bytes())
    assert b'CI tier passed' in suite and b'burst_coherence_masks_boundaries_refresh_reset passed' in suite
    result={'status':'passed','gate':'G4','bitstream_sha256':bitstream,
            'performance':performance,'reports_sha256':{name:sha(E/name) for name in names},
            'route_sha256':sha(E/'physical-sequence-route.json'),
            'suite_sha256':sha(E/'physical-sequence-suite.txt.gz'),
            'retained_sdram_evidence_sha256':sha(E/'physical-sdram-hs.json'),
            'checks':['complete frozen audio/vision kernels','retained full-range SDRAM evidence',
                      'compiler-managed hybrid ping-pong with full-model net benefit',
                      '64-byte burst DMA correctness and bandwidth',
                      'live-region guard, abort/reset and stalled simulation',
                      'prospective compiler-static timing prediction and held-out DMA fit'],
            'limits':['one deterministic primary input per model; complete-set quality and 10000 jobs are Phase 5',
                      'physical energy unavailable without instrumentation',
                      'clock below the broader 27/54-MHz aspirations; no SOTA or publication claim']}
    (E/'physical-sequence-closure.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result['performance'],indent=2))


if __name__=='__main__':main()
