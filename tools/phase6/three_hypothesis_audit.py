#!/usr/bin/env python3
"""Audit raw results and limitations of the three bounded research experiments.

Recomputes quality/counts from saved predictions, verifies frozen inputs and
native report hashes, checks schedule conservation and production invariance.
This audit is separate from the generators, but is not a new timing theorem.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'work/phase6/three-hypothesis-campaign-v1'
sys.path[:0] = [str(ROOT/'compiler'),str(ROOT/'tools/phase6')]
from program_image import load_image


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(ok,msg):
    if not ok:
        raise AssertionError(msg)


def main():
    before = read(BASE/'baseline.json')
    changed = [p for p,h in before['files'].items() if sha(ROOT/p) != h]
    require(not changed,'production files changed: '+str(changed))
    result = {'status':'running','scope':__doc__,'source_sha256':sha(Path(__file__)),
              'production_files_unchanged':len(before['files']),'classification':{},
              'limitations':['VWW native dense/factor timings use matched generic compilation, not the best tuned physical schedule.',
                  'Classification candidates were selected using48 labeled development examples from the original evaluation archive; reported new quality excludes these and the earlier256-example screen.',
                  'AD candidate selection used48 labeled development recordings from the original evaluation archive;200 disjoint recordings remain. This is a redesigned research protocol, not an official unchanged benchmark score.',
                  'Native external RAM and virtual UART are simulation models; there is no new routed area, board latency, energy or worst-case response-time proof.',
                  'Column-pruning arithmetic costs require gather/propagated compaction; zero-column native execution itself has no such saving.',
                  'Some arrival scenario labels represent identical actual job sequences; counts are coverage labels, not independent statistical trials.']}
    for name in ('kws','vww','ad'):
        src = ROOT/f'work/phase6/representation-screen-v1/{name}'
        screen = read(src/'report.json')
        image = ROOT/screen['export']['path']
        require(sha(image)==screen['export']['sha256'],'factor image changed')
        frozen = read(src/'selection-frozen-before-heldout.json')
        require(frozen==screen['selection'],'frozen selection mismatch')
        dev,held = frozen['dev_indices'],frozen['heldout_indices']
        require(len(set(dev))==len(dev) and len(set(held))==len(held) and not set(dev)&set(held),'split overlap')
        factor = load_image(image.read_bytes())
        require(factor.provenance['representation_screen']['original_layers']==sorted(map(int,frozen['selected_ranks'])),'factor layer selection changed')
        costs = screen['heldout_costs']['selected/int8']
        require(0<=costs['charged_limb_product_reduction_fraction']<1,'invalid product saving')
        if name=='ad':
            require(len(dev)==48 and len(held)==200,'AD population mismatch')
            continue
        quality = read(ROOT/f'work/phase6/representation-quality-v1/{name}/report.json')
        mask = read(ROOT/f'work/phase6/representation-quality-v1/{name}/mask-report.json')
        require(quality['source_sha256']==sha(ROOT/'tools/phase6/representation_quality.py'),'quality driver changed')
        require(mask['source_sha256']==sha(ROOT/'tools/phase6/representation_mask_control.py'),'mask driver changed')
        require(quality['factor_sha256']==sha(image),'quality image differs')
        labels = np.array(quality['labels'])
        ix = quality['evaluated_indices']
        require(not quality['smoke'] and len(ix)==len(set(ix)) and not set(ix)&(set(dev)|set(held)),'quality leakage')
        source_count = len(np.load(src/'mapped/labels.npy',mmap_mode='r'))
        require(set(ix)|set(dev)|set(held)==set(range(source_count)),'untouched coverage incomplete')
        require(mask['evaluated_indices']==ix and mask['labels']==quality['labels'],'mask populations differ')
        for policy,predictions in quality['predictions'].items():
            pred = np.array(predictions)
            q = quality['quality'][policy]
            require(len(pred)==len(labels) and q['correct']==int(np.count_nonzero(pred==labels)) and
                    q['accuracy']==float(np.mean(pred==labels)),'quality tally error')
        pred = np.array(mask['predictions'])
        require(mask['quality']['correct']==int(np.count_nonzero(pred==labels)),'mask tally error')
        require(mask['prior_subset_exact_predictions']==256,'mask is not frozen original control')
        require(quality['peak_rss_bytes']<1<<30 and mask['peak_rss_bytes']<1<<30,'final memory guard exceeded')
        result['classification'][name] = {'untouched_count':len(ix),'quality':quality['quality'],
            'column_mask_quality':mask['quality'],'factor_delta_interval':quality['intervals']['factor'],
            'factor_vs_mask_interval':mask['factor_comparison'],
            'peak_rss_bytes':max(quality['peak_rss_bytes'],mask['peak_rss_bytes'])}
    native = read(ROOT/'work/phase6/representation-native-v1/report.json')
    require(native['status']=='passed-native-full-model-comparison','native campaign incomplete')
    require(native['source_sha256']==sha(ROOT/'tools/phase6/representation_native.py'),'native driver changed')
    runs = 0
    for name,row in native['models'].items():
        for path,h in row['source_pins'].items():
            require(sha(ROOT/path)==h,'native model pin changed')
        for variant,v in row['variants'].items():
            require(v['geometry']['peak_data_scratch_bytes']<=32768,'SRAM overflow')
            require(len(v['runs'])==8,'native matrix incomplete')
            for r in v['runs']:
                raw_path = ROOT/f'work/phase6/representation-native-v1/native/{name}-{variant}-{r["sample"]}-{r["mode"]}-s{r["seed"]}.json'
                require(sha(raw_path)==r['native_report_sha256'],'raw native report changed')
                raw = read(raw_path)
                require(raw['status']=='passed' and raw['elapsed_cycles']==r['elapsed_cycles'],'native counter mismatch')
                fixture = v['fixtures'][r['sample']+'/'+r['mode']]
                require(raw['tensor_checks']==fixture['diagnostic_tensor_checks'],'missing tensor checks')
                require(fixture['program_bytes']<=32768 and fixture['payload_bytes']<=8*1024*1024,'program/payload overflow')
                for fn,h in fixture['fixture_files_sha256'].items():
                    require(sha(ROOT/r['fixture']/fn)==h,'native fixture changed')
                runs += 1
        for comp in row['comparisons']:
            require(abs(comp['elapsed_cycle_reduction_fraction']-
                (1-comp['factor_elapsed_cycles']/comp['dense_elapsed_cycles']))<1e-12,'native reduction error')
    result['representation_native'] = {'exact_runs':runs,'comparisons':{m:r['comparisons'] for m,r in native['models'].items()}}
    deadline = ROOT/'work/phase6/deadline-screen-v1'
    profiles = read(deadline/'profiles.json')
    dreport = read(deadline/'report.json')
    rows = read(deadline/'results.json')
    require(len(rows)==dreport['schedule_runs']==20640,'deadline count mismatch')
    keys = []
    for r in rows:
        keys.append(tuple(r[k] for k in ('case','ram_seed','word_cycles','fixed_cycles','program_mode','variant','policy')))
        require(r['completed']==r['jobs'] and r['finish_cycle']>=r['service_cycles']+r['overhead_cycles'],'schedule conservation failure')
        require(0<=r['misses']<=r['jobs'],'invalid misses')
        require(r['overhead_cycles']>=0,'negative overhead')
    require(len(keys)==len(set(keys)),'duplicated labeled schedule result')
    for model,variants in profiles.items():
        for variant,seeds in variants.items():
            for seed,p in seeds.items():
                require(sum(s['cycles'] for s in p['segments'])==p['native_cycles'],'native segment conservation')
                require(all(0<=s['save_dma_bytes']<=s['live_dma_bytes']<=32768 for s in p['segments']),'invalid checkpoint geometry')
    pause = read(ROOT/'work/phase6/deadline-pause-v1/report.json')
    require(pause['status']=='passed-native-pause' and len(pause['native_runs'])==12,'pause matrix incomplete')
    require(pause['driver_sha256']==sha(ROOT/'tools/phase6/deadline_pause.py'),'pause driver changed')
    ident = read(ROOT/'work/phase6/deadline-pause-v1/native-build/identity.json')
    for path,h in ident['source_sha256'].items():
        require(sha(path)==h,'pause source changed')
    require(sha(ROOT/'work/phase6/deadline-pause-v1/native.cpp')==ident['harness_sha256'],'pause harness changed')
    for r in pause['native_runs']:
        require(r['status']=='passed' and r['restored_byte_mismatches']==0,'pause failed preservation')
        if r['mode']!=3:
            require(r['tensor_checks']==2 and r['busy_guard_checks']==3,'pause safety checks incomplete')
            require(r['ad_deadline_missed']==(r['ad_response_global_native_cycles']>r['ad_relative_deadline_cycles']),'pause deadline tally')
    require(pause['negative_control']['returncode']!=0 and 'output mismatch' in pause['negative_control']['stderr'],'weak no-restore negative control')
    result['deadline'] = {'recorded_schedules':len(rows),'native_pause_exact_runs':12,
        'no_restore_negative_control':pause['negative_control'],'native_pause_witnesses':[r for r in pause['native_runs'] if r['mode']==0]}
    optimal = read(ROOT/'work/phase6/optimality-screen-v1/report.json')
    require(optimal['status']=='passed','optimality incomplete')
    gaps = []; families = 0
    for case in optimal['cases'].values():
        for budget,row in case['budgets'].items():
            c = row['certificate']
            require(c['verified_lower_bound']==c['verified_legal_upper'] and c['upper_over_lower']==1.,'finite family not certified')
            gaps.append(row['controls']['fixed_pair_strong']['gap_fraction']);families += 1
    require(families==18,'family count changed')
    result['optimality'] = {'certified_finite_families':families,'max_strong_control_cost_gap':max(gaps),
                           'model_scope':optimal['scope'],'limitations':optimal['model_limitations']}
    result['status'] = 'passed-raw-evidence-audit'
    inputs = list((ROOT/'tools/phase6').glob('*representation*.py'))+[
        ROOT/'tools/phase6/deadline_screen.py',ROOT/'tools/phase6/deadline_pause.py',
        ROOT/'tools/phase6/optimality_screen.py']
    result['current_experiment_sources_sha256'] = {str(p.relative_to(ROOT)):sha(p) for p in inputs}
    (BASE/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ('status','production_files_unchanged','optimality')},indent=2))


if __name__=='__main__':
    main()
