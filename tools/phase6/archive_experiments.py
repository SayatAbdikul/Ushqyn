#!/usr/bin/env python3
"""Archive short optimization evidence without claiming release qualification."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from variants import ROOT, check_frozen, sha
from run_screening import save_json, verify_files
from run_priority import read_rows
from quick_experiments import summarize

BASE=ROOT/'work/phase6/experiments-v1'
OUT=ROOT/'docs/research/evidence/phase6/experiments'


def archive(winner):
    check_frozen();OUT.mkdir(parents=True,exist_ok=True)
    artifacts={};candidates={};physical={}
    def save(path):
        relative=path.relative_to(BASE)
        destination=OUT/(str(relative)+'.gz');destination.parent.mkdir(parents=True,exist_ok=True)
        raw=path.read_bytes();destination.write_bytes(gzip.compress(raw,mtime=0))
        artifacts[str(relative)]=dict(file=str(destination.relative_to(ROOT)),
            uncompressed_sha256=hashlib.sha256(raw).hexdigest(),gzip_sha256=sha(destination))
    for directory in sorted(BASE.iterdir()):
        if not (directory/'engine.sv').exists():continue
        label=directory.name;candidate={'engine_sha256':sha(directory/'engine.sv')}
        for filename in ('engine.sv','identity.json','pll.v','host.sv','build.tcl'):
            if (directory/filename).exists():save(directory/filename)
        for stage in ('engine','native','edges','route'):
            path=directory/stage/'report.json'
            if path.exists():
                report=json.loads(path.read_text());candidate[stage]=report['status'];save(path)
                if stage!='route' and report['status']=='passed':
                    verify_files(ROOT,report['sources'])
                    if stage in ('engine','edges'):
                        xml=directory/stage/'results.xml'
                        cases=ET.parse(xml).findall('.//testcase')
                        if sha(xml)!=report['results_sha256'] or len(cases)!=(3 if stage=='engine' else 1) or any(c.find('failure') is not None or c.find('error') is not None for c in cases):
                            raise ValueError('regression report changed')
                        save(xml)
                if stage=='route' and 'bitstream' in report:
                    verify_files(ROOT,{report['bitstream']:report['bitstream_sha256']})
                    if report['engine_sha256']!=candidate['engine_sha256']:raise ValueError('route identity mismatch')
                    candidate['resources']=report['resources'];candidate['fmax_mhz']=report['routed_core_fmax_mhz']
                    candidate['clock_mhz']=report['core_clock_mhz']
                    for suffix,key in (('.rpt.txt','route_sha256'),('_tr_content.html','timing_sha256')):
                        evidence=directory/'route/phase6_tiled_host/impl/pnr'/('phase6_tiled_host'+suffix)
                        if sha(evidence)!=report[key]:raise ValueError('route report changed')
                        save(evidence)
            elif stage=='route' and (directory/stage/'build.log').exists():
                candidate[stage]='failed-synthesis' if 'RP0006' in (directory/stage/'build.log').read_text() else 'no-completed-report'
        if (directory/'route/build.log').exists():save(directory/'route/build.log')
        for path in (directory/'edges').glob('results-first.xml'):save(path)
        candidates[label]=candidate
    for directory in sorted(BASE.glob('quick-*')):
        report_path=directory/'report.json'
        if not report_path.exists():continue
        report=json.loads(report_path.read_text());plan_path=directory/'plan.json';plan=json.loads(plan_path.read_text())
        if report['status']!='passed-short-screen':raise ValueError(f'{directory.name} is not complete')
        if sha(plan_path)!=report['plan_sha256']:raise ValueError('board plan changed')
        runner=BASE/'runner-snapshots'/(plan['runner_sha256']+'.py')
        if not runner.exists() or sha(runner)!=plan['runner_sha256']:raise ValueError('historical physical runner missing')
        save(runner)
        rows=read_rows(directory/'records.jsonl')
        if sha(directory/'records.jsonl')!=report['records_sha256'] or summarize(rows,plan['labels'])!=report['summary']:
            raise ValueError('physical records changed')
        verify_files(ROOT/'work/phase6/priority-v1',plan['preserved_accuracy'])
        for programmed in report['programming']:
            path=directory/f'program-{programmed["variant"]}.log'
            if sha(path)!=programmed['log_sha256']:raise ValueError('programming log changed')
            save(path)
            image=plan['images'][programmed['variant']]
            verify_files(ROOT,{image['file']:image['sha256']})
        for file in ('plan.json','report.json','records.jsonl'):save(directory/file)
        physical[directory.name]=dict(inferences=report['completed'],
            schedule=plan.get('schedule_kind','original'),summary=report['summary'])
    for path in (BASE/'algorithms/report.json',BASE/'host-batch/report.json',BASE/'chain/fixtures.json'):
        if path.exists():save(path)
    for path in (BASE/'chain').glob('native-*/report.json'):
        report=json.loads(path.read_text())
        if report['status']!='passed':raise ValueError('chain RTL incomplete')
        verify_files(ROOT,report['sources']);save(path)
    if (BASE/'chain/fixtures.json').exists():
        for fixture in json.loads((BASE/'chain/fixtures.json').read_text())['fixtures']:
            root=BASE/'chain/fixtures'/fixture['name'];verify_files(root,fixture['files'])
            for name in fixture['files']:save(root/name)
    route=json.loads((BASE/winner/'route/report.json').read_text())
    if route['status']!='passed-route' or not any(winner in r['summary'] and r['schedule']=='original' for r in physical.values()):
        raise ValueError('winner needs routed and physical evidence')
    bitstream=ROOT/route['bitstream']
    release=ROOT/f'hardware/releases/phase6/experimental_{winner}.fs.gz'
    release.write_bytes(gzip.compress(bitstream.read_bytes(),mtime=0))
    reference=physical['quick-a']['summary']['writeback']
    comparisons={}
    for campaign, result in physical.items():
        if winner not in result['summary']:continue
        rows=result['summary'][winner]
        comparisons[campaign]=dict(schedule=result['schedule'],models={m:dict(**r,
            inferences_per_second=1000/r['median_ms'],
            speedup_vs_writeback=reference[m]['median_ms']/r['median_ms'],
            latency_reduction_percent=100*(1-r['median_ms']/reference[m]['median_ms'])) for m,r in rows.items()},
            geometric_mean_speedup=math.sqrt(math.prod(reference[m]['median_ms']/r['median_ms'] for m,r in rows.items())))
    recommended_schedules={m:min(comparisons,key=lambda c:comparisons[c]['models'][m]['median_ms']) for m in ('kws','vww')}
    summary=dict(schema=1,status='short-screens-complete',full_verification_complete=False,
        candidates=candidates,physical=physical,artifacts=artifacts,selected_candidate=winner,
        comparisons=comparisons,recommended_schedule_campaigns=recommended_schedules,
        candidate_image=dict(path=str(release.relative_to(ROOT)),gzip_sha256=sha(release),uncompressed_sha256=sha(bitstream)),
        preserved_accuracy_status=json.loads((ROOT/'work/phase6/priority-v1/report.json').read_text())['status'],
        scope='Short optimization screens and architecture prototypes; no full accuracy, long stress, energy or SOTA claim.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'rtl/phase6/experimental_engine.sv',
            *[ROOT/'tools/phase6'/n for n in ('experiments.py','quick_experiments.py','clock_experiment.py',
                                           'chain_resident.py','algorithm_experiments.py','host_batch_experiment.py')],
            ROOT/'test/phase6/test_experiment_edges.py',ROOT/'test/phase6/test_experiment_runners.py']})
    save_json(OUT/'summary.json',summary);check_frozen();print(json.dumps(dict(selected=winner,physical_inferences=sum(r['inferences'] for r in physical.values()),artifacts=len(artifacts))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('winner')
    args=parser.parse_args();archive(args.winner)
