#!/usr/bin/env python3
"""Seal the explicit profiling, negative compiler and physical RTL evidence.

Uses the shared archive format. Generated binaries remain hash-pinned; declared
fixtures and text evidence are embedded. No hardware or evidence mutation.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import archive_matched_baselines as archive
import physical_engine_candidate as physical

ROOT=physical.ROOT
BASE=physical.BASE
PROFILE=ROOT/'work/phase6/engine-profile-v1/final'


class EngineCollector(archive.Collector):
    """Resolve only explicitly declared per-record tensor check filenames.

    Original JSON bytes remain unchanged. The generic walker receives a local
    descriptive label after this adapter validates and captures its scoped file.
    """
    def __init__(self,*args,scoped_fixtures,**kwargs):
        super().__init__(*args,**kwargs)
        self.scoped_fixtures=scoped_fixtures
        self.scoped_tensor_checks=[]

    def walk(self,value,origin,parent,directory,upstream,top=False):
        if isinstance(value,dict) and value.get('stress_tensor_checks'):
            fixture=value.get('fixture')
            if fixture not in self.scoped_fixtures:raise ValueError('tensor check fixture is undeclared')
            files=self.scoped_fixtures[fixture]; checks=[]
            for check in value['stress_tensor_checks']:
                name=check['file'];expected=check['sha256']
                if name not in files or files[name]!=expected:raise ValueError('tensor check filename/hash not in fixture')
                target=archive.contained(self.root/fixture,name)
                if archive.hash_file(target)!=expected:raise ValueError('tensor check scoped bytes changed')
                self.add(str(target.relative_to(self.root)),expected,origin=origin,embed=True)
                checks.append({**{k:v for k,v in check.items() if k!='file'},'tensor_file_label':name})
                self.scoped_tensor_checks.append(dict(origin=origin,fixture=fixture,file=name,sha256=expected))
            value={**value,'stress_tensor_checks':checks}
        return super().walk(value,origin,parent,directory,upstream,top)


def regression():
    """Valid nested output binds to its fixture; wrong hash/path is rejected."""
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp).resolve();folder=root/'work/fixture';folder.mkdir(parents=True)
        target=folder/'output.bin';target.write_bytes(b'exact');sha=archive.hash_file(target)
        source=root/'report.json'
        value=dict(status='passed',records=[dict(fixture='work/fixture',
            stress_tensor_checks=[dict(file='output.bin',sha256=sha)])])
        source.write_bytes(archive.encoded(value))
        collector=EngineCollector(root,scoped_fixtures={'work/fixture':{'output.bin':sha}})
        collector.input(source,'report')
        if collector.limitations or not any(row['workspace_path']=='work/fixture/output.bin' for row in collector.entries.values()):
            raise AssertionError('scoped tensor check did not bind')
        if any(row['workspace_path']=='output.bin' for row in collector.entries.values()):
            raise AssertionError('tensor check escaped to root')
        for name,badsha in [('output.bin','0'*64),('../output.bin',sha)]:
            value['records'][0]['stress_tensor_checks'][0]=dict(file=name,sha256=badsha)
            source.write_bytes(archive.encoded(value))
            try:EngineCollector(root,scoped_fixtures={'work/fixture':{'output.bin':sha}}).input(source,'report')
            except ValueError:pass
            else:raise AssertionError('invalid scoped tensor check accepted')
    return dict(status='passed',cases=['scoped filename embeds declared exact bytes','wrong digest rejected','undeclared traversal rejected'])


def build(output,reports,evidence,fixtures):
    # Same manifest construction as archive_matched_baselines.build; only the
    # collector differs. Shared verifier and original archival source stay intact.
    historical=physical.common.read(BASE/'physical/matched-v1/plan.json')['original_checkout']
    collector=EngineCollector(ROOT,relocate_roots=(historical,),scoped_fixtures=fixtures,include_fixtures=True)
    for role,paths in (('report',reports),('evidence',evidence)):
        for path in paths:collector.input(path,role)
    collector.resolve_fixtures()
    collector.add(archive.__file__,origin='archive-method',required=True)
    artifacts={};packed={}
    for key,row in sorted(collector.entries.items()):
        row=dict(row)
        if key in collector.raw:
            gz=archive.compressed(collector.raw[key]);name='artifacts/'+row['sha256'][:2]+'/'+row['sha256']+'.gz'
            row.update(archive=name,archive_sha256=archive.digest(gz));packed[name]=gz
        artifacts[key]=row
    manifest=dict(schema=1,status='sealed-matched-baseline-archive',
        scope='Explicit completed engine profiling, compiler-negative, native, route and physical evidence; see limitations.',
        original_root=str(ROOT),relocation_roots=sorted(collector.aliases),
        archive_paths_relative_to='manifest_directory',inputs=collector.inputs,artifacts=artifacts,
        references=[json.loads(r) for r in sorted({json.dumps(r,sort_keys=True) for r in collector.links})],
        limits=dict(max_file_bytes=collector.max_file_bytes,max_total_bytes=collector.max_total_bytes,max_hash_bytes=collector.max_hash_bytes),
        limitations=sorted(collector.limitations,key=lambda r:(r['kind'],r['reference'],r['origin'])),
        fixture_policy='Embed declared fixture files within budgets.',
        generated_binary_policy='Generated binaries are hash-pinned here; two companion portable archives embed tested image/executable.',
        source_snapshot_policy='Historical snapshots never overwrite workspace originals.',
        scoped_tensor_checks=collector.scoped_tensor_checks,scoped_adapter_regression=regression())
    manifest['checks']=archive.semantic_checks(manifest,collector.raw)
    raw_manifest=archive.encoded(manifest)
    if (output/'manifest.json').exists() and (output/'manifest.json').read_bytes()!=raw_manifest:
        raise FileExistsError('preserve immutable prior archive')
    output.mkdir(parents=True,exist_ok=True)
    for name,data in sorted(packed.items()):
        path=archive.contained(output,name);path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_bytes()!=data:raise ValueError('existing compressed artifact differs')
        if not path.exists():path.write_bytes(data)
    (output/'manifest.json').write_bytes(raw_manifest)
    (output/'manifest.sha256').write_text(archive.digest(raw_manifest)+'\n')
    return manifest


def portable_verify(path):
    manifest=physical.common.read(path/'manifest.json')
    verification=physical.common.read(path/'verification.json')
    if (verification['status']!='passed' or physical.common.sha(path/'manifest.json')!=verification['manifest_sha256']
            or verification['files']!=len(manifest['files'])):
        raise ValueError('portable archive manifest/count differs')
    uncompressed=compressed=0
    for name,row in manifest['files'].items():
        packed=archive.contained(path,row['archive_path']).read_bytes()
        raw=gzip.decompress(packed)
        if (hashlib.sha256(packed).hexdigest()!=row['gzip_sha256'] or
                hashlib.sha256(raw).hexdigest()!=row['sha256'] or len(raw)!=row['bytes']):
            raise ValueError('portable archive artifact differs: '+name)
        uncompressed+=len(raw);compressed+=len(packed)
    if uncompressed!=verification['uncompressed_bytes'] or compressed!=verification['compressed_bytes']:
        raise ValueError('portable archive byte counts differ')
    return verification


def run(output):
    output=Path(output).resolve()
    reports=[PROFILE/'report.json',ROOT/'work/phase6/engine-candidate-compiler-v1/report.json',
        BASE/'native/report.json',BASE/'route27/report.json',BASE/'physical/screen-v1/report.json',
        BASE/'physical/matched-v1/report.json',BASE/'independent-audit/report.json',
        BASE/'independent-audit/report-matched.json']
    for path in reports:
        record=physical.common.read(path)
        if not archive.completed(record):raise ValueError('incomplete explicit evidence: '+str(path))
    profile=physical.common.read(PROFILE/'report.json')
    sources=dict(profile['identity']['input_files'])
    for name,expected in [('native_profile.cpp',profile['identity']['profile_harness_sha256']),
            ('profile_native',profile['identity']['profile_executable_sha256'])]:
        sources[physical.common.relative(PROFILE/name)]=expected
    for model,entry in profile['models'].items():
        for row in entry['profiles']:
            sources[row['path']]=row['sha256']
            enriched=physical.common.read(ROOT/row['path'])
            raw=PROFILE/f'{model}-s{row["stall_seed"]}.json.profile.json'
            sources[physical.common.relative(raw)]=enriched['raw_profile_sha256']
    for path in (PROFILE/'summary.json',PROFILE/'summary.md',
            ROOT/'tools/phase6/profile_engine_summary.py',Path(__file__).resolve()):
        sources[physical.common.relative(path)]=physical.common.sha(path)
    physical.common.verify(ROOT,sources)
    supplement=BASE/'physical/archive-inputs-v3.json'
    value=dict(status='passed',physical_board=False,
        scope='Explicit source/input binding for the completed observational engine profile',
        source_sha256=sources,profile_report=physical.common.relative(PROFILE/'report.json'),
        profile_report_sha256=physical.common.sha(PROFILE/'report.json'))
    if supplement.exists() and physical.common.read(supplement)!=value:
        raise ValueError('archive input supplement changed; preserve existing evidence')
    if not supplement.exists():physical.common.save(supplement,value)
    reports.append(supplement)
    evidence=[BASE/'independent-audit/NOTES.md',BASE/'independent-audit/audit.py',
        BASE/'independent-audit/audit_matched.py',
        Path(__file__).resolve(),ROOT/'tools/phase6/physical_engine_candidate_matched.py',
        ROOT/'work/phase6/engine-prior-work-v1/NOTES.md',ROOT/'work/phase6/engine-prior-work-v1/sources.json']
    fixtures={}
    for label in ('screen-v1','matched-v1'):
        plan=physical.common.read(BASE/'physical'/label/'plan.json')
        for model in plan['fixtures'].values():
            for row in model.values():fixtures[row['directory']]=row['files']
    manifest=build(output,reports,evidence,fixtures)
    result=archive.verify(output,ROOT,workspace=True)
    portable={}
    for label in ('screen-v1','matched-v1'):
        source=BASE/'physical'/label/'archive'
        expected=portable_verify(source)
        target=output/'portable'/label
        if not target.exists():shutil.copytree(source,target)
        actual=portable_verify(target)
        if actual!=expected:raise ValueError('durable portable archive differs')
        portable[label]=dict(actual,directory=str(target.relative_to(output)))
    result['portable_archives']=portable
    physical.common.save(Path(output)/'verification.json',result)
    return dict(result,manifest_sha256=physical.common.sha(Path(output)/'manifest.json'),
        limitations=manifest['limitations'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/research/evidence/phase6/engine-candidate-v2')
    args=parser.parse_args();print(json.dumps(run(args.output),indent=2,sort_keys=True))
