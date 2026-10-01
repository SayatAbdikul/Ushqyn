#!/usr/bin/env python3
"""Archive an explicit, completed matched-baseline evidence dependency closure.

Examples (supply final paths deliberately; no implicit campaign discovery):
  python3 tools/phase6/archive_matched_baselines.py --run --report REPORT.json \
      --report OTHER.json --plan CAMPAIGN/plan.json --evidence DECISION.md
  python3 tools/phase6/archive_matched_baselines.py --verify --verify-workspace

Only schema-referenced files are followed. --include-fixtures embeds declared
fixture binaries within size budgets; generated executables, bitstreams and
model files remain hash-pinned. Historical source versions may be recovered from
already sealed archives; their snapshots never overwrite workspace originals.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_OUT=ROOT/'docs/research/evidence/phase6/matched-baselines-v1'
HEX=re.compile(r'^[0-9a-f]{64}$')
TEXT_SUFFIXES={'.py','.sv','.v','.tcl','.gprj','.json','.jsonl','.txt','.xml',
               '.md','.html','.csv','.log','.h','.hpp','.cpp','.toml','.yaml','.yml'}
SOURCE_MAPS={'sources','source_sha256','source_hashes','compiler_sources','native_sources',
             'dependency_sha256','physical_dependencies','protocol_sources','source_pins',
             'evidence_sha256','compacted_source_sha256','route_sources','upstream_source_files'}
FILE_MAPS={'files','fixture_files','fixture_sha256','prior_files'}
IMPLICIT={'plan_sha256':'plan.json','records_sha256':'records.jsonl',
          'program_log_sha256':'program.log','results_sha256':'results.xml'}


def digest(data):return hashlib.sha256(data).hexdigest()


def hash_file(path):
    hasher=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):hasher.update(chunk)
    return hasher.hexdigest()


def encoded(value):return (json.dumps(value,indent=2,sort_keys=True)+'\n').encode()


def compressed(raw):
    stream=io.BytesIO()
    with gzip.GzipFile(filename='',mode='wb',fileobj=stream,compresslevel=6,mtime=0) as archive:
        archive.write(raw)
    return stream.getvalue()


def contained(base,name):
    path=(Path(base)/name).resolve()
    path.relative_to(Path(base).resolve())
    return path


def completed(document):
    status=document.get('status','')
    return isinstance(status,str) and (status.startswith('passed') or status in
        ('enumerated','verified','sealed-matched-baseline-archive'))


class Collector:
    def __init__(self,root,relocate_roots=(),max_file_bytes=16_000_000,
                 max_total_bytes=128_000_000,max_hash_bytes=512_000_000,include_fixtures=False):
        self.root=Path(root).resolve()
        self.aliases={str(self.root),*(str(Path(p)) for p in relocate_roots)}
        self.max_file_bytes=max_file_bytes;self.max_total_bytes=max_total_bytes
        self.max_hash_bytes=max_hash_bytes;self.total_bytes=0;self.hash_bytes=0
        self.entries={};self.raw={};self.links=[];self.limitations=[];self.inputs=[]
        self.snapshots=None;self.visited=set()
        self.include_fixtures=include_fixtures
        self.fixture_maps={};self.pending_fixtures=[]

    def limit(self,kind,reference,origin,detail):
        row=dict(kind=kind,reference=str(reference),origin=origin,detail=detail)
        if row not in self.limitations:self.limitations.append(row)

    def path(self,name,base=None):
        name=Path(name)
        if name.is_absolute():
            for alias in sorted(self.aliases,key=len,reverse=True):
                try:return contained(self.root,name.relative_to(alias))
                except ValueError:pass
            raise ValueError('absolute reference has no declared checkout relocation')
        return contained(self.root,(Path(base)/name if base else name))

    def snapshot_index(self):
        if self.snapshots is not None:return self.snapshots
        result={}
        # This indexes only small existing archive manifests, never work trees.
        evidence=self.root/'docs/research/evidence'
        for manifest_path in sorted(evidence.rglob('manifest.json')) if evidence.exists() else ():
            if manifest_path.stat().st_size>self.max_file_bytes:continue
            try:manifest=json.loads(manifest_path.read_bytes())
            except (ValueError,OSError):continue
            for name,row in manifest.get('artifacts',{}).items():
                if not isinstance(row,dict) or not HEX.fullmatch(str(row.get('sha256',''))):continue
                logical=row.get('workspace_path',name)
                if not row.get('archive'):continue
                base=manifest_path.parent if manifest.get('archive_paths_relative_to')=='manifest_directory' else self.root
                try:archive=contained(base,row['archive'])
                except ValueError:continue
                result.setdefault((logical,row['sha256']),[]).append((archive,row))
        self.snapshots=result
        return result

    def historical(self,name,expected):
        for archive,row in self.snapshot_index().get((name,expected),[]):
            if not archive.is_file() or row.get('bytes',self.max_file_bytes+1)>self.max_file_bytes:continue
            raw_gz=archive.read_bytes()
            if row.get('archive_sha256') and digest(raw_gz)!=row['archive_sha256']:continue
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(raw_gz)) as stream:data=stream.read(self.max_file_bytes+1)
            except (OSError,EOFError):continue
            if len(data)<=self.max_file_bytes and digest(data)==expected:
                return data,dict(kind='previous_archive',path=str(archive.relative_to(self.root)),
                                 archive_sha256=digest(raw_gz))
        return None,None

    def add(self,name,expected=None,*,origin,base=None,required=False,follow=True,embed=False):
        if expected is not None and not HEX.fullmatch(str(expected)):
            raise ValueError(f'invalid dependency digest: {name}')
        try:path=self.path(name,base)
        except ValueError as error:
            if required:raise
            self.limit('unresolved_path',name,origin,str(error));return None
        logical=str(path.relative_to(self.root))
        if expected is not None and logical+'@'+expected in self.entries and \
                (not embed or self.entries[logical+'@'+expected]['storage']=='gzip'):
            key=logical+'@'+expected
            self.links.append(dict(origin=origin,reference=str(name),artifact=key))
            return key
        actual=None;raw=None;capture=None;size=None
        if path.is_file():
            size=path.stat().st_size
            if self.hash_bytes+size<=self.max_hash_bytes:
                actual=hash_file(path);self.hash_bytes+=size
            elif required:raise ValueError(f'archive hashing budget exhausted: {logical}')
        if expected is None:expected=actual
        if expected is None:
            if required:raise ValueError(f'missing explicit input: {logical}')
            self.limit('unavailable_dependency',logical,origin,'No file and no digest to pin.');return None
        key=logical+'@'+expected
        self.links.append(dict(origin=origin,reference=str(name),artifact=key))
        if key in self.entries and (not embed or self.entries[key]['storage']=='gzip'):return key
        matches=actual==expected
        if matches:
            capture=dict(kind='workspace',path=logical)
            if (path.suffix in TEXT_SUFFIXES or embed) and size<=self.max_file_bytes:raw=path.read_bytes()
        elif path.suffix in TEXT_SUFFIXES or embed:
            raw,capture=self.historical(logical,expected)
            if raw is not None:size=len(raw)
        if capture is None:
            if required:raise ValueError(f'explicit input identity unavailable: {logical}')
            reason='missing' if not path.is_file() else 'hash-budget' if actual is None else 'workspace-sha-mismatch'
            self.limit('unavailable_dependency',logical,origin,reason)
            self.entries[key]=dict(workspace_path=logical,sha256=expected,bytes=size,
                storage='unavailable',workspace_policy='require_match',reason=reason)
            return key
        policy='require_match' if capture['kind']=='workspace' else 'historical_snapshot'
        item=dict(workspace_path=logical,sha256=expected,bytes=size,capture=capture,workspace_policy=policy)
        if raw is not None and self.total_bytes+len(raw)<=self.max_total_bytes:
            item['storage']='gzip';self.raw[key]=raw;self.total_bytes+=len(raw)
        else:
            item['storage']='hash_only'
            item['reason']='generated_binary' if path.suffix not in TEXT_SUFFIXES and not embed else 'archive-size-budget'
            if path.suffix in TEXT_SUFFIXES or embed:
                self.limit('requested_content_not_embedded',logical,origin,item['reason'])
        self.entries[key]=item
        if follow and path.suffix=='.json' and key in self.raw:
            self.visit(key)
        return key

    def visit(self,key):
        if key in self.visited:return
        self.visited.add(key)
        value=json.loads(self.raw[key])
        if not isinstance(value,(dict,list)):return
        path=Path(self.entries[key]['workspace_path'])
        self.walk(value,key,path.parent,None,None,top=True)
        if isinstance(value,dict) and value.get('physical_board') is True and \
                value.get('status','').startswith('passed'):
            for filename in ('plan.json','records.jsonl','seal.json','program.log'):
                self.add(str(path.parent/filename),origin=key)
        if isinstance(value,dict) and value.get('status')=='passed-route' and value.get('bitstream'):
            prefix=str(Path(value['bitstream']).with_suffix(''))
            for suffix,field in (('.rpt.txt','route_sha256'),('_tr_content.html','timing_sha256')):
                if field in value:self.add(prefix+suffix,value[field],origin=key)

    def walk(self,value,origin,parent,directory,upstream,top=False):
        if isinstance(value,list):
            for item in value:self.walk(item,origin,parent,directory,upstream)
            return
        if not isinstance(value,dict):return
        for key in ('original_root','original_checkout','current_checkout'):
            if isinstance(value.get(key),str) and Path(value[key]).is_absolute():self.aliases.add(value[key])
        revision=value.get('revision')
        if isinstance(revision,str) and re.fullmatch(r'[0-9a-f]{40}',revision):
            candidate=Path('work/phase6/defines-source')/('DeFiNES-'+revision)
            if (self.root/candidate).is_dir():upstream=candidate
        for key in ('directory','fixture_directory','prior_directory'):
            if isinstance(value.get(key),str):
                try:directory=self.path(value[key]).relative_to(self.root)
                except ValueError:self.limit('unresolved_directory',value[key],origin,'Provide --relocate-root for the original checkout.');directory=None
                break
        if directory is None:
            label=value.get('label',value.get('name'))
            if isinstance(label,str) and label and '/' not in label:
                candidate=parent/'fixtures'/label
                if (self.root/candidate).is_dir():directory=candidate
        for key,items in value.items():
            if key in SOURCE_MAPS and isinstance(items,dict):
                for name,expected in items.items():
                    if HEX.fullmatch(str(expected)):self.add(name,expected,origin=origin)
            elif key in FILE_MAPS and isinstance(items,dict):
                file_directory=directory
                if key=='prior_files' and isinstance(value.get('prior_directory'),str):
                    try:file_directory=self.path(value['prior_directory']).relative_to(self.root)
                    except ValueError:file_directory=None
                identities={name:sha for name,sha in items.items() if HEX.fullmatch(str(sha))}
                if file_directory is not None and identities:
                    signature=tuple(sorted(identities.items()))
                    self.fixture_maps.setdefault(signature,set()).add(str(file_directory))
                elif identities and all('/' not in name for name in identities):
                    self.pending_fixtures.append((identities,origin,key));continue
                for name,expected in items.items():
                    if HEX.fullmatch(str(expected)):
                        if file_directory is None and '/' not in name:
                            self.limit('unresolved_fixture_directory',key,origin,
                                       'File hashes have no declared fixture directory; original report retains the identities.')
                        else:self.add(name,expected,origin=origin,base=file_directory,embed=self.include_fixtures)
        if isinstance(value.get('path'),str) and HEX.fullmatch(str(value.get('sha256',''))):
            self.add(value['path'],value['sha256'],origin=origin,base=upstream)
        for key,reference in value.items():
            if not isinstance(reference,str):continue
            expected=value.get(key+'_sha256')
            if key.endswith('_file'):expected=value.get(key[:-5]+'_sha256',expected)
            if key=='file':expected=value.get('sha256',expected)
            if HEX.fullmatch(str(expected)) and (Path(reference).suffix or '/' in reference):
                # Paths in report fields are checkout-relative; only fixture
                # file maps and implicit physical companions are directory-relative.
                self.add(reference,expected,origin=origin)
        for field,filename in IMPLICIT.items():
            physical=value.get('physical_board') is True or 'records_sha256' in value or \
                     Path(self.entries[origin]['workspace_path']).name=='seal.json'
            if top and HEX.fullmatch(str(value.get(field,''))) and field[:-7] not in value and \
                    (physical or field=='results_sha256'):
                self.add(str(parent/filename),value[field],origin=origin)
        if 'report_sha256' in value and 'report' not in value and \
                self.entries[origin]['workspace_path'].endswith('/seal.json'):
            self.add(str(parent/'report.json'),value['report_sha256'],origin=origin)
        # GPRJ/raw route sources are already covered by hardware source maps.
        # Native rows are retained inside reports; also collect an exact matching
        # standalone native record when the standard fixture-relative path exists.
        if directory is not None and value.get('status')=='passed' and 'elapsed_cycles' in value:
            seed=value.get('stall_seed',value.get('seed'))
            if type(seed) is int:
                paths=(directory/f'native-s{seed}.json',directory.parent/f'native-{directory.name}-s{seed}.json',
                       parent/f'{directory.name}-native-s{seed}.json')
                for native in paths:
                    file=self.root/native
                    if file.is_file() and file.stat().st_size<=self.max_file_bytes:
                        try:same=json.loads(file.read_bytes())==value
                        except ValueError:same=False
                        # The identical row was traversed with its fixture context
                        # above; a standalone record can omit that inherited path.
                        if same:self.add(str(native),origin=origin,follow=False);break
        for key,child in value.items():
            if key in SOURCE_MAPS or key in FILE_MAPS and isinstance(child,dict):continue
            if key=='fixtures' and isinstance(child,dict):
                folder=None
                if isinstance(value.get('fixture_root'),str):
                    try:folder=self.path(value['fixture_root']).relative_to(self.root)
                    except ValueError:pass
                for label,row in child.items():
                    local=parent/'fixtures'/label
                    fallback=local if (self.root/local).is_dir() else directory
                    self.walk(row,origin,parent,folder/label if folder else fallback,upstream)
            else:self.walk(child,origin,parent,directory,upstream)

    def input(self,path,role):
        key=self.add(path,origin='explicit:'+role,required=True)
        if key not in self.raw:
            raise ValueError(f'explicit input cannot be embedded within archive budgets: {path}')
        if role in ('report','plan'):
            value=json.loads(self.raw[key])
            if role=='report' and not completed(value):raise ValueError(f'explicit report is incomplete: {path}')
            if role=='plan' and value.get('status') not in ('prepared',None):
                raise ValueError(f'unexpected campaign plan status: {path}')
        self.inputs.append(dict(artifact=key,role=role))

    def resolve_fixtures(self):
        """Bind old native rows only to full matching file maps already in closure."""
        pending=self.pending_fixtures;self.pending_fixtures=[]
        for identities,origin,field in pending:
            directories=self.fixture_maps.get(tuple(sorted(identities.items())),set())
            if not directories:
                self.limit('unresolved_fixture_directory',field,origin,
                           'No declared fixture directory with exactly these file identities was found; original report retains the hashes.')
                continue
            # Any directory has byte-identical files. Retain the deterministic
            # first binding as the explicit provenance link, without a tree scan.
            directory=min(directories)
            for name,expected in identities.items():
                self.add(name,expected,origin=origin,base=directory,embed=self.include_fixtures)


def semantic_checks(manifest,raw):
    """Verify signed matched-board records and medians using archived bytes."""
    by_path={row['workspace_path']:key for key,row in manifest['artifacts'].items() if key in raw}
    checks=[]
    for entry in manifest['inputs']:
        if entry['role']!='report':continue
        report=json.loads(raw[entry['artifact']])
        if not completed(report):raise ValueError('archived report is incomplete')
        if report.get('status')!='passed-matched-short-screen':continue
        parent=Path(manifest['artifacts'][entry['artifact']]['workspace_path']).parent
        def data(filename,expected=None):
            key=str(parent/filename)+'@'+expected if expected else by_path.get(str(parent/filename))
            try:return raw[key]
            except KeyError:raise ValueError(f'missing physical archive dependency: {parent/filename}') from None
        plan=json.loads(data('plan.json',report['plan_sha256']));seal=json.loads(data('seal.json'))
        for key,filename in (('plan','plan.json'),('report','report.json'),('records','records.jsonl')):
            if seal[key+'_sha256']!=digest(data(filename,seal[key+'_sha256'])):raise ValueError('physical seal mismatch: '+filename)
        if report['plan_sha256']!=seal['plan_sha256'] or report['records_sha256']!=seal['records_sha256']:
            raise ValueError('physical report/seal mismatch')
        if digest(data('program.log',report['program_log_sha256']))!=report['program_log_sha256']:
            raise ValueError('physical programming log mismatch')
        rows=[]
        for line in data('records.jsonl',report['records_sha256']).splitlines():
            row=json.loads(line);signature=row.pop('record_sha256')
            if signature!=digest(json.dumps(row,sort_keys=True,separators=(',',':')).encode()):
                raise ValueError('physical record signature mismatch')
            if row['output_hex']!=row['expected_hex'] or row['input_readback_verified'] is not True or \
                    row['bitstream_sha256']!=plan['hardware']['image']['sha256']:
                raise ValueError('physical exactness or image identity mismatch')
            rows.append(row)
        if rows!=report['records'] or len(rows)!=report['completed'] or len(rows)!=plan['planned']:
            raise ValueError('physical record coverage mismatch')
        medians={}
        for policy,models in report['summary'].items():
            medians[policy]={}
            for model,summary in models.items():
                representative=summary.get('measured_policy',policy)
                samples=[r['elapsed_cycles'] for r in rows if r['policy']==representative and
                         r['model']==model and r['kind']=='timed']
                if len(samples)!=plan['repeats'] or statistics.median(samples)!=summary['median_cycles']:
                    raise ValueError('physical median or repeat count mismatch')
                medians[policy][model]=summary['median_cycles']
        checks.append(dict(report=entry['artifact'],physical_records=len(rows),medians=medians))
    return checks


def build(root,output,reports=(),plans=(),evidence=(),relocate_roots=(),**limits):
    if not reports:raise ValueError('at least one explicit final --report is required')
    collector=Collector(root,relocate_roots,**limits)
    for role,paths in (('report',reports),('plan',plans),('evidence',evidence)):
        for path in paths:collector.input(path,role)
    collector.resolve_fixtures()
    script=Path(__file__).resolve()
    if script.is_relative_to(collector.root):collector.add(str(script.relative_to(collector.root)),origin='archive-method',required=True)
    artifacts={}
    packed={}
    for key,row in sorted(collector.entries.items()):
        row=dict(row)
        if key in collector.raw:
            gz=compressed(collector.raw[key]);name='artifacts/'+row['sha256'][:2]+'/'+row['sha256']+'.gz'
            row.update(archive=name,archive_sha256=digest(gz));packed[name]=gz
        artifacts[key]=row
    manifest=dict(schema=1,status='sealed-matched-baseline-archive',
        scope='Explicit completed matched-baseline reports and their declared dependencies; see limitations.',
        original_root=str(collector.root),relocation_roots=sorted(collector.aliases),
        archive_paths_relative_to='manifest_directory',inputs=collector.inputs,artifacts=artifacts,
        references=sorted({json.dumps(r,sort_keys=True) for r in collector.links}),
        limits=dict(max_file_bytes=collector.max_file_bytes,max_total_bytes=collector.max_total_bytes,
                    max_hash_bytes=collector.max_hash_bytes),
        limitations=sorted(collector.limitations,key=lambda r:(r['kind'],r['reference'],r['origin'])),
        fixture_policy='Embed declared fixture files within budgets.' if collector.include_fixtures else 'Hash-pin generated fixture binaries; use --include-fixtures to embed.',
        generated_binary_policy='Generated executables, bitstreams and model files are hash-pinned only.',
        source_snapshot_policy='Historical source versions are captured separately by logical path and digest; never rewrite originals.')
    manifest['references']=[json.loads(row) for row in manifest['references']]
    manifest['checks']=semantic_checks(manifest,collector.raw)
    output=Path(output).resolve();raw_manifest=encoded(manifest)
    if (output/'manifest.json').exists() and (output/'manifest.json').read_bytes()!=raw_manifest:
        raise FileExistsError('archive differs from existing seal; choose a fresh --output')
    output.mkdir(parents=True,exist_ok=True)
    for name,data in sorted(packed.items()):
        path=contained(output,name);path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_bytes()!=data:raise ValueError('existing compressed artifact differs')
        if not path.exists():path.write_bytes(data)
    (output/'manifest.json').write_bytes(raw_manifest)
    (output/'manifest.sha256').write_text(digest(raw_manifest)+'\n')
    return manifest


def verify(output,root=ROOT,workspace=False):
    output=Path(output).resolve();root=Path(root).resolve()
    raw_manifest=(output/'manifest.json').read_bytes()
    if (output/'manifest.sha256').read_text().strip()!=digest(raw_manifest):raise ValueError('manifest seal mismatch')
    manifest=json.loads(raw_manifest)
    if manifest.get('schema')!=1 or manifest.get('status')!='sealed-matched-baseline-archive':
        raise ValueError('unknown matched archive schema/status')
    raw={};unavailable=[];historical=0
    for key,row in manifest['artifacts'].items():
        if key!=row['workspace_path']+'@'+row['sha256']:raise ValueError('artifact identity mismatch')
        contained(root,row['workspace_path'])
        if row['storage']=='gzip':
            gz=contained(output,row['archive']).read_bytes()
            if digest(gz)!=row['archive_sha256'] or gz[4:8]!=b'\0'*4:raise ValueError('gzip seal/mtime mismatch')
            with gzip.GzipFile(fileobj=io.BytesIO(gz)) as stream:data=stream.read(row['bytes']+1)
            if len(data)!=row['bytes'] or digest(data)!=row['sha256'] or compressed(data)!=gz:
                raise ValueError('compressed payload mismatch')
            raw[key]=data
        elif row['storage'] not in ('hash_only','unavailable'):raise ValueError('unknown artifact storage')
        if row['workspace_policy']=='historical_snapshot':historical+=1
        elif workspace:
            path=contained(root,row['workspace_path'])
            if not path.is_file() or hash_file(path)!=row['sha256']:unavailable.append(row['workspace_path'])
    for link in manifest['references']:
        if link['artifact'] not in manifest['artifacts']:raise ValueError('unresolved manifest reference')
    if semantic_checks(manifest,raw)!=manifest['checks']:raise ValueError('archive semantic checks changed')
    if unavailable:raise ValueError('workspace dependencies missing or changed: '+', '.join(sorted(set(unavailable))))
    return dict(status='verified',archived_files=len(raw),hash_pinned_files=len(manifest['artifacts'])-len(raw),
        historical_source_snapshots=historical,limitations=manifest['limitations'],
        workspace_checked=workspace,manifest_sha256=digest(raw_manifest),checks=manifest['checks'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run',action='store_true');mode.add_argument('--verify',action='store_true')
    parser.add_argument('--report',type=Path,action='append',default=[])
    parser.add_argument('--plan',type=Path,action='append',default=[])
    parser.add_argument('--evidence',type=Path,action='append',default=[])
    parser.add_argument('--relocate-root',type=Path,action='append',default=[])
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--verify-workspace',action='store_true')
    parser.add_argument('--include-fixtures',action='store_true',help='embed explicitly declared fixture files within size budgets')
    parser.add_argument('--max-file-bytes',type=int,default=16_000_000)
    parser.add_argument('--max-total-bytes',type=int,default=128_000_000)
    parser.add_argument('--max-hash-bytes',type=int,default=512_000_000)
    args=parser.parse_args()
    if min(args.max_file_bytes,args.max_total_bytes,args.max_hash_bytes)<1:parser.error('budgets must be positive')
    if args.run:
        build(ROOT,args.output,args.report,args.plan,args.evidence,args.relocate_root,
            max_file_bytes=args.max_file_bytes,max_total_bytes=args.max_total_bytes,max_hash_bytes=args.max_hash_bytes,
            include_fixtures=args.include_fixtures)
    print(json.dumps(verify(args.output,ROOT,args.verify_workspace),indent=2,sort_keys=True))


if __name__=='__main__':main()
