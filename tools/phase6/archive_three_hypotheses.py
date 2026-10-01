#!/usr/bin/env python3
"""Stream a portable evidence pack; exclude data maps, builds and executables."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/'docs/research/evidence/phase6/three-hypotheses-v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):
            h.update(part)
    return h.hexdigest()


def main():
    paths = {p for p in (ROOT/'tools/phase6').glob('*representation*.py')}
    paths.update(ROOT/'tools/phase6'/p for p in (
        'deadline_screen.py','deadline_pause.py','optimality_screen.py',
        'three_hypothesis_audit.py','archive_three_hypotheses.py'))
    paths.add(ROOT/'docs/research/PHASE_6_THREE_HYPOTHESIS_CAMPAIGN_2026_10_01.md')
    before = json.loads((ROOT/'work/phase6/three-hypothesis-campaign-v1/baseline.json').read_text())
    paths.update(ROOT/p for p in before['files'])
    branches = ('representation-screen-v1','representation-quality-v1',
                'representation-native-v1','deadline-screen-v1',
                'deadline-pause-v1','optimality-screen-v1','three-hypothesis-campaign-v1')
    for name in branches:
        for p in (ROOT/'work/phase6'/name).rglob('*'):
            if not p.is_file() or any(s in ('mapped','native-build','__pycache__') for s in p.parts):
                continue
            if p.suffix in ('.json','.md','.py','.cpp','.sv','.uq2','.log') and not p.name.endswith('.partial.json'):
                paths.add(p)
    # Include identity manifests without generated/native binary objects.
    for name in ('deadline-screen-v1','deadline-pause-v1'):
        paths.add(ROOT/f'work/phase6/{name}/native-build/identity.json')
    files = {str(p.relative_to(ROOT)):p for p in sorted(paths)}
    selected = Path('/Users/sayat/.codex/worktrees/21f0/tinyML_accelerator/work/phase6/engine-candidate-rtl-v2/engine.sv')
    files['selected-engine/engine.sv'] = selected
    DEST.mkdir(parents=True,exist_ok=True)
    manifest = {'schema':1,'scope':'portable sources, frozen model programs and raw evidence; data/build binaries excluded',
                'files':{k:{'bytes':p.stat().st_size,'sha256':sha(p)} for k,p in files.items()}}
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    archive = DEST/'evidence.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=6) as t:
        for key,p in files.items():
            t.add(p,arcname=key,recursive=False)
    checked = 0
    with tarfile.open(archive,'r:gz') as t:
        for item in t:
            if not item.isfile() or item.name not in manifest['files']:
                raise ValueError('unexpected archive entry')
            h = hashlib.sha256()
            with t.extractfile(item) as stream:
                for part in iter(lambda:stream.read(1<<20),b''):
                    h.update(part)
            if h.hexdigest()!=manifest['files'][item.name]['sha256']:
                raise ValueError('archive hash mismatch')
            checked += 1
    if checked!=len(files):
        raise ValueError('archive incomplete')
    proof = {'status':'passed-streaming-archive-verification','files_checked':checked,
             'manifest_sha256':sha(DEST/'manifest.json'),'archive_sha256':sha(archive),
             'archive_bytes':archive.stat().st_size}
    (DEST/'verification.json').write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
    print(json.dumps(proof,indent=2))


if __name__=='__main__':
    main()
