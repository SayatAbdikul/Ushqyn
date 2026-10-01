#!/usr/bin/env python3
"""Stream and verify the completed decision round's source/raw evidence archive."""
import hashlib
import json
from pathlib import Path
import resource
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'docs/research/evidence/phase6/decision-round-v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    audit = json.loads((ROOT / 'work/phase6/decision-round-v1/audit.json').read_text())
    if audit['status'] != 'passed-evidence-audit':
        raise ValueError('evidence must pass audit before archive')
    paths = set()
    for name in ('representation_optimized.py', 'deadline_policy.py', 'deadline_resident.py',
                 'deadline_screen.py', 'deadline_pause.py', 'decision_round_audit.py',
                 'finalize_decision_round.py', 'archive_decision_round.py'):
        paths.add(ROOT / 'tools/phase6' / name)
    for name in ('PHASE_6_IMPLEMENTATION_DECISION_ROUND_2026_10_01.md',
                 'PHASE_6_DECISION_ROUND_PRIOR_ART_2026_10_01.md'):
        paths.add(ROOT / 'docs/research' / name)
    baseline = json.loads((ROOT / 'work/phase6/decision-round-v1/baseline.json').read_text())
    for group in ('production_files', 'frozen_model_files'):
        paths.update(ROOT / p for p in baseline[group])
    for branch in ('representation-optimized-v1', 'deadline-policy-v1',
                   'deadline-resident-v1', 'decision-round-v1', 'deadline-screen-v1'):
        for p in (ROOT / 'work/phase6' / branch).rglob('*'):
            if not p.is_file() or any(part in ('__pycache__', 'native-build', 'mapped') for part in p.parts):
                continue
            if 'physical' in p.parts and p.suffix not in ('.json', '.log', '.tcl', '.txt', '.html'):
                continue
            if p.suffix in ('.json', '.md', '.py', '.cpp', '.sv', '.v', '.uq2', '.log', '.bin', '.txt', '.cst', '.sdc', '.tcl', '.html'):
                paths.add(p)
    for branch in ('deadline-resident-v1', 'deadline-screen-v1'):
        paths.add(ROOT / f'work/phase6/{branch}/native-build/identity.json')
        buildlog = ROOT / f'work/phase6/{branch}/native-build/build.log'
        if buildlog.exists():
            paths.add(buildlog)
    paths.add(ROOT / 'test/phase6/native.cpp')
    files = {str(p.relative_to(ROOT)): p for p in sorted(paths)}
    fixtures = json.loads((ROOT / 'work/phase6/deadline-policy-v1/fixtures.json').read_text())
    reference = Path('/Users/sayat/.codex/worktrees/21f0/tinyML_accelerator')
    for variants in fixtures.values():
        for folder in variants.values():
            folder = Path(folder)
            if folder.is_relative_to(ROOT):
                continue
            for p in folder.iterdir():
                if p.is_file() and p.suffix not in ('.onnx', '.npz'):
                    files['external-reference/' + str(p.relative_to(reference))] = p
    DEST.mkdir(parents=True, exist_ok=True)
    files['evidence-pack-README.md'] = DEST / 'README.md'
    manifest = dict(schema=1, scope='Pinned sources, fixtures and raw evidence; datasets/toolchains/build binaries excluded',
        files={k: dict(bytes=p.stat().st_size, sha256=sha(p)) for k, p in sorted(files.items())})
    (DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    archive = DEST / 'evidence.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=6) as packed:
        for key, path in sorted(files.items()):
            packed.add(path, arcname=key, recursive=False)
    checked = set()
    with tarfile.open(archive, 'r:gz') as packed:
        for member in packed:
            if not member.isfile() or member.name not in manifest['files'] or member.name in checked:
                raise ValueError('unexpected archive member')
            h = hashlib.sha256()
            with packed.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1 << 20), b''):
                    h.update(block)
            expected = manifest['files'][member.name]
            if h.hexdigest() != expected['sha256'] or member.size != expected['bytes']:
                raise ValueError('archive member differs')
            checked.add(member.name)
    if checked != set(files):
        raise ValueError('archive coverage incomplete')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    if rss >= 1 << 30:
        raise ValueError('archive memory guard exceeded')
    proof = dict(status='passed-streaming-archive-verification', files_checked=len(checked),
        manifest_sha256=sha(DEST / 'manifest.json'), archive_sha256=sha(archive),
        archive_bytes=archive.stat().st_size, peak_python_rss_bytes=rss)
    (DEST / 'verification.json').write_text(json.dumps(proof, indent=2, sort_keys=True) + '\n')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
