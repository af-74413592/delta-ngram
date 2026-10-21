#!/usr/bin/env python3
"""Seal an immutable, credential-free snapshot of the completed native runs."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time

ROOT = Path('/workspace/delta-ngram-native-20261020')
STAGE = Path('/workspace/delta-ngram-backup-stage-20261021-native')
DIRECTORIES = ('reports', 'data', 'ops', 'delta-ngram', 'sources', 'logs')
IGNORED_DIRS = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.cache', 'node_modules'}
TEXT_SUFFIXES = {'.py', '.sh', '.json', '.jsonl', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.md', '.txt', '.log'}
PRIORITY = [
    'reports/native_main_training_08/joint-policy-stage-2-alpha100_rmsnorm-model_milestone-100000000.pt',
    'reports/native_main_training_08/joint-policy-stage-3-final.pt',
    'reports/native_main_training_08/joint-latest.pt',
    'reports/native_main_training_07/joint-stage-3-alpha099_rmsnorm-nominal-40000000.pt',
    'reports/native_main_training_07/joint-stage-4-final.pt',
    'reports/native_main_training_07/joint-latest.pt',
    'data/native_dev64/plan.json',
]
SECRET_PATTERNS = [
    re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----'),
    re.compile(rb'\b(?:ghp_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,}|hf_[A-Za-z0-9]{25,})\b'),
    re.compile(rb'''["']?(?:applicationKey|application_key|authorizationToken|aws_secret_access_key|secret_access_key|api_key|access_token)["']?\s*[:=]\s*["']([A-Za-z0-9_+/=-]{24,})["']''', re.I),
]


def sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def signature(path: Path) -> tuple[int, int, int, int]:
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def excluded(path: Path) -> str | None:
    if any(part in IGNORED_DIRS for part in path.parts):
        return 'generated_cache_or_git_metadata'
    low = path.name.lower()
    if low.endswith(('.pyc', '.pyo', '.b64')):
        return 'compiled_cache_or_duplicate_transfer'
    if low in {'.env', '.netrc', '.npmrc', '.pypirc', 'credentials', 'rclone.conf', 'id_rsa', 'id_ed25519', '.b2_account_info'}:
        return 'credential_filename'
    if low.endswith(('.pem', '.key')) or (low.startswith('.') and ('auth' in low or 'credential' in low)):
        return 'credential_filename'
    return None


def check_obvious_secret(path: Path, relative: str) -> bool:
    if relative.startswith('data/') or (path.suffix.lower() not in TEXT_SUFFIXES and path.suffix):
        return False
    # This is a narrow credential scan, not a filter on experiment "token" fields.
    overlap = b''
    with path.open('rb') as handle:
        while block := handle.read(1024 * 1024):
            chunk = overlap + block
            for pattern in SECRET_PATTERNS:
                for match in pattern.finditer(chunk):
                    value = match.group(1) if match.lastindex else match.group(0)
                    lowered = value.lower()
                    if any(word in lowered for word in (b'example', b'placeholder', b'your_', b'xxxxxxxx', b'yyyyyyyy')):
                        continue
                    return True
            overlap = chunk[-512:]
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--stage', type=Path, default=STAGE)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    stage = args.stage.resolve()
    if stage.exists() or stage.parent != Path('/workspace') or stage == root:
        raise ValueError('Stage must be a new direct child of /workspace')
    stage.mkdir()
    metadata = stage / 'metadata'
    metadata.mkdir()
    started = time.monotonic()
    exclusions = []
    candidates = []
    symlinks = []
    for name in DIRECTORIES:
        folder = root / name
        if not folder.is_dir():
            raise ValueError(f'Required directory missing: {name}')
        for directory, folders, files in os.walk(folder, followlinks=False):
            base = Path(directory)
            for sub in list(folders):
                path = base / sub
                reason = excluded(path)
                if path.is_symlink():
                    folders.remove(sub)
                    symlinks.append(path.relative_to(root).as_posix())
                elif reason:
                    folders.remove(sub)
                    exclusions.append({'relative_path': path.relative_to(root).as_posix() + '/', 'reason': reason})
            for name in files:
                path = base / name
                relative = path.relative_to(root).as_posix()
                reason = excluded(path)
                if path.is_symlink():
                    symlinks.append(relative)
                elif reason:
                    exclusions.append({'relative_path': relative, 'reason': reason})
                elif stat.S_ISREG(path.stat().st_mode):
                    candidates.append((relative, path))
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix in {'.py', '.sh'}:
            candidates.append((path.name, path))
        elif path.is_file():
            exclusions.append({'relative_path': path.name, 'reason': 'duplicate_transfer_archive_or_unrequested_root_file'})
    exclusions.append({'relative_path': 'b2_core/', 'reason': 'previously_sealed_1733_file_backup_already_in_B2'})
    if symlinks:
        (metadata / 'PAUSED.json').write_text(json.dumps({'reason': 'symlink_review_required', 'paths': symlinks}, indent=2) + '\n')
        raise RuntimeError('Symlinks require explicit inventory review; only paths recorded in metadata/PAUSED.json')
    names = {name for name, _ in candidates}
    if len(names) != len(candidates) or any(name not in names for name in PRIORITY):
        raise ValueError('Missing priority file or duplicate destination')
    for evaluation in ('reports/native_main_training_07/evaluations/', 'reports/native_main_training_08/evaluations/'):
        if not any(name.startswith(evaluation) and name.endswith('/report.json') for name in names):
            raise ValueError(f'Missing full evaluation reports: {evaluation}')
    priority_index = {name: index for index, name in enumerate(PRIORITY)}
    candidates.sort(key=lambda pair: (priority_index.get(pair[0], len(PRIORITY)), pair[0]))
    print(json.dumps({'event': 'inventory', 'files': len(candidates), 'bytes': sum(p.stat().st_size for _, p in candidates), 'exclusions': len(exclusions)}), flush=True)
    flagged = [name for name, path in candidates if check_obvious_secret(path, name)]
    if flagged:
        (metadata / 'PAUSED.json').write_text(json.dumps({'reason': 'possible_credentials_in_text', 'paths': flagged}, indent=2) + '\n')
        print(json.dumps({'event': 'paused', 'reason': 'possible_credentials_in_text', 'paths': flagged}), flush=True)
        raise RuntimeError('Possible credentials: only paths recorded, values never printed')
    records = []
    methods = Counter()
    manifest = metadata / 'member_manifest.jsonl'
    with manifest.open('w', encoding='utf-8', newline='\n') as out:
        for index, (relative, source) in enumerate(candidates, 1):
            before = signature(source)
            digest = sha256(source)
            if signature(source) != before:
                raise RuntimeError(f'Source changed during hash: {relative}')
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
                method = 'hardlink'
            except OSError:
                shutil.copy2(source, target)
                method = 'copy2'
                if sha256(target) != digest:
                    raise RuntimeError(f'Copied content differs: {relative}')
            if signature(source) != before or target.stat().st_size != before[2]:
                raise RuntimeError(f'Source changed during staging: {relative}')
            record = {'relative_path': relative, 'size_bytes': before[2], 'sha256': digest, 'source_path': str(source), 'source_mtime_ns': before[3], 'method': method}
            out.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
            records.append(record)
            methods[method] += 1
            if index % 250 == 0 or index == len(candidates):
                out.flush()
                print(json.dumps({'event': 'staged', 'files': index, 'total_files': len(candidates), 'bytes': sum(x['size_bytes'] for x in records), 'elapsed_seconds': time.monotonic() - started}), flush=True)
    # Confirm every source still matches its stat identity after the complete pass.
    for record in records:
        source = Path(record['source_path'])
        current = source.stat()
        if current.st_size != record['size_bytes'] or current.st_mtime_ns != record['source_mtime_ns']:
            raise RuntimeError(f'Source changed after staging: {record["relative_path"]}')
    checksum_file = metadata / 'member_sha256.txt'
    checksum_file.write_text(''.join(f'{x["sha256"]}  {x["relative_path"]}\n' for x in records), encoding='utf-8')
    groups = {}
    for record in records:
        group = record['relative_path'].split('/')[0]
        groups.setdefault(group, {'files': 0, 'size_bytes': 0})
        groups[group]['files'] += 1
        groups[group]['size_bytes'] += record['size_bytes']
    staged = {
        'protocol': 'qwen38_native_immutable_backup_stage_v1',
        'status': 'staged_sha256_verified',
        'source_root': str(root), 'stage_root': str(stage),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'payload_files': len(records), 'payload_bytes': sum(x['size_bytes'] for x in records),
        'groups': groups, 'methods': dict(methods),
        'source_stats_verified_before_after_hash_and_at_completion': True,
        'obvious_credential_scan': 'passed_without_printing_values',
        'priority_files': [x for x in records if x['relative_path'] in priority_index],
        'exclusions': exclusions,
        'seals': {p.relative_to(stage).as_posix(): {'size_bytes': p.stat().st_size, 'sha256': sha256(p)} for p in (manifest, checksum_file)},
        'metadata_self_excluded_from_member_manifest': True,
        'immutable_source_contract': 'All selected native training, diagnostics and evaluation runs have ended; do not edit hardlinked source or stage payload files.',
        'upload_performed': False, 'elapsed_seconds': time.monotonic() - started,
    }
    (metadata / 'STAGED.json').write_text(json.dumps(staged, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'complete', 'payload_files': len(records), 'payload_bytes': staged['payload_bytes'], 'stage_root': str(stage), 'seals': staged['seals'], 'staged_json_sha256': sha256(metadata / 'STAGED.json')}), flush=True)


if __name__ == '__main__':
    main()
