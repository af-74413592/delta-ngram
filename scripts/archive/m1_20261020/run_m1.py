"""Supervised M1 preparation, immutable baseline, training and endpoint audit."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path('/workspace/delta-ngram-m1')
OPS = Path('/workspace/delta-ngram-transfer-20261019/multidomain')
DATA = Path('/workspace/delta-ngram-data/multidomain_20261019')
OUTPUT = Path('/workspace/delta-ngram-runs/m1_event_mixture_seed17_20261019')
CONFIG = ROOT / 'configs/m1_event_mixture_seed17.yaml'
MERGED = DATA / 'merged_event_v5'
STATUS = OPS / 'm1_pipeline_status.json'
stage = 'starting'
process = None


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def progress():
    metrics = OUTPUT / 'metrics.jsonl'
    latest = {}
    if metrics.exists():
        with metrics.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row['event'] == 'train':
                    latest['training'] = {key: row[key] for key in ('step', 'counters', 'nll', 'step_seconds', 'source_loss_tokens')}
                elif row['event'] == 'checkpoint':
                    latest['checkpoint'] = row
                elif row['event'] in ('evaluation', 'final_evaluation'):
                    latest['evaluation'] = {key: row.get(key) for key in ('step', 'training_loss_tokens', 'source_macro_delta_empty')}
    return latest


def save(**extra):
    write_json(STATUS, {'stage': stage, 'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                        'pid': None if process is None else process.pid, **extra})


def snapshot_first_checkpoint(latest):
    checkpoint = latest.get('checkpoint', {})
    if checkpoint.get('training_loss_tokens', 0) < 1_000_000:
        return
    destination = OPS / 'first_m1_checkpoint'
    if (destination / 'manifest.json').exists():
        return
    destination.mkdir(exist_ok=True)
    # Opening the atomically replaced file pins its inode while the trainer continues.
    with (OUTPUT / 'training.pt').open('rb') as source, (destination / 'training.pt').open('xb') as target:
        shutil.copyfileobj(source, target)
    for name in ('config.json', 'data_manifest.json', 'writer_init.pt'):
        shutil.copy2(OUTPUT / name, destination / name)
    import torch
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(destination / 'training.pt', map_location='cpu', weights_only=True)
    write_json(destination / 'manifest.json', {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'step': payload['step'],
        'counters': payload['counters'], 'source_loss_tokens': payload['sampler_state']['source_loss_tokens'],
        'files': {path.name: {'bytes': path.stat().st_size, 'sha256': sha(path)}
                  for path in destination.iterdir() if path.is_file()},
        'scope': 'Frozen server-side copy; off-instance backup requires transfer and hash verification.'})


def run(name, command, *, maximum_seconds, training=False, allowed=(0,)):
    global stage, process
    stage = name
    log = OPS / (name + '.log')
    print(json.dumps({'event': 'stage_start', 'stage': stage, 'command': command}), flush=True)
    with log.open('x', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        started = time.monotonic()
        while process.poll() is None:
            if time.monotonic() - started > maximum_seconds:
                process.terminate()
                raise TimeoutError(name + ' exceeded wall limit')
            latest = progress() if training else {}
            if training:
                snapshot_first_checkpoint(latest)
            save(log=str(log), elapsed_seconds=time.monotonic() - started, **latest)
            time.sleep(15)
        if process.returncode not in allowed:
            raise RuntimeError(f'{name} exited {process.returncode}; see {log}')
    print(json.dumps({'event': 'stage_exit', 'stage': stage, 'returncode': process.returncode}), flush=True)


try:
    assert not STATUS.exists(), 'Do not overwrite an existing pipeline status'
    assert not subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip()
    stage = 'waiting_for_frozen_joint_manifest'
    assert CONFIG.is_file()
    waiting = time.monotonic()
    while not (MERGED / 'data_manifest.json').exists():
        status_file = OPS / 'event_merge_v5_status.json'
        if status_file.exists() and json.loads(status_file.read_text()).get('status') == 'failed':
            raise RuntimeError('Final joint corpus audit failed; M1 must not start')
        if time.monotonic() - waiting > 3600:
            raise TimeoutError('Final joint corpus audit did not finish within one hour')
        save(elapsed_seconds=time.monotonic() - waiting)
        time.sleep(15)
    stage = 'checking_frozen_joint_manifest'
    assert json.loads((MERGED / 'data_manifest.json').read_text())['ready']
    run('m1_preflight', [sys.executable, '-u', str(OPS / 'preflight_m1.py')], maximum_seconds=3600)
    expected = json.loads((OPS / 'm1_preflight.json').read_text())
    assert expected['passed']
    run('m1_p1_baseline', [sys.executable, '-u', str(ROOT / 'scripts/evaluate_raw_checkpoint.py'),
        '--config', str(CONFIG), '--checkpoint', '/workspace/delta-ngram-runs/p1_code_seed17_20261019/writer.pt',
        '--output', str(OPS / 'p1_writer_on_m1_dev.json'), '--threads', '8'], maximum_seconds=7200)
    baseline = json.loads((OPS / 'p1_writer_on_m1_dev.json').read_text())
    assert baseline['validation_sample_sha256'] == expected['validation']['validation_sample_sha256']
    assert baseline['validation_protocol_sha256'] == expected['validation']['validation_protocol_sha256']
    run('m1_train', [sys.executable, '-u', str(ROOT / 'scripts/train_raw_deterministic.py'),
                    '--config', str(CONFIG), '--threads', '8'], maximum_seconds=90000, training=True, allowed=(0, 2))
    stage = 'auditing_training_endpoint'
    summary = json.loads((OUTPUT / 'summary.json').read_text())
    sampling = expected['sampling']
    assert summary['budget_endpoint'] and summary['counters']['loss_tokens'] == sampling['loss_tokens']
    assert summary['completed_steps'] == sampling['steps']
    assert summary['source_loss_tokens'] == sampling['source_loss_tokens']
    order_hash = hashlib.sha256()
    with (OUTPUT / 'metrics.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['event'] == 'train':
                order_hash.update((json.dumps({key: row[key] for key in ('step', 'batch_loss_tokens', 'windows')},
                                  sort_keys=True, separators=(',', ':')) + '\n').encode())
    assert order_hash.hexdigest() == sampling['order_sha256']
    assert sha(CONFIG) == expected['config_sha256']
    assert subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip() == expected['git_commit']
    stage = 'training_budget_reached_pending_registered_evaluations'
    save(endpoint_audit_passed=True, loss_tokens=summary['counters']['loss_tokens'],
         unspent_loss_tokens=summary['unspent_loss_tokens'], order_sha256=order_hash.hexdigest(),
         source_loss_tokens=summary['source_loss_tokens'],
         scope='Training endpoint only; online recent-news continuation, sealed final and M2 remain separate gates.')
except BaseException as error:
    if process is not None and process.poll() is None:
        process.terminate()
    save(failed_stage=stage, error_type=type(error).__name__, error=str(error), status='failed')
    traceback.print_exc()
    raise
