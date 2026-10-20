"""Resume the existing alpha suite with the user's accepted M1 slowdown."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path('/workspace/delta-ngram-alpha')
OPS = Path('/workspace/delta-ngram-transfer-20261019/multidomain')
M1 = Path('/workspace/delta-ngram-runs/m1_event_mixture_seed17_20261019')
OUT = Path('/workspace/delta-ngram-runs/alpha_m1_prefix_parallel_seed17_20261019')
CONFIG = ROOT / 'configs/m1_event_mixture_seed17.yaml'
STATUS = OPS / 'alpha_parallel_status.json'
ENTRY = OPS / 'alpha_cuda_entry.py'
VARIANTS = ('fixed_099', 'fixed_095', 'learned_global_099', 'evidence_block_099')
stage, process, shared = 'validating_user_requested_parallel_resume', None, {}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save(**extra):
    write_json(STATUS, {'stage': stage, 'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'pid': None if process is None else process.pid, **shared, **extra})


def main_status():
    result = json.loads((OPS / 'm1_pipeline_status.json').read_text())
    if result.get('status') == 'failed':
        raise RuntimeError('M1 failed; stop alpha for inspection')
    return result


def progress(destination):
    metrics = destination / 'metrics.jsonl'
    if not metrics.exists():
        return None
    data = metrics.read_bytes()
    complete = data[:data.rfind(b'\n') + 1]
    for line in reversed(complete.splitlines()):
        row = json.loads(line)
        if row.get('event') == 'train':
            return {'step': row['step'], 'loss_tokens': row['counters']['loss_tokens'], 'nll': row['nll']}
    return None


def run(variant, arguments, destination):
    global stage, process
    stage = 'alpha_parallel_' + variant
    log = OPS / (stage + '.log')
    command = [sys.executable, '-u', str(ENTRY), *arguments]
    with log.open('x') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        started = time.monotonic()
        while process.poll() is None:
            main = main_status()
            save(log=str(log), command=command, variant=variant, elapsed_seconds=time.monotonic() - started,
                 m1_step=main.get('training', {}).get('step'), alpha_progress=progress(destination))
            if time.monotonic() - started > 4 * 3600:
                process.terminate()
                raise TimeoutError(stage + ' exceeded four hours')
            time.sleep(10)
    if process.returncode:
        raise RuntimeError(f'{stage} exited {process.returncode}; see {log}')


try:
    previous = json.loads(STATUS.read_text())
    assert previous['stage'] == 'parallel_probe_did_not_pass_waiting_for_m1_endpoint'
    assert previous.get('pid') is None
    assert not subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip()
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == previous['git_commit']
    assert CONFIG.read_bytes() == Path('/workspace/delta-ngram-m1/configs/m1_event_mixture_seed17.yaml').read_bytes()
    assert sha(ENTRY) == previous['launcher_sha256']
    prefix = OUT / 'm1_first_5m_prefix.json'
    prefix_data = json.loads(prefix.read_text())
    probe = OUT / 'fixed_099_probe20'
    probe_summary = json.loads((probe / 'summary.json').read_text())
    assert probe_summary['completed_steps'] == 20 and not probe_summary['complete']
    assert probe_summary['prefix_sha256'] == prefix_data['manifest_sha256']
    assert all(not (OUT / variant).exists() for variant in VARIANTS)
    main_status()
    override = {'registered_at_utc': datetime.now(timezone.utc).isoformat(),
        'authority': 'User explicitly requested parallel execution after reviewing the measured 29% M1 throughput reduction and 38% estimated combined throughput increase.',
        'decision': 'Continue M1 plus one alpha variant concurrently; alpha variants remain sequential.',
        'supersedes': 'Only the M1 >=80% throughput scheduling gate; scientific protocol and budgets unchanged.',
        'resume_from': str(probe / 'training.pt'), 'resume_step': 20,
        'resume_loss_tokens': probe_summary['counters']['loss_tokens'],
        'prefix_sha256': prefix_data['manifest_sha256'], 'orchestrator_sha256': sha(__file__),
        'alpha_cuda_allocator_fraction': 0.45}
    override_path = OUT / 'parallel_user_override_20261019.json'
    with override_path.open('x') as stream:
        json.dump(override, stream, indent=2)
        stream.write('\n')
    shared = {key: previous[key] for key in ('git_commit', 'output_dir', 'parallelism',
        'alpha_cuda_allocator_fraction', 'resource_scope', 'launcher_sha256', 'm1_milestone_step',
        'm1_prefix_loss_tokens', 'm1_checkpoint', 'frozen_metrics_sha256', 'parallel_probe_audit')}
    shared.update(orchestrator_sha256=sha(__file__), scheduling_rule=override['decision'],
                  user_override_active=True, user_override_record=str(override_path))
    common = ['--config', str(CONFIG), '--writer-init', str(M1 / 'writer_init.pt'), '--prefix-manifest', str(prefix)]
    results = {}
    for variant in VARIANTS:
        destination = OUT / variant
        arguments = ['train', *common, '--variant', variant, '--output', str(destination)]
        if variant == 'fixed_099':
            arguments += ['--resume', str(probe / 'training.pt')]
        run(variant, arguments, destination)
        summary = json.loads((destination / 'summary.json').read_text())
        assert summary['complete'] and summary['counters']['loss_tokens'] == prefix_data['actual_loss_tokens']
        results[variant] = summary
    stage, process = 'alpha_training_complete_pending_registered_evaluations', None
    save(variants=list(results), loss_tokens_per_variant=prefix_data['actual_loss_tokens'],
         total_additional_loss_tokens=sum(row['counters']['loss_tokens'] for row in results.values()),
         scope='Training only. Frozen self/empty/donor, retention, revision and pruning evaluations remain required.')
except BaseException as error:
    if process is not None and process.poll() is None:
        process.terminate()
    save(status='failed', error_type=type(error).__name__, error=str(error), failed_stage=stage)
    traceback.print_exc()
    raise
