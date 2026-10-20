"""Start a bounded alpha/M1 overlap after the actual first-5M prefix exists."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
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
stage, process, shared = 'waiting_for_m1_first_5m', None, {}


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
        raise RuntimeError('M1 failed; stop the overlapping alpha process for inspection')
    return result


def rows(path):
    if not path.exists():
        return []
    # Read a byte snapshot and ignore only a possibly incomplete trailing line.
    data = path.read_bytes()
    complete = data[:data.rfind(b'\n') + 1]
    return [json.loads(line) for line in complete.splitlines() if line]


def training_rows():
    return [row for row in rows(M1 / 'metrics.jsonl') if row.get('event') == 'train']


def throughput(records):
    elapsed = sum(row['step_seconds'] for row in records)
    return {'steps': len(records), 'tokens': sum(row['batch_loss_tokens'] for row in records),
        'summed_step_seconds': elapsed,
        'tokens_per_second': sum(row['batch_loss_tokens'] for row in records) / elapsed if elapsed else None,
        'median_step_seconds': statistics.median(row['step_seconds'] for row in records) if records else None}


def wait_until(predicate, label, limit=28 * 3600):
    global stage, process
    stage, process = label, None
    started = time.monotonic()
    while True:
        status = main_status()
        if predicate(status):
            return status
        if time.monotonic() - started > limit:
            raise TimeoutError(label)
        save(m1_stage=status['stage'], waiting_seconds=time.monotonic() - started)
        time.sleep(15)


def run(label, arguments, *, probe=None):
    global stage, process
    stage = label
    log = OPS / (label + '.log')
    command = [sys.executable, '-u', str(ENTRY), *arguments]
    start_m1_step = None
    with log.open('x') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        started = time.monotonic()
        while process.poll() is None:
            status = main_status()
            alpha_progress = None
            if probe is not None:
                progress = [row for row in rows(probe / 'metrics.jsonl') if row.get('event') == 'train']
                if progress:
                    alpha_progress = {'step': progress[-1]['step'], 'loss_tokens': progress[-1]['counters']['loss_tokens']}
                    if start_m1_step is None:
                        start_m1_step = status.get('training', {}).get('step', 0)
            save(log=str(log), command=command, elapsed_seconds=time.monotonic() - started,
                 m1_step=status.get('training', {}).get('step'), alpha_progress=alpha_progress)
            if time.monotonic() - started > 4 * 3600:
                process.terminate()
                raise TimeoutError(label + ' exceeded four hours')
            time.sleep(10)
    if process.returncode:
        raise RuntimeError(f'{label} exited {process.returncode}; see {log}')
    end_m1_step = main_status().get('training', {}).get('step', 0)
    return {'start_m1_step': start_m1_step, 'end_m1_step': end_m1_step}


try:
    assert not STATUS.exists() and not OUT.exists(), 'Use a fresh parallel suite output'
    assert not subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip()
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    assert CONFIG.read_bytes() == Path('/workspace/delta-ngram-m1/configs/m1_event_mixture_seed17.yaml').read_bytes()
    old = json.loads((OPS / 'alpha_pipeline_status.json').read_text())
    assert old.get('stage') == 'superseded_by_parallel_first_5m_queue'
    shared = {'git_commit': commit, 'output_dir': str(OUT), 'parallelism': 'M1 plus at most one alpha variant',
        'alpha_cuda_allocator_fraction': 0.45, 'resource_scope': 'PyTorch allocator cap, not a hardware partition',
        'launcher_sha256': sha(ENTRY), 'orchestrator_sha256': sha(__file__),
        'scheduling_rule': '20-step fixed-0.99 probe; continue overlap if M1 throughput >=80% of prior rate and summed throughput >=110%; otherwise resume after M1 endpoint'}
    wait_until(lambda s: s.get('checkpoint', {}).get('training_loss_tokens', 0) >= 5_000_000,
               'waiting_for_m1_first_5m')
    candidates = [(int(path.stem.rsplit('_tokens_', 1)[1]), path)
                  for path in M1.glob('training_step_*_tokens_*.pt')]
    checkpoint = min((pair for pair in candidates if pair[0] >= 5_000_000), key=lambda pair: pair[0])[1]
    nominal_actual = int(checkpoint.stem.rsplit('_tokens_', 1)[1])
    observed = training_rows()
    prefix_rows = [row for row in observed if row['counters']['loss_tokens'] <= nominal_actual]
    assert prefix_rows[-1]['counters']['loss_tokens'] == nominal_actual
    milestone_step = prefix_rows[-1]['step']
    assert [row['step'] for row in prefix_rows] == list(range(1, milestone_step + 1))
    baseline = throughput(prefix_rows[-30:])
    OUT.mkdir()
    metrics = OUT / 'm1_metrics_through_first_5m.jsonl'
    metrics.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in prefix_rows))
    prefix = OUT / 'm1_first_5m_prefix.json'
    shared.update(m1_milestone_step=milestone_step, m1_prefix_loss_tokens=nominal_actual,
                  m1_checkpoint=str(checkpoint), frozen_metrics_sha256=sha(metrics))
    write_json(OUT / 'scheduling_registration.json', {**shared, 'm1_baseline': baseline,
        'scientific_protocol': 'Same unchanged full-budget sampler, writer initialization, 5M optimizer prefix, variants, and evaluation rules; only resource scheduling changes.'})
    run('alpha_parallel_build_prefix', ['build-prefix', '--config', str(CONFIG), '--m1-checkpoint', str(checkpoint),
        '--m1-metrics', str(metrics), '--writer-init', str(M1 / 'writer_init.pt'), '--output', str(prefix)])
    prefix_data = json.loads(prefix.read_text())
    wait_until(lambda s: s.get('training', {}).get('step', 0) > milestone_step or s.get('endpoint_audit_passed') is True,
               'waiting_for_m1_training_after_5m_evaluation')
    probe = OUT / 'fixed_099_probe20'
    common = ['--config', str(CONFIG), '--writer-init', str(M1 / 'writer_init.pt'), '--prefix-manifest', str(prefix)]
    overlap = run('alpha_parallel_probe20', ['train', *common, '--variant', 'fixed_099',
                  '--output', str(probe), '--stop-after-steps', '20'], probe=probe)
    probe_summary = json.loads((probe / 'summary.json').read_text())
    assert probe_summary['completed_steps'] == 20 and not probe_summary['complete']
    start_step = overlap['start_m1_step']
    observed_overlap = [row for row in training_rows()
                        if start_step is not None and start_step < row['step'] <= overlap['end_m1_step']]
    measured = throughput(observed_overlap)
    alpha_rate = probe_summary['counters']['loss_tokens'] / probe_summary['elapsed_seconds']
    main_ratio = measured['tokens_per_second'] / baseline['tokens_per_second'] if measured['tokens_per_second'] else 0
    aggregate_ratio = (measured['tokens_per_second'] + alpha_rate) / baseline['tokens_per_second'] if measured['tokens_per_second'] else 0
    continue_parallel = len(observed_overlap) >= 8 and main_ratio >= 0.80 and aggregate_ratio >= 1.10
    audit = {'m1_before': baseline, 'm1_during_probe': measured, 'alpha_probe_tokens_per_second': alpha_rate,
        'm1_throughput_ratio': main_ratio, 'estimated_combined_throughput_ratio': aggregate_ratio,
        'continue_parallel': continue_parallel, 'overlap_steps': overlap,
        'limitations': 'Short mixed-batch scheduling measurement; not a matched-batch speed benchmark. Does not select models by task performance.',
        'probe_checkpoint_reused_by_exact_resume': True}
    write_json(OUT / 'parallel_probe_audit.json', audit)
    shared['parallel_probe_audit'] = audit
    if not continue_parallel:
        wait_until(lambda s: s.get('endpoint_audit_passed') is True,
                   'parallel_probe_did_not_pass_waiting_for_m1_endpoint')
    results = {}
    for variant in VARIANTS:
        destination = OUT / variant
        arguments = ['train', *common, '--variant', variant, '--output', str(destination)]
        if variant == 'fixed_099':
            arguments += ['--resume', str(probe / 'training.pt')]
        run('alpha_parallel_' + variant, arguments)
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
