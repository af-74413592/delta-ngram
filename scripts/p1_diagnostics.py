"""Prepare fixed P1 diagnostic pairs, then evaluate one frozen 5M writer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def prepare(args):
    from delta_ngram.data.raw_text import load_documents
    from delta_ngram.diagnostic_manifest import build_diagnostic_manifest

    baseline = read_json(Path(args.baseline_run) / 'data_manifest.json')
    dataset = Path(args.dataset).resolve()
    dataset_hash = sha256(dataset)
    if dataset_hash != baseline['dataset_sha256']:
        raise ValueError('Diagnostic corpus differs from the registered development corpus')
    documents = load_documents(dataset, tokenizer_identity=baseline['tokenizer_fingerprint'],
                               require_record_identity=True)
    manifest = build_diagnostic_manifest(
        documents, baseline['validation_sample'], block_size=512, prefix_tokens=128,
        seed=104729, donors_per_target=2, donor_pool_policy='validation_first',
        dataset_sha256=dataset_hash,
        baseline_validation_protocol_sha256=baseline['validation_protocol_sha256'])
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f'Keep the fixed manifest immutable: {output}')
    save_json(output, manifest)
    coverage = manifest['coverage']
    compact = {name: {key: value for key, value in counts.items() if not key.startswith('by_')}
               for name, counts in {'primary': coverage['primary'], **coverage['after']}.items()}
    print(json.dumps({'manifest': str(output), 'sha256': sha256(output),
                      'coverage': compact}, allow_nan=False))


def evaluate(args):
    # Establish deterministic settings before importing torch or creating CUDA state.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    import torch
    from delta_ngram.config import ExperimentConfig
    from delta_ngram.data.raw_text import load_documents
    from delta_ngram.p1_diagnostics import evaluate_diagnostics, load_diagnostic_checkpoint
    from delta_ngram.raw_lm import tokenizer_identity, _json_hash
    from delta_ngram.train import create_runtime
    from summarize_p1 import audit_run

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    repo = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Commit source changes before a recorded diagnostic run')
    config = ExperimentConfig.load(args.config)
    training_run = Path(args.training_run).resolve()
    original_config = read_json(training_run / 'config.json')
    original_summary = read_json(training_run / 'summary.json')
    endpoint_audit = audit_run(training_run, replication=True)
    if (endpoint_audit['identity_check']['status'] != 'passed' or
            endpoint_audit.get('endpoint', {}).get('status') not in ('passed', 'not_met')):
        raise ValueError(f'Final checkpoint artifacts failed the endpoint audit: {endpoint_audit}')
    baseline_evaluation = original_summary['last_evaluation']
    actual = original_summary['counters']['loss_tokens']
    unspent = original_summary['unspent_loss_tokens']
    if not (original_summary['budget_endpoint'] is True and
            original_summary['requested_loss_tokens'] == 5_000_000 and
            actual + unspent == 5_000_000 and 0 <= unspent < 3584 and
            original_summary['stop_reason'] in ('loss_token_budget', 'remaining_budget_below_complete_trajectory')):
        raise ValueError('The requested writer is not at the completed nominal 5M endpoint')
    checkpoint = training_run / 'writer.pt'
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use an empty output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    dataset = Path(config.values['raw']['dataset'])
    dataset_hash = sha256(dataset)
    if dataset_hash != original_config['raw_identity']['dataset_sha256']:
        raise ValueError('Training and diagnostic dataset SHA256 differ')
    manifest = read_json(args.manifest)
    original_manifest = read_json(training_run / 'data_manifest.json')
    bindings = {
        'dataset_sha256': dataset_hash,
        'tokenizer_identity': original_manifest['tokenizer_fingerprint'],
        'baseline_validation_protocol_sha256': original_manifest['validation_protocol_sha256'],
        'baseline_validation_sample_sha256': original_manifest['validation_sample_sha256'],
        'baseline_validation_sample': original_manifest['validation_sample'],
    }
    if any(manifest.get(key) != expected for key, expected in bindings.items()):
        raise ValueError('Diagnostic manifest does not bind the original fixed development sample')
    started = time.monotonic()
    launch = {
        'protocol': 'p1_frozen_diagnostics_launch_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'evaluation_git_commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'training_launch': read_json(training_run.with_name(training_run.name + '.launch.json')),
        'training_run': str(training_run), 'training_seed': config.values['seed'],
        'training_actual_loss_tokens': actual, 'training_unspent_loss_tokens': unspent,
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
        'dataset_sha256': dataset_hash, 'manifest_path': str(Path(args.manifest).resolve()),
        'manifest_sha256': sha256(args.manifest), 'config_sha256': sha256(args.config),
        'evaluation_batch_size': 1, 'original_evaluation_batch_size': config.values['raw']['batch_size'],
        'threads': args.threads, 'deterministic_algorithms': True,
        'torch': str(torch.__version__), 'cuda': torch.version.cuda,
        'max_wall_seconds': args.max_wall_seconds,
        'source_sha256': {str(p.relative_to(repo / 'src')): sha256(p)
                          for p in sorted((repo / 'src/delta_ngram').rglob('*.py'))},
    }
    save_json(output / 'launch.json', launch)
    save_json(output / 'manifest.json', manifest)
    print(json.dumps({'event': 'launch', 'output_dir': str(output),
                      'seed': config.values['seed'], 'manifest_sha256': launch['manifest_sha256']}, allow_nan=False), flush=True)

    def append(path, value):
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')

    def check_deadline():
        if time.monotonic() - started > args.max_wall_seconds:
            raise TimeoutError('Diagnostic wall limit reached; partial logs are not a complete result')

    def block_callback(record):
        append(output / 'block_metrics.jsonl', record)
        # Callbacks follow a completed logical operation. Preserve its actual
        # work before aborting so partial cost accounting does not lose a block.
        check_deadline()

    def progress_callback(record):
        event = {'event': 'progress', 'elapsed_seconds': time.monotonic() - started, **record}
        append(output / 'progress.jsonl', event)
        print(json.dumps(event, allow_nan=False), flush=True)
        check_deadline()

    try:
        method, _ = create_runtime(config)
        metadata = load_diagnostic_checkpoint(method, config, checkpoint,
                                               expected_identity=original_config['raw_identity'])
        if (metadata['checkpoint_step'] != original_summary['completed_steps'] or
                metadata['training_loss_tokens'] != actual):
            raise ValueError('Writer checkpoint is not the recorded final optimizer boundary')
        fingerprint = 'sha256:' + _json_hash(tokenizer_identity(method))
        documents = load_documents(dataset, method.tokenizer, tokenizer_identity=fingerprint,
                                   require_record_identity=True)
        result = evaluate_diagnostics(
            method, documents, manifest, bootstrap_seed=104729, bootstrap_samples=2000,
            block_callback=block_callback, progress_callback=progress_callback,
            baseline_evaluation=baseline_evaluation, checkpoint_metadata=metadata)
        result['run_provenance'] = launch
        result['whole_run_elapsed_seconds'] = time.monotonic() - started
        if method.device.type == 'cuda':
            result['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(method.device)
            result['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved(method.device)
        save_json(output / 'results.json', result)
        print(json.dumps({'event': 'finished', 'results': str(output / 'results.json'),
                          'elapsed_seconds': result['whole_run_elapsed_seconds']}, allow_nan=False), flush=True)
    except BaseException as error:
        save_json(output / 'failure.json', {'complete': False, 'type': type(error).__name__,
                                           'message': str(error), 'elapsed_seconds': time.monotonic() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    builder = commands.add_parser('prepare', help='Freeze targets and donor pairs without GPU work')
    builder.add_argument('--baseline-run', required=True)
    builder.add_argument('--dataset', required=True)
    builder.add_argument('--output', required=True)
    builder.set_defaults(handler=prepare)
    runner = commands.add_parser('evaluate', help='Evaluate one completed 5M checkpoint with fixed pairs')
    runner.add_argument('--config', required=True)
    runner.add_argument('--training-run', required=True)
    runner.add_argument('--manifest', required=True)
    runner.add_argument('--output', required=True)
    runner.add_argument('--threads', type=int, default=8)
    runner.add_argument('--max-wall-seconds', type=float, default=7200)
    runner.set_defaults(handler=evaluate)
    args = parser.parse_args()
    if args.command == 'evaluate' and (args.threads < 1 or args.max_wall_seconds <= 0):
        parser.error('threads and max-wall-seconds must be positive')
    args.handler(args)


if __name__ == '__main__':
    main()
