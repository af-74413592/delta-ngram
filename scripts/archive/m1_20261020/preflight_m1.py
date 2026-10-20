"""Audit the frozen corpus, complete sampling budget, and real GPU gradient path."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '8'
ROOT = Path('/workspace/delta-ngram-m1')
OPS = Path('/workspace/delta-ngram-transfer-20261019/multidomain')
DATA = Path('/workspace/delta-ngram-data/multidomain_20261019/merged_event_v5')
sys.path.insert(0, str(ROOT / 'scripts'))
import torch
import delta_ngram
from delta_ngram.config import ExperimentConfig
from delta_ngram.checkpoints import load_checkpoint, restore_writer
from delta_ngram.data.raw_text import RawWindowSampler
from delta_ngram.raw_corpora import load_corpora, trajectory_pools
from delta_ngram.raw_lm import tokenizer_identity, _json_hash, trajectory_loss
from delta_ngram.train import create_runtime
from evaluate_raw_checkpoint import prepare_validation

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def tensor_sha(state):
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value.update(name.encode())
        value.update(str(tensor.dtype).encode())
        value.update(str(tuple(tensor.shape)).encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()

started = time.monotonic()
assert Path(delta_ngram.__file__).resolve().is_relative_to(ROOT / 'src'), 'Use the frozen M1 source checkout'
torch.set_num_threads(8)
torch.use_deterministic_algorithms(True)
assert not subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip()
commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
config = ExperimentConfig.load(ROOT / 'configs/m1_event_mixture_seed17.yaml')
raw = config.values['raw']
manifest = json.loads((DATA / 'data_manifest.json').read_text())
capacity = json.loads((DATA / 'training_capacity_report.json').read_text())
assert manifest['ready'] is True and capacity['passed'] is True
assert capacity['planned_loss_tokens'] == raw['loss_tokens'] == 100_000_000
assert capacity['maximum_train_cell_passes'] == 3.0
assert manifest['training_source_weights'] == raw['source_weights']
assert raw['eval_donor'] and raw['event_dataset']
assert manifest['donor_coverage']['passed']
for source, weights in capacity['conditional_length_weights'].items():
    assert {int(key): value for key, value in weights.items()} == raw['source_length_weights'][source]
for name, info in manifest['files'].items():
    path = DATA / name
    assert path.stat().st_size == info['bytes'] and sha(path) == info['sha256'], name
method, _ = create_runtime(config)
initial_path = Path(config.values['run']['checkpoint'])
initial = load_checkpoint(initial_path)
assert initial['step'] == 0 and 'optimizer_state_dict' not in initial
restore_writer(method, initial, config)
before = {name: tensor.detach().cpu().clone() for name, tensor in method.writer.state_dict().items()}
assert all(torch.equal(before[name], tensor) for name, tensor in initial['writer_state_dict'].items())
fingerprint = 'sha256:' + _json_hash(tokenizer_identity(method))
assert fingerprint == manifest['tokenizer']['fingerprint']
documents, events = load_corpora(raw, method.tokenizer, fingerprint,
    vocabulary_size=method.reader.model.get_input_embeddings().weight.shape[0])
windows = trajectory_pools(raw, documents, events)
sampler = RawWindowSampler(windows, config.values['seed'], source_weights=raw['source_weights'],
                           source_length_weights=raw['source_length_weights'])
sample_hash = hashlib.sha256()
steps = consumed = 0
while consumed < raw['loss_tokens']:
    batch = sampler.next_batch(raw['batch_size'] * raw['gradient_accumulation_steps'],
                               remaining_loss_tokens=raw['loss_tokens'] - consumed)
    if not batch:
        break
    steps += 1
    count = sum(window.loss_tokens for window in batch)
    sample_hash.update((json.dumps({'step': steps, 'batch_loss_tokens': count,
        'windows': [{'document_id': w.document_id, 'start': w.start, 'blocks': w.blocks} for w in batch]},
        sort_keys=True, separators=(',', ':')) + '\n').encode())
    consumed += count
assert 0 <= raw['loss_tokens'] - consumed < 512
sampling = {'steps': steps, 'loss_tokens': consumed, 'unspent_loss_tokens': raw['loss_tokens'] - consumed,
            'order_sha256': sample_hash.hexdigest(), 'fingerprint': sampler.fingerprint,
            'terminal_status': sampler.last_status,
            'source_loss_tokens': sampler.source_loss_tokens,
            'cells': [{'blocks': count, 'source': source, 'loss_tokens': sampler.cell_loss_tokens[count, source],
                       'windows': len(sampler._cells[count, source]), 'epochs': sampler._epochs[count, source]}
                      for count, source in sampler._cells]}
validation, validation_metadata, donors = prepare_validation(config, method, include_donors=True)
assert all(count >= 64 for count in validation_metadata['selected_validation_groups_per_source'].values())
assert validation_metadata['validation_protocol']['conditions'] == ['self', 'empty', 'donor']
# One representative from each configured cell; no optimizer is created or stepped.
probe = [max(items, key=lambda w: w.loss_tokens) for items in sampler._cells.values()]
# Exercise the largest actual event batch, including maximum BPTT horizon.
event_candidates = [w for items in sampler._cells.values() for w in items if w.source == 'news_event']
worst = sorted(event_candidates, key=lambda w: (w.actual_blocks, w.loss_tokens), reverse=True)[:raw['batch_size']]
assert len(worst) == raw['batch_size']
probe = worst + probe
probe_targets = sum(w.loss_tokens for w in probe)
assert probe_targets <= 250_000
method.train()
method.writer.zero_grad(set_to_none=True)
loss_sum = 0.0
for start in range(0, len(probe), raw['batch_size']):
    result = trajectory_loss(method, probe[start:start + raw['batch_size']],
                             prefix_tokens=raw['prefix_tokens'], checkpoint_reader=True)
    loss = result.loss_sum / probe_targets
    assert torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    loss_sum += float(result.loss_sum.detach())
    del loss, result
assert all(not p.requires_grad and p.grad is None for p in method.reader.model.parameters())
grad_norm = float(torch.nn.utils.clip_grad_norm_(method.parameters(), 1.0, error_if_nonfinite=True))
assert grad_norm > 0
assert all(torch.equal(before[name], tensor.detach().cpu()) for name, tensor in method.writer.state_dict().items())
report = {'passed': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'git_commit': commit,
          'config_sha256': sha(config.source), 'dataset_manifest_sha256': sha(DATA / 'data_manifest.json'),
          'dataset_sha256': sha(raw['dataset']), 'tokenizer_fingerprint': fingerprint,
          'event_dataset_sha256': sha(raw['event_dataset']),
          'initial_checkpoint_sha256': sha(initial_path), 'initial_writer_tensor_sha256': tensor_sha(before),
          'initial_step': 0, 'fresh_optimizer_required': True, 'sampling': sampling,
          'validation': validation_metadata,
          'gpu_gradient_probe': {'target_tokens': probe_targets, 'optimizer_steps': 0,
              'nll': loss_sum / probe_targets, 'grad_norm_before_clip': grad_norm,
              'worst_event_actual_blocks': [w.actual_blocks for w in worst],
              'worst_event_target_tokens': [w.loss_tokens for w in worst],
              'writer_parameters_exactly_unchanged': True, 'reader_frozen_no_gradients': True,
              'scope': 'P0 engineering access; not trained writer quality'},
          'torch': str(torch.__version__), 'cuda': torch.version.cuda,
          'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
          'elapsed_seconds': time.monotonic() - started}
with (OPS / 'm1_preflight.json').open('x') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({'passed': True, 'sampling': sampling, 'gradient_probe': report['gpu_gradient_probe']},
                 allow_nan=False), flush=True)
