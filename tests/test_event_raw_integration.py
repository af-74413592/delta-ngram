"""The real trainer and frozen evaluator must consume event masks identically."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest
import torch
import yaml

from delta_ngram.config import ExperimentConfig
from delta_ngram.data.event_streams import EventDocument
from delta_ngram.data.raw_text import _digest
from delta_ngram.raw_corpora import load_corpora
from delta_ngram.raw_lm import _json_hash, tokenizer_identity, train_raw
from delta_ngram.train import create_runtime
from test_raw_lm import config_for
from test_event_streams import article, serialized

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_raw_checkpoint import evaluate_checkpoint


def prepare_case(tmp_path):
    model, _ = create_runtime(config_for(tmp_path))
    fingerprint = 'sha256:' + _json_hash(tokenizer_identity(model))
    rows = [{'id': f'raw-{split}-{i}', 'group_id': f'raw-{split}-{i}', 'source': 'code',
             'split': split, 'text': 'abc xyz\n' * 64} for split in ('train', 'validation') for i in range(2)]
    (tmp_path / 'documents.jsonl').write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
    events = []
    for index in range(4):
        split, year = ('train', 2017) if index < 2 else ('validation', 2018)
        articles = tuple(article(2 * index + j, length=128 + index + j,
                                 day=f'{year}-01-0{j + 1}', token=3 + j) for j in range(2))
        members = [item.id for item in articles]
        events.append(EventDocument(f'event-{index}', 'news-family:' + _digest(sorted(members)),
            'news_event', split, articles, fingerprint, {'family_members': members, 'stream_article_ids': members}))
    (tmp_path / 'events.jsonl').write_text('\n'.join(json.dumps(serialized(event)) for event in events), encoding='utf-8')
    base = {'seed': 17, 'model': {'smoke': True, 'device': 'cpu'}, 'reader': {'buckets': 17},
            'writer': {'hidden_dim': 12}, 'run': {'output_dir': 'full', 'max_wall_seconds': 120},
            'raw': {'dataset': 'documents.jsonl', 'event_dataset': 'events.jsonl',
                    'block_size': 128, 'prefix_tokens': 8, 'block_counts': [2, 4],
                    'source_weights': {'code': .5, 'news_event': .5},
                    'source_length_weights': {'code': {2: 1.}, 'news_event': {4: 1.}},
                    'macro_source_groups': {'code': 'code', 'news_event': 'news'},
                    'batch_size': 2, 'loss_tokens': 2000, 'checkpoint_every_tokens': 256,
                    'eval_windows': 4, 'eval_min_groups_per_source': 2, 'eval_donor': True,
                    'eval_every_tokens': 1000, 'eval_bootstrap_samples': 20}}
    def config(name, **raw):
        values = {**base, 'run': {**base['run'], 'output_dir': name}, 'raw': {**base['raw'], **raw}}
        path = tmp_path / (name + '.yaml')
        path.write_text(yaml.safe_dump(values), encoding='utf-8')
        return ExperimentConfig.load(path)
    return config, fingerprint, events


def test_event_training_exact_resume_and_frozen_donor_protocol(tmp_path):
    config, _, _ = prepare_case(tmp_path)
    full = train_raw(config('full'))
    assert full['source_loss_tokens']['code'] > 0 and full['source_loss_tokens']['news_event'] > 0
    assert sum(full['source_loss_tokens'].values()) == full['counters']['loss_tokens']
    assert full['budget_endpoint']
    train_raw(config('partial', max_steps=2))
    resumed = train_raw(config('resumed', resume=str(tmp_path / 'partial/training.pt')))
    assert full['counters'] == resumed['counters']
    assert full['source_loss_tokens'] == resumed['source_loss_tokens']
    left = torch.load(tmp_path / 'full/training.pt', weights_only=True)
    right = torch.load(tmp_path / 'resumed/training.pt', weights_only=True)
    assert left['sampler_state'] == right['sampler_state']
    for key, tensor in left['writer_state_dict'].items():
        assert torch.equal(tensor, right['writer_state_dict'][key])
    evaluation = evaluate_checkpoint(config('frozen'), tmp_path / 'full/writer.pt')
    last = full['last_evaluation']
    assert evaluation['validation_sample_sha256'] == last['validation_sample_sha256']
    assert evaluation['validation_protocol_sha256'] == last['validation_protocol_sha256']
    assert evaluation['donor_sample_sha256'] == last['donor_sample_sha256']
    for key in ('self_nll', 'empty_nll', 'donor_nll', 'source_macro_delta_donor'):
        assert evaluation[key] == last[key]
    assert evaluation['optimizer_steps'] == evaluation['backward_calls'] == 0
    assert evaluation['integrity_checks']['writer_state_dict_exact_equal']


def test_mixed_group_alias_cannot_hide_an_event_family_overlap(tmp_path):
    config, fingerprint, events = prepare_case(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / 'documents.jsonl').read_text().splitlines()]
    rows[0]['group_id'] = 'mixed-family:new-id'
    rows[0]['metadata'] = {'original_group_id': events[0].group_id}
    (tmp_path / 'documents.jsonl').write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
    model, _ = create_runtime(config('overlap'))
    with pytest.raises(ValueError, match='excluded'):
        load_corpora(config('overlap').values['raw'], model.tokenizer, fingerprint)
