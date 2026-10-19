"""Donor histories must change memory without changing scored target prefixes."""
from types import SimpleNamespace

import pytest
import torch

from delta_ngram.data.raw_text import TrajectoryWindow
from delta_ngram.raw_evaluation import select_donors, window_key, macro_statistics
from delta_ngram.raw_lm import evaluate_raw, trajectory_loss
from delta_ngram.train import create_runtime
from test_raw_lm import config_for, window


def test_donor_uses_foreign_writes_and_identical_target_queries(tmp_path):
    method, _ = create_runtime(config_for(tmp_path))
    target = window(method, 'abc xyz\n' * 4, 'target')
    donor = window(method, 'xyz abc\n' * 4, 'donor')
    traces = {}
    for condition in ('self', 'empty', 'donor'):
        traces[condition] = []
        with torch.no_grad():
            result = trajectory_loss(method, [target], prefix_tokens=4, memory_mode=condition,
                **({'donor_windows': [donor]} if condition == 'donor' else {}),
                observer=lambda kind, block, state, ids, mask, labels: traces[condition].append(
                    (kind, block, ids.clone(), None if labels is None else labels.clone())))
        assert result.loss_tokens == 24
        assert not result.loss_sum.requires_grad
    queries = lambda key: [item for item in traces[key] if item[0] == 'read']
    for baseline, empty, swapped in zip(queries('self'), queries('empty'), queries('donor')):
        assert torch.equal(baseline[2], empty[2]) and torch.equal(baseline[2], swapped[2])
        assert torch.equal(baseline[3], empty[3]) and torch.equal(baseline[3], swapped[3])
    writes = [item for item in traces['donor'] if item[0] == 'write']
    assert len(writes) == 3
    for _, block, ids, _ in writes:
        assert ids.tolist()[0] == list(donor.token_ids[block * 8:(block + 1) * 8])
    with pytest.raises(ValueError, match='different groups'):
        trajectory_loss(method, [target], memory_mode='donor', donor_windows=[target], prefix_tokens=4)


def test_donors_are_fixed_independent_and_shape_matched():
    items = [TrajectoryWindow(str(i), str(i), source, 0, tuple(range(16)), 2)
             for source, i in [('code', 0), ('code', 1), ('news', 2), ('news', 3)]]
    first = select_donors(items, 17)
    assert {key: window_key(value) for key, value in first.items()} == {
        key: window_key(value) for key, value in select_donors(list(reversed(items)), 17).items()}
    for target in items:
        donor = first[window_key(target)]
        assert donor.source == target.source and donor.group_id != target.group_id
    with pytest.raises(ValueError, match='no matched'):
        select_donors(items[:1])


def test_evaluation_reports_content_specific_benefit_without_gradients(monkeypatch):
    import delta_ngram.raw_lm as module
    items = [TrajectoryWindow(str(i), str(i), 'news', 0, tuple(range(16)), 2) for i in range(64)]
    calls = []
    def fake(method, batch, *, prefix_tokens, memory_mode='self', donor_windows=None):
        assert not torch.is_grad_enabled()
        if memory_mode == 'donor':
            assert all(left.group_id != right.group_id for left, right in zip(batch, donor_windows))
        count = sum(item.loss_tokens for item in batch)
        calls.append(memory_mode)
        return SimpleNamespace(loss_tokens=count, loss_sum=torch.tensor(count * {'self': 1., 'donor': 2., 'empty': 4.}[memory_mode]))
    monkeypatch.setattr(module, 'trajectory_loss', fake)
    result = evaluate_raw(SimpleNamespace(eval=lambda: None), items, prefix_tokens=4, batch_size=4,
                          report_donor=True, bootstrap_samples=100)
    assert set(calls) == {'self', 'empty', 'donor'}
    assert result['delta_empty'] == 3. and result['delta_donor'] == 1.
    assert result['source_macro_delta_donor'] == 1.
    assert result['checkpoint_selection_eligible'] is True
    assert result['optimizer_steps'] == result['backward_calls'] == 0
    assert len(result['donor_sample']) == len(items)


def test_news_strata_share_one_macro_category():
    records = [dict(group_id=source, source=source, loss_tokens=10,
                    self_loss_sum=10., empty_loss_sum=empty, donor_loss_sum=donor)
               for source, empty, donor in [('code', 30., 20.), ('news', 20., 30.), ('news_event', 50., 30.)]]
    result = macro_statistics(records, {'code': 'code', 'news': 'news', 'news_event': 'news'},
                              bootstrap_seed=17, bootstrap_samples=50, donor=True)
    assert result['source_macro_delta_donor'] == 1.5
    assert result['source_macro_delta_empty'] == 2.25
    assert result['macro_categories'] == ['code', 'news']
    assert result['checkpoint_selection_eligible'] is False  # insufficient independent families


def test_generic_bias_alone_does_not_pass_selection():
    records = [dict(group_id=str(i), source='code', loss_tokens=10,
                    self_loss_sum=10., empty_loss_sum=20., donor_loss_sum=10.) for i in range(64)]
    result = macro_statistics(records, None, bootstrap_seed=17, bootstrap_samples=100, donor=True)
    assert result['source_macro_delta_empty'] == 1.
    assert result['source_macro_delta_donor'] == 0.
    assert result['checkpoint_selection_eligible'] is False
