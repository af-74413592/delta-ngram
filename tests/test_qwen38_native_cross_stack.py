import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'native_qwen38'))
import qwen38_native_a800_forward as forward
import qwen38_native_cross_stack as probe
from delta_ngram.alpha_memory import LazyDecayState
from delta_ngram.qwen38_native import Qwen38PLELayout
from delta_ngram.qwen38_vllm_snapshot import save_vllm_snapshot


def test_prefix_padding_keeps_actual_decision_position():
    batch, valid = probe.prompt_batch([2, 3, 4], pad_token_id=9, device='cpu')
    assert batch['input_ids'].tolist() == [[2, 3, 4, 9]]
    assert batch['position_ids'].tolist() == [[0, 1, 2, 3]]
    assert valid.tolist() == [[True, True, True, False]]
    with pytest.raises(ValueError, match='invalid native probe prefix'):
        probe.prompt_batch([True], pad_token_id=9, device='cpu')


def test_last_real_logits_are_gathered_in_vocab_order(monkeypatch):
    batch, valid = probe.prompt_batch([2, 3, 4], pad_token_id=9, device='cpu')
    logits = torch.zeros(1, 4, probe.VOCAB // 2, dtype=torch.bfloat16)
    logits[:, 2] = 3.
    logits[:, 3] = 99.  # Padding must never select the next token.
    def schedule(**kwargs):
        out, collect = kwargs['forward_step_func'](None, lambda **unused: logits)
        return [collect(out)[1]]
    def gather(parts, local, group):
        parts[0].copy_(local)
        parts[1].copy_(local + 10.)
    ps = SimpleNamespace(is_pipeline_last_stage=lambda: True,
        get_tensor_model_parallel_world_size=lambda: 2,
        get_tensor_model_parallel_group=lambda: 'tp',
        get_pipeline_model_parallel_last_rank=lambda: 6,
        get_pipeline_model_parallel_group=lambda: 'pp')
    dist = SimpleNamespace(all_gather=gather, broadcast=lambda *args, **kwargs: None)
    monkeypatch.setattr(forward, 'runtime', lambda session: (dist, ps, schedule))
    session = SimpleNamespace(models=[object()], adapter=None)
    value = forward.native_forward(session, batch, valid, return_logits=True)['next_logits']
    assert value.shape == (1, probe.VOCAB) and value.dtype == torch.float32
    assert torch.equal(value[:, :probe.VOCAB // 2], torch.full((1, probe.VOCAB // 2), 3.))
    assert torch.equal(value[:, probe.VOCAB // 2:], torch.full((1, probe.VOCAB // 2), 13.))
    with pytest.raises(ValueError, match='mutually exclusive'):
        forward.native_forward(session, batch, valid, evidence=True, return_logits=True)
    with pytest.raises(ValueError, match='contiguous valid prefix'):
        forward.native_forward(session, batch, torch.tensor([[True, False, True, False]]), return_logits=True)


def test_argmax_ties_use_lowest_vocabulary_id():
    logits = torch.zeros(probe.VOCAB)
    logits[3] = logits[7] = 2.
    result = probe.logit_summary(logits)
    assert result['argmax_token_id'] == 3 and result['top1_top2_margin'] == 0.


def test_live_state_is_verified_against_exported_snapshot(tmp_path):
    layout = Qwen38PLELayout(tuple(range(0, 64, 4)), (4,) * 16)
    state = LazyDecayState.empty(batch_size=1, width=layout.row_width, row_capacity=layout.row_capacity)
    identity = {key: {'test': True} for key in
                ('base_model', 'tokenizer', 'addressing', 'writer', 'operator', 'trajectory')}
    receipt = save_vllm_snapshot(tmp_path/'delta.pt', state=state, layout=layout, identity=identity)
    path = tmp_path/'contract.json'
    path.write_text(json.dumps(receipt), encoding='utf-8')
    result = probe.verify_snapshot_state(state, layout, path)
    assert result['actual_live_state_matches_snapshot']
    receipt['snapshot_sha256'] = '0' * 64
    path.write_text(json.dumps(receipt), encoding='utf-8')
    with pytest.raises(ValueError, match='SHA'):
        probe.verify_snapshot_state(state, layout, path)
