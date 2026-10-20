"""Compare disposable-reader VJPs with an ordinary fully retained writer graph."""
import copy
import hashlib

import pytest
import torch
from torch.nn import functional as F

from delta_ngram.alpha_memory import BlockAlpha, LazyDecayState
from delta_ngram.qwen38_native import write_native_block
from delta_ngram.qwen38_training import (
    GradientScaleProbe, NativeTrajectoryGradients, TPGradientPolicy, writer_evidence_features,
)
from delta_ngram.qwen38_writer import NativeForwardWriter


RUNTIME = hashlib.sha256(b"CPU analytic fixture, not a GPU runtime receipt").hexdigest()


def policy(reduction="replica", world_size=1):
    # Independent scalar reference: f(theta)=theta**3 + theta at theta=.7.
    derivative = 3 * .7 ** 2 + 1
    local = derivative / world_size if reduction == "sum" else derivative
    return TPGradientPolicy(
        reduction, world_size, RUNTIME, "shared-theta-scalar-direction",
        tuple(GradientScaleProbe(
            step, (((.7 + step) ** 3 + .7 + step)
                   - ((.7 - step) ** 3 + .7 - step)) / (2 * step),
            (local,) * world_size) for step in (.001, .0005)),
        rtol=1e-5, atol=1e-8,
    )


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def fixture(alpha):
    torch.manual_seed(731)
    writer = NativeForwardWriter(
        width=3, input_dim=5, hidden_dim=7, heads_per_order=2,
        orders=(2, 3), target_scale=torch.tensor([.8, 1., 1.2, 1.4]))
    decay = BlockAlpha(width=5, kind="fixed", initial_alpha=alpha)
    features = [torch.randn(2, 4, 5, requires_grad=True) for _ in range(3)]
    masks = [torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool),
             torch.tensor([[1, 1, 1, 0], [0, 0, 0, 0]], dtype=torch.bool),
             torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)]
    # Collisions within each head, same physical IDs under different owners.
    rows = torch.tensor([[[1, 4, 7, 10], [1, 5, 7, 11], [2, 4, 8, 10], [1, 4, 7, 10]],
                         [[1, 4, 7, 10], [2, 5, 8, 11], [1, 4, 7, 10], [2, 4, 8, 10]]])
    rows = [torch.where(mask[..., None], rows, 0) for mask in masks]
    query_rows = torch.tensor([[[1, 4, 7, 10], [1, 4, 7, 10], [2, 5, 8, 11]],
                               [[1, 4, 7, 10], [2, 5, 8, 11], [1, 4, 7, 10]]])
    targets = [torch.tensor([[0, 1, 2], [1, 2, -100]]),
               torch.tensor([[2, 1, -100], [0, 1, 2]]),
               torch.tensor([[1, -100, 2], [2, 1, -100]]),
               torch.tensor([[2, 0, 1], [0, -100, -100]])]
    reader = (torch.randn(12, 6) * .2, torch.randn(6, 3) * .3,
              torch.randn(2, 3, 12) * .1)
    return writer, decay, features, masks, rows, query_rows, targets, reader


def read_loss(state, rows, targets, reader):
    # Nonlinear frozen reader with a temporal path; no dependence on writer
    # parameters except through the effective sparse delta lookup.
    delta = state.lookup(rows)[0].flatten(-2)
    w1, w2, base = reader
    projected = (base + delta) @ w1
    value = projected * torch.sigmoid(projected.roll(1, dims=-1))
    normalized = value * (value.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    shifted = F.pad(normalized[:, :-1], (0, 0, 1, 0))
    logits = (value + .2 * shifted) @ w2
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100,
                           reduction="sum"), int(targets.ne(-100).sum())


@pytest.mark.parametrize("alpha", [1., .99])
@pytest.mark.parametrize("feature_mode", ["none", "rmsnorm"])
@pytest.mark.parametrize("reader_divisor", ["sum", "query_tokens", "group_tokens"])
def test_writer_vjp_equals_retained_graph_for_all_parameters(alpha, feature_mode, reader_divisor):
    initial, decay, features, masks, rows, query_rows, targets, reader = fixture(alpha)
    total_tokens = sum(int(target.ne(-100).sum()) for target in targets)
    group_tokens = total_tokens * 2 + 7  # Another trajectory's share must NOT disappear.
    results = []
    final_states = []
    for bridged in (False, True):
        writer = copy.deepcopy(initial)
        state = LazyDecayState.empty(2, 3, 12)
        ledger = NativeTrajectoryGradients(
            total_loss_tokens=group_tokens, expected_loss_tokens=total_tokens,
            policy=policy(), runtime_identity_sha256=RUNTIME) if bridged else None
        objectives = []
        query_index = 0
        snapshots = []
        for block in range(3):
            hidden = writer_evidence_features(features[block], mode=feature_mode, eps=1e-6)
            state = write_native_block(writer, state, hidden, rows[block], masks[block], decay)
            snapshots.append(state)
            # Read the first snapshot twice, including after the next write.
            reads = [state] if block != 1 else [state, snapshots[0]]
            for snapshot in reads:
                target = targets[query_index]
                ticket = ledger.prepare_query(snapshot) if bridged else None
                loss, count = read_loss(ticket.reader_state if bridged else snapshot,
                                        query_rows, target, reader)
                if bridged:
                    divisor = {"sum": 1, "query_tokens": count, "group_tokens": group_tokens}[reader_divisor]
                    (loss / divisor).backward()
                    assert all(parameter.grad is None for parameter in writer.parameters())
                    ledger.record_query(ticket, loss_sum=loss, loss_tokens=count,
                                        reader_loss_divisor=divisor)
                    assert ticket.leaf_values.grad is None
                else:
                    objectives.append(loss)
                query_index += 1
        if bridged:
            report = ledger.backward_writer()
            assert report["optimizer_group_loss_tokens"] == group_tokens
            assert report["loss_tokens"] == total_tokens and report["writer_vjp_queries"] == 4
        else:
            (sum(objectives) / group_tokens).backward()
        results.append({name: parameter.grad.clone() for name, parameter in writer.named_parameters()})
        final_states.append(state)
        assert all(hidden.grad is None for hidden in features)
    assert results[0].keys() == results[1].keys()
    for name in results[0]:
        torch.testing.assert_close(results[1][name], results[0][name], rtol=3e-5, atol=2e-7, msg=name)
    for name in ("values", "counts", "block_clock", "cumulative_log_alpha",
                 "row_last_written_block", "row_log_alpha_at_write"):
        assert torch.equal(getattr(final_states[0], name), getattr(final_states[1], name))
    assert final_states[1].block_clock.tolist() == [3, 2]


def test_policy_requires_informative_matching_scale_and_runtime_receipt():
    accepted = policy("mean", 2)
    assert accepted.receipt()["accepted"] and len(accepted.receipt()["receipt_sha256"]) == 64
    with pytest.raises(ValueError, match="disagrees"):
        TPGradientPolicy("sum", 2, RUNTIME, "same", accepted.probes, 1e-5, 1e-8)
    with pytest.raises(ValueError, match="two finite-difference"):
        TPGradientPolicy("mean", 2, RUNTIME, "same", accepted.probes[:1], 1e-5, 1e-8)
    with pytest.raises(ValueError, match="different runtime"):
        NativeTrajectoryGradients(total_loss_tokens=1, expected_loss_tokens=1,
                                  policy=accepted, runtime_identity_sha256="0" * 64)
    with pytest.raises(ValueError, match="uninformative"):
        TPGradientPolicy("replica", 1, RUNTIME, "zero",
                         (GradientScaleProbe(.1, 0., (0.,)), GradientScaleProbe(.2, 0., (0.,))),
                         1e-5, 1e-8)


@pytest.mark.parametrize("reduction", ["replica", "sum", "mean"])
def test_explicit_tp_policy_and_reducer_do_not_hide_a_factor(reduction):
    writer, decay, features, masks, rows, query_rows, targets, reader = fixture(.99)
    state = write_native_block(writer, LazyDecayState.empty(2, 3, 12),
                               features[0], rows[0], masks[0], decay)
    count = int(targets[0].ne(-100).sum())
    expected = torch.autograd.grad(read_loss(state, query_rows, targets[0], reader)[0],
                                   state.values, retain_graph=True)[0] / count
    ledger = NativeTrajectoryGradients(total_loss_tokens=count, expected_loss_tokens=count,
                                      policy=policy(reduction, 2), runtime_identity_sha256=RUNTIME)
    ticket = ledger.prepare_query(state)
    loss, _ = read_loss(ticket.reader_state, query_rows, targets[0], reader)
    # For SUM emulate each TP rank contributing half; for replicated/mean,
    # each already owns the complete logical gradient.
    (loss / (2 if reduction == "sum" else 1)).backward()
    if reduction != "replica":
        with pytest.raises(ValueError, match="explicit reducer"):
            ledger.record_query(ticket, loss_sum=loss, loss_tokens=count, reader_loss_divisor=1)
    reducer = None if reduction == "replica" else (
        lambda local, selected: local * 2 if selected.reduction == "sum" else local)
    ledger.record_query(ticket, loss_sum=loss, loss_tokens=count,
                        reader_loss_divisor=1, reducer=reducer)
    torch.testing.assert_close(ticket._cotangent, expected)
    ledger.backward_writer()
    assert writer.target_head[-1].weight.grad.abs().sum() > 0


def test_ticket_budget_detach_and_mutation_guards():
    writer, decay, features, masks, rows, query_rows, targets, reader = fixture(1.)
    state = write_native_block(writer, LazyDecayState.empty(2, 3, 12),
                               features[0], rows[0], masks[0], decay)
    ledger = NativeTrajectoryGradients(total_loss_tokens=5, expected_loss_tokens=5,
                                      policy=policy(), runtime_identity_sha256=RUNTIME)
    with pytest.raises(ValueError, match="detached"):
        ledger.prepare_query(state.detach())
    ticket = ledger.prepare_query(state)
    with pytest.raises(RuntimeError, match="must finish"):
        ledger.backward_writer()
    loss, count = read_loss(ticket.reader_state, query_rows, targets[0], reader)
    loss.backward()
    with pytest.raises(ValueError, match="budget"):
        ledger.record_query(ticket, loss_sum=loss, loss_tokens=6, reader_loss_divisor=1)
    ledger.record_query(ticket, loss_sum=loss, loss_tokens=count, reader_loss_divisor=1)
    with pytest.raises(RuntimeError, match="already recorded"):
        ledger.record_query(ticket, loss_sum=loss, loss_tokens=count, reader_loss_divisor=1)
    with torch.no_grad():
        state.values.add_(0)
    with pytest.raises(RuntimeError, match="modified in place"):
        ledger.backward_writer()


def test_empty_query_and_feature_preprocessing_are_explicit():
    ledger = NativeTrajectoryGradients(total_loss_tokens=5, expected_loss_tokens=5,
                                      policy=policy(), runtime_identity_sha256=RUNTIME)
    ticket = ledger.prepare_query(LazyDecayState.empty(2, 3, 12))
    assert not ticket.requires_reader_backward
    ledger.record_query(ticket, loss_sum=9., loss_tokens=5, reader_loss_divisor=5)
    assert ledger.backward_writer()["writer_vjp_queries"] == 0
    with pytest.raises(RuntimeError, match="already finished"):
        ledger.backward_writer()
    hidden = torch.randn(2, 3, 5, requires_grad=True)
    plain = writer_evidence_features(hidden, mode="none", eps=1e-6)
    normalized = writer_evidence_features(hidden, mode="rmsnorm", eps=1e-6)
    assert not plain.requires_grad and not normalized.requires_grad
    torch.testing.assert_close(plain, hidden.detach())
    torch.testing.assert_close(normalized, F.rms_norm(hidden.detach(), (5,), eps=1e-6))
    saved = plain.clone()
    with torch.no_grad():
        hidden.zero_()  # Reusing a PP receive buffer must not alter writer features.
    torch.testing.assert_close(plain, saved, rtol=0, atol=0)
    with pytest.raises(ValueError, match="epsilon"):
        writer_evidence_features(hidden, mode="rmsnorm", eps=0)
