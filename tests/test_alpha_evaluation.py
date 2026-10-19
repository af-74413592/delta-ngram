"""Synthetic CPU checks only; no real continuation cohort or model results."""
from copy import deepcopy
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from delta_ngram.alpha_evaluation import (AlphaEvaluationBackend, _historical_trajectory,
    evaluate_alpha_historical, evaluate_revision, file_sha256, load_joint_alpha, pruning_query_report,
    score_revision_candidates)
from delta_ngram.alpha_memory import BlockAlpha, make_alpha_policy
from delta_ngram.checkpoints import load_checkpoint, save_checkpoint
from delta_ngram.data.event_streams import EventArticle, EventTrajectory
from delta_ngram.data.raw_text import TrajectoryWindow
from delta_ngram.raw_lm import evaluate_raw, trajectory_loss
from delta_ngram.recent_news_evaluation import evaluate_continuation
from delta_ngram.train import create_runtime
from test_alpha_lm import items
from test_raw_lm import config_for
from test_recent_news_evaluation import example, prepare, annotations, cli as plain_recent_cli
from test_evaluate_raw_checkpoint import evaluate_script as plain_raw_cli

SPEC = importlib.util.spec_from_file_location("evaluate_alpha_checkpoint",
    Path(__file__).resolve().parents[1] / "scripts/evaluate_alpha_checkpoint.py")
cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cli
SPEC.loader.exec_module(cli)


def paired_payload(example, variant="learned_global_099"):
    config, method, _, path, _, original = example
    old = load_checkpoint(original)
    alpha = make_alpha_policy(variant, method.reader.width)
    identity = {"protocol": "delta_ngram_alpha_lm_v1", "variant": variant, "alpha": alpha.metadata(),
                "config_sha256": file_sha256(config.source), "prefix_sha256": "a" * 64}
    metadata = {**old["config"], "requires_alpha_payload": True, "alpha_identity": identity}
    writer_path, alpha_path = path.parent / "alpha-writer.pt", path.parent / "alpha.pt"
    save_checkpoint(writer_path, method.writer, metadata, 3, protocol="delta_ngram_alpha_lm_v1")
    payload = {"format": "delta_ngram_alpha_policy_v1", "identity": identity, "step": 3,
        "variant": variant, "policy": alpha.metadata(), "state_dict": alpha.state_dict(),
        "writer_sha256": file_sha256(writer_path), "online_requires_frozen_writer_and_alpha": True}
    torch.save(payload, alpha_path)
    return writer_path, alpha_path, payload


@pytest.mark.parametrize("variant", ["fixed_099", "fixed_095", "learned_global_099", "evidence_block_099"])
def test_joint_loader_validates_pair_and_freezes_every_parameter(example, variant):
    config, *_ = example
    writer, alpha, _ = paired_payload(example, variant)
    runtime = load_joint_alpha(config, writer, alpha)
    runtime.verify_files()
    assert runtime.manifest["joint_identity_verified"]
    assert runtime.manifest["variant"] == variant
    assert all(not p.requires_grad and p.grad is None for module in
               (runtime.method.writer, runtime.method.reader.model, runtime.alpha) for p in module.parameters())


@pytest.mark.parametrize("mutation", ["writer_hash", "identity", "step", "policy", "config"])
def test_joint_mismatch_fails_before_runtime_model_creation(example, monkeypatch, mutation):
    import delta_ngram.alpha_evaluation as module
    config, *_ = example
    writer, alpha, payload = paired_payload(example)
    if mutation == "writer_hash":
        payload["writer_sha256"] = "0" * 64
    elif mutation == "identity":
        payload["identity"] = {**payload["identity"], "prefix_sha256": "b" * 64}
    elif mutation == "step":
        payload["step"] += 1
    elif mutation == "policy":
        payload["policy"] = {**payload["policy"], "initial_alpha": .95}
    else:
        config.source.write_bytes(config.source.read_bytes() + b"\n# changed\n")
    torch.save(payload, alpha)
    monkeypatch.setattr(module, "create_runtime", lambda *a: pytest.fail("malformed pair loaded a model"))
    with pytest.raises(ValueError):
        load_joint_alpha(config, writer, alpha)


def test_plain_d1_evaluators_refuse_alpha_writer(example):
    config, _, _, path, digest, _ = example
    writer, _, _ = paired_payload(example)
    with pytest.raises(ValueError, match="paired alpha.pt"):
        plain_raw_cli.evaluate_checkpoint(config, writer)
    with pytest.raises(ValueError, match="paired alpha.pt"):
        plain_recent_cli.evaluate_checkpoint(config, writer, path, registry_sha256=digest)


@pytest.mark.parametrize("kind", ["raw", "event", "mixed"])
@pytest.mark.parametrize("mode", ["self", "empty", "donor"])
def test_unit_alpha_historical_matches_d1_all_masks_diagnostics_and_donors(tmp_path, kind, mode):
    method, _ = create_runtime(config_for(tmp_path))
    windows = items(method, kind)
    donors = []
    for w in windows:
        if isinstance(w, EventTrajectory):
            donors.append(EventTrajectory("donor-" + w.document_id, "donor-" + w.group_id, w.source,
                tuple(EventArticle("donor-" + a.id, a.token_ids, {}) for a in w.articles), w.blocks, w.block_size))
        else:
            donors.append(TrajectoryWindow("donor-" + w.document_id, "donor-" + w.group_id,
                                          w.source, w.start, w.token_ids, w.blocks))
    kwargs = dict(prefix_tokens=4, memory_mode=mode, donor_windows=donors if mode == "donor" else None)
    backend = AlphaEvaluationBackend(BlockAlpha(method.reader.width, initial_alpha=1))
    backend.start(method)
    with torch.no_grad():
        expected = trajectory_loss(method, windows, **kwargs)
        actual = _historical_trajectory(method, windows, backend=backend, **kwargs)
    assert actual.counters == expected.counters
    assert actual.first_article_block == expected.first_article_block
    torch.testing.assert_close(actual.loss_sum, expected.loss_sum, rtol=0, atol=0)
    torch.testing.assert_close(actual.state.values, expected.state.values, rtol=0, atol=0)
    assert actual.loss_tokens == expected.loss_tokens
    backend.finish(method)


def test_unit_alpha_historical_summary_and_donor_manifest_match_core(tmp_path):
    method, _ = create_runtime(config_for(tmp_path))
    source = tuple(method.tokenizer.encode("abc xyz\n" * 60))
    windows = [TrajectoryWindow(name, name, "code", 0, source[:32], 4) for name in ("a", "b")]
    windows += [EventTrajectory(name, name, "news_event", tuple(
        EventArticle(name + str(i), source[:130], {}) for i in range(3)), 8, 128) for name in ("c", "d")]
    kwargs = dict(prefix_tokens=4, batch_size=2, bootstrap_samples=20, report_donor=True, report_per_source=True)
    expected = evaluate_raw(method, windows, **kwargs)
    backend = AlphaEvaluationBackend(BlockAlpha(method.reader.width, initial_alpha=1))
    actual = evaluate_alpha_historical(method, windows, backend, **kwargs)
    for key, value in expected.items():
        assert actual[key] == value, key
    assert actual["memory_backend"]["alpha_parameters_frozen_no_gradients"]


def test_unit_alpha_recent_same_distances_directions_and_scores(example, monkeypatch):
    _, method, *_ = example
    prepared = prepare(example)
    prefix = annotations(example, prepared)
    expected = evaluate_continuation(method, prepared, bootstrap_samples=10, prefix_annotations=prefix)
    backend = AlphaEvaluationBackend(BlockAlpha(method.reader.width, initial_alpha=1))
    def forbidden(*args, **kwargs):
        pytest.fail("online evaluation used optimizer/backward")
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    actual = evaluate_continuation(method, prepared, bootstrap_samples=10, prefix_annotations=prefix,
                                   memory_backend=backend)
    assert actual["retention_curve"] == expected["retention_curve"]
    assert actual["execution"] == expected["execution"]
    assert actual["direction_control"] == expected["direction_control"]
    assert actual["prefix_restatement_groups"] == expected["prefix_restatement_groups"]
    assert actual["memory_backend"]["query_advances_clock"] is False


def test_prune_reports_real_archive_cost_effective_rms_and_nll_without_live_mutation(example):
    _, method, *_ = example
    prepared = prepare(example)
    backend = AlphaEvaluationBackend(make_alpha_policy("fixed_095", method.reader.width))
    backend.start(method)
    state = backend.empty(method)
    state, _ = backend.write_article(method, state, prepared.tokens["a_early"], source_id="a_early")
    state, _ = backend.write_article(method, state, prepared.tokens["b_early"], source_id="b_early")
    before = state.values.clone(), state.block_clock.clone(), state.cumulative_log_alpha.clone()
    report = pruning_query_report(method, state, prepared.tokens["b_later"], source_id="b_later")
    assert report["live_lazy_state_unchanged"] and not report["online_ram_freed"] and not report["resumable"]
    assert report["thresholds"] == [0., 1e-4, 1e-3, 1e-2, 1e-1]
    assert report["reports"][0]["nll_difference_from_unpruned"] == 0
    for row in report["reports"]:
        assert row["serialized_bytes"] > row["raw_tensor_bytes"] >= 0
        assert row["readout"]["measured"] and row["effective_delta_rms_max"] >= 0
        assert row["query_score"]["target_tokens"] == len(prepared.tokens["b_later"]) - 128
    for left, right in zip(before, (state.values, state.block_clock, state.cumulative_log_alpha)):
        assert torch.equal(left, right)
    backend.finish(method)


@pytest.mark.parametrize("variant", ["fixed_099", "fixed_095", "learned_global_099", "evidence_block_099"])
def test_loader_rejects_changed_registered_constant(example, variant):
    config, *_ = example
    writer, alpha, payload = paired_payload(example, variant)
    payload["state_dict"]["fixed_log_alpha"] = torch.tensor(-.5, dtype=torch.float64)
    torch.save(payload, alpha)
    with pytest.raises(ValueError, match="fixed_log_alpha"):
        load_joint_alpha(config, writer, alpha)


def revision_annotations(example, prepared):
    from delta_ngram.recent_news_evaluation import prepare_prefix_annotations
    from test_recent_news_evaluation import write_json
    labels = annotations(example, prepared)
    value = json.loads(labels.path.read_bytes())
    value["annotations"][0]["revision"] = {"eligible": True, "type": "status_update",
        "property": "synthetic marker", "revision_query": "What is the current synthetic marker?",
        "earlier_answer": "abxy", "later_answer": "TARGET",
        "earlier_span": {"source_id": "a_early", "char_start": 0, "char_end_exclusive": 4,
                         "text_sha256": hashlib.sha256(b"abxy").hexdigest()},
        "later_span": {"source_id": "a_later", "char_start": 5, "char_end_exclusive": 11,
                       "text_sha256": hashlib.sha256(b"TARGET").hexdigest()}}
    digest = write_json(labels.path, value)
    return prepare_prefix_annotations(labels.path, prepared, example[1], expected_sha256=digest)


def test_revision_ordered_evidence_all_distances_queries_read_only_and_actual_traces(example, monkeypatch):
    _, method, *_ = example
    prepared = prepare(example)
    labels = revision_annotations(example, prepared)
    traces, reads = [], []
    backend = AlphaEvaluationBackend(make_alpha_policy("evidence_block_099", method.reader.width),
                                     alpha_observer=traces.append)
    def forbidden(*args, **kwargs):
        pytest.fail("revision evaluation used optimizer/backward")
    for name in ("AdamW", "Adam", "SGD"):
        monkeypatch.setattr(torch.optim, name, forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    def observe(row):
        assert not torch.is_grad_enabled()
        if row["kind"] == "write":
            start, ids = row["article_token_start"], row["ids"][0].tolist()
            assert ids == list(prepared.tokens[row["source_id"]][start:start + len(ids)])
        else:
            reads.append(row)
    result = evaluate_revision(method, prepared, labels, backend, bootstrap_samples=10, observer=observe)
    assert result["independent_families"] == 1
    assert result["optimizer_steps"] == result["backward_calls"] == 0
    assert set(result["retention_curve"]) == {"0", "1", "4", "8", "16"}
    assert set(result["by_revision_type"]) == {"status_update"}
    assert len(reads) == 21
    assert {r["stage"] for r in traces} == {"old_evidence", "old_plus_new_evidence"}
    assert {r["memory_mode"] for r in traces} == {"self", "donor"}
    assert all(r["pair_id"] == "a" and r["actual_new_tokens"] > 0 and r["raw_logit"] for r in traces)
    prefix_hashes = set()
    for row in result["per_family"]:
        distance = row["unrelated_article_count"]
        for stage, new in (("old_evidence", False), ("old_plus_new_evidence", True)):
            for arm, anchor in (("self", "a"), ("donor", "b")):
                score = row["stages"][stage][arm]
                written = [anchor + "_early"] + ([anchor + "_later"] if new and arm == "self" else [])
                written += ["bg" + str(i) for i in range(distance)]
                assert score["write_source_ids"] == written
                assert score["actual_written_tokens"] == sum(len(prepared.tokens[s]) for s in written)
                assert score["owner_block_clock"] == [score["actual_written_blocks"]]
                prefix_hashes.add(score["prefix_token_sha256"])
    assert len(prefix_hashes) == 1
    # The donor's later article is after target a's query time in this fixture.
    # Only its registered early article is a valid fixed donor control.
    assert all(row["source_id"] != "b_later" for row in traces)
    for row in result["per_family"]:
        assert row["stages"]["old_evidence"]["donor"] == row["stages"]["old_plus_new_evidence"]["donor"]


def test_revision_requires_explicit_curated_query_and_detects_clock_mutation(example, monkeypatch):
    import delta_ngram.news_eval as news_eval
    _, method, *_ = example
    prepared = prepare(example)
    labels = revision_annotations(example, prepared)
    del labels.by_pair["a"]["revision"]["revision_query"]
    with pytest.raises(ValueError, match="curator-written"):
        evaluate_revision(method, prepared, labels, bootstrap_samples=2)
    backend = AlphaEvaluationBackend(make_alpha_policy("fixed_099", method.reader.width))
    backend.start(method)
    state = backend.empty(method)
    original = news_eval.score_candidates
    def corrupt(*args):
        result = original(*args)
        args[1].block_clock.add_(1)
        return result
    monkeypatch.setattr(news_eval, "score_candidates", corrupt)
    with pytest.raises(RuntimeError, match="block_clock"):
        score_revision_candidates(method, state, "marker?", ["a", "b"])


def test_unequal_donor_article_boundaries_count_each_applied_alpha_segment(tmp_path):
    method, _ = create_runtime(config_for(tmp_path))
    tokens = tuple(method.tokenizer.encode("abc xyz\n" * 100))
    def event(name, lengths):
        return EventTrajectory(name, name, "news_event", tuple(
            EventArticle(name + str(i), tokens[:length], {}) for i, length in enumerate(lengths)), 8, 128)
    target, donor = event("target", (129, 130, 131)), event("donor", (140, 140, 140))
    traces = []
    backend = AlphaEvaluationBackend(BlockAlpha(method.reader.width, initial_alpha=1), alpha_observer=traces.append)
    backend.start(method)
    own = _historical_trajectory(method, [target], backend=backend, prefix_tokens=4)
    swapped = _historical_trajectory(method, [target], backend=backend, prefix_tokens=4,
                                     memory_mode="donor", donor_windows=[donor])
    with torch.no_grad():
        expected = trajectory_loss(method, [target], prefix_tokens=4, memory_mode="donor", donor_windows=[donor])
    assert own.counters["write_blocks"] == 5 and swapped.counters["write_blocks"] == 7
    assert own.state.block_clock.tolist() == [5] and swapped.state.block_clock.tolist() == [7]
    torch.testing.assert_close(swapped.loss_sum, expected.loss_sum, rtol=0, atol=0)
    assert sum(len(row["alpha"]) for row in traces if row["memory_mode"] == "donor") == 7
    assert all(row["owners"][0]["evidence_document_id"] == "donor"
               for row in traces if row["memory_mode"] == "donor")


@pytest.fixture
def trained_pair(tmp_path):
    from delta_ngram.alpha_training import build_m1_prefix, train_alpha
    from delta_ngram.raw_lm import train_raw
    documents = [{"id": str(i), "group_id": str(i), "source": "code",
        "split": "train" if i < 3 else "validation", "text": "abc xyz\n" * 24} for i in range(5)]
    (tmp_path / "documents.jsonl").write_text("\n".join(json.dumps(d) for d in documents), encoding="utf-8")
    config = config_for(tmp_path, "m1", raw={"loss_tokens": 128, "max_steps": 1,
        "eval_donor": True, "eval_bootstrap_samples": 10})
    summary = train_raw(config)
    prefix = tmp_path / "prefix.json"
    build_m1_prefix(config, m1_checkpoint=tmp_path / "m1/training.pt", m1_metrics=tmp_path / "m1/metrics.jsonl",
        writer_init=tmp_path / "m1/writer_init.pt", output=prefix, nominal_loss_tokens=summary["counters"]["loss_tokens"])
    train_alpha(config, variant="fixed_099", writer_init=tmp_path / "m1/writer_init.pt",
                prefix_manifest=prefix, output=tmp_path / "alpha")
    return config, prefix, tmp_path / "alpha/writer.pt", tmp_path / "alpha/alpha.pt", tmp_path / "m1/data_manifest.json"


def test_historical_cli_requires_actual_complete_prefix_and_exact_m1_validation(trained_pair, monkeypatch):
    config, prefix, writer, alpha, manifest = trained_pair
    def forbidden(*args, **kwargs):
        pytest.fail("evaluation used optimizer/backward")
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    result = cli.evaluate_checkpoint(config, writer, alpha, prefix_manifest=prefix,
                                     suite="historical", m1_validation_manifest=manifest)
    historical = result["results"]["historical"]
    assert historical["conditions"] == ["self", "empty", "donor"]
    assert historical["m1_validation_sample_and_protocol_exact_match"]
    assert result["m1_prefix"]["exact_complete_prefix_verified"]
    assert result["alpha_trace"] and not result["all_registered_suites_executed"]
    assert {row["memory_mode"] for row in result["alpha_trace"]} == {"self", "donor"}
    bad = json.loads(manifest.read_bytes())
    bad["validation_sample_sha256"] = "0" * 64
    manifest.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from M1"):
        cli.evaluate_checkpoint(config, writer, alpha, prefix_manifest=prefix,
                                suite="historical", m1_validation_manifest=manifest)


def test_alpha_cli_writes_immutable_output(trained_pair, tmp_path):
    config, prefix, writer, alpha, manifest = trained_pair
    output = tmp_path / "evaluated.json"
    argv = ["--config", str(config.source), "--checkpoint", str(writer), "--alpha", str(alpha),
        "--prefix-manifest", str(prefix), "--suite", "historical", "--m1-validation-manifest", str(manifest),
        "--threads", "1", "--output", str(output)]
    assert cli.main(argv) == 0
    result = json.loads(output.read_bytes())
    assert result["run"]["deterministic_algorithms"] and result["run"]["requested_threads"] == 1
    assert result["run"]["threads"] == torch.get_num_threads()
    with pytest.raises(FileExistsError):
        cli.main(argv)


def test_cli_rejects_partial_prefix_even_when_writer_and_alpha_pair_match(trained_pair):
    config, prefix, writer, alpha, manifest = trained_pair
    payload = load_checkpoint(writer)
    payload["step"] = 0
    payload["config"]["training_loss_tokens"] = 0
    torch.save(payload, writer)
    policy = torch.load(alpha, weights_only=True)
    policy.update(step=0, writer_sha256=file_sha256(writer))
    torch.save(policy, alpha)
    with pytest.raises(ValueError, match="not completed"):
        cli.evaluate_checkpoint(config, writer, alpha, prefix_manifest=prefix,
                                suite="historical", m1_validation_manifest=manifest)


def synthetic_news_prefix_pair(example):
    from delta_ngram.alpha_training import _json_hash
    config, method, *_ = example
    writer, alpha, payload = paired_payload(example, "fixed_099")
    old = load_checkpoint(writer)
    value = {"format": "delta_ngram_m1_optimizer_prefix_v1", "config_sha256": file_sha256(config.source),
        "m1_identity": old["config"]["raw_identity"], "writer_init_sha256": "1" * 64,
        "m1_checkpoint_sha256": "2" * 64, "full_loss_token_budget": config.values["raw"]["loss_tokens"],
        "stop_step": 3, "groups": [{}, {}, {}], "actual_loss_tokens": 192, "nominal_loss_tokens": 192}
    value["manifest_sha256"] = _json_hash(value)
    path = writer.parent / "prefix.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    identity = {**payload["identity"], "prefix_sha256": value["manifest_sha256"],
                "writer_init_sha256": value["writer_init_sha256"], "raw_identity_sha256": _json_hash(value["m1_identity"])}
    metadata = {**old["config"], "alpha_identity": identity, "training_loss_tokens": 192}
    save_checkpoint(writer, method.writer, metadata, 3, protocol="delta_ngram_alpha_lm_v1")
    payload.update(identity=identity, writer_sha256=file_sha256(writer))
    torch.save(payload, alpha)
    return writer, alpha, path


def test_news_cli_prepares_frozen_inputs_runs_pruning_and_revision_without_raw_corpus(example):
    config, _, _, registry, digest, _ = example
    prepared = prepare(example)
    labels = revision_annotations(example, prepared)
    writer, alpha, prefix = synthetic_news_prefix_pair(example)
    arguments = dict(prefix_manifest=prefix, registry_path=registry, registry_sha256=digest,
                     annotations_path=labels.path, annotations_sha256=labels.sha256, bootstrap_samples=2)
    continuation = cli.evaluate_checkpoint(config, writer, alpha, suite="continuation", **arguments)
    assert set(continuation["results"]) == {"continuation"}
    records = continuation["results"]["continuation"]["per_family"]
    assert len(records) == 10
    for row in records:
        for arm in ("self", "donor"):
            pruning = row["conditions"][arm]["memory_diagnostics"]["pruning"]
            assert len(pruning["reports"]) == 5
            assert pruning["live_lazy_state_unchanged"]
    assert all(r["evaluation"] == "continuation" and "source_id" in r for r in continuation["alpha_trace"])
    revision = cli.evaluate_checkpoint(config, writer, alpha, suite="revision", **arguments)
    assert revision["results"]["revision"]["independent_families"] == 1
    assert all(r["evaluation"] == "revision" and "stage" in r for r in revision["alpha_trace"])


@pytest.mark.parametrize("variant", ["fixed_095", "learned_global_099", "evidence_block_099"])
def test_nonunit_historical_matches_training_lazy_schedule(tmp_path, variant):
    from delta_ngram.alpha_lm import alpha_trajectory_loss
    method, _ = create_runtime(config_for(tmp_path))
    windows = items(method, "mixed")
    alpha = make_alpha_policy(variant, method.reader.width)
    with torch.no_grad():
        if alpha.head is not None:
            alpha.head.weight.fill_(.03)
        expected = alpha_trajectory_loss(method, windows, alpha, prefix_tokens=4)
    backend = AlphaEvaluationBackend(alpha)
    backend.start(method)
    actual = _historical_trajectory(method, windows, backend=backend, prefix_tokens=4)
    torch.testing.assert_close(actual.loss_sum, expected.loss_sum, rtol=0, atol=0)
    torch.testing.assert_close(actual.state.values, expected.state.values, rtol=0, atol=0)
    torch.testing.assert_close(actual.state.cumulative_log_alpha, expected.state.cumulative_log_alpha, rtol=0, atol=0)
    # The evaluator also scores the registered later-article first block with
    # a real prefix; the training loss path intentionally omits that read.
    diagnostic_input = sum(min(len(article.token_ids), window.block_size)
        for window in windows if isinstance(window, EventTrajectory)
        for article in window.articles[1:] if len(article.token_ids) > 4)
    assert actual.counters == {**expected.counters,
        "reader_input_tokens": expected.counters["reader_input_tokens"] + diagnostic_input}
