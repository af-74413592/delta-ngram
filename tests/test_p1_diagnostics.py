"""Target shifts, frozen interventions, paired groups, and honest next-block coverage."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delta_ngram.checkpoints import save_checkpoint
from delta_ngram.config import ExperimentConfig
from delta_ngram.data.raw_text import RawDocument, TrajectoryWindow
from delta_ngram.diagnostic_manifest import build_diagnostic_manifest
from delta_ngram.p1_diagnostics import (evaluate_diagnostics, load_diagnostic_checkpoint,
                                       per_target_cross_entropy, recurrence_mask, _summarize)
from delta_ngram.raw_lm import (PROTOCOL, _identity, _json_hash, _writer_metadata,
                               evaluate_raw, tokenizer_identity, trajectory_loss)
from delta_ngram.train import create_runtime


def setup_case(tmp_path, *, document_blocks=18, second_language=False):
    values = {"seed": 17, "model": {"smoke": True, "device": "cpu"},
              "reader": {"buckets": 17}, "writer": {"hidden_dim": 12},
              "raw": {"block_size": 4, "prefix_tokens": 2, "block_counts": [2, 4],
                      "batch_size": 4, "eval_windows": 2, "eval_seed": 104729,
                      "eval_bootstrap_samples": 31}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(values), encoding="utf-8")
    config = ExperimentConfig.load(config_path)
    method, args = create_runtime(config)
    fingerprint = "sha256:" + _json_hash(tokenizer_identity(method))

    def document(name, offset, *, source="code/python", split="validation"):
        tokens = tuple(3 + (i * 7 + offset) % 83 for i in range(document_blocks * 4))
        return RawDocument(name, "group-" + name, source, split, tokens, tokenizer_identity=fingerprint)

    documents = [document("target", 0), document("donor-a", 13), document("donor-b", 31)]
    targets = [documents[0]]
    if second_language:
        documents.extend([document("java-target", 4, source="code/java"),
                          document("java-dev", 17, source="code/java"),
                          document("java-train", 39, source="code/java", split="train")])
        targets.append(documents[-3])
    samples = [{"document_id": doc.id, "group_id": doc.group_id, "source": doc.source,
                "start": 0, "blocks": 4, "loss_tokens": 12,
                "token_sha256": _json_hash(doc.token_ids[:16])} for doc in targets]
    raw = config.values["raw"]
    validation = {"name": "group_first_fixed_dev_paired_bootstrap_v1", "selection": "group_first_round_robin",
                  "group_key": "group_id", "seed": raw["eval_seed"], "window_limit": raw["eval_windows"],
                  "bootstrap_samples": raw["eval_bootstrap_samples"], "minimum_groups": 64,
                  "evaluation_batching": "within_group", "batch_size": 4, "prefix_tokens": 2}
    manifest = build_diagnostic_manifest(documents, samples, block_size=4, prefix_tokens=2,
                                          dataset_sha256="a" * 64,
                                          baseline_validation_protocol_sha256=_json_hash(validation))
    return config, method, args, documents, manifest


def checkpoint(tmp_path, config, method, args, manifest):
    identity = _identity(config, method, manifest["dataset_sha256"], tokenizer_identity(method))
    identity["validation"] = {"validation_protocol_sha256": manifest["baseline_validation_protocol_sha256"],
                              "validation_sample_sha256": manifest["baseline_validation_sample_sha256"]}
    metadata = _writer_metadata(config, method, args, identity)
    metadata["training_loss_tokens"] = 1232
    path = tmp_path / "writer.pt"
    save_checkpoint(path, method.writer, metadata, 7, protocol=PROTOCOL)
    return path, json.loads(json.dumps(identity))


def test_per_token_ce_uses_previous_logit_and_ignores_prefix_and_last_logit():
    logits = torch.tensor([[[8., 0., 0.], [0., 8., 0.], [0., 0., 8.], [1., 2., 3.]]])
    labels = torch.tensor([[-100, -100, 1, 2]])
    losses, mask = per_target_cross_entropy(logits, labels)
    assert mask.tolist() == [[False, True, True]]
    expected = -F.log_softmax(logits.float(), dim=-1)[0, [1, 2], [1, 2]]
    torch.testing.assert_close(losses[mask], expected, rtol=0, atol=0)
    changed = logits.clone()
    changed[:, -1] = 1000 * torch.randn_like(changed[:, -1])
    torch.testing.assert_close(per_target_cross_entropy(changed, labels)[0], losses, rtol=0, atol=0)
    assert losses[0, 0] == 0


def test_recurrence_excludes_direct_prefix_current_block_and_future():
    tokens = (5, 6, 7, 8, 8, 5, 9, 9, 9, 10, 11, 12)
    assert recurrence_mask(tokens, 1, 4, 1) == (False, True, False, False)
    assert recurrence_mask(tokens[:8] + (5, 5, 5, 5), 1, 4, 1) == (False, True, False, False)


def test_all_conditions_same_targets_causal_order_and_real_compute_ledger(tmp_path, monkeypatch):
    config, method, args, documents, manifest = setup_case(tmp_path)
    original = {name: value.detach().clone() for name, value in method.writer.state_dict().items()}
    target = documents[0]
    item = TrajectoryWindow(target.id, target.group_id, target.source, 0, target.token_ids[:16], 4)
    with torch.no_grad():
        self_expected = float(trajectory_loss(method, [item], prefix_tokens=2).mean)
        empty_expected = float(trajectory_loss(method, [item], prefix_tokens=2, memory_mode="empty").mean)
    original_evaluation = evaluate_raw(method, [item], prefix_tokens=2, batch_size=1, bootstrap_samples=31)
    inputs = []
    forward = method.reader.forward

    def capture(ids, mask, state):
        inputs.append(ids.detach().cpu().tolist()[0])
        return forward(ids, mask, state)

    monkeypatch.setattr(method.reader, "forward", capture)
    blocks, progress = [], []
    result = evaluate_diagnostics(method, documents, manifest, bootstrap_samples=31,
                                  block_callback=blocks.append, progress_callback=progress.append)
    assert result["complete"] and result["primary"]["loss_tokens"] == 12
    conditions = result["primary"]["conditions"]
    assert set(conditions) == {"self", "empty", "context", "donor_1", "donor_2", "donor_mean"}
    assert all(item["loss_tokens"] == 12 for item in conditions.values())
    assert conditions["self"]["nll"] == pytest.approx(self_expected, abs=1e-6)
    assert conditions["empty"]["nll"] == pytest.approx(empty_expected, abs=1e-6)
    assert conditions["self"]["nll"] == pytest.approx(original_evaluation["self_nll"], abs=1e-6)
    assert conditions["empty"]["nll"] == pytest.approx(original_evaluation["empty_nll"], abs=1e-6)
    assert result["primary"]["comparisons"]["delta_empty"]["delta"] == pytest.approx(
        original_evaluation["delta_empty"], abs=1e-6)
    assert conditions["donor_mean"]["loss_sum"] == pytest.approx(
        (conditions["donor_1"]["loss_sum"] + conditions["donor_2"]["loss_sum"]) / 2)
    for block in range(1, 4):
        short = list(target.token_ids[block * 4 - 2:(block + 1) * 4])
        assert inputs[(block - 1) * 5:(block - 1) * 5 + 4] == [short] * 4
        assert inputs[(block - 1) * 5 + 4] == list(target.token_ids[:(block + 1) * 4])
    for condition in ("self", "donor_1", "donor_2"):
        writes = [item for item in blocks if item["cohort"] == "primary" and item["kind"] == "write"
                  and item["condition"] == condition]
        assert [item["block"] for item in writes] == [0, 1, 2]
        for block in range(1, 4):
            query_pos = next(i for i, record in enumerate(blocks) if record["cohort"] == "primary"
                             and record["kind"] == "query" and record["condition"] == condition and record["block"] == block)
            assert blocks.index(writes[block - 1]) < query_pos
            if block < 3:
                assert query_pos < blocks.index(writes[block])
    assert [r["target_block_number"] for r in blocks if r["cohort"] == "after_write"
            and r["kind"] == "query" and r["condition"] == "self"] == [5, 9, 17]
    ledger = result["budget"]["by_cohort_and_condition"]
    assert ledger["primary"]["self"]["reader_input_tokens"] == 18
    assert ledger["primary"]["context"]["reader_input_tokens"] == 8 + 12 + 16
    assert ledger["primary"]["self"]["evidence_encoded_tokens"] == 12
    assert ledger["primary"]["context"]["written_tokens"] == 0
    assert ledger["after_write"]["self"]["written_tokens"] == 16 * 4
    assert ledger["after_write"]["self"]["scored_targets"] == 3 * 4
    assert progress[-1]["stage"] == "complete"
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in method.parameters())
    assert all(parameter.grad is None for parameter in method.reader.model.parameters())
    assert all(torch.equal(original[name], value) for name, value in method.writer.state_dict().items())
    assert result["by_recurrence"]["present_before_local_prefix"]["loss_tokens"] + result["by_recurrence"]["absent_before_local_prefix"]["loss_tokens"] == 12
    json.dumps(result, allow_nan=False)
    json.dumps(blocks, allow_nan=False)


def test_after_write_missing_natural_target_is_explicit_and_not_replaced(tmp_path):
    _, method, _, documents, manifest = setup_case(tmp_path, document_blocks=8)
    records = []
    result = evaluate_diagnostics(method, documents, manifest, bootstrap_samples=7, block_callback=records.append)
    assert result["after_write"]["4"]["metrics"]["loss_tokens"] == 4
    for point in ("8", "16"):
        assert result["after_write"][point]["coverage"]["available_targets"] == 0
        assert result["after_write"][point]["metrics"]["loss_tokens"] == 0
        assert result["after_write"][point]["metrics"]["comparisons"]["delta_empty"]["status"] == "unavailable"
    assert {item["n_written"] for item in result["after_unavailable_targets"]} == {8, 16}
    assert max(item["block"] for item in records if item["cohort"] == "after_write") == 4


def test_train_donor_fallback_has_dev_only_sensitivity_without_more_inference(tmp_path):
    _, method, _, documents, manifest = setup_case(tmp_path, second_language=True)
    result = evaluate_diagnostics(method, documents, manifest, bootstrap_samples=11)
    assert result["primary"]["windows"] == 2
    assert result["all_dev_donors"]["windows"] == 1
    assert result["all_dev_donors"]["loss_tokens"] == 12
    assert result["primary_coverage"]["donor_train_fallback_slots"] == 1
    assert set(result["by_language"]) == {"code/java", "code/python"}
    assert result["budget"]["by_cohort_and_condition"]["primary"]["donor_2"]["score_calls"] == 6


def test_insufficient_two_donors_keeps_all_primary_targets_and_reports_unavailable(tmp_path):
    _, method, _, documents, manifest = setup_case(tmp_path)
    documents = documents[:2]
    manifest = build_diagnostic_manifest(documents, manifest["baseline_validation_sample"], block_size=4,
                                          prefix_tokens=2, donor_pool_policy="validation_only")
    result = evaluate_diagnostics(method, documents, manifest, bootstrap_samples=7)
    assert result["primary"]["loss_tokens"] == 12
    assert set(result["primary"]["conditions"]) == {"self", "empty", "context"}
    assert result["primary"]["comparisons"]["delta_donor"]["status"] == "unavailable"


def test_group_bootstrap_weights_tokens_and_averages_donors_before_difference():
    records = [{"window_id": "short", "group_id": "short", "source": "code/python", "loss_tokens": 1,
                "loss_sums": {"self": 1., "empty": 3., "donor_mean": 2., "context": .5}},
               {"window_id": "long", "group_id": "long", "source": "code/python", "loss_tokens": 9,
                "loss_sums": {"self": 18., "empty": 9., "donor_mean": 9., "context": 8.}}]
    result = _summarize(records, seed=4, samples=2000)
    assert result["comparisons"]["delta_empty"]["delta"] == pytest.approx(-.7)
    assert result["comparisons"]["delta_donor"]["delta"] == pytest.approx(-.8)
    assert result["comparisons"]["delta_empty"]["ci_lower"] == -1
    assert result["comparisons"]["delta_empty"]["ci_upper"] == 2
    assert result["comparisons"]["delta_empty"]["status"] == "insufficient_independent_groups"


def test_checkpoint_load_binds_training_identity_and_baseline_recompute(tmp_path):
    config, method, args, documents, manifest = setup_case(tmp_path)
    path, identity = checkpoint(tmp_path, config, method, args, manifest)
    metadata = load_diagnostic_checkpoint(method, config, path, expected_identity=identity)
    assert metadata["checkpoint_step"] == 7 and metadata["training_loss_tokens"] == 1232
    baseline = {"validation_sample_sha256": manifest["baseline_validation_sample_sha256"],
                "validation_protocol_sha256": manifest["baseline_validation_protocol_sha256"],
                "step": 7, "training_loss_tokens": 1232, "loss_tokens": 12, "self_nll": 5., "empty_nll": 5.1}
    result = evaluate_diagnostics(method, documents, manifest, bootstrap_samples=7,
                                  checkpoint_metadata=metadata, baseline_evaluation=baseline)
    assert result["original_evaluation_batch_size"] == 4 and result["evaluation_batch_size"] == 1
    assert result["baseline_numerical_comparison"]["status"] == "evaluated"
    current = result["primary"]["conditions"]["self"]["nll"]
    assert result["baseline_numerical_comparison"]["self"]["difference"] == current - 5.
    wrong = deepcopy(identity)
    wrong["seed"] += 1
    with pytest.raises(ValueError, match="selected training run"):
        load_diagnostic_checkpoint(method, config, path, expected_identity=wrong)
    wrong = deepcopy(metadata)
    wrong["raw_identity"]["validation"]["validation_sample_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="fixed development identity"):
        evaluate_diagnostics(method, documents, manifest, checkpoint_metadata=wrong)
    baseline["step"] = 6
    with pytest.raises(ValueError, match="loaded writer checkpoint"):
        evaluate_diagnostics(method, documents, manifest, checkpoint_metadata=metadata, baseline_evaluation=baseline)


def test_tampered_manifest_rejected_before_inference_and_callback_can_abort(tmp_path, monkeypatch):
    _, method, _, documents, manifest = setup_case(tmp_path)
    altered = deepcopy(manifest)
    altered["primary_windows"][0]["donors"][0]["start"] += 4
    with pytest.raises(ValueError, match="content hash"):
        evaluate_diagnostics(method, documents, altered)
    before = {name: value.detach().clone() for name, value in method.writer.state_dict().items()}

    def abort(record):
        if record["kind"] == "paired_score":
            raise TimeoutError("external wall budget")

    with pytest.raises(TimeoutError, match="external wall budget"):
        evaluate_diagnostics(method, documents, manifest, block_callback=abort)
    assert all(torch.equal(before[name], value) for name, value in method.writer.state_dict().items())


def test_context_length_guard_refuses_silent_history_truncation(tmp_path):
    _, method, _, documents, manifest = setup_case(tmp_path)
    method.reader.model.config.max_position_embeddings = 12
    with pytest.raises(ValueError, match="refusing to truncate"):
        evaluate_diagnostics(method, documents, manifest, bootstrap_samples=7)
