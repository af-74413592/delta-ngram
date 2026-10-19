"""CPU checks of original-only writes, causal targets, accounting, and identity."""
from copy import deepcopy
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from delta_ngram.checkpoints import save_checkpoint
from delta_ngram.config import ExperimentConfig
from delta_ngram.raw_lm import _json_hash, tokenizer_identity
from delta_ngram.train import create_runtime
import delta_ngram.recent_news_evaluation as news

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluate_recent_news_continuation", ROOT / "scripts/evaluate_recent_news_continuation.py")
cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cli
SPEC.loader.exec_module(cli)


def write_json(path, value):
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode())
    return news.file_sha256(path)


@pytest.fixture
def example(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"seed": 17, "model": {"smoke": True, "device": "cpu"},
        "reader": {"buckets": 17}, "writer": {"hidden_dim": 12},
        "raw": {"block_size": 512, "prefix_tokens": 128}}), encoding="utf-8")
    config = ExperimentConfig.load(config_path)
    method, args = create_runtime(config)
    originals = tmp_path / "originals"
    originals.mkdir()

    def source(name, text, timestamp, family=None):
        path = originals / (name + ".txt")
        path.write_bytes(text.encode())
        return {"id": name, "url": "https://example.org/" + name, "published_at": timestamp,
            "available_at": timestamp, "retrieved_at": "2026-10-19", "status": 200,
            "extracted_text_relpath": "originals/" + path.name, "extracted_text_sha256": news.file_sha256(path),
            "token_count": len(method.tokenizer.encode(text, add_special_tokens=False)),
            **({"event_family": family} if family else {})}

    sources = [source("a_early", "abxy" * 130, "2025-08-02"),
               source("a_later", "abxy " + "TARGET a xy\n" * 48, "2025-09-02"),
               source("b_early", "Independent B", "2025-08-03"),
               source("b_later", "B query " * 18, "2025-09-03")]
    background = [source("bg" + str(i), "Background " + str(i), "2025-07-01", "background" + str(i)) for i in range(17)]
    indexed = {s["id"]: s for s in sources + background}
    pairs = []
    for name, other in [("a", "b"), ("b", "a")]:
        early, later, donor = indexed[name + "_early"], indexed[name + "_later"], indexed[other + "_early"]
        pair = {"id": name, "event_family": name, "domain": "science", "earlier_source_id": early["id"],
            "later_source_id": later["id"], "query_at": later["available_at"],
            "donor_source_id": donor["id"], "donor_event_family": other,
            "earlier_write_token_count": early["token_count"], "earlier_write_blocks": (early["token_count"] + 511) // 512,
            "donor_write_token_count": donor["token_count"], "donor_write_blocks": (donor["token_count"] + 511) // 512,
            "scoring": {"text_source_id": later["id"], "initial_prefix_tokens": 128, "target_start_token": 128,
                "target_end_token_exclusive": later["token_count"], "maximum_read_window_tokens": 512,
                "maximum_target_tokens_per_window": 384, "score_all_remaining_tokens": True,
                "article_memory_write_allowed": False},
            "distractor_source_ids": [s["id"] for s in background[:16]], "interference_conditions": []}
        for count in news.DISTANCES:
            filler_tokens = sum(s["token_count"] for s in background[:count])
            condition = {"unrelated_article_count": count, "distractor_source_ids": pair["distractor_source_ids"][:count],
                "distractor_token_count": filler_tokens, "distractor_block_count": count,
                "empty_actual_written_articles": 0, "empty_actual_written_tokens": 0, "empty_actual_written_blocks": 0}
            for arm, anchor in [("self", early), ("donor", donor)]:
                condition.update({arm + "_actual_written_articles": count + 1,
                    arm + "_actual_written_tokens": filler_tokens + anchor["token_count"],
                    arm + "_actual_written_blocks": count + (anchor["token_count"] + 511) // 512})
            pair["interference_conditions"].append(condition)
        pairs.append(pair)
    registry = {"protocol": "recent_news_continuation_v1", "evaluation_only": True, "split": "dev",
        "model": {key: config.values["model"][key] for key in ("name_or_path", "revision")},
        "tokenization": {"revision": config.values["model"]["revision"], "add_special_tokens": False,
                         "write_block_tokens": 512, "maximum_tokens_per_write_article": None},
        "online_invariants": {"parameters_fixed": True, "optimizer_allowed": False,
                              "backward_allowed": False, "autograd_gradient_updates_allowed": False},
        "sources": sources, "background_sources": background, "pairs": pairs}
    registry_path = tmp_path / "registry.json"
    registry_hash = write_json(registry_path, registry)
    identity = tokenizer_identity(method)
    checkpoint = tmp_path / "writer.pt"
    metadata = {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "width": method.reader.width, "orders": list(method.reader.orders),
                "package_config": config.values, "raw_identity": {"tokenizer": identity},
                "tokenizer_fingerprint": "sha256:" + _json_hash(identity), "training_loss_tokens": 0}
    save_checkpoint(checkpoint, method.writer, metadata, 0)
    return config, method, registry, registry_path, registry_hash, checkpoint


def prepare(example):
    config, method, _, path, digest, _ = example
    return news.prepare_continuation(path, method, tokenizer_source=config.values["model"]["name_or_path"],
        tokenizer_revision=config.values["model"]["revision"], expected_sha256=digest)


def annotations(example, prepared):
    _, method, registry, path, digest, _ = example
    rows = []
    for pair in registry["pairs"]:
        ids = prepared.tokens[pair["later_source_id"]][:128]
        prefix = method.tokenizer.decode(ids, skip_special_tokens=False)
        early = "abxy" if pair["id"] == "a" else "Independent B"
        span = {"source_id": pair["earlier_source_id"], "char_start": 0,
                "char_end_exclusive": len(early), "text_sha256": hashlib.sha256(early.encode()).hexdigest()}
        fact = {"fact": "The synthetic marker is " + early, "earlier_span": span}
        if pair["id"] == "a":
            fact["later_prefix_span"] = {**span, "source_id": pair["later_source_id"]}
        rows.append({"pair_id": pair["id"], "event_family": pair["event_family"],
            "label": "restated" if pair["id"] == "a" else "not_restated",
            "rationale": "Synthetic fixture marker checked against the original prefix.", "key_facts": [fact],
            "later_prefix_token_ids_sha256": _json_hash(ids),
            "later_prefix_text_sha256": hashlib.sha256(prefix.encode()).hexdigest()})
    document = {"protocol": "recent_news_prefix_annotations_v1", "evaluation_only": True, "split": "dev",
        "parent_registry_file_sha256": digest, "tokenizer_revision": registry["model"]["revision"],
        "initial_prefix_tokens": 128, "model_results_seen": False, "annotations": rows}
    output = path.parent / "annotations.json"
    digest = write_json(output, document)
    return news.prepare_prefix_annotations(output, prepared, method, expected_sha256=digest)


def test_preparation_checks_frozen_bytes_tokens_and_portable_manifest(example):
    prepared = prepare(example)
    assert prepared.manifest["independent_families"] == 2
    assert prepared.manifest["archive_files_verified"] == 21
    assert all(s["token_ids_sha256"] for s in prepared.manifest["sources"])
    config, method, _, path, digest, _ = example
    with pytest.raises(ValueError, match="registry file SHA256"):
        news.prepare_continuation(path, method, tokenizer_source="x", tokenizer_revision="x", expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="tokenizer source/revision"):
        news.prepare_continuation(path, method, tokenizer_source="other", tokenizer_revision=config.values["model"]["revision"], expected_sha256=digest)
    text = path.parent / "originals/a_early.txt"
    text.write_bytes(text.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="evaluation input changed"):
        prepared.verify_files()
    with pytest.raises(ValueError, match="extracted_text SHA256"):
        prepare(example)


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r.update(split="final_holdout"), "only evaluation-only DEV"),
    (lambda r: r["sources"][0].update(available_at="2025-09-03", updated_at="2025-09-03"), "strictly causal"),
    (lambda r: r["pairs"][0].update(donor_source_id="a_later"), "donor must"),
    (lambda r: r["pairs"][0]["scoring"].update(article_memory_write_allowed=True), "target scoring"),
    (lambda r: r["sources"][0].update(token_count=1), "tokenizer count"),
    (lambda r: r["pairs"][0]["interference_conditions"][-1].update(self_actual_written_blocks=17), "self write budget"),
    (lambda r: r["sources"][0].update(extracted_text_relpath="../../outside.txt"), "path escapes"),
])
def test_invalid_registries_fail_before_evaluation(example, mutation, match):
    config, method, registry, path, _, _ = example
    mutation(registry)
    digest = write_json(path, registry)
    with pytest.raises(ValueError, match=match):
        news.prepare_continuation(path, method, tokenizer_source=config.values["model"]["name_or_path"],
            tokenizer_revision=config.values["model"]["revision"], expected_sha256=digest)


def test_windows_cover_full_target_once_and_never_future_context():
    windows = list(news.scoring_windows(1000))
    assert windows == [(0, 128, 512), (384, 512, 896), (768, 896, 1000)]
    covered = [token for _, start, end in windows for token in range(start, end)]
    assert covered == list(range(128, 1000))
    assert all(start - context == 128 and end - context <= 512 for context, start, end in windows)


def test_forward_only_all_arms_exact_writes_targets_and_frozen_parameters(example, monkeypatch):
    _, method, _, _, _, _ = example
    prepared = prepare(example)
    writes, reads = [], []
    def observe(row):
        assert not torch.is_grad_enabled()
        assert all(not p.requires_grad and p.grad is None for m in (method.writer, method.reader.model) for p in m.parameters())
        (writes if row["kind"] == "write" else reads).append(row)
    def forbidden(*args, **kwargs):
        pytest.fail("evaluation used an optimizer or backward")
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    before = {k: v.clone() for k, v in method.writer.state_dict().items()}
    result = news.evaluate_continuation(method, prepared, bootstrap_samples=40, observer=observe)
    assert result["optimizer_steps"] == result["backward_calls"] == 0
    assert len(result["per_family"]) == 10 and len(result["retention_curve"]) == 5
    assert {r["source_id"] for r in writes}.isdisjoint({"a_later", "b_later"})
    assert {r["source_id"] for r in reads} == {"a_later", "b_later"}
    assert len(writes) == result["execution"]["write_calls"] == 70
    assert all(torch.equal(v, method.writer.state_dict()[k]) for k, v in before.items())
    assert all(r["ids"].shape[1] <= 512 and torch.all(r["labels"][:, :128] == -100) for r in reads)
    # Every arm and distance uses exactly the same reader input and target mask.
    for source_id in ("a_later", "b_later"):
        matching = [r for r in reads if r["source_id"] == source_id]
        for start in {r["target_start"] for r in matching}:
            windows = [r for r in matching if r["target_start"] == start]
            assert len(windows) == 41  # empty plus five distances x two anchors x four direction views
            assert all(torch.equal(r["ids"], windows[0]["ids"]) and torch.equal(r["labels"], windows[0]["labels"]) for r in windows)
    # Article starts never hash across the preceding article; second blocks do.
    for row in writes:
        tokens = prepared.tokens[row["source_id"]]
        start = row["article_token_start"]
        context = max(0, start - max(method.reader.orders) + 1)
        ids = torch.tensor([tokens[context:start + row["ids"].shape[1]]])
        expected = method.reader.adapter.addresses(ids, torch.ones_like(ids))[:, start - context:]
        torch.testing.assert_close(row["rows"], expected)
    first = next(r for r in result["per_family"] if r["pair_id"] == "a" and r["unrelated_article_count"] == 16)
    assert first["conditions"]["self"]["actual_written_blocks"] == 18
    assert first["conditions"]["self"]["actual_written_articles"] == 17
    assert first["conditions"]["empty"]["actual_written_blocks"] == 0
    assert first["delta_donor"] == first["conditions"]["donor"]["nll"] - first["conditions"]["self"]["nll"]
    assert [p["seed"] for p in result["direction_control"]["plans"]] == [17, 29, 43]
    assert first["conditions"]["self_perm"]["nll"] == sum(
        r["nll"] for r in first["conditions"]["self_perm"]["per_seed"].values()) / 3
    # Permuted read views must not feed back into the future write trajectory.
    state = news._empty(method)
    for source_id in ["a_early", *prepared.registry["pairs"][0]["distractor_source_ids"]]:
        state, _ = news.write_article(method, state, prepared.tokens[source_id], source_id=source_id)
    replayed = news.score_later_article(method, state, prepared.tokens["a_later"], source_id="a_later")
    assert replayed["nll"] == first["conditions"]["self"]["nll"]


def test_query_does_not_write_and_future_perturbation_preserves_earlier_loss(example):
    _, method, _, _, _, _ = example
    prepared = prepare(example)
    state, _ = news.write_article(method, news._empty(method), prepared.tokens["a_early"], source_id="a_early")
    original = prepared.tokens["a_later"]
    changed = original[:512] + tuple(method.tokenizer.encode("Z" * (len(original) - 512)))
    first = news.score_later_article(method, state, original, source_id="a_later")
    later = news.score_later_article(method, state, changed, source_id="a_later")
    assert first["first_target_loss_sum"] == later["first_target_loss_sum"]
    assert first["target_tokens"] == len(original) - 128


def test_memory_mutation_during_query_is_detected(example, monkeypatch):
    _, method, _, _, _, _ = example
    prepared = prepare(example)
    state, _ = news.write_article(method, news._empty(method), prepared.tokens["a_early"], source_id="a_early")
    original = news.score_block
    def altered(reader, state, *args, **kwargs):
        value = original(reader, state, *args, **kwargs)
        state.values.add_(1)
        return value
    monkeypatch.setattr(news, "score_block", altered)
    with pytest.raises(RuntimeError, match="query reader mutated memory"):
        news.score_later_article(method, state, prepared.tokens["a_later"], source_id="a_later")


def test_family_macro_and_retention_bootstrap_are_paired_not_token_weighted():
    rows = []
    for family, count, self_nll, gain in [("a", 10, 1., 2.), ("b", 1000, 3., 6.)]:
        for distance in news.DISTANCES:
            loss = distance / 16
            conditions = {}
            for arm, nll in [("self", self_nll + loss), ("empty", self_nll + gain), ("donor", self_nll + gain / 2)]:
                conditions[arm] = {"nll": nll, "loss_sum": nll * count, "target_tokens": count,
                    "first_target_nll": nll, "actual_written_articles": 0, "actual_written_tokens": 0, "actual_written_blocks": 0}
            rows.append({"event_family": family, "unrelated_article_count": distance, "conditions": conditions,
                         "delta_empty": gain - loss, "delta_donor": gain / 2 - loss})
    result = news.summarize_continuation(rows, bootstrap_samples=100)
    assert result == news.summarize_continuation(rows, bootstrap_samples=100)
    assert result["retention_curve"][0]["self_nll"] == 2
    assert result["retention_curve"][0]["delta_empty"] == 4
    assert result["retention_curve"][-1]["retention_change_empty"] == -1
    assert result["retention_curve"][-1]["retention_change_empty_ci95"] == [-1, -1]


@pytest.mark.parametrize("mutation,match", [("writer_data", "writer state_dict changed"),
                                           ("reader", "reader parameters changed")])
def test_parameter_mutation_is_detected(example, mutation, match):
    _, method, _, _, _, _ = example
    prepared = prepare(example)
    changed = False
    def observe(row):
        nonlocal changed
        if not changed and row["kind"] == "read":
            changed = True
            if mutation == "writer_data":
                next(method.writer.parameters()).data.add_(1)
            else:
                next(method.reader.model.parameters()).add_(1)
    with pytest.raises(RuntimeError, match=match):
        news.evaluate_continuation(method, prepared, bootstrap_samples=10, observer=observe)


def test_unpinned_revision_and_dirty_recorded_run_fail_before_model_loading(example, monkeypatch):
    import delta_ngram.train as training
    config, _, _, path, digest, checkpoint = example
    monkeypatch.setattr(training, "create_runtime", lambda *args: pytest.fail("loaded reader before identity check"))
    config.values["model"].update(smoke=False, revision="main")
    with pytest.raises(ValueError, match="immutable model commit"):
        cli.evaluate_checkpoint(config, checkpoint, path, registry_sha256=digest)
    config.values["model"]["revision"] = "a" * 40
    monkeypatch.setattr(cli, "git_identity", lambda: {"commit": "a" * 40, "dirty": True})
    with pytest.raises(RuntimeError, match="Commit repository changes"):
        cli.evaluate_checkpoint(config, checkpoint, path, registry_sha256=digest,
                                annotations_path="unread.json", annotations_sha256="0" * 64)


def test_frozen_prefix_spans_and_grouped_direction_statistics(example):
    _, method, _, _, _, _ = example
    prepared = prepare(example)
    labels = annotations(example, prepared)
    result = news.evaluate_continuation(method, prepared, prefix_annotations=labels, bootstrap_samples=20)
    assert result["prefix_restatement_groups"]["restated"]["families"] == 1
    assert result["prefix_restatement_groups"]["not_restated"]["families"] == 1
    assert result["prefix_restatement_groups"]["unknown"]["families"] == 0
    assert result["manifest"]["prefix_annotations"]["file_sha256"] == labels.sha256
    document = json.loads(labels.path.read_bytes())
    document["annotations"][0]["key_facts"][0]["later_prefix_span"]["char_end_exclusive"] = 129
    digest = write_json(labels.path, document)
    with pytest.raises(ValueError, match="outside fixed prefix"):
        news.prepare_prefix_annotations(labels.path, prepared, method, expected_sha256=digest)


def test_three_permutation_seeds_are_averaged_within_each_family(example):
    _, method, _, _, _, _ = example
    result = news.evaluate_continuation(method, prepare(example), bootstrap_samples=20)
    row = result["retention_curve"][0]
    expected = sum(record["direction_delta_self"] for record in result["per_family"]
                   if record["unrelated_article_count"] == 0) / 2
    assert row["direction_delta_self"] == expected
    assert row["families"] == 2  # not six independent seed observations
    assert set(row["direction_by_seed"]) == {"17", "29", "43"}


def test_checkpoint_cli_style_identity_immutability_and_no_training_data(example):
    config, _, _, registry_path, digest, checkpoint = example
    original_values = deepcopy(config.values)
    # The evaluation checkpoint argument overrides unrelated initialization/resume paths.
    config.values["run"]["checkpoint"] = "missing.pt"
    config.values["raw"]["resume"] = "missing-resume.pt"
    config.values["raw"]["dataset"] = "must-not-be-read.jsonl"
    result = cli.evaluate_checkpoint(config, checkpoint, registry_path, registry_sha256=digest, bootstrap_samples=20)
    assert result["complete"] and result["model"]["matches_registry_reader"]
    assert result["checkpoint"]["sha256"] == news.file_sha256(checkpoint)
    assert result["manifest"]["registry_file_sha256"] == digest
    assert result["checkpoint"]["checkpoint_tokenizer_fingerprint_verified"]
    assert result["integrity_checks"]["target_never_written"]
    assert config.values["model"] == original_values["model"]


def test_cli_output_is_immutable_and_deterministic_settings_recorded(example, tmp_path):
    config, _, _, registry_path, digest, checkpoint = example
    output = tmp_path / "result.json"
    command = [sys.executable, str(ROOT / "scripts/evaluate_recent_news_continuation.py"),
        "--config", str(config.source), "--checkpoint", str(checkpoint), "--registry", str(registry_path),
        "--registry-sha256", digest, "--output", str(output), "--threads", "2", "--bootstrap-samples", "20"]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["run"]["deterministic_algorithms"]
    assert result["run"]["cublas_workspace_config"] == ":4096:8"
    previous = output.read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert repeated.returncode != 0 and "FileExistsError" in repeated.stderr
    assert previous == output.read_bytes()
