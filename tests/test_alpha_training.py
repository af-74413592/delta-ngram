"""Audited optimizer-prefix locking and exact tiny CPU alpha resume."""
import json
import importlib.util
from pathlib import Path

import pytest
import torch

from delta_ngram.alpha_training import (ALPHA_FORMAT, build_m1_prefix, file_sha256,
                                         freeze_alpha_online, train_alpha, _load)
from delta_ngram.alpha_memory import make_alpha_policy
from delta_ngram.raw_lm import _json_hash, train_raw
from delta_ngram.train import create_runtime
from test_raw_lm import config_for, assert_tree_equal


@pytest.fixture
def baseline(tmp_path):
    config = config_for(tmp_path, "m1", raw={"loss_tokens": 512, "max_steps": 3,
        "checkpoint_reader": True, "gradient_accumulation_steps": 2, "eval_every_tokens": 512})
    summary = train_raw(config, monitor_blocks=False)
    actual = summary["counters"]["loss_tokens"]
    manifest = tmp_path / "m1-prefix.json"
    result = build_m1_prefix(config, m1_checkpoint=tmp_path / "m1/training.pt",
        m1_metrics=tmp_path / "m1/metrics.jsonl", writer_init=tmp_path / "m1/writer_init.pt",
        output=manifest, nominal_loss_tokens=actual)
    return config, manifest, result, tmp_path / "m1/writer_init.pt"


@pytest.mark.parametrize("variant", ["fixed_099", "fixed_095", "learned_global_099", "evidence_block_099"])
def test_exact_optimizer_sampler_alpha_and_rng_resume(tmp_path, baseline, variant):
    config, manifest, prefix, writer_init = baseline
    kwargs = dict(config=config, variant=variant, writer_init=writer_init, prefix_manifest=manifest)
    full = train_alpha(**kwargs, output=tmp_path / "full")
    partial = train_alpha(**kwargs, output=tmp_path / "partial", stop_after_steps=1)
    resumed = train_alpha(**kwargs, output=tmp_path / "resumed", resume=tmp_path / "partial/training.pt")
    assert full["complete"] and resumed["complete"] and not partial["complete"]
    assert full["counters"] == resumed["counters"] == prefix["baseline_counters"]
    assert full["unchanged_full_loss_token_budget"] == 512 > full["actual_prefix_loss_tokens"]
    expected = torch.load(tmp_path / "full/training.pt", weights_only=True)
    restored = torch.load(tmp_path / "resumed/training.pt", weights_only=True)
    assert_tree_equal(expected, restored)
    alpha = torch.load(tmp_path / "full/alpha.pt", weights_only=True)
    assert alpha["format"] == ALPHA_FORMAT and alpha["online_requires_frozen_writer_and_alpha"]
    assert alpha["writer_sha256"] == file_sha256(tmp_path / "full/writer.pt")
    metrics = [json.loads(line) for line in (tmp_path / "full/metrics.jsonl").read_text().splitlines()]
    assert [r["batch_sha256"] for r in metrics] == [g["batch_sha256"] for g in prefix["groups"]]
    if variant.startswith("learned") or variant.startswith("evidence"):
        assert any(any(v > 0 for v in r["alpha_parameter_update_l2"].values()) for r in metrics)
    assert not full["evaluation_performed"]


def test_prefix_rejects_later_checkpoint_and_altered_actual_log(tmp_path, baseline):
    config, _, prefix, writer_init = baseline
    kwargs = dict(config=config, m1_checkpoint=tmp_path / "m1/training.pt",
        m1_metrics=tmp_path / "m1/metrics.jsonl", writer_init=writer_init, output=tmp_path / "bad-prefix.json")
    with pytest.raises(ValueError, match="first complete optimizer group"):
        build_m1_prefix(**kwargs, nominal_loss_tokens=1)
    rows = [json.loads(line) for line in (tmp_path / "m1/metrics.jsonl").read_text().splitlines()]
    next(r for r in rows if r["event"] == "train")["windows"][0]["start"] += 1
    damaged = tmp_path / "damaged.jsonl"
    damaged.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    kwargs["m1_metrics"] = damaged
    with pytest.raises(ValueError, match="sampler replay differs"):
        build_m1_prefix(**kwargs, nominal_loss_tokens=prefix["actual_loss_tokens"])


def test_prefix_rejects_shortened_budget_or_nonzero_init(tmp_path, baseline):
    config, manifest, prefix, writer_init = baseline
    changed = config_for(tmp_path, "changed", raw={**config.values["raw"], "loss_tokens": prefix["actual_loss_tokens"]})
    with pytest.raises(ValueError, match="full raw.loss_tokens"):
        train_alpha(changed, variant="fixed_099", writer_init=writer_init, prefix_manifest=manifest,
                    output=tmp_path / "changed-out")
    with pytest.raises(ValueError, match="step-zero"):
        train_alpha(config, variant="fixed_099", writer_init=tmp_path / "m1/writer.pt", prefix_manifest=manifest,
                    output=tmp_path / "changed-init")


def test_prefix_compares_strong_batch_identity_not_only_total_tokens(tmp_path, baseline):
    config, manifest, _, writer_init = baseline
    damaged = json.loads(manifest.read_text())
    damaged["groups"][0]["windows"][0]["token_sha256"] = "0" * 64
    damaged.pop("manifest_sha256")
    damaged["manifest_sha256"] = _json_hash(damaged)
    path = tmp_path / "damaged-prefix.json"
    path.write_text(json.dumps(damaged), encoding="utf-8")
    with pytest.raises(ValueError, match="batch differs"):
        train_alpha(config, variant="fixed_099", writer_init=writer_init, prefix_manifest=path,
                    output=tmp_path / "bad")


def test_resume_rejects_changed_variant_and_sampler_boundary(tmp_path, baseline):
    config, manifest, _, writer_init = baseline
    kwargs = dict(config=config, writer_init=writer_init, prefix_manifest=manifest)
    train_alpha(**kwargs, variant="learned_global_099", output=tmp_path / "partial", stop_after_steps=1)
    resume = tmp_path / "partial/training.pt"
    with pytest.raises(ValueError, match="resume identity differs"):
        train_alpha(**kwargs, variant="fixed_099", output=tmp_path / "wrong", resume=resume)
    payload = torch.load(resume, weights_only=True)
    payload["step"] = 2
    altered = tmp_path / "altered.pt"
    torch.save(payload, altered)
    with pytest.raises(ValueError, match="loss tokens differ"):
        train_alpha(**kwargs, variant="learned_global_099", output=tmp_path / "wrong-step", resume=altered)


def test_online_boundary_freezes_writer_and_policy(tmp_path):
    method, _ = create_runtime(config_for(tmp_path))
    alpha = make_alpha_policy("evidence_block_099", method.reader.width)
    freeze_alpha_online(method, alpha)
    assert not method.writer.training and not alpha.training
    assert not any(p.requires_grad for p in method.writer.parameters())
    assert not any(p.requires_grad for p in alpha.parameters())


def test_explicit_m1_init_allows_frozen_config_with_prior_init_path(tmp_path, baseline):
    config, manifest, _, writer_init = baseline
    # M1's frozen YAML retains its P1 initialization field. Alpha must load
    # the explicit M1 export and not open/reapply this stale initialization.
    config.values["run"]["checkpoint"] = str(tmp_path / "not-opened-p1-init.pt")
    result = train_alpha(config, variant="fixed_099", writer_init=writer_init,
        prefix_manifest=manifest, output=tmp_path / "zero", stop_after_steps=0)
    assert result["completed_steps"] == 0


def test_checkpoint_load_allows_torch_version_metadata(tmp_path):
    path = tmp_path / "version.pt"
    torch.save({"version": torch.__version__, "tensor": torch.ones(1)}, path)
    result = _load(path)
    assert str(result["version"]) == str(torch.__version__)
    assert torch.equal(result["tensor"], torch.ones(1))


def test_json_manifest_preserves_numeric_source_length_key_identity(tmp_path):
    config = config_for(tmp_path, "conditional", raw={"loss_tokens": 10_000, "max_steps": 1,
        "block_counts": [2, 4, 16], "source_weights": {"code": 1.0},
        "source_length_weights": {"code": {2: .4, 4: .3, 16: .3}}, "eval_every_tokens": 10_000})
    baseline = train_raw(config, monitor_blocks=False)
    out = tmp_path / "conditional"
    path = tmp_path / "conditional-prefix.json"
    prefix = build_m1_prefix(config, m1_checkpoint=out / "training.pt", m1_metrics=out / "metrics.jsonl",
        writer_init=out / "writer_init.pt", output=path, nominal_loss_tokens=baseline["counters"]["loss_tokens"])
    result = train_alpha(config, variant="fixed_099", writer_init=out / "writer_init.pt",
                         prefix_manifest=path, output=tmp_path / "conditional-alpha")
    assert result["complete"] and result["counters"] == prefix["baseline_counters"]


def test_cli_records_deterministic_launch_and_runs_frozen_prefix(tmp_path, baseline, monkeypatch):
    config, manifest, _, writer_init = baseline
    script = Path(__file__).parents[1] / "scripts/train_alpha_ablation.py"
    spec = importlib.util.spec_from_file_location("alpha_cli_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.subprocess, "check_output",
                        lambda args, **kwargs: "test-commit\n" if args[-1] == "HEAD" else "")
    for key in ("CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM"):
        monkeypatch.setenv(key, "test-restored-after-call")
    previous, threads = torch.are_deterministic_algorithms_enabled(), torch.get_num_threads()
    out = tmp_path / "cli"
    try:
        assert module.main(["train", "--config", str(config.source), "--variant", "fixed_099",
            "--writer-init", str(writer_init), "--prefix-manifest", str(manifest), "--output", str(out),
            "--stop-after-steps", "1", "--threads", "8"]) == 0
        launch = json.loads(out.with_name("cli.launch.json").read_text())
        assert launch["deterministic_algorithms"] is True and launch["threads"] == 8
        assert launch["cublas_workspace_config"] == ":4096:8"
        saved = torch.load(out / "training.pt", weights_only=True)
        assert saved["identity"]["execution"]["deterministic_algorithms"] is True
        assert saved["identity"]["execution"]["source_sha256"]
        torch.use_deterministic_algorithms(False)
        with pytest.raises(ValueError, match="resume identity differs"):
            train_alpha(config, variant="fixed_099", writer_init=writer_init, prefix_manifest=manifest,
                        output=tmp_path / "different-execution", resume=out / "training.pt")
    finally:
        torch.use_deterministic_algorithms(previous)
        torch.set_num_threads(threads)
