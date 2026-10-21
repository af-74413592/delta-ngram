"""Isolation, pinned-source, and conditional cache-spec patch checks."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/native_qwen38/prepare_vllm_local.py"
spec = importlib.util.spec_from_file_location("prepare_vllm_local_under_test", SCRIPT)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

PLE = '''import torch
class Qwen4ExpPLELayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        compilation_config = type("Config", (), {"static_forward_context": {}})()
        prefix = "tiny.ple"
        compilation_config.static_forward_context[prefix] = self

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def forward(
        self, hidden_states, input_ids, query_start_loc, ngram_context,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1)
        if input_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError("shape")
        return hidden_states + 3
'''
MODEL = '''import torch
class Model:
    @classmethod
    def get_ple_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_gdn_mamba_state_dtype_from_config(cls, config):
        return (torch.bfloat16, torch.float32)
'''


@pytest.fixture
def installed(tmp_path, monkeypatch):
    source = tmp_path / "site-packages/vllm"
    source.mkdir(parents=True)
    inputs = {"models/qwen4_exp/nvidia/ple_layer.py": PLE,
              "models/qwen4_exp/nvidia/model.py": MODEL}
    pins, files = {}, {}
    for relative, text in inputs.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        pins[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        files["vllm/" + relative] = {"bytes": path.stat().st_size, "sha256": pins[relative]}
    monkeypatch.setattr(prepare, "PATCH_SHA256", pins)
    (source / "_C.abi3.so").write_bytes(b"stand-in compiled extension for copy isolation")
    (source / "__init__.py").write_text("__version__ = '0.30.0'\n", encoding="utf-8")
    (source / "__pycache__").mkdir()
    (source / "__pycache__/old.pyc").write_bytes(b"stale")
    metadata = source.parent / "vllm-0.30.0.dist-info/METADATA"
    metadata.parent.mkdir()
    metadata.write_text("Name: vllm\nVersion: 0.30.0\n", encoding="utf-8")
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(json.dumps({"files": files}), encoding="utf-8")
    repo = tmp_path / "repo"
    for name in ("qwen38_vllm_ple.py", "qwen38_vllm_snapshot.py"):
        helper = repo / "src/delta_ngram" / name
        helper.parent.mkdir(parents=True, exist_ok=True)
        helper.write_text('"""Helper identity fixture."""\n', encoding="utf-8")
    monkeypatch.setattr(prepare, "REPO_ROOT", repo)
    monkeypatch.delenv("DELTA_NGRAM_VLLM_BINDING", raising=False)
    monkeypatch.delenv("DELTA_NGRAM_VLLM_BINDING_SHA256", raising=False)
    return source, manifest, tmp_path / "fork"


def test_prepare_copies_independent_files_and_binds_the_actual_result(installed):
    source, manifest, output = installed
    originals = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    receipt = prepare.prepare_fork(source, manifest, output)
    assert receipt["upstream_version"] == "0.30.0"
    assert Path(receipt["package_root"]) == output / "vllm"
    assert set(receipt["patched_files"]) == set(prepare.PATCH_SHA256)
    assert (output / "vllm/_C.abi3.so").read_bytes() == (source / "_C.abi3.so").read_bytes()
    assert not (output / "vllm/__pycache__").exists()
    for path, data in originals.items():
        assert path.read_bytes() == data
    for relative, record in receipt["files"].items():
        path = output / "vllm" / relative
        assert record == {"bytes": path.stat().st_size, "sha256": prepare.sha256(path)}
    # Writing the fork cannot mutate an upstream hardlink/symlink target.
    (output / "vllm/_C.abi3.so").write_bytes(b"changed only in copied fork")
    assert (source / "_C.abi3.so").read_bytes() == originals[source / "_C.abi3.so"]
    assert json.loads((output / "fork_manifest.json").read_text())["source_modified"] is False


def test_changed_upstream_source_fails_before_any_fork_write(installed):
    source, manifest, output = installed
    path = source / "models/qwen4_exp/nvidia/ple_layer.py"
    path.write_bytes(path.read_bytes() + b"\n# upstream changed\n")
    with pytest.raises(ValueError, match="differs from the read-only export"):
        prepare.prepare_fork(source, manifest, output)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["existing", "inside_source", "source_ancestor"])
def test_output_must_be_new_and_outside_source(installed, kind):
    source, manifest, output = installed
    if kind == "existing":
        output.mkdir()
    elif kind == "inside_source":
        output = source / "fork"
    else:
        output = source.parent
    with pytest.raises(ValueError, match="new directory outside"):
        prepare.prepare_fork(source, manifest, output)


def test_partial_python_only_export_cannot_be_mistaken_for_an_installed_fork(installed):
    source, manifest, output = installed
    (source / "_C.abi3.so").unlink()
    with pytest.raises(ValueError, match="complete installed package"):
        prepare.prepare_fork(source, manifest, output)
    assert not output.exists()


def test_patch_checks_content_even_without_external_manifest(installed):
    source, _, _ = installed
    relative = "models/qwen4_exp/nvidia/ple_layer.py"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepare.patch_source(relative, (source / relative).read_bytes() + b" ")


def namespace_for(raw):
    class Calculator:
        @staticmethod
        def short_conv_state_dtype(*args):
            return (torch.bfloat16,)

    namespace = {"MambaStateDtypeCalculator": Calculator, "VllmConfig": object}
    exec(compile(raw, "patched_fixture.py", "exec"), namespace)
    return namespace


def test_disabled_branch_preserves_native_result_and_cache_dtype(installed, monkeypatch):
    source, _, _ = installed
    relative = "models/qwen4_exp/nvidia/ple_layer.py"
    namespace = namespace_for(prepare.patch_source(relative, (source / relative).read_bytes()))
    ple = namespace["Qwen4ExpPLELayer"]()
    assert ple.weight.requires_grad
    ple.model_config = type("Config", (), {"dtype": torch.bfloat16})()
    ple.cache_config = type("Cache", (), {"mamba_cache_dtype": "auto"})()
    hidden = torch.ones(2, 8)
    assert torch.equal(ple.forward(hidden, torch.tensor([1, 2]), None, None), hidden + 3)
    assert ple.get_state_dtype() == (torch.bfloat16,)
    monkeypatch.setenv("DELTA_NGRAM_VLLM_BINDING", "/new/binding.json")
    with pytest.raises(RuntimeError, match="changed after vLLM import"):
        ple.get_state_dtype()


def test_enabled_branch_changes_both_ple_specs_but_not_gdn(installed, monkeypatch):
    import delta_ngram.qwen38_vllm_ple as split

    source, _, _ = installed
    monkeypatch.setenv("DELTA_NGRAM_VLLM_BINDING", "/already-set/binding.json")
    monkeypatch.setenv("DELTA_NGRAM_VLLM_BINDING_SHA256", "a" * 64)
    monkeypatch.setattr(split, "split_enabled", lambda: True)
    ple_name = "models/qwen4_exp/nvidia/ple_layer.py"
    model_name = "models/qwen4_exp/nvidia/model.py"
    ple_namespace = namespace_for(prepare.patch_source(ple_name, (source / ple_name).read_bytes()))
    model_namespace = namespace_for(prepare.patch_source(model_name, (source / model_name).read_bytes()))
    ple = ple_namespace["Qwen4ExpPLELayer"]()
    assert not ple.weight.requires_grad
    assert ple.get_state_dtype() == (torch.float32,)
    assert model_namespace["Model"].get_ple_mamba_state_dtype_from_config(None) == (torch.float32,)
    assert model_namespace["Model"].get_gdn_mamba_state_dtype_from_config(None) == (torch.bfloat16, torch.float32)
