"""The live fork stays distinct from the existing fixed-snapshot fork."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/native_qwen38"))
import prepare_vllm_live as live


def test_live_fork_reseals_only_new_copy(tmp_path, monkeypatch):
    output = tmp_path / "live"
    repo = tmp_path / "repo"
    helper = repo / "src/delta_ngram/qwen38_vllm_live.py"
    helper.parent.mkdir(parents=True); helper.write_text("# live fixture\n")
    monkeypatch.setattr(live.fixed, "REPO_ROOT", repo)
    original = "import os\nfrom delta_ngram.qwen38_vllm_ple import split_enabled\nx = 'DELTA_NGRAM_VLLM_BINDING_SHA256'\n"
    def fixed_copy(source, manifest, target):
        package = target / "vllm"; package.mkdir(parents=True)
        (package / "model.py").write_text(original)
        return {"package_root": str(package), "patched_files": ["model.py"], "files": {}, "required_runtime": {}}
    monkeypatch.setattr(live.fixed, "prepare_fork", fixed_copy)
    result = live.prepare_fork(tmp_path / "upstream", tmp_path / "source.json", output)
    text = (output / "vllm/model.py").read_text()
    assert "qwen38_vllm_live" in text and "DELTA_NGRAM_VLLM_LIVE_BINDING_SHA256" in text
    assert "qwen38_vllm_ple" not in text and "DELTA_NGRAM_VLLM_BINDING" not in text
    assert result["live_helper"]["sha256"] == live.fixed.sha256(helper)
    assert result["files"]["model.py"]["sha256"] == live.fixed.sha256(output / "vllm/model.py")
    assert json.loads((output / "fork_manifest.json").read_text())["protocol"] == "qwen38_vllm_live_two_chain_fp32_v1"
