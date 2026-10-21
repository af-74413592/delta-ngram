"""Copy the audited installed vLLM package into a new, isolated local fork.

No install, compilation, import of vLLM, CUDA context, or write to the source
package occurs. All package files, including compiled extensions, are copied.
The two Python edits are conditional on an immutable fixed-delta binding.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
from typing import Any


UPSTREAM_VERSION = "0.30.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
PATCH_SHA256 = {
    "models/qwen4_exp/nvidia/ple_layer.py": "b81f9372eca168157c7f3cf5646d61eb0f139b64c9420588d8288672f18a3fd2",
    "models/qwen4_exp/nvidia/model.py": "2d9d98054dfa4e2c2c434526ef6fbdceede06b35c38a8ea6dccb42161a666dfb",
}

_MODULE_GUARD = '''

# delta-ngram: fixed-delta inference branch; upstream path stays available.
import os as _delta_ngram_os

_DELTA_NGRAM_BINDING_AT_IMPORT = (
    _delta_ngram_os.environ.get("DELTA_NGRAM_VLLM_BINDING"),
    _delta_ngram_os.environ.get("DELTA_NGRAM_VLLM_BINDING_SHA256"),
)


def _delta_ngram_split_enabled():
    current = (
        _delta_ngram_os.environ.get("DELTA_NGRAM_VLLM_BINDING"),
        _delta_ngram_os.environ.get("DELTA_NGRAM_VLLM_BINDING_SHA256"),
    )
    if current != _DELTA_NGRAM_BINDING_AT_IMPORT:
        raise RuntimeError("fixed-delta binding changed after vLLM import; start a new process")
    if current == (None, None):
        return False
    from delta_ngram.qwen38_vllm_ple import split_enabled
    return split_enabled()
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace_once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise ValueError("audited patch anchor is missing or ambiguous")
    return text.replace(before, after, 1)


def patch_source(relative: str, raw: bytes) -> bytes:
    """Apply only the reviewed edits, with a pinned upstream content guard."""
    expected = PATCH_SHA256.get(relative)
    if expected is None or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"upstream source SHA256 mismatch: {relative}")
    text = raw.decode("utf-8")
    if relative.endswith("/ple_layer.py"):
        before = "        compilation_config.static_forward_context[prefix] = self\n"
        after = (before + "        if _delta_ngram_split_enabled():\n"
                 "            self.requires_grad_(False)\n")
        text = _replace_once(text, before, after)
        before = """    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype("""
        after = """    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        if _delta_ngram_split_enabled():
            return (torch.float32,)
        return MambaStateDtypeCalculator.short_conv_state_dtype("""
        text = _replace_once(text, before, after)
        before = """    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1)
        if input_ids.shape[0] != hidden_states.shape[0]:"""
        after = """    ) -> torch.Tensor:
        if _delta_ngram_split_enabled():
            from delta_ngram.qwen38_vllm_ple import split_ple_forward
            return split_ple_forward(
                self, hidden_states, input_ids, query_start_loc, ngram_context
            )
        input_ids = input_ids.reshape(-1)
        if input_ids.shape[0] != hidden_states.shape[0]:"""
        text = _replace_once(text, before, after)
    else:
        before = """    def get_ple_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype("""
        after = """    def get_ple_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, ...]:
        if _delta_ngram_split_enabled():
            return (torch.float32,)
        return MambaStateDtypeCalculator.short_conv_state_dtype("""
        text = _replace_once(text, before, after)
    result = text.rstrip() + "\n" + _MODULE_GUARD
    ast.parse(result, filename=relative)
    return result.encode("utf-8")


def _source_inventory(source: Path) -> dict[str, int]:
    files = {}
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError(f"source package contains a symlink; provide a materialized package: {relative}")
        if path.is_file():
            files[relative.as_posix()] = path.stat().st_size
        elif not path.is_dir():
            raise ValueError(f"unsupported source package entry: {relative}")
    if not files or not any(name.endswith(".so") for name in files):
        raise ValueError("source must be the complete installed package, including compiled .so files")
    return files


def _verify_exported_manifest(source: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("source manifest has no files map")
    checked = {}
    for name, item in files.items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or "\\" in name:
            raise ValueError("unsafe source manifest path")
        if not rel.parts or rel.parts[0] != "vllm":
            continue
        local_rel = PurePosixPath(*rel.parts[1:])
        path = source.joinpath(*local_rel.parts)
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"installed source differs from the read-only export: {name}")
        checked[local_rel.as_posix()] = item["sha256"]
    for name, expected in PATCH_SHA256.items():
        if checked.get(name) != expected:
            raise ValueError(f"source manifest does not bind the audited source: {name}")
    return checked


def prepare_fork(source_package: Path, source_manifest: Path, output: Path) -> dict[str, Any]:
    source_input = Path(source_package).expanduser()
    if source_input.is_symlink():
        raise ValueError("source package itself must not be a symlink")
    source = source_input.resolve(strict=True)
    destination = Path(output).expanduser().resolve()
    if source.name != "vllm" or not source.is_dir():
        raise ValueError("--source-package must name the installed vllm package directory")
    if destination.exists() or destination == source or destination in source.parents or source in destination.parents:
        raise ValueError("--output must be a new directory outside the source package")
    manifest_path = Path(source_manifest).expanduser().resolve(strict=True)
    files_before = _source_inventory(source)
    checked = _verify_exported_manifest(source, manifest_path)
    metadata = source.parent / f"vllm-{UPSTREAM_VERSION}.dist-info" / "METADATA"
    if not metadata.is_file() or f"\nVersion: {UPSTREAM_VERSION}\n" not in metadata.read_text(encoding="utf-8"):
        raise ValueError("source is not the installed vLLM 0.30.0 distribution")
    helper = REPO_ROOT / "src/delta_ngram/qwen38_vllm_ple.py"
    snapshot_helper = REPO_ROOT / "src/delta_ngram/qwen38_vllm_snapshot.py"
    helpers = {"helper": helper, "snapshot_helper": snapshot_helper}
    for name, path in helpers.items():
        if not path.is_file():
            raise ValueError(f"required {name} is missing: {path}")
    patched = {name: patch_source(name, (source / name).read_bytes()) for name in PATCH_SHA256}
    if shutil.disk_usage(destination.parent).free < sum(files_before.values()) + 16 * 1024 * 1024:
        raise ValueError("not enough free space to copy the installed vLLM package")
    destination.mkdir(parents=False, exist_ok=False)
    package_root = destination / "vllm"
    shutil.copytree(source, package_root, symlinks=False,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for relative, raw in patched.items():
        (package_root / relative).write_bytes(raw)
    inventory = {}
    for relative, size_before in files_before.items():
        path = package_root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"fork is not an independent ordinary file: {relative}")
        digest = sha256(path)
        if relative not in patched and (path.stat().st_size != size_before or digest != sha256(source / relative)):
            raise ValueError(f"source changed while copying: {relative}")
        inventory[relative] = {"bytes": path.stat().st_size, "sha256": digest}
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_version": UPSTREAM_VERSION,
        "package_root": str(package_root),
        "source_package": str(source),
        "source_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "verified_exported_sources": checked,
        "files": inventory,
        "patched_files": sorted(patched),
        "helper": {"path": str(helper.resolve()), "sha256": sha256(helper)},
        "snapshot_helper": {"path": str(snapshot_helper.resolve()), "sha256": sha256(snapshot_helper)},
        "prepare_script_sha256": sha256(Path(__file__)),
        "source_modified": False,
        "symlinks": False,
        "compiled_extensions": "copied without rebuilding",
        "pythonpath": [str(destination), str(REPO_ROOT / "src")],
        "required_runtime": {"enforce_eager": True, "enable_prefix_caching": False,
                             "mamba_cache_mode": "none", "data_parallel_size": 1,
                             "tensor_parallel_size": 8, "pipeline_parallel_size": 1,
                             "speculative_config": None},
    }
    (destination / "fork_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_fork(args.source_package, args.source_manifest, args.output)
    print(json.dumps({"package_root": result["package_root"],
                      "fork_manifest": str(Path(result["package_root"]).parent / "fork_manifest.json"),
                      "files": len(result["files"]), "patched_files": result["patched_files"],
                      "pythonpath": result["pythonpath"]}, indent=2))


if __name__ == "__main__":
    main()
