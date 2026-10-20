#!/usr/bin/env python3
"""Reproduce the audited Swift stack over NeMo 26.08, without replacing CUDA binaries.

Run on the new Linux host with /opt/venv/bin/python. Default action creates an
isolated system-site venv and audits pip's proposed changes; it does not install.
--action install repeats the audit, then installs the pinned Swift overlay.
No model download, model construction, GPU allocation, or training.
The parent image /opt/venv remains untouched. Existing compatible target venvs
can be reused (e.g. plan followed by install); incompatible ones fail closed.
Optional --source-cache-dir shallow-fetches each exact official Git commit once;
pip dry-run and installation use editable checkouts with verified Git identity.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlparse


SWIFT = "21ada46205e0abfc6ab33ef8b6dd260cc4eccf8d"
BRIDGE = "3a5ab71264ca0c98db9eab7cc4de58aa57e065e3"
GIT_SOURCES = {
    "ms-swift": ("https://github.com/modelscope/ms-swift.git", SWIFT),
    "mcore-bridge": ("https://github.com/modelscope/mcore-bridge.git", BRIDGE),
}
EXPECTED_BINARY = {
    "torch": "2.13.0a0+8145d630e8.nv26.6.54250401",
    "torchvision": "0.27.0a0+499ca510.nv26.6.54250401",
    "triton": "3.7.0+gitb7fa781f.nv26.6",
    "transformer-engine": "2.14.1+366798ef",
    "flash-attn": "2.7.4.post1+git5231d95fe1.54157505",
    "causal-conv1d": "1.6.0", "numpy": "1.26.4", "apex": "0.1",
}
# The 2026-09-28 successful pip report, plus the subsequent datasets fix and
# explicit pins for inherited Python packages recorded in the image inventory.
OVERLAY = {
    "megatron-core": "0.18.0", "transformers": "5.16.0", "flash-linear-attention": "0.5.2",
    "fla-core": "0.5.2", "qwen-vl-utils": "0.0.14", "datasets": "4.7.0",
    "ms-swift": "4.6.0.dev0", "mcore-bridge": "1.7.0.dev0", "modelscope": "1.40.1",
    "simplejson": "4.1.2", "sortedcontainers": "2.4.0", "tokenizers": "0.23.2", "trl": "0.29.1",
    "addict": "2.4.0", "attrdict": "2.0.1", "av": "18.1.0", "binpacking": "2.0.1",
    "cpm-kernels": "1.0.11", "dacite": "1.9.2", "json-repair": "0.63.5", "nltk": "3.10.3",
    "openai": "3.19.2", "oss2": "2.19.1", "rouge": "1.0.1", "transformers-stream-generator": "0.0.5",
    "zstandard": "0.25.0", "aliyun-python-sdk-core": "2.16.1", "aliyun-python-sdk-kms": "2.16.5",
    "anyio": "4.15.1", "crcmod": "1.7", "httpx2": "2.13.1", "httpcore2": "2.13.1", "idna": "3.20",
    "jiter": "0.17.0", "modelscope-hub": "0.4.5", "pycryptodome": "3.23.0",
    "typing-extensions": "4.16.0", "truststore": "0.10.4",
    "peft": "0.18.1", "accelerate": "1.11.0", "huggingface-hub": "1.9.2",
}


def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def protected(name):
    name = normalized(name)
    return name in EXPECTED_BINARY or name.startswith(("nvidia-", "transformer-engine-")) or name == "torchaudio"


PROBE = r'''
import importlib.metadata as md, json, re, site, sys
norm = lambda value: re.sub(r"[-_.]+", "-", value).lower()
names = sorted({norm(d.metadata["Name"]) for d in md.distributions() if d.metadata.get("Name")})
# version(name) resolves the actually selected distribution; do not combine
# duplicate system and /opt/venv installations into contradictory constraints.
packages = {name: md.version(name) for name in names}
print(json.dumps({"python":list(sys.version_info[:3]), "executable":sys.executable,
                  "prefix":sys.prefix,"base_prefix":sys.base_prefix,"site":site.getsitepackages(),
                  "packages":packages}))
'''


def run(command, *, log=None, env):
    if log is None:
        return subprocess.check_output([str(c) for c in command], text=True, env=env)
    print("Running:", " ".join(str(c) for c in command), flush=True)
    with Path(log).open("w", encoding="utf-8") as handle:
        result = subprocess.run([str(c) for c in command], stdout=handle, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}; inspect {log}")
    return None


def check_protected(before, after):
    for name, version in before["packages"].items():
        if protected(name) and after["packages"].get(name) != version:
            raise ValueError(f"Protected binary resolution changed: {name} {version} -> {after['packages'].get(name)}")


def audit_plan(plan, before, source_records=None):
    for entry in plan.get("install", []):
        meta = entry["metadata"]
        name, version = normalized(meta["name"]), meta["version"]
        if protected(name):
            raise ValueError(f"pip proposed touching protected binary {name}=={version}; stop and audit the image")
        if name not in OVERLAY or version != OVERLAY[name]:
            raise ValueError(f"pip proposed an unrecorded overlay {name}=={version}; inspect dependencies before proceeding")
        if name in ("ms-swift", "mcore-bridge"):
            expected = SWIFT if name == "ms-swift" else BRIDGE
            source = entry.get("download_info", {})
            cached = (source_records or {}).get(name)
            editable_ok = bool(cached and source.get("dir_info", {}).get("editable")
                               and urlparse(source.get("url", "")).scheme == "file"
                               and Path(unquote(urlparse(source["url"]).path)).resolve() == Path(cached["path"]).resolve()
                               and cached["commit_id"] == expected)
            if source.get("vcs_info", {}).get("commit_id") != expected and not editable_ok:
                raise ValueError(f"unverified source commit for {name}")


def verify_source_cache(path, *, official_url, commit, env):
    """Require the pinned commit, clean tree/index, and unchanged official origin."""
    path = Path(path).resolve()
    origin = run(["git", "-C", path, "config", "--get", "remote.origin.url"], env=env).strip()
    head = run(["git", "-C", path, "rev-parse", "--verify", "HEAD^{commit}"], env=env).strip()
    status = run(["git", "-C", path, "status", "--porcelain", "--untracked-files=no"], env=env)
    if origin != official_url or head != commit or status.strip():
        raise ValueError(f"Source cache identity/clean-tree check failed: {path}")
    shallow = run(["git", "-C", path, "rev-parse", "--is-shallow-repository"], env=env).strip()
    return {"path": str(path), "official_url": official_url, "commit_id": head,
            "worktree_clean": True, "shallow": shallow == "true", "pip_url": "git+" + path.as_uri() + "@" + head}


def cache_git_source(cache_dir, *, name, official_url, commit, report, env):
    """Fetch only the requested commit and its full tree, with no ancestor history.

    Install the verified checkout in editable mode. Its real Git HEAD, official
    origin and tracked source are authenticated before and after installation.
    An interrupted init/fetch can resume only while HEAD is still unborn and
    the recorded origin/tree remain unchanged; no reset or deletion is used.
    """
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("invalid source cache name or commit")
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}-{commit}"
    report = Path(report)
    if not path.exists():
        run(["git", "init", path], log=report / f"source-{name}-init.log", env=env)
        run(["git", "-C", path, "config", "core.autocrlf", "false"], env=env)
        run(["git", "-C", path, "remote", "add", "origin", official_url], env=env)
    elif not (path / ".git").is_dir():
        raise ValueError(f"Refusing to reuse non-repository source cache: {path}")
    origin = run(["git", "-C", path, "config", "--get", "remote.origin.url"], env=env).strip()
    status = run(["git", "-C", path, "status", "--porcelain", "--untracked-files=all"], env=env)
    if origin != official_url or status.strip():
        raise ValueError(f"Refusing to modify a changed source cache: {path}")
    has_head = subprocess.run(["git", "-C", str(path), "rev-parse", "--verify", "HEAD"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env).returncode == 0
    if not has_head:
        # This one shallow network fetch avoids pip's repeated partial/full
        # history clones. HTTP/1.1 also avoids problematic HTTP/2 proxy streams;
        # this is a transport fallback, not a diagnosis of the prior stall.
        run(["git", "-C", path, "-c", "http.version=HTTP/1.1",
             "-c", "http.lowSpeedLimit=1024", "-c", "http.lowSpeedTime=120",
             "fetch", "--depth=1", "--no-tags", "origin", commit],
            log=report / f"source-{name}-fetch.log", env=env)
        fetched = run(["git", "-C", path, "rev-parse", "--verify", "FETCH_HEAD^{commit}"], env=env).strip()
        if fetched != commit:
            raise ValueError("Git fetch returned a different commit")
        run(["git", "-C", path, "-c", "advice.detachedHead=false", "checkout", "--detach", commit],
            log=report / f"source-{name}-checkout.log", env=env)
    return verify_source_cache(path, official_url=official_url, commit=commit, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("plan", "install"), default="plan")
    parser.add_argument("--base-python", type=Path, default=Path("/opt/venv/bin/python"))
    parser.add_argument("--venv", type=Path, default=Path("/workspace/venvs/qwen38-swift"))
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--source-cache-dir", type=Path,
                        help="Optional shallow official Git cache shared by pip plan and install")
    args = parser.parse_args()
    if sys.platform != "linux":
        raise SystemExit("Run this bootstrap on the Linux NeMo host; it does not modify local Windows Python")
    report = args.report_dir or Path("/workspace/cl-research/reports") / (
        "runpod_swift_bootstrap_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    report.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INPUT="1")
    # Inherited task PYTHONPATH must not select research code or old package overlays.
    env.pop("PYTHONPATH", None)
    before = json.loads(run([args.base_python, "-c", PROBE], env=env))
    (report / "nemo_before.json").write_text(json.dumps(before, indent=2) + "\n")
    if before["python"][:2] != [3, 12]:
        raise ValueError("The accepted stack uses Python 3.12")
    for name, expected in EXPECTED_BINARY.items():
        if before["packages"].get(name) != expected:
            raise ValueError(f"NeMo binary differs from accepted image: {name}={before['packages'].get(name)}, expected {expected}")
    target = args.venv.resolve()
    if target == Path(before["prefix"]).resolve() or target == Path(before["base_prefix"]).resolve():
        raise ValueError("Target must be separate from the original image environment")
    if not target.exists():
        run([args.base_python, "-m", "venv", "--system-site-packages", target], log=report / "venv.log", env=env)
    config = target / "pyvenv.cfg"
    if not config.is_file() or "include-system-site-packages = true" not in config.read_text():
        raise ValueError("Existing target is not a system-site-packages venv")
    python = target / "bin/python"
    site_dir = target / "lib/python3.12/site-packages"
    inherited = [p for p in before["site"] if Path(p).is_relative_to(Path(before["prefix"])) and Path(p).is_dir()]
    # Process original .pth files as well as its site directory, while keeping
    # this venv's own site-packages earlier in the import search order.
    pth_content = "import site; " + "; ".join(f"site.addsitedir({p!r})" for p in inherited) + "\n"
    pth = site_dir / "inherit_nemo_bootstrap.pth"
    if pth.exists() and pth.read_text() != pth_content:
        raise ValueError("Existing NeMo inheritance file has a different contract")
    pth.write_text(pth_content)
    selected = json.loads(run([python, "-c", PROBE], env=env))
    check_protected(before, selected)
    binary_constraints = report / "preserve_nemo_binaries.txt"
    binary_constraints.write_text("".join(f"{name}=={version}\n" for name, version in sorted(before["packages"].items()) if protected(name)))
    overlay_constraints = report / "accepted_overlay.txt"
    overlay_constraints.write_text("".join(f"{name}=={version}\n" for name, version in sorted(OVERLAY.items())))
    # Explicitly request the resolved overlay too, so retained image packages
    # cannot silently differ from the accepted versions merely because their
    # version ranges still satisfy Swift's requirements.
    requested = [f"{name}=={version}" for name, version in sorted(OVERLAY.items())
                 if name not in ("ms-swift", "mcore-bridge")]
    source_records = {}
    for name, (official_url, commit) in GIT_SOURCES.items():
        if args.source_cache_dir is not None:
            record = cache_git_source(args.source_cache_dir, name=name, official_url=official_url,
                                      commit=commit, report=report, env=env)
            source_records[name] = record
            # Editable installs use the authenticated checkout directly. This
            # avoids a recursive partial-clone failure of git+file in this image.
            # pip records dir_info.editable; HEAD/origin/tracked-tree are checked.
            requested.extend(["--editable", record["path"]])
        else:
            url = f"git+{official_url}@{commit}"
            requested.append(f"{name} @ {url}")
    (report / "source_cache.json").write_text(json.dumps(source_records, indent=2) + "\n")
    command = [python, "-m", "pip", "install", "--no-build-isolation", "--only-binary=megatron-core", "-c", binary_constraints,
               "-c", overlay_constraints, *requested]
    run([*command, "--dry-run", "--report", report / "plan.json"], log=report / "plan.log", env=env)
    audit_plan(json.loads((report / "plan.json").read_text()), before, source_records)
    print(f"Audited plan: {report}; protected CUDA binaries unchanged", flush=True)
    if args.action == "plan":
        print("PLAN ONLY: no packages installed. Re-run with --action install and a fresh report directory.")
        return
    # Reject tracked source changes between the audit and installation. pip
    # records editable-directory provenance; the real checkout proves its commit.
    for record in source_records.values():
        verify_source_cache(record["path"], official_url=record["official_url"], commit=record["commit_id"], env=env)
    run([*command, "--report", report / "installed.json"], log=report / "install.log", env=env)
    audit_plan(json.loads((report / "installed.json").read_text()), before, source_records)
    for record in source_records.values():
        verify_source_cache(record["path"], official_url=record["official_url"], commit=record["commit_id"], env=env)
    after = json.loads(run([python, "-c", PROBE], env=env))
    check_protected(before, after)
    for name, expected in OVERLAY.items():
        if after["packages"].get(name) != expected:
            raise ValueError(f"Overlay mismatch after installation: {name}")
    (report / "swift_after.json").write_text(json.dumps(after, indent=2) + "\n")
    imports = r'''
import importlib.metadata as md, json, subprocess
from urllib.parse import urlparse, unquote
from datasets.features import Json, List
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForConditionalGeneration
from mcore_bridge.model.gpts.qwen4_exp import Qwen4ExpBridge
import transformer_engine.pytorch, flash_attn, causal_conv1d, fla
expected = {"ms-swift": "SWIFT_COMMIT", "mcore-bridge": "BRIDGE_COMMIT"}
for name, commit in expected.items():
    provenance = json.loads(md.distribution(name).read_text("direct_url.json") or "{}")
    found = provenance.get("vcs_info", {}).get("commit_id")
    if found is None and provenance.get("dir_info", {}).get("editable"):
        parsed = urlparse(provenance["url"])
        assert parsed.scheme == "file"
        path = unquote(parsed.path)
        found = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
        assert not subprocess.check_output(["git", "-C", path, "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    assert found == commit, (name, provenance)
print("CPU import/provenance checks passed. No GPU kernel was run.")
'''.replace("SWIFT_COMMIT", SWIFT).replace("BRIDGE_COMMIT", BRIDGE)
    run([python, "-c", imports], log=report / "imports.log", env=env)
    # Preserve all inherited conflicts as evidence; NeMo's transformer pin is
    # intentionally overridden only inside this separate Swift environment.
    with (report / "pip_check.log").open("w") as handle:
        checked = subprocess.run([python, "-m", "pip", "check"], stdout=handle, stderr=subprocess.STDOUT, env=env)
    (report / "COMPLETE.json").write_text(json.dumps({"venv": str(target), "binary_versions_preserved": True,
        "cpu_imports_passed": True, "pip_check_exit": checked.returncode, "gpu_kernels_tested": False,
        "scope": "fixed Swift overlay; original NeMo environment unchanged"}, indent=2) + "\n")
    print(f"Installed and checked: {target}. Reports: {report}. GPU short acceptance remains separate.")


if __name__ == "__main__":
    main()
