"""Build a separately sealed live-state fork from the audited original vLLM."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import prepare_vllm_local as fixed


def prepare_fork(source_package, source_manifest, output):
    result = fixed.prepare_fork(source_package, source_manifest, output)
    root = Path(result["package_root"])
    for relative in result["patched_files"]:
        path = root / relative
        text = path.read_text(encoding="utf-8")
        text = text.replace("DELTA_NGRAM_VLLM_BINDING", "DELTA_NGRAM_VLLM_LIVE_BINDING")
        text = text.replace("delta_ngram.qwen38_vllm_ple", "delta_ngram.qwen38_vllm_live")
        text = text.replace("fixed-delta", "request-bound live-delta")
        ast.parse(text, filename=relative)
        path.write_text(text, encoding="utf-8", newline="\n")
        result["files"][relative] = {"sha256": fixed.sha256(path), "bytes": path.stat().st_size}
    helper = fixed.REPO_ROOT / "src/delta_ngram/qwen38_vllm_live.py"
    result["live_helper"] = {"path": str(helper.resolve()), "sha256": fixed.sha256(helper)}
    result["protocol"] = "qwen38_vllm_live_two_chain_fp32_v1"
    result["prepare_script_sha256"] = fixed.sha256(Path(__file__))
    result["required_runtime"].update(max_num_seqs=1, enable_chunked_prefill=False,
                                       use_v2_model_runner=True)
    (Path(output) / "fork_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_fork(args.source_package, args.source_manifest, args.output)
    print(json.dumps({"package_root": result["package_root"], "files": len(result["files"]),
                      "protocol": result["protocol"]}))
