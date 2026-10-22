"""HumanEval execution in capability-isolated CPython WASI; no Docker needed.

The host never executes candidate Python. Validation checks every official
canonical solution before this runtime can score a model's completions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

from humaneval_probe import (DATA_URL, PROBE_IDS, REVISION, TASK_IDS, file_hash,
    load_locked_data, project_tasks, scope_ids, validate_tasks, write_json)
from evaluate_humaneval_isolated import result_metrics, transition_counts

EVALUATOR_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "humaneval"
sys.path.insert(0, str(EVALUATOR_DIR))
from wasi_worker import FUEL, MEMORY_BYTES, OUTPUT_BYTES, WASM_SHA256, WASMTIME_VERSION

RUNTIME_URL = ("https://github.com/vmware-labs/webassembly-language-runtimes/releases/download/"
               "python/3.12.0%2B20231211-040d5a6/python-3.12.0.wasm")
RUNTIME_BYTES = 26267204
TASK_WALL_SECONDS = 10.0
STARTUP_SECONDS = 90.0


def protocol():
    return {"version": "humaneval-wasi-v1", "harness_revision": REVISION,
            "python": "3.12.0-wasi", "wasmtime": WASMTIME_VERSION,
            "wasm_sha256": WASM_SHA256, "fuel_per_task": FUEL,
            "memory_bytes": MEMORY_BYTES, "output_bytes_per_stream": OUTPUT_BYTES,
            "task_wall_seconds": TASK_WALL_SECONDS, "workers": 1,
            "preopened_directories": [], "inherited_environment": False,
            "inherited_stdin": False, "network_capabilities": False,
            "worker_sha256": file_hash(EVALUATOR_DIR / "wasi_worker.py"),
            "controller_sha256": file_hash(Path(__file__))}


def protocol_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare_runtime(directory):
    directory.mkdir(parents=True, exist_ok=True)
    runtime = directory / "python-3.12.0.wasm"
    if runtime.exists():
        if file_hash(runtime) != WASM_SHA256:
            raise ValueError("existing runtime checksum differs; nothing overwritten")
    else:
        with urllib.request.urlopen(RUNTIME_URL, timeout=120) as response:
            payload = response.read(RUNTIME_BYTES + 1)
        if len(payload) != RUNTIME_BYTES or hashlib.sha256(payload).hexdigest() != WASM_SHA256:
            raise ValueError("downloaded runtime size/checksum mismatch")
        temporary = runtime.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(runtime)
    manifest = {"source_url": RUNTIME_URL, "runtime_sha256": WASM_SHA256,
                "runtime_bytes": RUNTIME_BYTES, "runtime": str(runtime.resolve()),
                "required_package": f"wasmtime=={WASMTIME_VERSION}"}
    write_json(directory / "runtime_manifest.json", manifest)
    return manifest


class Worker:
    """A replaceable trusted host process; guest blocking calls cannot hang us."""
    def __init__(self, runtime):
        self.runtime = runtime.resolve()
        self.process = None
        self.lines = None

    def start(self):
        self.close()
        # No credential environment is passed even to the trusted WASM host.
        # PYTHONPATH supports a dedicated --target dependency directory.
        env = {key: value for key, value in os.environ.items()
               if key in {"PATH", "SYSTEMROOT", "SystemRoot", "WINDIR", "TEMP", "TMP", "PYTHONPATH",
                          "PROCESSOR_ARCHITECTURE", "PROCESSOR_ARCHITEW6432", "PROCESSOR_IDENTIFIER"}}
        env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        process = subprocess.Popen([sys.executable, "-u", str(EVALUATOR_DIR / "wasi_worker.py"),
                                    "--runtime", str(self.runtime)],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   encoding="utf-8", env=env, bufsize=1)
        self.process = process
        self.lines = queue.Queue()
        output_queue = self.lines

        def read_lines():
            try:
                for line in process.stdout:
                    output_queue.put(line)
            except ValueError:
                pass  # close() may close a terminated process's pipe.
            finally:
                output_queue.put(None)

        threading.Thread(target=read_lines, daemon=True).start()
        try:
            ready = self.receive(STARTUP_SECONDS)
        except Exception:
            self.close()
            raise
        if ready != {"ready": True, "wasmtime": WASMTIME_VERSION, "wasm_sha256": WASM_SHA256}:
            self.close()
            raise RuntimeError("WASI worker did not complete its startup handshake")

    def receive(self, timeout):
        line = self.lines.get(timeout=timeout)
        if line is None:
            self.process.wait(timeout=1)
            diagnostic = self.process.stderr.read(4096)
            raise RuntimeError("WASI worker exited without a result: " + diagnostic)
        return json.loads(line)

    def check(self, task_id, program):
        if self.process is None or self.process.poll() is not None:
            self.start()
        started = time.perf_counter()
        request = {"task_id": task_id, "program": program, "nonce": uuid.uuid4().hex + ":"}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        try:
            result = self.receive(TASK_WALL_SECONDS)
        except queue.Empty:
            self.close()
            return {"task_id": task_id, "status": "completed", "passed": False,
                    "result": "wall_timeout", "elapsed_seconds": time.perf_counter()-started}
        if result.get("task_id") != task_id:
            raise ValueError("worker returned the wrong task ID")
        return result

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=15)
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                if stream:
                    stream.close()
            self.process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.close()


def build_program(problem, completion):
    # Same prompt + completion + tests + check(entry_point) as official HumanEval.
    return problem["prompt"] + completion + "\n" + problem["test"] + "\ncheck(" + problem["entry_point"] + ")"


def validate_runtime(args):
    args.output.mkdir(parents=True, exist_ok=False)
    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        payload = response.read(2 * 1024 * 1024 + 1)
    rows = validate_tasks([json.loads(line) for line in gzip.decompress(payload).decode().splitlines() if line.strip()])
    results = []
    started = time.perf_counter()
    checks = {
        "no_environment": "import os; assert dict(os.environ) == {}",
        "no_host_files": "\n".join([
            "for path in ['/etc/passwd', '/root/.ssh/id_rsa', 'C:/Windows/win.ini']:",
            "    try: open(path).read()", "    except OSError: pass",
            "    else: raise AssertionError('host file visible')"]),
        "no_host_writes": "\n".join([
            "try: open('/guest_must_not_create.txt', 'w').write('x')",
            "except OSError: pass", "else: raise AssertionError('host write allowed')"]),
        "no_process": "\n".join([
            "import subprocess", "try: subprocess.run(['echo', 'must-not-run'])",
            "except (OSError, NotImplementedError): pass", "else: raise AssertionError('spawn allowed')"]),
        "no_network": "\n".join([
            "try:", "    import socket", "    socket.create_connection(('127.0.0.1', 80), timeout=0.1)",
            "except (ImportError, OSError, AttributeError): pass", "else: raise AssertionError('network allowed')"]),
    }
    with Worker(args.runtime) as worker:
        security = [worker.check(name, code) for name, code in checks.items()]
        if not all(item.get("passed") is True for item in security):
            raise RuntimeError(f"WASI capability checks failed: {security}")
        negative = worker.check("known_wrong", "assert 1 == 2")
        runaway = worker.check("runaway_loop", "while True: pass")
        if negative.get("passed") is not False or runaway.get("passed") is not False:
            raise RuntimeError("WASI checker accepted a known incorrect/runaway program")
        for problem in rows:
            result = worker.check(problem["task_id"], build_program(problem, problem["canonical_solution"]))
            results.append(result)
            print(json.dumps({"canonical_task": problem["task_id"], "passed": result["passed"]}), flush=True)
    write_json(args.output / "canonical_results.json", results)
    summary = {"status": "validated" if all(row.get("passed") is True for row in results) else "failed",
               "runtime_protocol": protocol(), "runtime_protocol_sha256": protocol_hash(protocol()),
               "canonical_tasks": len(results), "canonical_passed": sum(row.get("passed") is True for row in results),
               "dataset_source_sha256": hashlib.sha256(payload).hexdigest(), "source_url": DATA_URL,
               "capability_checks": security, "elapsed_seconds": time.perf_counter()-started,
               "negative_control": negative, "runaway_control": runaway,
               "canonical_results_sha256": file_hash(args.output / "canonical_results.json"),
               "validated_utc": datetime.now(timezone.utc).isoformat()}
    write_json(args.output / "summary.json", summary)
    if summary["status"] != "validated":
        raise RuntimeError(f"canonical compatibility check failed: {summary['canonical_passed']}/164")
    return summary


def verify_validation(directory, manifest):
    validation = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (validation.get("status") != "validated" or validation.get("canonical_passed") != 164
            or validation.get("canonical_tasks") != 164
            or validation["runtime_protocol_sha256"] != protocol_hash(protocol())
            or validation["dataset_source_sha256"] != manifest["source_gzip_sha256"]
            or not all(row.get("passed") is True for row in validation.get("capability_checks", []))
            or len(validation.get("capability_checks", [])) != 5
            or validation.get("negative_control", {}).get("passed") is not False
            or validation.get("runaway_control", {}).get("passed") is not False
            or file_hash(directory / "canonical_results.json") != validation["canonical_results_sha256"]):
        raise ValueError("runtime validation is missing, stale, or differs from the dataset")
    return validation


def score(args):
    problems, manifest = load_locked_data(args.data_dir, "problems.json")
    validation = verify_validation(args.validation, manifest)
    source = json.loads((args.samples_dir / "summary.json").read_text(encoding="utf-8"))
    if (source["status"] != "completed" or source["data_manifest"] != manifest
            or file_hash(args.samples_dir / "humaneval_completions.json") != source["completions_sha256"]):
        raise ValueError("generation incomplete, changed, or based on a different dataset")
    source_ids = scope_ids(source["scope"])
    if source["task_ids"] != list(source_ids) or source["protocol"]["version"] != 2:
        raise ValueError("generation scope/protocol differs from locked version")
    selected = scope_ids(args.scope)
    samples = validate_tasks(json.loads((args.samples_dir / "humaneval_completions.json").read_text(encoding="utf-8")), source_ids)
    samples = project_tasks(samples, selected)
    problems = {row["task_id"]: row for row in project_tasks(problems, selected)}
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"execution_status": "running", "pass_at_1": None,
               "protocol_sha256": source["protocol_sha256"],
               "execution_protocol": protocol(), "execution_protocol_sha256": protocol_hash(protocol()),
               "model_identity": source["model_identity"], "source_checkpoint": source["source_checkpoint"],
               "source_generation": str(args.samples_dir), "source_scope": source["scope"],
               "scope": args.scope, "task_ids": list(selected), "task_count": len(selected),
               "generation_timing": source.get("timing"), "validation_sha256": file_hash(args.validation / "summary.json"),
               "canonical_compatibility": f"{validation['canonical_passed']}/164",
               "completed_tasks": 0, "infrastructure_errors": 0}
    write_json(args.output / "summary.json", summary)
    results = []
    started = time.perf_counter()
    try:
        with Worker(args.runtime) as worker:
            for sample in samples:
                result = worker.check(sample["task_id"], build_program(problems[sample["task_id"]], sample["completion"]))
                result.update(generation_truncated=sample.get("truncated"), generated_tokens=sample.get("generated_tokens"))
                results.append(result)
                write_json(args.output / "per_question.json", results)
                if result["status"] != "completed" or type(result.get("passed")) is not bool:
                    summary["infrastructure_errors"] += 1
                    raise RuntimeError(f"evaluator infrastructure failure on {sample['task_id']}")
                summary["completed_tasks"] = len(results)
                write_json(args.output / "summary.json", summary)
                print(json.dumps({"task_id": sample["task_id"], "passed": result["passed"], "result": result["result"]}), flush=True)
        summary.update(execution_status="completed", **result_metrics(results, selected),
                       probe32=result_metrics(results, PROBE_IDS),
                       per_question_sha256=file_hash(args.output / "per_question.json"))
        if args.reference:
            reference = json.loads((args.reference / "summary.json").read_text(encoding="utf-8"))
            if (reference["execution_status"] != "completed"
                    or reference["protocol_sha256"] != summary["protocol_sha256"]
                    or reference.get("execution_protocol_sha256") != summary["execution_protocol_sha256"]
                    or reference["model_identity"] != summary["model_identity"]
                    or file_hash(args.reference / "per_question.json") != reference["per_question_sha256"]):
                raise ValueError("reference model, decoding, executor, or result checksum differs")
            previous = validate_tasks(json.loads((args.reference / "per_question.json").read_text(encoding="utf-8")), scope_ids(reference["scope"]))
            common = selected if reference["scope"] == args.scope else PROBE_IDS
            summary["transitions"] = transition_counts(previous, results, common)
            summary["probe32_transitions"] = transition_counts(previous, results, PROBE_IDS)
    except Exception as exc:
        summary.update(execution_status="failed", pass_at_1=None, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter()-started
        summary["seconds_per_task"] = summary["elapsed_seconds"]/len(selected) if summary["execution_status"] == "completed" else None
        write_json(args.output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-runtime")
    prepare.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate-runtime")
    validate.add_argument("--runtime", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("score")
    for name in ("runtime", "validation", "data-dir", "samples-dir", "output"):
        execute.add_argument("--" + name, type=Path, required=True)
    execute.add_argument("--scope", choices=("probe", "full"), default="probe")
    execute.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.command == "prepare-runtime":
        result = prepare_runtime(args.output)
    elif args.command == "validate-runtime":
        result = validate_runtime(args)
    else:
        result = score(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
