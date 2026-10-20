"""One-shot local handoff after an identified source run exits naturally.

Never sends signals, alters the running source, or retries a failed GPU job.
The immutable spec binds the exact new code, command, paths, and source PID.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def process_identity(pid):
    root = Path("/proc") / str(pid)
    try:
        stat = (root / "stat").read_text()
        tail = stat[stat.rindex(")") + 2:].split()
        return {"state": tail[0], "start_ticks": int(tail[19]),
                "cmdline": (root / "cmdline").read_bytes().replace(b"\0", b" ").decode()}
    except FileNotFoundError:
        return None


def source_alive(spec):
    current = process_identity(spec["source_pid"])
    if current is None:
        return False
    if current["start_ticks"] != spec["source_start_ticks"]:
        raise RuntimeError("source PID was reused; manual inspection required")
    if current["state"] == "Z":
        return False
    if spec["source_command_contains"] not in current["cmdline"]:
        raise RuntimeError("source command identity differs")
    return True


def check_files(spec):
    for path, expected in spec["frozen_files_sha256"].items():
        if sha(path) != expected:
            raise RuntimeError("frozen handoff input changed: " + path)


def check_command(spec):
    command = spec["command"]
    valued = {"--resume-checkpoint", "--source-budget-ledger", "--source-runtime-identity",
              "--cache-root", "--model-path", "--verification", "--delta-repo", "--output",
              "--initial-writer", "--dev64-plan", "--source-input-manifest", "--source-complete",
              "--budget-basis", "--stop-after-group-file"}
    switches = {"--execute", "--resume-shared-cap-final"}
    parsed = {}
    index = 7
    while index < len(command):
        flag = command[index]
        if flag in parsed or flag not in valued | switches:
            raise RuntimeError("unknown, abbreviated or duplicate launch flag: " + flag)
        if flag in switches:
            parsed[flag] = True
            index += 1
        else:
            if index + 1 >= len(command) or command[index + 1].startswith("--"):
                raise RuntimeError("launch flag lacks a value: " + flag)
            parsed[flag] = command[index + 1]
            index += 2
    required = valued - {"--stop-after-group-file"}
    if not required.issubset(parsed) or not switches.issubset(parsed):
        raise RuntimeError("launch command is missing required flags")
    expected = {
        "--output": spec["output"], "--budget-basis": "model",
        "--resume-checkpoint": str(Path(spec["source_output"]) / "joint-latest.pt"),
        "--source-budget-ledger": str(Path(spec["source_output"]) / "budget_ledger.json"),
        "--source-complete": str(Path(spec["source_output"]) / "COMPLETE.json"),
        "--source-input-manifest": str(Path(spec["source_output"]) / "input_manifest.json"),
    }
    for flag, value in expected.items():
        if parsed[flag] != value:
            raise RuntimeError("launch command differs: " + flag)
    if command.count("--execute") != 1 or command.count("--resume-shared-cap-final") != 1:
        raise RuntimeError("handoff must execute exactly one explicit old-final migration")
    if Path(spec["source_output"]).resolve() == Path(spec["output"]).resolve():
        raise RuntimeError("new run cannot overwrite source output")
    if command[:6] != [spec["python"], "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "4"]:
        raise RuntimeError("handoff must use the authorized four-rank launcher")
    if command[6] != spec["runner"]:
        raise RuntimeError("runner path differs")
    if (Path(spec["runner"]).name != "qwen38_native_train_staged_alpha100.py"
            or spec["runner"] not in spec["frozen_files_sha256"]):
        raise RuntimeError("runner must be the frozen authorized alpha100 continuation")


def validate_source_completion(spec):
    source = Path(spec["source_output"])
    complete = json.loads((source / "COMPLETE.json").read_bytes())
    ledger = json.loads((source / "budget_ledger.json").read_bytes())
    cp = source / "joint-latest.pt"
    if complete["status"] != "shared_budget_complete_group_boundary":
        raise RuntimeError("source stopped for a different reason")
    if complete["latest_checkpoint"]["sha256"] != sha(cp):
        raise RuntimeError("source final checkpoint digest differs")
    if complete["budget"] != ledger:
        raise RuntimeError("source final checkpoint budget and durable ledger differ")
    unused = 100000000 - ledger["total_optimizer_target_accesses"]
    if (unused < 0 or complete["unused_shared_optimizer_budget"] != unused
            or type(complete["next_complete_group_target_cost"]) is not int
            or complete["next_complete_group_target_cost"] <= unused):
        raise RuntimeError("source has not reached the whole-group shared-cap endpoint")
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--check-spec", action="store_true")
    args = parser.parse_args()
    raw = args.spec.read_bytes()
    spec = json.loads(raw)
    check_command(spec)
    check_files(spec)
    if args.check_spec:
        print(json.dumps({"valid": True, "source_alive": source_alive(spec),
                          "source_complete_exists": (Path(spec["source_output"]) / "COMPLETE.json").exists(),
                          "new_CE_targets": 0, "optimizer_updates": 0}))
        return
    import fcntl
    inputs = Path(spec["inputs"])
    inputs.mkdir(parents=True, exist_ok=True)
    lock = (inputs / "supervisor.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (inputs / "launch.claim").exists() or (inputs / "launch.json").exists():
        raise RuntimeError("a launch attempt already exists; refuse automatic retry")
    receipt = {"protocol": "qwen38_alpha100_one_shot_handoff_v1", "supervisor_pid": os.getpid(),
               "armed_utc": utc(), "spec_sha256": hashlib.sha256(raw).hexdigest(),
               "status": "waiting_for_source_natural_completion", "source_pid": spec["source_pid"],
               "source_output": spec["source_output"], "output": spec["output"],
               "automatic_retry": False, "source_signals_sent": 0}
    status_path = inputs / "handoff_status.json"
    atomic_json(status_path, receipt)
    try:
        deadline = time.monotonic() + spec.get("wait_timeout_seconds", 43200)
        while source_alive(spec):
            if (inputs / "CANCEL_HANDOFF").exists():
                raise RuntimeError("handoff cancelled by marker")
            if time.monotonic() >= deadline:
                raise TimeoutError("source completion deadline expired")
            time.sleep(5)
        receipt["source_exit_observed_utc"] = utc()
        source_start = datetime.fromisoformat(spec["source_started_utc"])
        source_seconds = (datetime.now(timezone.utc) - source_start).total_seconds()
        receipt["source_job_occupancy"] = {
            "launch_utc": spec["source_started_utc"], "exit_observed_utc": receipt["source_exit_observed_utc"],
            "wall_seconds": source_seconds, "reserved_gpu_hours": 4 * source_seconds / 3600,
            "gpus": 4, "exit_observation_lag_bound_seconds": 5,
            "definition": "job launch to observed process exit; includes loading, evaluation, checkpoints and waits"}
        complete = validate_source_completion(spec)
        atomic_json(inputs / "source_final_complete.json", complete)
        check_files(spec)
        if (inputs / "CANCEL_HANDOFF").exists():
            raise RuntimeError("handoff cancelled before launch")
        if Path(spec["output"]).exists():
            raise RuntimeError("target output exists; refuse duplicate launch")
        busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
        if busy:
            raise RuntimeError("GPU compute processes remain after source exit: " + busy)
        receipt["status"] = "validating_final_source_on_cpu"
        atomic_json(status_path, receipt)
        env = dict(os.environ, **spec["environment"])
        validation_env = dict(env, CUDA_VISIBLE_DEVICES="")
        validation_command = [spec["python"], spec["runner"], *spec["command"][7:]]
        validation_command[validation_command.index("--execute")] = "--validate"
        with (inputs / "validation.log").open("w") as log:
            subprocess.run(validation_command, cwd=spec["cwd"], env=validation_env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        if (inputs / "CANCEL_HANDOFF").exists():
            raise RuntimeError("handoff cancelled after validation")
        check_files(spec)
        check_command(spec)
        if validate_source_completion(spec) != complete:
            raise RuntimeError("source completion changed during CPU validation")
        if Path(spec["output"]).exists():
            raise RuntimeError("target output appeared during validation; refuse launch")
        # Persist an irreversible attempt claim BEFORE spawning. A crash requires
        # human/agent inspection; it can never silently create a second GPU job.
        with (inputs / "launch.claim").open("x") as claim:
            claim.write(utc() + "\n")
            claim.flush()
            os.fsync(claim.fileno())
        started = time.monotonic()
        launch_utc = utc()
        with (inputs / "launch.log").open("x") as log:
            child = subprocess.Popen(spec["command"], cwd=spec["cwd"], env=env,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            launch = {"started_utc": launch_utc, "pid": child.pid, "command": spec["command"],
                      "environment": spec["environment"], "spec_sha256": receipt["spec_sha256"],
                      "source_complete_sha256": sha(inputs / "source_final_complete.json"),
                      "source_final_checkpoint": complete["latest_checkpoint"],
                      "budget_basis": "model", "target_model_targets": 100000000}
            atomic_json(inputs / "launch.json", launch)
            receipt.update(status="continuation_launched", launched_utc=launch_utc, child_pid=child.pid)
            atomic_json(status_path, receipt)
            returncode = child.wait()
        elapsed = time.monotonic() - started
        complete_exists = (Path(spec["output"]) / "COMPLETE.json").exists()
        exit_status = ("continuation_complete" if returncode == 0 and complete_exists else
                       "continuation_failed" if returncode else "continuation_incomplete_exit")
        receipt.update(status=exit_status, returncode=returncode, exit_utc=utc(),
                       continuation_job_occupancy={"gpus": 4, "wall_seconds": elapsed,
                                                  "reserved_gpu_hours": elapsed * 4 / 3600,
                                                  "definition": "full child process lifetime; not kernel-active GPU time"},
                       complete_exists=complete_exists)
        atomic_json(status_path, receipt)
    except BaseException as error:
        receipt.update(status="handoff_failed", failure_utc=utc(), error_type=type(error).__name__,
                       error=str(error))
        atomic_json(status_path, receipt)
        raise


if __name__ == "__main__":
    main()
