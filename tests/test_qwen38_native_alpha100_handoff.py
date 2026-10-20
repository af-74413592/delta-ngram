"""Isolated CPU tests for the one-shot handoff; no real process is launched."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPT = (Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38" / "ops"
          / "handoff_alpha100_after_shared_cap.py")
MODULE_SPEC = importlib.util.spec_from_file_location("alpha100_handoff_cpu_test", SCRIPT)
handoff = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(handoff)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.fixture
def spec(tmp_path):
    source = tmp_path / "source07"
    source.mkdir()
    checkpoint = source / "joint-latest.pt"
    checkpoint.write_bytes(b"opaque completed checkpoint fixture")
    ledger = {"total_optimizer_target_accesses": 99_999_983,
              "total_diagnostic_target_accesses": 59_620,
              "total_evaluation_target_accesses": 8_789_252}
    write_json(source / "budget_ledger.json", ledger)
    write_json(source / "input_manifest.json", {"source": "immutable-run07"})
    write_json(source / "runtime_identity.json", {"runtime": "immutable-run07"})
    write_json(source / "COMPLETE.json", {
        "status": "shared_budget_complete_group_boundary", "budget": ledger,
        "latest_checkpoint": {"path": checkpoint.name, "bytes": checkpoint.stat().st_size,
                              "sha256": handoff.sha(checkpoint)},
        "unused_shared_optimizer_budget": 17, "next_complete_group_target_cost": 12_345,
    })
    runner = tmp_path / "qwen38_native_train_staged_alpha100.py"
    runner.write_text("# Frozen test fixture; never executed.\n", encoding="utf-8")
    result = {
        "source_pid": 86316, "source_start_ticks": 123456,
        "source_command_contains": "native_main_training_07",
        "source_started_utc": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        "source_output": str(source), "output": str(tmp_path / "extension08"),
        "inputs": str(tmp_path / "handoff-inputs"), "cwd": str(tmp_path),
        "python": sys.executable, "runner": str(runner), "wait_timeout_seconds": 60,
        "environment": {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "PLE_FUSED_KERNEL": "0"},
        "frozen_files_sha256": {str(runner): handoff.sha(runner), str(SCRIPT): handoff.sha(SCRIPT)},
    }
    arguments = {
        "--resume-checkpoint": str(checkpoint),
        "--source-budget-ledger": str(source / "budget_ledger.json"),
        "--source-runtime-identity": str(source / "runtime_identity.json"),
        "--source-input-manifest": str(source / "input_manifest.json"),
        "--source-complete": str(source / "COMPLETE.json"),
        "--cache-root": str(tmp_path / "cache"), "--model-path": str(tmp_path / "model"),
        "--verification": str(tmp_path / "verification.json"), "--delta-repo": str(tmp_path),
        "--initial-writer": str(tmp_path / "initial.pt"), "--dev64-plan": str(tmp_path / "plan.json"),
        "--output": result["output"], "--budget-basis": "model",
    }
    result["command"] = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                         "--nproc_per_node", "4", str(runner)]
    result["command"] += [part for pair in arguments.items() for part in pair]
    result["command"] += ["--resume-shared-cap-final", "--execute"]
    return result


def alive_identity(spec, *, state="S", start_ticks=None, cmdline=None):
    return {"state": state,
            "start_ticks": spec["source_start_ticks"] if start_ticks is None else start_ticks,
            "cmdline": spec["source_command_contains"] if cmdline is None else cmdline}


@pytest.fixture
def rig(spec, tmp_path, monkeypatch):
    """Every process, GPU inspection, wait and signal boundary is replaced."""
    value = SimpleNamespace(
        spec=spec, spec_path=tmp_path / "spec.json", events=[], clock=0.,
        identities=[None], validation_calls=[], popen_calls=[], gpu_output="",
        validation_error=None, validation_hook=None, popen_error=None,
        child_returncode=0, child_complete=True, lock_error=None)

    def process_identity(pid):
        assert pid == spec["source_pid"]
        current = value.identities.pop(0) if len(value.identities) > 1 else value.identities[0]
        value.events.append(("source_probe", current))
        return copy.deepcopy(current)

    def sleep(seconds):
        assert seconds == 5
        value.events.append(("wait", seconds))
        value.clock += seconds

    def flock(fd, flags):
        value.events.append(("lock", flags))
        if value.lock_error:
            raise value.lock_error

    def inspect_gpu(command, **kwargs):
        assert command == ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
        assert kwargs == {"text": True}
        value.events.append(("gpu_inspect", None))
        return value.gpu_output

    def validate(command, **kwargs):
        value.validation_calls.append((copy.deepcopy(command), dict(kwargs)))
        value.events.append(("cpu_validate", None))
        assert command[:2] == [spec["python"], spec["runner"]]
        assert "torch.distributed.run" not in command
        assert command.count("--validate") == 1 and "--execute" not in command
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["cwd"] == spec["cwd"] and kwargs["check"] is True
        if value.validation_hook:
            value.validation_hook()
        if value.validation_error:
            raise value.validation_error
        return SimpleNamespace(returncode=0)

    original_fsync = os.fsync

    def fsync(fd):
        original_fsync(fd)
        value.events.append(("claim_fsync", None))

    def popen(command, **kwargs):
        value.popen_calls.append((copy.deepcopy(command), dict(kwargs)))
        value.events.append(("launch", None))
        assert command == spec["command"]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
        assert kwargs["start_new_session"] is True
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert (Path(spec["inputs"]) / "launch.claim").read_text().strip()
        assert any(event == "claim_fsync" for event, _ in value.events[:-1])
        if value.popen_error:
            raise value.popen_error

        def wait():
            value.events.append(("child_wait", None))
            value.clock += 90
            if value.child_complete:
                output = Path(spec["output"])
                output.mkdir()
                write_json(output / "COMPLETE.json", {"status": "complete"})
            return value.child_returncode

        return SimpleNamespace(pid=98765, wait=wait)

    def forbidden_signal(*args, **kwargs):
        pytest.fail("the handoff must never signal or stop an existing process")

    monkeypatch.setattr(handoff, "process_identity", process_identity)
    monkeypatch.setattr(handoff.time, "monotonic", lambda: value.clock)
    monkeypatch.setattr(handoff.time, "sleep", sleep)
    monkeypatch.setattr(handoff.subprocess, "check_output", inspect_gpu)
    monkeypatch.setattr(handoff.subprocess, "run", validate)
    monkeypatch.setattr(handoff.subprocess, "Popen", popen)
    monkeypatch.setattr(handoff.os, "fsync", fsync)
    monkeypatch.setattr(handoff.os, "kill", forbidden_signal)
    if hasattr(handoff.os, "killpg"):
        monkeypatch.setattr(handoff.os, "killpg", forbidden_signal)
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=flock))

    def run(*, check_only=False):
        write_json(value.spec_path, spec)
        argv = [str(SCRIPT), "--spec", str(value.spec_path)]
        if check_only:
            argv.append("--check-spec")
        monkeypatch.setattr(sys, "argv", argv)
        return handoff.main()

    value.run = run
    value.status = lambda: read_json(Path(spec["inputs"]) / "handoff_status.json")
    return value


def test_process_identity_parses_real_stat_layout_with_parentheses(tmp_path, monkeypatch):
    root = tmp_path / "proc"
    process = root / "42"
    process.mkdir(parents=True)
    tail = ["S", *(["0"] * 18), "7654321"]
    (process / "stat").write_text("42 (name with ) parentheses) " + " ".join(tail))
    (process / "cmdline").write_bytes(b"python\0train.py\0")
    original_path = Path
    monkeypatch.setattr(handoff, "Path", lambda path: root if path == "/proc" else original_path(path))
    assert handoff.process_identity(42) == {
        "state": "S", "start_ticks": 7654321, "cmdline": "python train.py "}
    assert handoff.process_identity(43) is None


@pytest.mark.parametrize("state", ["S", "Z"])
def test_reused_pid_is_rejected_even_if_replacement_is_zombie(spec, monkeypatch, state):
    monkeypatch.setattr(handoff, "process_identity", lambda pid: alive_identity(
        spec, state=state, start_ticks=spec["source_start_ticks"] + 1))
    with pytest.raises(RuntimeError, match="PID was reused"):
        handoff.source_alive(spec)


def test_same_source_zombie_or_missing_pid_counts_as_exit(spec, monkeypatch):
    monkeypatch.setattr(handoff, "process_identity", lambda pid: alive_identity(spec, state="Z", cmdline=""))
    assert handoff.source_alive(spec) is False
    monkeypatch.setattr(handoff, "process_identity", lambda pid: None)
    assert handoff.source_alive(spec) is False


def test_live_source_command_identity_must_match(spec, monkeypatch):
    monkeypatch.setattr(handoff, "process_identity", lambda pid: alive_identity(spec, cmdline="unrelated task"))
    with pytest.raises(RuntimeError, match="command identity"):
        handoff.source_alive(spec)


@pytest.mark.parametrize("extra", [
    ["--budget-basis=charged"], ["--budget-basis", "charged"], ["--budget-b", "model"],
    ["--output=/different"], ["--validate"], ["--execute"],
])
def test_command_rejects_alternate_duplicate_or_abbreviated_flags(spec, extra):
    spec["command"] += extra
    with pytest.raises(RuntimeError):
        handoff.check_command(spec)


@pytest.mark.parametrize("change", ["charged", "ranks", "old_runner", "unfrozen_runner", "source_output", "missing_value"])
def test_command_binds_model_new_runner_four_ranks_and_new_output(spec, change):
    command = spec["command"]
    if change == "charged":
        command[command.index("--budget-basis") + 1] = "charged"
    elif change == "ranks":
        command[5] = "1"
    elif change == "old_runner":
        old = spec["runner"]
        spec["runner"] = str(Path(old).with_name("qwen38_native_train_staged_alternating.py"))
        command[6] = spec["runner"]
        spec["frozen_files_sha256"][spec["runner"]] = spec["frozen_files_sha256"][old]
    elif change == "unfrozen_runner":
        del spec["frozen_files_sha256"][spec["runner"]]
    elif change == "source_output":
        spec["output"] = spec["source_output"]
        command[command.index("--output") + 1] = spec["output"]
    else:
        command.remove("model")
    with pytest.raises(RuntimeError):
        handoff.check_command(spec)


@pytest.mark.parametrize("corruption", ["status", "checkpoint", "ledger", "unused", "next_group", "over_budget"])
def test_source_requires_natural_cap_boundary_matching_hash_and_ledger(spec, corruption):
    source = Path(spec["source_output"])
    complete = read_json(source / "COMPLETE.json")
    if corruption == "status":
        complete["status"] = "stop_after_group_file"
    elif corruption == "checkpoint":
        (source / "joint-latest.pt").write_bytes(b"different checkpoint")
    elif corruption == "ledger":
        ledger = read_json(source / "budget_ledger.json")
        ledger["total_optimizer_target_accesses"] -= 1
        write_json(source / "budget_ledger.json", ledger)
    elif corruption == "unused":
        complete["unused_shared_optimizer_budget"] += 1
    elif corruption == "next_group":
        complete["next_complete_group_target_cost"] = complete["unused_shared_optimizer_budget"]
    else:
        complete["budget"]["total_optimizer_target_accesses"] = 100_000_001
        complete["unused_shared_optimizer_budget"] = -1
        write_json(source / "budget_ledger.json", complete["budget"])
    write_json(source / "COMPLETE.json", complete)
    with pytest.raises(RuntimeError):
        handoff.validate_source_completion(spec)


def test_check_spec_is_read_only_and_does_not_validate_or_launch(rig, capsys):
    rig.identities = [alive_identity(rig.spec)]
    rig.run(check_only=True)
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] and report["source_alive"]
    assert report["new_CE_targets"] == report["optimizer_updates"] == 0
    assert not Path(rig.spec["inputs"]).exists()
    assert rig.validation_calls == rig.popen_calls == []


def test_waits_for_actual_source_exit_then_claims_and_launches_once(rig):
    source = Path(rig.spec["source_output"])
    original_files = {path.name: path.read_bytes() for path in source.iterdir()}
    rig.identities = [alive_identity(rig.spec), alive_identity(rig.spec), None]
    rig.run()
    events = [event for event, _ in rig.events]
    assert events.count("wait") == 2
    assert events.index("cpu_validate") > max(i for i, event in enumerate(events) if event == "source_probe")
    assert events.index("claim_fsync") < events.index("launch")
    assert len(rig.validation_calls) == len(rig.popen_calls) == 1
    assert {path.name: path.read_bytes() for path in source.iterdir()} == original_files
    assert not any("STOP" in path.name for path in source.iterdir())
    status = rig.status()
    assert status["status"] == "continuation_complete"
    assert status["automatic_retry"] is False and status["source_signals_sent"] == 0
    assert status["continuation_job_occupancy"]["wall_seconds"] == 90
    assert status["continuation_job_occupancy"]["reserved_gpu_hours"] == .1
    before = (Path(rig.spec["inputs"]) / "launch.json").read_bytes()
    with pytest.raises(RuntimeError, match="attempt already exists"):
        rig.run()
    assert len(rig.validation_calls) == len(rig.popen_calls) == 1
    assert (Path(rig.spec["inputs"]) / "launch.json").read_bytes() == before


@pytest.mark.parametrize("reason", ["timeout", "cancel"])
def test_wait_failure_never_signals_or_starts_a_process(rig, reason):
    rig.identities = [alive_identity(rig.spec)]
    if reason == "timeout":
        rig.spec["wait_timeout_seconds"] = 4
        expected = TimeoutError
    else:
        inputs = Path(rig.spec["inputs"])
        inputs.mkdir()
        (inputs / "CANCEL_HANDOFF").touch()
        expected = RuntimeError
    with pytest.raises(expected):
        rig.run()
    assert rig.validation_calls == rig.popen_calls == []
    assert rig.status()["source_signals_sent"] == 0
    assert not (Path(rig.spec["inputs"]) / "launch.claim").exists()


def test_lock_contention_prevents_a_second_supervisor(rig):
    rig.lock_error = BlockingIOError("supervisor already owns the lock")
    with pytest.raises(BlockingIOError):
        rig.run()
    assert rig.validation_calls == rig.popen_calls == []


@pytest.mark.parametrize("obstruction", ["output", "busy_gpu", "validation_failure", "existing_claim", "existing_launch"])
def test_prelaunch_failure_does_not_spawn_or_retry(rig, obstruction):
    if obstruction == "output":
        Path(rig.spec["output"]).mkdir()
    elif obstruction == "busy_gpu":
        rig.gpu_output = "1234\n5678\n"
    elif obstruction == "validation_failure":
        rig.validation_error = subprocess.CalledProcessError(1, ["CPU-validator"])
    else:
        inputs = Path(rig.spec["inputs"])
        inputs.mkdir()
        name = "launch.claim" if obstruction == "existing_claim" else "launch.json"
        (inputs / name).write_text("original attempt", encoding="utf-8")
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        rig.run()
    assert rig.popen_calls == []
    if obstruction in ("existing_claim", "existing_launch"):
        assert rig.validation_calls == []
        assert (inputs / name).read_text(encoding="utf-8") == "original attempt"
    else:
        assert not (Path(rig.spec["inputs"]) / "launch.claim").exists()


@pytest.mark.parametrize("mutation", ["code", "output", "source_complete", "cancel"])
def test_validation_to_launch_gap_is_rechecked(rig, mutation):
    def mutate():
        if mutation == "code":
            Path(rig.spec["runner"]).write_text("changed after validation", encoding="utf-8")
        elif mutation == "output":
            Path(rig.spec["output"]).mkdir()
        elif mutation == "cancel":
            (Path(rig.spec["inputs"]) / "CANCEL_HANDOFF").touch()
        else:
            path = Path(rig.spec["source_output"]) / "COMPLETE.json"
            complete = read_json(path)
            complete["unregistered_change"] = True
            write_json(path, complete)
    rig.validation_hook = mutate
    with pytest.raises(RuntimeError):
        rig.run()
    assert len(rig.validation_calls) == 1 and rig.popen_calls == []
    assert not (Path(rig.spec["inputs"]) / "launch.claim").exists()


def test_spawn_failure_consumes_claim_and_cannot_be_retried(rig):
    rig.popen_error = OSError("spawn failed")
    with pytest.raises(OSError, match="spawn failed"):
        rig.run()
    assert (Path(rig.spec["inputs"]) / "launch.claim").exists()
    assert rig.status()["status"] == "handoff_failed"
    with pytest.raises(RuntimeError, match="attempt already exists"):
        rig.run()
    assert len(rig.validation_calls) == len(rig.popen_calls) == 1


def test_existing_launch_log_is_not_overwritten(rig):
    inputs = Path(rig.spec["inputs"])
    inputs.mkdir()
    path = inputs / "launch.log"
    path.write_text("preserved earlier log", encoding="utf-8")
    with pytest.raises(FileExistsError):
        rig.run()
    assert path.read_text(encoding="utf-8") == "preserved earlier log"
    assert rig.popen_calls == []


@pytest.mark.parametrize("returncode,complete,status", [
    (1, False, "continuation_failed"), (0, False, "continuation_incomplete_exit"),
])
def test_child_failure_or_incomplete_exit_is_recorded_without_retry(rig, returncode, complete, status):
    rig.child_returncode, rig.child_complete = returncode, complete
    rig.run()
    assert len(rig.popen_calls) == 1
    assert rig.status()["status"] == status
    assert rig.status()["returncode"] == returncode
    assert rig.status()["automatic_retry"] is False
    with pytest.raises(RuntimeError, match="attempt already exists"):
        rig.run()
    assert len(rig.popen_calls) == 1
