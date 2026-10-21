"""Run the registered full-only branch once; residual was cancelled by the user."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def digest(path):
    raw = Path(path).read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def arg(spec, key):
    return spec["args"][spec["args"].index(key) + 1]


def validate_code(spec, manifest):
    for name, receipt in manifest["code"].items():
        assert digest(Path(spec["repo"]) / name) == receipt, name
    assert digest(manifest["source80_checkpoint"]["path"]) == {
        key: manifest["source80_checkpoint"][key] for key in ("bytes", "sha256")}


def verify_completion(output):
    output = Path(output)
    completed = read(output / "COMPLETE.json")
    assert completed["status"] == completed["phase"] == "complete"
    assert completed["completed_groups"] == 835
    assert completed["completed_targets"] == 20015113
    assert completed["budget"]["optimizer_targets_completed"] == 20015113
    assert completed["budget"]["optimizer_targets_dispatched"] == 20015113
    latest = completed["latest_checkpoint"]
    assert digest(latest["path"]) == {key: latest[key] for key in ("bytes", "sha256")}
    assert latest["completed_groups"] == 835 and latest["phase"] == "complete"
    node = completed["node_checkpoint"]
    assert node is not None
    assert digest(node["path"]) == {key: node[key] for key in ("bytes", "sha256")}
    terminal = completed["terminal_checkpoint"]
    assert digest(terminal["path"]) == {key: terminal[key] for key in ("bytes", "sha256")}
    evaluation = completed["evaluation_receipt"]
    assert evaluation["required_runs"] == evaluation["completed_runs"] == 2
    assert len(evaluation["runs"]) == 2
    for run in evaluation["runs"]:
        evaluation_dir = Path(run["output"])
        report = read(evaluation_dir / "report.json")
        receipt = read(evaluation_dir / "COMPLETE.json")
        assert report["complete"] is True and receipt["complete"] is True
        assert canonical(report) == run["report_sha256"]
        assert digest(evaluation_dir / "report.json")["sha256"] == receipt["report_sha256"]
        assert report["checkpoint"]["branch_checkpoint"] == terminal
    return {"completion_receipt": digest(output / "COMPLETE.json"),
            "latest_checkpoint": latest, "evaluation_receipt": evaluation}


def main():
    decision = read(ROOT / "training_decision.json")
    assert decision["status"] == "registered_for_execution"
    diagnostic = Path(decision["supplemental_output"])
    diagnostic_complete = read(diagnostic / "COMPLETE.json")
    assert diagnostic_complete["complete"] is True
    assert digest(diagnostic / "report.json") == diagnostic_complete["report"]
    assert diagnostic_complete["backward_calls"] == diagnostic_complete["optimizer_updates"] == 0
    assert diagnostic_complete["CE_targets"] == 12200
    assert digest(diagnostic / "report.json")["sha256"] == decision["supplemental_report_sha256"]
    status = {"protocol": "qwen38_memory_full_supervisor_v1", "started_utc": now(),
              "status": "registered", "automatic_retry": False,
              "decision": digest(ROOT / "training_decision.json"), "jobs": []}
    try:
        specs, manifests = {}, {}
        for branch in ("full",):
            assert (ROOT / f"{branch}_validation.exitcode").read_text().strip() == "0"
            spec = read(ROOT / f"{branch}_spec.json")
            validation = read(ROOT / f"{branch}_validation.log")
            manifest = validation["manifest"]
            assert validation["status"] == "memory_fork_CPU_valid"
            assert manifest["policy"]["complete_groups"] == 835
            assert manifest["policy"]["additional_training_targets"] == 20015113
            assert manifest["policy"]["diagnostic_G_action"] == "monitor_only_never_automatic_stop"
            assert manifest["policy"]["endpoint_DEV64_runs"] == 2
            assert manifest["policy"]["optimizer"] == "inherited_AdamW"
            assert manifest["policy"]["optimizer_options"]["lr"] == .0003
            assert manifest["policy"]["paired_loss"]["action"] == "monitor"
            assert manifest["policy"]["paired_loss"]["automatic_stop"] is False
            assert arg(spec, "--stop-after-group-file") == arg(spec, "--output") + "/stop-after-group"
            expected_baseline = "matched_base" if branch == "residual" else "actual_base"
            assert manifest["policy"]["residual_difference_baseline"] == expected_baseline
            assert arg(spec, "--residual-difference-baseline") == expected_baseline
            assert manifest["code_manifest_sha256"] == decision["training_code_manifest_sha256"]
            assert not Path(arg(spec, "--output")).exists()
            validate_code(spec, manifest)
            specs[branch], manifests[branch] = spec, manifest
        with (ROOT / "launch.claim").open("x") as claim:
            claim.write(now())
            claim.flush()
            os.fsync(claim.fileno())
        for branch in ("full",):
            spec = specs[branch]
            validate_code(spec, manifests[branch])
            busy = subprocess.check_output([
                "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
            assert not busy, f"GPU processes are still active: {busy}"
            environment = dict(os.environ)
            environment.update(CUDA_VISIBLE_DEVICES="0,1,2,3", CUDA_DEVICE_MAX_CONNECTIONS="1",
                NCCL_NVLS_ENABLE="0", OMP_NUM_THREADS="8", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                PYTHONPATH=spec["repo"] + "/src:" + spec["repo"] + "/scripts/native_qwen38")
            command = [spec["python"], "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "4",
                       *spec["args"], "--model-path", spec["model_path"],
                       "--verification", spec["verification"], "--execute"]
            job = {"branch": branch, "started_utc": now(), "command": command,
                   "output": arg(spec, "--output"), "status": "running"}
            with (ROOT / f"{branch}_launch.log").open("w") as log:
                process = subprocess.Popen(command, cwd=spec["repo"], env=environment,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                job["pid"] = process.pid
                status["jobs"].append(job)
                status["status"] = "running_" + branch
                write(ROOT / "full_status.json", status)
                result = process.wait()
            job.update(returncode=result, exited_utc=now())
            assert result == 0, f"{branch} child exited {result}; no automatic retry"
            job["verification"] = verify_completion(job["output"])
            job["status"] = "complete"
            write(ROOT / "full_status.json", status)
        status.update(status="complete", exited_utc=now())
        write(ROOT / "full_status.json", status)
    except BaseException as error:
        status.update(status="failed", exited_utc=now(), error_type=type(error).__name__,
                      message=str(error), traceback=traceback.format_exc())
        write(ROOT / "full_status.json", status)
        write(ROOT / "FAILURE.json", status)
        raise


if __name__ == "__main__":
    main()
