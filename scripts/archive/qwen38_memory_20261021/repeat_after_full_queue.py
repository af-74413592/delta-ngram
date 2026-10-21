"""Run the registered repeat-only control after the full-only four-GPU training job."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parent
PAIR = ROOT.parent / "native_memory_full_inputs_03"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    raw = Path(path).read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def write(value):
    value["updated_utc"] = now()
    p = ROOT / "queue_status.json"
    temporary = p.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, p)


def main():
    status = {"status": "waiting_for_CPU_validation", "automatic_retry": False,
              "purpose": "reference versus independent-repeat final-memory NLL; zero artificial perturbations",
              "training_gate": False, "execution_order": ["full", "repeat_only_control"]}
    try:
        with (ROOT / "queue.claim").open("x") as claim:
            claim.write(now())
        write(status)
        for _ in range(120):
            if (ROOT / "validation.exitcode").exists():
                break
            time.sleep(5)
        assert (ROOT / "validation.exitcode").read_text().strip() == "0"
        validation = read(ROOT / "validation.log")
        assert validation["status"] == "repeat_probe_CPU_valid"
        spec = read(ROOT / "spec.json")
        manifest = validation["manifest"]
        status["status"] = "queued_after_full_training"
        status["validation"] = digest(ROOT / "validation.log")
        write(status)
        while True:
            pair = read(PAIR / "full_status.json")
            assert pair["status"] != "failed", "full training failed or stopped; repeat control not launched"
            if pair["status"] == "complete":
                break
            time.sleep(30)
        assert [job["branch"] for job in pair["jobs"]] == ["full"]
        assert all(job["status"] == "complete" and job["returncode"] == 0 for job in pair["jobs"])
        for name, receipt in manifest["code"].items():
            assert digest(Path(spec["repo"]) / name) == receipt, name
        source = manifest["source80_checkpoint"]
        assert digest(source["path"]) == {key: source[key] for key in ("bytes", "sha256")}
        busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        assert not busy, f"GPU processes still active: {busy}"
        output = Path(spec["args"][spec["args"].index("--output") + 1])
        assert not output.exists()
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES="0,1,2,3", CUDA_DEVICE_MAX_CONNECTIONS="1",
                   NCCL_NVLS_ENABLE="0", OMP_NUM_THREADS="8",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                   PYTHONPATH=spec["repo"] + "/src:" + spec["repo"] + "/scripts/native_qwen38")
        command = [spec["python"], "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "4",
                   *spec["args"], "--model-path", spec["model_path"], "--verification", spec["verification"], "--execute"]
        with (ROOT / "launch.log").open("w") as log:
            child = subprocess.Popen(command, cwd=spec["repo"], env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            status.update(status="running_repeat_only_control", pid=child.pid, started_utc=now())
            write(status)
            result = child.wait()
        status["returncode"] = result
        assert result == 0, "repeat-only control failed; no automatic retry"
        complete = read(output / "COMPLETE.json")
        assert complete["complete"] is True and complete["CE_targets"] == 12200
        assert complete["backward_calls"] == complete["optimizer_updates"] == 0
        assert digest(output / "report.json") == complete["report"]
        status.update(status="complete", completion=complete, exited_utc=now())
        write(status)
    except BaseException as error:
        status.update(status="failed", error=str(error), traceback=traceback.format_exc())
        write(status)
        raise


if __name__ == "__main__":
    main()
