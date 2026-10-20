"""Bounded, registered 21-checkpoint DEV64 evaluation queue; default read-only.

Use --execute --workers 4 under supervisor after registering the full inventory.
Each worker calls the unchanged frozen continuation CLI in its own process.
Control file accepts {"workers": 4, "pause_launch": false, "stop": false}.
Changing concurrency does not change job order or choose jobs from their scores.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path("/workspace/delta-ngram-evaluation-20261019")
OPS = Path("/workspace/delta-ngram-transfer-20261019/multidomain")
OUT = Path("/workspace/delta-ngram-runs/m1_dev64_curve_20261020")
NEWS = Path("/workspace/delta-ngram-data/recent_news_dev64_20261019")
POLICY = OPS / "m1_checkpoint_policy_20261020.json"
INVENTORY = OPS / "m1_dev64_curve_inventory_20261020.json"
REGISTRATION = OPS / "m1_dev64_curve_registration_20261020.json"
CONTROL = OPS / "m1_dev64_curve_control_20261020.json"
STATUS = OPS / "m1_dev64_curve_status_20261020.json"
ENTRY = OPS / "evaluation_cuda_entry.py"
CONFIG = ROOT / "configs/m1_event_mixture_seed17.yaml"
REGISTRY = NEWS / "recent_news_continuation64_20261019.json"
ANNOTATIONS = NEWS / "recent_news_prefix_annotations64_20261019.json"
EXPECTED_COMMIT = "167d24783161be7842bbb9aec52f2c7804191ba9"
PINS = {
    POLICY: "d043211e2993aabc781ba3b021d8c20e299279511841044c1d49c6cb37a29c94",
    INVENTORY: "710686bd0c1f77be618b9a4b06c7a62ed5ac4046767c9c03af0602e5feb5d4e9",
    ENTRY: "96bcc448c42175ab834b9abf59476eb64e78ffd7caf6010a9346394b83424e41",
    CONFIG: "aa4319dd172b545563e705f644c167675c0c6ebede9cd7be5ed9d5018519530a",
    REGISTRY: "9ea663ad70131e272522cbad8f6c0fa20bc03f41b324fe7b71d0fb8f4fbfccc2",
    ANNOTATIONS: "7164e9f8b6f23ebfeaaccc2687a6a8b6ad222803f705bacd1a203a2596fdbcce",
}
active = {}
completed = []
failed = []


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def command(job):
    return [
        sys.executable, "-u", str(ENTRY), "evaluate_recent_news_continuation.py",
        "--config", str(CONFIG), "--checkpoint", job["checkpoint"],
        "--registry", str(REGISTRY), "--registry-sha256", PINS[REGISTRY],
        "--annotations", str(ANNOTATIONS), "--annotations-sha256", PINS[ANNOTATIONS],
        "--threads", "8", "--bootstrap-seed", "20261019", "--bootstrap-samples", "10000",
        "--output", str(OUT / (job["name"] + ".json")),
    ]


def preflight(registration_sha256, workers):
    if sha(REGISTRATION) != registration_sha256:
        raise RuntimeError("Curve registration SHA256 mismatch")
    reg = load(REGISTRATION)
    if reg["protocol"] != "m1_dev64_curve_registration_20261020_v1":
        raise RuntimeError("Unknown curve registration")
    for path, expected in PINS.items():
        if sha(path) != expected:
            raise RuntimeError("Pinned input changed: " + str(path))
    if reg["base_policy_sha256"] != PINS[POLICY] or reg["inventory_sha256"] != PINS[INVENTORY]:
        raise RuntimeError("Curve registration does not inherit the pinned policy/inventory")
    if reg["execute_final"] is not False or reg["selection_from_curve_allowed"] is not False:
        raise RuntimeError("FINAL execution and result-based checkpoint selection must be forbidden")
    if reg["maximum_workers"] != 8 or reg["initial_workers"] != 4 or workers != 4:
        raise RuntimeError("Start with exactly four workers; registered ceiling is eight")
    if subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip() != EXPECTED_COMMIT:
        raise RuntimeError("Frozen evaluation source commit changed")
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Frozen evaluation source checkout is dirty")
    inv, base, registry = load(INVENTORY), load(POLICY), load(REGISTRY)
    order = inv["expected_execution_order_nominal_loss_tokens"]
    if reg["execution_order_nominal_loss_tokens"] != order:
        raise RuntimeError("Curve execution order changed")
    if sorted(order) != list(range(0, 100000001, 5000000)):
        raise RuntimeError("All 21 registered milestones must be evaluated exactly once")
    if registry["split"] != "dev" or len({p["event_family"] for p in registry["pairs"]}) != 64:
        raise RuntimeError("Expected the fixed 64-family DEV registry")
    if base["final_policy"]["execute_in_this_dev64_queue"] is not False:
        raise RuntimeError("Base policy must forbid FINAL in this queue")
    for role in ("primary", "secondary"):
        if reg[role + "_checkpoint_sha256"] != base[role]["sha256"]:
            raise RuntimeError("Primary/secondary checkpoint roles changed")
    by_nominal = {r["nominal_loss_tokens"]: r for r in inv["checkpoints"]}
    if len(by_nominal) != 21 or inv["zero_initialization"]["all_writer_tensors_exact_equal"] is not True:
        raise RuntimeError("Missing checkpoint or unverified initialization")
    if inv["zero_initialization"]["step"] != 0 or by_nominal[0]["step"] != 0:
        raise RuntimeError("The zero-token checkpoint is not the actual initialization")
    jobs = []
    for nominal in order:
        job = dict(by_nominal[nominal])
        if sha(job["checkpoint"]) != job["sha256"]:
            raise RuntimeError("Checkpoint bytes changed: " + job["label"])
        job["name"] = f"m1_{nominal // 1000000:03d}m_continuation"
        job["role"] = "primary" if nominal == 100000000 else "secondary" if nominal == 65000000 else "curve_only"
        jobs.append(job)
    for role, nominal in (("primary", 100000000), ("secondary", 65000000)):
        for key in ("checkpoint", "sha256", "step", "training_loss_tokens"):
            if by_nominal[nominal][key] != base[role][key]:
                raise RuntimeError("Inventory differs from base policy: " + role + "." + key)
    return reg, jobs


def control():
    value = load(CONTROL)
    if type(value.get("workers")) is not int or not 1 <= value["workers"] <= 8:
        raise RuntimeError("Control workers must be an integer from one to eight")
    if any(type(value.get(k, False)) is not bool for k in ("pause_launch", "stop")):
        raise RuntimeError("Control pause_launch/stop must be boolean")
    return value


def progress(item):
    path = item["log"]
    with path.open(encoding="utf-8", errors="replace") as stream:
        stream.seek(item["log_offset"])
        lines = stream.read().splitlines()
        item["log_offset"] = stream.tell()
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("stage") in ("prepared", "pair_complete"):
                item["progress"] = {k: row[k] for k in ("stage", "pairs_complete", "pairs_total", "families") if k in row}


def save_status(stage, pending, limits, **extra):
    value = {
        "stage": stage, "updated_at_utc": now(), "output_dir": str(OUT),
        "running": [{"label": x["job"]["label"], "pid": x["process"].pid,
                     "elapsed_seconds": time.monotonic() - x["started"],
                     "log": str(x["log"]), **x.get("progress", {})} for x in active.values()],
        "completed": completed, "failed": failed, "pending": [j["label"] for j in pending],
        "control": limits, **extra,
    }
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(STATUS)
    print(json.dumps(value), flush=True)


def validate_result(job):
    output = OUT / (job["name"] + ".json")
    result = load(output)
    if not (result["complete"] and result["evaluation_only"] and result["split"] == "dev"):
        raise RuntimeError("Incomplete/non-DEV output")
    if result["optimizer_steps"] or result["backward_calls"]:
        raise RuntimeError("Optimization occurred")
    if result["independent_families"] != 64 or len(result["per_family"]) != 320:
        raise RuntimeError("Incomplete family-distance coverage")
    if result["conditions"] != ["self", "empty", "donor", "self_perm", "donor_perm"]:
        raise RuntimeError("Registered conditions changed")
    cp = result["checkpoint"]
    for key in ("sha256", "step"):
        if cp[key] != job[key]:
            raise RuntimeError("Result checkpoint identity mismatch")
    # The real M1 initialization predates this metadata field; CLI reports null.
    if cp["training_loss_tokens"] != job["checkpoint_training_loss_tokens"]:
        raise RuntimeError("Result checkpoint training-token metadata mismatch")
    if result["writer_tensor_sha256"] != job["writer_tensor_sha256"]:
        raise RuntimeError("Result writer tensor identity mismatch")
    if completed and result["manifest_sha256"] != completed[0]["manifest_sha256"]:
        raise RuntimeError("DEV64 manifest changed across curve points")
    if sha(job["checkpoint"]) != job["sha256"]:
        raise RuntimeError("Checkpoint changed during evaluation")
    return {"label": job["label"], "role": job["role"], "output": str(output),
            "sha256": sha(output), "manifest_sha256": result["manifest_sha256"],
            "nominal_loss_tokens": job["nominal_loss_tokens"],
            "training_loss_tokens": job["training_loss_tokens"], "step": job["step"]}


def stop_active():
    for item in active.values():
        if item["process"].poll() is None:
            item["process"].terminate()
    deadline = time.monotonic() + 20
    for item in active.values():
        try:
            item["process"].wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            item["process"].kill()
        item["stream"].close()


def interrupted(signum, _frame):
    raise KeyboardInterrupt("Termination signal " + str(signum))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    registration, jobs = preflight(args.registration_sha256, args.workers)
    if not args.execute:
        print(json.dumps({"preflight_passed": True, "mode": "read_only", "milestones": len(jobs),
                          "commands": [command(j) for j in jobs], "output_dir": str(OUT)}, indent=2))
        return 0
    if OUT.exists() or STATUS.exists():
        raise FileExistsError("Existing run/status is immutable; no automatic deletion, overwrite, or rerun")
    OUT.mkdir(parents=True, exist_ok=False)
    for source, name in ((REGISTRATION, "registration.json"), (INVENTORY, "inventory.json")):
        with (OUT / name).open("xb") as stream:
            stream.write(source.read_bytes())
    write_new(OUT / "runtime.json", {
        "created_at_utc": now(), "runner_sha256": sha(__file__), "source_commit": EXPECTED_COMMIT,
        "initial_workers": args.workers, "maximum_workers": 8, "threads_per_worker": 8,
        "allocator_fraction_per_worker": 0.45, "jobs": jobs,
    })
    if not CONTROL.exists():
        write_new(CONTROL, {"workers": args.workers, "pause_launch": False, "stop": False})
    if control()["workers"] != 4:
        raise RuntimeError("The first launch must use four workers")
    env = os.environ.copy()
    env.update(HF_HOME="/workspace/.hf_home", PYTHONPATH=str(ROOT / "src"), PYTHONUNBUFFERED="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="8")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    pending, limits = list(jobs), control()
    try:
        while pending or active:
            limits = control()
            if limits.get("stop", False):
                raise KeyboardInterrupt("Explicit stop requested in the control file")
            for name, item in list(active.items()):
                progress(item)
                code = item["process"].poll()
                if code is None:
                    if time.monotonic() - item["started"] > 7200:
                        if not item.get("timed_out"):
                            item["process"].terminate()
                            item["timed_out"] = time.monotonic()
                        elif time.monotonic() - item["timed_out"] > 20:
                            item["process"].kill()
                    continue
                item["stream"].close()
                del active[name]
                try:
                    if code or item.get("timed_out"):
                        raise RuntimeError("Worker exit code " + str(code) + "; see its immutable log")
                    completed.append(validate_result(item["job"]))
                except Exception as error:
                    failed.append({"label": item["job"]["label"], "error": str(error), "log": str(item["log"])})
            while pending and len(active) < limits["workers"] and not limits.get("pause_launch", False):
                job = pending.pop(0)
                if sha(job["checkpoint"]) != job["sha256"]:
                    raise RuntimeError("Checkpoint changed before launch")
                path = OUT / (job["name"] + ".log")
                stream = path.open("x", encoding="utf-8")
                proc = subprocess.Popen(command(job), cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                active[job["name"]] = {"job": job, "process": proc, "stream": stream,
                    "log": path, "log_offset": 0, "started": time.monotonic()}
            save_status("evaluating_curve", pending, limits)
            if pending or active:
                time.sleep(10)
        # Source/data/role pins are checked again, without observing scores.
        preflight(args.registration_sha256, args.workers)
        completion = {"complete": len(completed) == 21 and not failed, "completed_at_utc": now(),
                      "completed": completed, "failed": failed, "evaluation_only": True,
                      "optimizer_steps": 0, "backward_calls": 0}
        write_new(OUT / "completion.json", completion)
        save_status("curve_complete" if completion["complete"] else "curve_finished_with_failures",
                    pending, limits, complete=completion["complete"])
        return 0 if completion["complete"] else 1
    except BaseException as error:
        stop_active()
        save_status("interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    pending, limits, complete=False, error_type=type(error).__name__, error=str(error),
                    completed_results_preserved=True)
        traceback.print_exc()
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
