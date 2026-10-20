"""One-shot incremental backup of runner-validated DEV64 results.

Read the remote status once, package only newly completed JSON results plus
metadata, verify the archive and each member, and preserve conflicting files.
No GPU process, control file, registration, or frozen evaluator is modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tarfile
import uuid


REPO = Path(__file__).resolve().parents[1]
LOCALOPS = REPO.parent / "delta-ngram-runs/multidomain_launch_20261019"
DESTINATION = LOCALOPS / "dev64_curve_results_20261020"
HOST = "root@69.176.92.121"
PORT = "55432"
REMOTE_BATCHES = "/workspace/delta-ngram-transfer-20261019/multidomain/dev64_download_batches_20261020"
RESULT_NAME = re.compile(r"m1_\d{3}m_continuation\.json")
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
               "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2"]

REMOTE_PROGRAM = r'''
import hashlib,io,json,pathlib,re,sys,tarfile
from datetime import datetime,timezone
OPS=pathlib.Path("/workspace/delta-ngram-transfer-20261019/multidomain")
OUT=pathlib.Path("/workspace/delta-ngram-runs/m1_dev64_curve_20261020")
STATUS=OPS/"m1_dev64_curve_status_20261020.json"
request=json.load(sys.stdin)
batch=request["batch"]
assert re.fullmatch(r"c\d{2}_\d{8}T\d{6}Z_[0-9a-f]{8}",batch)
minimum=request["minimum_completed"]
assert minimum in (8,12,16,20,21)
known=request["known_results"]
assert all(re.fullmatch(r"m1_\d{3}m_continuation\.json",k) and re.fullmatch(r"[0-9a-f]{64}",v)
           for k,v in known.items())
status_bytes=STATUS.read_bytes()
status=json.loads(status_bytes)
completed=status["completed"]
if len(completed)<minimum:
 print(json.dumps({"action":"not_ready","completed":len(completed),"minimum_completed":minimum,
                   "stage":status["stage"],"updated_at_utc":status["updated_at_utc"]}))
 sys.exit(0)
assert status["output_dir"]==str(OUT)
assert len(completed)==len({entry["output"] for entry in completed})<=21
payloads={}
files=[]
runner_hashes={}
def digest(data):
 return hashlib.sha256(data).hexdigest()
def add(name,data,role,runner_sha256=None):
 assert pathlib.PurePosixPath(name).name==name and name not in payloads
 payloads[name]=data
 entry={"name":name,"bytes":len(data),"sha256":digest(data),"role":role}
 if runner_sha256 is not None: entry["runner_sha256"]=runner_sha256
 files.append(entry)
for entry in completed:
 path=pathlib.Path(entry["output"])
 assert path.resolve().parent==OUT.resolve()
 assert re.fullmatch(r"m1_\d{3}m_continuation\.json",path.name)
 assert re.fullmatch(r"[0-9a-f]{64}",entry["sha256"])
 runner_hashes[path.name]=entry["sha256"]
 if path.name in known:
  assert known[path.name]==entry["sha256"],"local result differs from runner: "+path.name
  continue
 data=path.read_bytes()
 assert digest(data)==entry["sha256"],"remote result differs from runner: "+path.name
 result=json.loads(data)
 assert result.get("complete") is True and result.get("split")=="dev" and result.get("evaluation_only") is True
 assert result.get("optimizer_steps")==result.get("backward_calls")==0
 assert result.get("independent_families")==64 and len(result["per_family"])==320
 add(path.name,data,"complete_dev64_result",entry["sha256"])
assert set(known)<=set(runner_hashes),"local results absent from runner status"
for name in ("runtime.json","registration.json","inventory.json"):
 add(name,(OUT/name).read_bytes(),"immutable_run_metadata")
add("m1_checkpoint_policy_20261020.json",(OPS/"m1_checkpoint_policy_20261020.json").read_bytes(),
    "immutable_policy")
add("status_"+batch+".json",status_bytes,"status_snapshot")
completion_path=OUT/"completion.json"
complete_final=(status.get("stage")=="curve_complete" and status.get("complete") is True
                and len(completed)==21 and not status.get("failed"))
if complete_final:
 completion_bytes=completion_path.read_bytes()
 completion=json.loads(completion_bytes)
 assert completion.get("complete") is True and len(completion["completed"])==21 and not completion["failed"]
 assert {pathlib.Path(r["output"]).name:r["sha256"] for r in completion["completed"]}==runner_hashes
 add("completion.json",completion_bytes,"final_completion")
 add("status.json",status_bytes,"final_status")
elif completion_path.exists():
 add("completion_"+batch+".json",completion_path.read_bytes(),"completion_snapshot")
manifest_name="download_manifest_"+batch+".json"
manifest={"protocol":"m1_dev64_incremental_download_v1","batch":batch,
          "created_utc":datetime.now(timezone.utc).isoformat(),"minimum_completed":minimum,
          "completed_count":len(completed),"new_result_count":sum(f["role"]=="complete_dev64_result" for f in files),
          "complete_final":complete_final,"runner_status_sha256":digest(status_bytes),
          "runner_result_sha256":runner_hashes,"files":files}
payloads[manifest_name]=(json.dumps(manifest,indent=2,allow_nan=False)+"\n").encode()
directory=OPS/"dev64_download_batches_20261020"
directory.mkdir(exist_ok=True)
archive=directory/(batch+".tar.gz")
assert not archive.exists()
with archive.open("xb") as output:
 with tarfile.open(fileobj=output,mode="w:gz",compresslevel=1) as tar:
  for name,data in payloads.items():
   item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o644;item.mtime=0
   tar.addfile(item,io.BytesIO(data))
with archive.open("rb") as stream:
 archive_sha=hashlib.file_digest(stream,"sha256").hexdigest()
print(json.dumps({"action":"archive_ready","archive":str(archive),"archive_sha256":archive_sha,
                  "archive_bytes":archive.stat().st_size,"manifest":manifest_name,
                  "manifest_sha256":digest(payloads[manifest_name]),
                  "completed_count":len(completed),"new_result_count":manifest["new_result_count"],
                  "complete_final":complete_final}))
'''


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_checked(command, *, stdin=None, timeout=60):
    completed = subprocess.run(command, input=stdin, text=True, capture_output=True, timeout=timeout)
    if completed.returncode:
        raise RuntimeError(f"{command[0]} failed ({completed.returncode}): {completed.stderr[-1500:]}")
    return completed.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--minimum-completed", type=int, choices=(8, 12, 16, 20, 21))
    mode.add_argument("--resume-transfer", type=Path, help="Retry SCP from a saved transfer descriptor; no SSH/status poll")
    parser.add_argument("--transport", choices=("scp", "ssh-cat"), default="scp",
                        help="ssh-cat preserves archive bytes through SSH stdout without SFTP/SCP")
    args = parser.parse_args()
    destination = DESTINATION.resolve()
    require(destination.parent == LOCALOPS.resolve(), "Destination escaped local operations directory")
    destination.mkdir(parents=True, exist_ok=True)
    if args.resume_transfer:
        require(args.resume_transfer.resolve().parent == destination, "Transfer descriptor outside destination")
        transfer = json.loads(args.resume_transfer.read_bytes())
        require(transfer["host"] == HOST and transfer["port"] == PORT, "Transfer endpoint changed")
        request, response = transfer["request"], transfer["remote_response"]
        batch = request["batch"]
        require(re.fullmatch(r"c\d{2}_\d{8}T\d{6}Z_[0-9a-f]{8}", batch), "Unsafe batch name")
    else:
        known = {p.name: sha(p) for p in sorted(destination.glob("m1_*m_continuation.json"))
                 if RESULT_NAME.fullmatch(p.name)}
        batch = (f"c{args.minimum_completed:02d}_"
                 + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                 + "_" + uuid.uuid4().hex[:8])
        request = {"batch": batch, "minimum_completed": args.minimum_completed, "known_results": known}
        with (destination / ("download_request_" + batch + ".json")).open(
                "x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
                "host": HOST, "port": PORT, "request": request,
                "script_sha256": sha(Path(__file__).resolve())}, indent=2) + "\n")
        ssh = ["ssh", "-p", PORT, *SSH_OPTIONS, HOST, "python3 -c " + shlex.quote(REMOTE_PROGRAM)]
        try:
            response = json.loads(run_checked(ssh, stdin=json.dumps(request), timeout=55))
        except Exception as error:
            with (destination / ("download_failure_" + batch + ".json")).open(
                    "x", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
                    "batch": batch, "phase": "ssh_prepare", "error": str(error),
                    "existing_local_results_preserved": True}, indent=2) + "\n")
            raise
    if response["action"] == "not_ready":
        print(json.dumps(response))
        return
    require(response["action"] == "archive_ready", "Unexpected remote response")
    expected_remote = REMOTE_BATCHES + "/" + batch + ".tar.gz"
    require(response["archive"] == expected_remote, "Unexpected remote archive path")
    require(re.fullmatch(r"[0-9a-f]{64}", response["archive_sha256"]), "Invalid archive digest")
    archive = destination / (batch + ".tar.gz")
    transfer_path = destination / ("download_transfer_" + batch + ".json")
    if not transfer_path.exists():
        transfer = {"created_utc": datetime.now(timezone.utc).isoformat(),
                    "host": HOST, "port": PORT, "request": request, "remote_response": response,
                    "script_sha256": sha(Path(__file__).resolve())}
        with transfer_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(transfer, indent=2, allow_nan=False) + "\n")
    partial = destination / (batch + ".tar.gz.part")
    if archive.exists():
        require(sha(archive) == response["archive_sha256"], "Existing archive differs; preserved")
    else:
        attempt = {"started_utc": datetime.now(timezone.utc).isoformat(), "batch": batch,
                   "partial_path": str(partial), "archive_sha256_expected": response["archive_sha256"],
                   "transport": args.transport}
        try:
            if not partial.exists() or sha(partial) != response["archive_sha256"]:
                if args.transport == "scp":
                    run_checked(["scp", "-P", PORT, *SSH_OPTIONS, HOST + ":" + expected_remote, str(partial)], timeout=60)
                else:
                    with partial.open("wb") as stream:
                        copied = subprocess.run(
                            ["ssh", "-p", PORT, *SSH_OPTIONS, HOST, "cat -- " + shlex.quote(expected_remote)],
                            stdout=stream, stderr=subprocess.PIPE, timeout=60)
                    require(copied.returncode == 0, "SSH binary transfer failed: "
                            + copied.stderr.decode("utf-8", errors="replace")[-1500:])
            require(partial.stat().st_size == response["archive_bytes"]
                    and sha(partial) == response["archive_sha256"], "Partial archive integrity mismatch")
            require(not archive.exists(), "Archive appeared during transfer; preserved")
            partial.rename(archive)
            attempt["verified"] = True
        except Exception as error:
            attempt.update(verified=False, error=str(error), partial_exists=partial.exists())
            raise
        finally:
            attempt["finished_utc"] = datetime.now(timezone.utc).isoformat()
            with (destination / ("download_transfer_attempts_" + batch + ".jsonl")).open(
                    "a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(attempt, allow_nan=False) + "\n")
    require(archive.stat().st_size == response["archive_bytes"], "Archive size mismatch")
    require(sha(archive) == response["archive_sha256"], "Archive SHA256 mismatch; extraction aborted")
    payloads = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            path = PurePosixPath(member.name)
            require(member.isfile() and path.name == member.name and not path.is_absolute()
                    and "\\" not in member.name and member.name not in payloads,
                    "Unsafe/duplicate archive member")
            require(0 <= member.size <= 25_000_000, "Unexpectedly large result member")
            payloads[member.name] = tar.extractfile(member).read()
    manifest_name = response["manifest"]
    require(manifest_name == "download_manifest_" + batch + ".json", "Unexpected manifest name")
    require(sha_bytes(payloads[manifest_name]) == response["manifest_sha256"], "Manifest SHA mismatch")
    manifest = json.loads(payloads[manifest_name])
    require(manifest["batch"] == batch and manifest["completed_count"] == response["completed_count"],
            "Manifest/response batch mismatch")
    expected_members = {item["name"] for item in manifest["files"]} | {manifest_name}
    require(set(payloads) == expected_members, "Unexpected or missing archive member")
    for item in manifest["files"]:
        data = payloads[item["name"]]
        require(len(data) == item["bytes"] and sha_bytes(data) == item["sha256"],
                "Member digest mismatch: " + item["name"])
        if item["role"] == "complete_dev64_result":
            require(RESULT_NAME.fullmatch(item["name"]) and item["sha256"] == item["runner_sha256"]
                    == manifest["runner_result_sha256"][item["name"]], "Runner/member digest mismatch")
    # Validate all destinations and conflicts before writing any extracted file.
    for name, data in payloads.items():
        target = (destination / name).resolve()
        require(target.parent == destination, "Extraction target escaped destination")
        if target.exists():
            require(target.is_file() and sha(target) == sha_bytes(data),
                    "Existing file differs; preserved without overwrite: " + name)
    installed, unchanged = [], []
    for name, data in payloads.items():
        target = destination / name
        if target.exists():
            require(sha(target) == sha_bytes(data), "Destination changed during installation")
            unchanged.append(name)
        else:
            with target.open("xb") as stream:
                stream.write(data)
            require(sha(target) == sha_bytes(data), "Installed member failed verification")
            installed.append(name)
    for name, expected in manifest["runner_result_sha256"].items():
        require(RESULT_NAME.fullmatch(name) and sha(destination / name) == expected,
                "Local completed result differs from runner: " + name)
    receipt = {
        "protocol": "m1_dev64_incremental_local_verification_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "batch": batch,
        "archive": str(archive), "archive_sha256": response["archive_sha256"],
        "archive_bytes": response["archive_bytes"], "archive_sha256_verified": True,
        "all_member_sha256_verified": True, "runner_result_sha256_verified": True,
        "completed_count": len(manifest["runner_result_sha256"]),
        "new_result_count": response["new_result_count"], "complete_final": response["complete_final"],
        "installed": installed, "unchanged_same_sha256": unchanged,
        "script_sha256": sha(Path(__file__).resolve()),
    }
    receipt_path = destination / ("download_verification_" + batch + ".json")
    with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"completed_count": receipt["completed_count"],
                      "new_result_count": receipt["new_result_count"],
                      "complete_final": receipt["complete_final"],
                      "archive_sha256": receipt["archive_sha256"],
                      "verification": str(receipt_path)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise
