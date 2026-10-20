"""Prepare the five-file DEV64 increment; SSH writes require explicit --execute."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

OPS = Path(__file__).resolve().parent
ROOT = "/workspace/delta-ngram-dev64-context-backup-20261020"
PREFIX = "b2://clclcl/cl-research/delta-ngram/20261020-dev64-context/"
BASE = "fcd056414fd148425940d76c845b39e0ec14e381"
RECEIPTS = OPS / "dev64_context_backup_receipt_20261020"
CHUNK = 1024 * 1024
REMOTE = r'''
import base64,datetime,hashlib,json,os,pathlib,stat,sys
ROOT=pathlib.Path("/workspace/delta-ngram-dev64-context-backup-20261020")
M="metadata/member_manifest.jsonl";C="metadata/sha256.txt";F="metadata/COMPLETE.json"
SEALS={M,C,F}
ALLOWED={"m1_dev64_context_addon_20261020.zip","dev64_context_delta_20261020.bundle",
 "metadata/package_receipt.json","RESTORE.md","metadata/verify_context_backup_20261020.py"}
q=json.loads(sys.stdin.buffer.readline())
def require(v,m):
 if not v:raise RuntimeError(m)
def h(data):return hashlib.sha256(data).hexdigest()
def safe(relative):
 require(relative in ALLOWED|SEALS or relative.removesuffix(".upload.part") in ALLOWED,"Path outside whitelist")
 require(ROOT.resolve()==ROOT and not ROOT.is_symlink(),"Redirected backup root")
 p=ROOT/relative
 for node in (p,*p.parents):
  if node==ROOT.parent:break
  require(not node.is_symlink(),"Symlink in backup path")
 require(p.resolve().is_relative_to(ROOT),"Escaping backup path")
 return p
def info(p):
 if not p.exists():return None
 require(p.is_file() and not p.is_symlink(),"Expected regular file")
 before=p.stat();digest=hashlib.sha256()
 with p.open("rb") as stream:
  for chunk in iter(lambda:stream.read(8388608),b""):digest.update(chunk)
 after=p.stat()
 require((before.st_size,before.st_mtime_ns,before.st_ino)==
 (after.st_size,after.st_mtime_ns,after.st_ino),"File changed during hash")
 return {"size_bytes":after.st_size,"sha256":digest.hexdigest()}
def paths():
 found={}
 if ROOT.exists():
  for p in ROOT.rglob("*"):
   require(not p.is_symlink(),"Linked backup member")
   mode=p.lstat().st_mode
   require(stat.S_ISDIR(mode) or stat.S_ISREG(mode),"Special backup member")
   if p.is_file():found[p.relative_to(ROOT).as_posix()]=p
 return found
expected=q["expected"]
require(set(expected)==ALLOWED,"Unexpected payload set")
op=q["op"]
if op in ("inspect","append","finalize"):
 name=q["name"];p=safe(name);part=safe(name+".upload.part")
 require(name in ALLOWED,"Unexpected transfer member")
 if op=="inspect":result={"final":info(p),"part":info(part)}
 else:
  require(not safe(F).exists(),"Frozen backup must not change")
  require(not p.exists(),"Existing final member must not be overwritten")
  if op=="append":
   require(0<q["size_bytes"]<=1048576,"Invalid chunk size")
   data=sys.stdin.buffer.read(q["size_bytes"])
   require(len(data)==q["size_bytes"] and h(data)==q["sha256"],"Chunk SHA differs")
   previous=info(part) or {"size_bytes":0,"sha256":h(b"")}
   require(previous==q["previous"],"Resume prefix changed")
   require(previous["size_bytes"]+len(data)<=expected[name]["size_bytes"],"Chunk exceeds payload")
   part.parent.mkdir(parents=True,exist_ok=True)
   with part.open("ab") as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
   result={"final":None,"part":info(part)}
  elif op=="finalize":
   require(info(part)==expected[name],"Finished payload SHA differs")
   part.rename(p);result={"final":info(p),"part":None}
elif op=="seal":
 found=paths()
 require(set(found) in (ALLOWED,ALLOWED|SEALS),"Unexpected, missing or partial backup files")
 stamps={}
 for name in sorted(ALLOWED):
  p=safe(name);require(info(p)==expected[name],"Final payload differs: "+name)
  s=p.stat();stamps[name]=(s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)
 rows=[{"relative_path":name,**expected[name]} for name in sorted(ALLOWED)]
 manifest=b"".join((json.dumps(row,sort_keys=True)+"\n").encode() for row in rows)
 checksums=("".join(row["sha256"]+"  "+row["relative_path"]+"\n" for row in rows)+h(manifest)+"  "+M+"\n").encode()
 if F not in found:
  complete={
   "protocol":"m1_dev64_context_backup_20261020_v1","complete":True,
   "created_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
   "object_storage_prefix":"b2://clclcl/cl-research/delta-ngram/20261020-dev64-context/",
   "source_root":str(ROOT),"git_bundle_prerequisite_commit":"fcd056414fd148425940d76c845b39e0ec14e381",
   "git_tip":q["git_tip"],"transfer_plan_sha256":q["plan_sha256"],
   "payload_files":len(rows),"payload_bytes":sum(r["size_bytes"] for r in rows),
   "total_objects":len(rows)+3,"total_bytes":0,
   "seals":{M:{"size_bytes":len(manifest),"sha256":h(manifest)},
            C:{"size_bytes":len(checksums),"sha256":h(checksums)}},
   "scope":"Small independent context increment; no old model, old run or full code bundle copied.",
   "seal_policy":"No member changes after sealing. COMPLETE is pinned by an external local receipt.",
   "b2_upload_verified":False,"b2_restore_verified":False}
  for unused in range(12):
   data=(json.dumps(complete,sort_keys=True,indent=2)+"\n").encode()
   total=complete["payload_bytes"]+len(manifest)+len(checksums)+len(data)
   if total==complete["total_bytes"]:break
   complete["total_bytes"]=total
  else:raise RuntimeError("Byte-total fixed point failed")
  seal_data={M:manifest,C:checksums,F:data}
  for name in (M,C,F):
   with safe(name).open("xb") as stream:stream.write(seal_data[name]);stream.flush();os.fsync(stream.fileno())
 else:
  seal_data={name:safe(name).read_bytes() for name in SEALS}
  complete=json.loads(seal_data[F])
  require(seal_data[M]==manifest and seal_data[C]==checksums,"Frozen seal content differs")
  require(complete["transfer_plan_sha256"]==q["plan_sha256"] and complete["git_tip"]==q["git_tip"],"Frozen plan differs")
 final=paths()
 require(set(final)==ALLOWED|SEALS,"Final object set differs")
 for name,stamp in stamps.items():
  s=final[name].stat()
  require((s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)==stamp,"Payload changed while sealing")
 require(complete["seals"]=={name:{"size_bytes":len(seal_data[name]),"sha256":h(seal_data[name])}
  for name in (M,C)},"Final seal hash pins differ")
 require(complete["total_objects"]==len(final),"Final object total differs")
 require(sum(p.stat().st_size for p in final.values())==complete["total_bytes"],"Final byte total differs")
 result={"complete":True,"payload_files":len(rows),"payload_bytes":complete["payload_bytes"],
  "total_objects":len(final),"total_bytes":complete["total_bytes"],
  "seals":{name:{"sha256":h(data),"size_bytes":len(data)} for name,data in seal_data.items()},
  "seal_bytes":{name:base64.b64encode(data).decode() for name,data in seal_data.items()}}
else:raise RuntimeError("Unsupported operation")
print(json.dumps(result),flush=True)
'''


def digest(path, length=None):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = length
        while remaining is None or remaining:
            chunk = stream.read(CHUNK if remaining is None else min(CHUNK, remaining))
            if not chunk:
                break
            result.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return result.hexdigest()


def save_immutable(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError("Existing local receipt conflicts: " + str(path))
    else:
        with path.open("xb") as stream:
            stream.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addon-zip", type=Path, required=True)
    parser.add_argument("--delta-bundle", type=Path, required=True)
    parser.add_argument("--package-receipt", type=Path, required=True)
    parser.add_argument("--restore-readme", type=Path, required=True)
    parser.add_argument("--git-tip", required=True)
    parser.add_argument("--execute", action="store_true", help="Upload only the five pinned files, then seal the new independent stage")
    args = parser.parse_args()
    if not re.fullmatch("[a-f0-9]{40}", args.git_tip):
        parser.error("git-tip must be the exact 40-hex report commit")
    sources = {
        "m1_dev64_context_addon_20261020.zip": args.addon_zip.resolve(),
        "dev64_context_delta_20261020.bundle": args.delta_bundle.resolve(),
        "metadata/package_receipt.json": args.package_receipt.resolve(),
        "RESTORE.md": args.restore_readme.resolve(),
        "metadata/verify_context_backup_20261020.py": OPS / "verify_context_backup_20261020.py",
    }
    expected = {name: {"size_bytes": path.stat().st_size, "sha256": digest(path)}
                for name, path in sources.items()}
    if any(row["size_bytes"] <= 0 for row in expected.values()):
        parser.error("All five authorized payloads must be nonempty files")
    plan = {"protocol": "m1_dev64_context_transfer_plan_20261020_v1",
        "remote_root": ROOT, "object_storage_prefix": PREFIX, "git_base": BASE, "git_tip": args.git_tip,
        "members": [{"relative_path": name, "local_path": str(sources[name]), **expected[name]} for name in sorted(sources)]}
    plan_bytes = (json.dumps(plan, sort_keys=True, indent=2) + "\n").encode()
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    save_immutable(RECEIPTS / "transfer_plan.json", plan_bytes)
    if not args.execute:
        print(json.dumps({"prepared": True, "network_used": False, "remote_modified": False,
                          "transfer_plan": str(RECEIPTS / "transfer_plan.json"), "plan_sha256": plan_sha}), flush=True)
        return

    def call(request, data=b""):
        request = {**request, "expected": expected, "git_tip": args.git_tip, "plan_sha256": plan_sha}
        command = ["ssh", "-p", "31123", "-o", "BatchMode=yes", "-o",
            "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15", "-o",
            "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2",
            "root@ssh6.vast.ai", "python3 -c " + shlex.quote(REMOTE)]
        result = subprocess.run(command, input=json.dumps(request).encode() + b"\n" + data,
                                capture_output=True, timeout=45)
        if result.returncode:
            raise RuntimeError("SSH operation failed: " + result.stderr.decode(errors="replace")[-1200:])
        return json.loads(result.stdout)

    def inspect(name):
        for attempt in range(3):
            try:
                return call({"op": "inspect", "name": name})
            except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError):
                if attempt == 2:
                    raise
                time.sleep(1)

    for name, path in sources.items():
        state = inspect(name)
        failures = 0
        while state["final"] != expected[name]:
            if state["final"] is not None:
                raise RuntimeError("Conflicting final member preserved: " + name)
            previous = state["part"] or {"size_bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
            offset = previous["size_bytes"]
            if offset > expected[name]["size_bytes"] or digest(path, offset) != previous["sha256"]:
                raise RuntimeError("Conflicting partial member preserved: " + name)
            try:
                if offset == expected[name]["size_bytes"]:
                    state = call({"op": "finalize", "name": name})
                else:
                    with path.open("rb") as stream:
                        stream.seek(offset)
                        data = stream.read(CHUNK)
                    state = call({"op": "append", "name": name, "previous": previous,
                                  "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}, data)
                    print(json.dumps({"file": name, "confirmed_bytes": state["part"]["size_bytes"],
                                      "total_bytes": expected[name]["size_bytes"]}), flush=True)
                failures = 0
            except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError):
                failures += 1
                if failures >= 3:
                    raise
                time.sleep(1)
                state = inspect(name)
        if path.stat().st_size != expected[name]["size_bytes"] or digest(path) != expected[name]["sha256"]:
            raise RuntimeError("Local payload changed: " + name)
    sealed = call({"op": "seal"})
    for name, encoded in sealed.pop("seal_bytes").items():
        data = base64.b64decode(encoded, validate=True)
        if hashlib.sha256(data).hexdigest() != sealed["seals"][name]["sha256"]:
            raise RuntimeError("Seal download SHA differs")
        save_immutable(RECEIPTS / name, data)
    receipt = {**sealed, "remote_root": ROOT, "object_storage_prefix": PREFIX,
               "transfer_plan_sha256": plan_sha, "all_payload_sha256_verified": True,
               "old_stages_modified": False, "b2_upload_performed": False, "shutdown_performed": False}
    receipt_data = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    save_immutable(RECEIPTS / "finalization_receipt.json", receipt_data)
    print(json.dumps({**receipt, "local_receipt": str(RECEIPTS / "finalization_receipt.json"),
                      "local_receipt_sha256": hashlib.sha256(receipt_data).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
