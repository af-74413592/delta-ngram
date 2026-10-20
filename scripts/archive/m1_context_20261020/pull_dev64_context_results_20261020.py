"""Read completed context results and logs via SSH; verify every byte locally."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile

OPS = Path(__file__).resolve().parent
DEST = OPS / "dev64_context_results_20261020"
REMOTE = r'''
import datetime,hashlib,io,json,pathlib,subprocess,sys,tarfile
root=pathlib.Path("/workspace/delta-ngram-runs/m1_dev64_context_20261020")
repo=pathlib.Path("/workspace/delta-ngram-context-evaluation-20261020")
name="delta_ngram_dev64_context_20261020"
def require(v,m):
 if not v:raise RuntimeError(m)
def h(b):return hashlib.sha256(b).hexdigest()
status=subprocess.run(["supervisorctl","status",name],capture_output=True,text=True,timeout=10).stdout.strip()
require("EXITED" in status,"Evaluation has not exited")
result=json.loads((root/"context.json").read_bytes())
require(result["complete"] and result["execution"]["context_cells"]==320,"Evaluation incomplete")
require(result["execution"]["context_read_windows"]==865,"Window count differs")
require(result["empty_sanity"]["all_64_families_passed"],"Empty sanity failed")
paths={}
for p in root.rglob("*"):
 require(not p.is_symlink(),"Linked result path")
 require(".part" not in p.name and not p.name.endswith(".tmp"),"Incomplete result path")
 if p.is_file():paths[p.relative_to(root).as_posix()]=p
paths.update({
 "logs/"+name+".log":pathlib.Path("/var/log")/(name+".log"),
 "logs/portal/"+name+".log":pathlib.Path("/var/log/portal")/(name+".log"),
 "execution/supervisor_wrapper.sh":pathlib.Path("/opt/supervisor-scripts")/(name+".sh"),
 "execution/supervisor_config.conf":pathlib.Path("/etc/supervisor/conf.d")/(name+".conf"),
 "execution/evaluate_dev64_full_context_20261020.py":repo/"scripts/evaluate_dev64_full_context_20261020.py"})
payload={};members=[]
for relative,p in sorted(paths.items()):
 require(p.is_file() and not p.is_symlink(),"Expected regular result or log: "+str(p))
 before=p.stat(); data=p.read_bytes();after=p.stat()
 require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),"File changed during read")
 payload[relative]=data
 members.append({"relative_path":relative,"source_path":str(p),"size_bytes":len(data),"sha256":h(data)})
require(h(payload["execution/evaluate_dev64_full_context_20261020.py"])=="0c0c5902452351f702c004dc07682247d1f4ee3152732bc20c861a0e03768130","Executed runner differs")
manifest={"protocol":"dev64_context_result_download_20261020_v1",
 "created_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "source_root":str(root),"source_commit":subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip(),
 "supervisor_status":status,"members":members,"member_files":len(members),
 "member_bytes":sum(r["size_bytes"] for r in members),
 "result_complete":True,"read_only_remote_collection":True}
payload["download_manifest.json"]=(json.dumps(manifest,indent=2)+"\n").encode()
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode="w:gz") as archive:
 for relative,data in sorted(payload.items()):
  info=tarfile.TarInfo(relative);info.size=len(data);info.mtime=0;info.mode=0o644
  archive.addfile(info,io.BytesIO(data))
data=buffer.getvalue()
header={"size_bytes":len(data),"sha256":h(data)}
sys.stdout.buffer.write((json.dumps(header)+"\n").encode()+data)
sys.stdout.buffer.flush()
'''

command = ["ssh", "-p", "31123", "-o", "BatchMode=yes", "-o",
           "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
           "root@ssh6.vast.ai", "python3 -c " + shlex.quote(REMOTE)]
response = subprocess.run(command, capture_output=True, timeout=55)
if response.returncode:
    raise RuntimeError(response.stderr.decode(errors="replace")[-3000:])
header_bytes, archive_bytes = response.stdout.split(b"\n", 1)
header = json.loads(header_bytes)
assert len(archive_bytes) == header["size_bytes"]
assert hashlib.sha256(archive_bytes).hexdigest() == header["sha256"]
with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
    extracted = {}
    for member in archive.getmembers():
        relative = PurePosixPath(member.name)
        assert member.isfile() and not relative.is_absolute() and ".." not in relative.parts
        assert member.name == relative.as_posix() and member.name not in extracted
        extracted[member.name] = archive.extractfile(member).read()
manifest = json.loads(extracted["download_manifest.json"])
expected = {row["relative_path"]: row for row in manifest["members"]}
assert set(extracted) == set(expected) | {"download_manifest.json"}
for name, row in expected.items():
    assert len(extracted[name]) == row["size_bytes"]
    assert hashlib.sha256(extracted[name]).hexdigest() == row["sha256"]
assert manifest["source_commit"] == "e1df5725fd2a4d6c9f523ad024ade3b7eba7a8cc"
result = json.loads(extracted["context.json"])
preflight = json.loads(extracted["preflight.json"])
assert result["complete"] and preflight["registration_sha256"] == result["preflight_registration_sha256"]
assert result["preflight_registration_sha256"] == "9fcb452ba4a6276fa9029e01e6b22abf27f20792ecebb135109852e5c2cdc9b5"
assert hashlib.sha256(extracted["preflight.json"]).hexdigest() == result["preflight_sha256"]
DEST.mkdir(parents=True, exist_ok=True)
for name, data in extracted.items():
    path = DEST / name
    assert path.resolve().is_relative_to(DEST.resolve())
    assert not path.exists() or path.read_bytes() == data, "Existing local conflict: " + name
for name in ["context.json", "preflight.json", *sorted(set(extracted) - {"context.json", "preflight.json"})]:
    path = DEST / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(extracted[name])
archive_path = DEST / "download_payload.tar.gz"
assert not archive_path.exists() or archive_path.read_bytes() == archive_bytes
if not archive_path.exists():
    archive_path.write_bytes(archive_bytes)
receipt = {
    "complete": True, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    "local_root": str(DEST), "member_files": manifest["member_files"],
    "member_bytes": manifest["member_bytes"], "all_member_sha256_verified": True,
    "archive": header, "source_commit": manifest["source_commit"],
    "context_sha256": expected["context.json"]["sha256"],
    "preflight_sha256": expected["preflight.json"]["sha256"],
    "download_manifest_sha256": hashlib.sha256(extracted["download_manifest.json"]).hexdigest(),
}
receipt_path = DEST / "download_verification.json"
if not receipt_path.exists():
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt), flush=True)
