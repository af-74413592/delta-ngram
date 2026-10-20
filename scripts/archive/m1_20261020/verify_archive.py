"""Verify historical archive payloads; never import or execute archived sources."""
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


archive = Path(__file__).resolve().parent
root = archive.parents[2]
manifest_path = archive / "provenance_manifest.json"
expected_manifest_sha = (archive / "provenance_manifest.json.sha256").read_text(
    encoding="ascii").split()[0]
assert digest(manifest_path) == expected_manifest_sha, "Manifest digest differs"
manifest = json.loads(manifest_path.read_bytes())
for row in manifest["files"]:
    path = (root / row["repository_relative_path"]).resolve()
    assert path.is_relative_to(root) and path.is_file(), row["repository_relative_path"]
    assert path.stat().st_size == row["bytes"], row["repository_relative_path"]
    assert digest(path) == row["sha256"], row["repository_relative_path"]
print("Verified", len(manifest["files"]), "historical archive payloads; no source executed")
