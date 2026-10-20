"""CPU contract checks; no native model, Torch, tokenizer or datasets loaded."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from array import array


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"
sys.path.insert(0, str(SCRIPTS))
import qwen38_native_norm_probe as probe


class ObservationContracts(unittest.TestCase):
    def fixture(self, root):
        samples, docs, events, payload = [], [], [], bytearray()

        def entry(identifier, source, group, count):
            ids = list(range(1000, 1000 + count))
            values = array("I", ids)
            if sys.byteorder != "little":
                values.byteswap()
            data = values.tobytes()
            item = {"id": identifier, "source": source, "split": "train",
                    "group_id": group, "offset_tokens": len(payload) // 4,
                    "token_count": count, "token_sha256": probe.canonical_hash(ids),
                    "token_bytes_sha256": probe.hashlib.sha256(data).hexdigest(),
                    "text_sha256": "f" * 64}
            payload.extend(data)
            return item

        for source in probe.SOURCES:
            for number in range(4):
                identifier = f"{source}-{number}"
                sample = {"id": identifier, "sample_id": "cal:" + identifier,
                          "group_id": identifier, "source": source, "split": "train",
                          "eligible": True, "selection_hash": f"{number:064x}"}
                if source != "news_event":
                    item = entry(identifier, source, identifier, 1024)
                    docs.append(item)
                    sample.update(kind="ordinary", start=0, end=1024, blocks=2,
                                  text_sha256=item["text_sha256"], token_sha256=item["token_sha256"])
                else:
                    first = entry(identifier + ":history", source, identifier, 609)
                    target = entry(identifier + ":target", source, identifier, 128)
                    events.append({"id": identifier, "group_id": identifier,
                                   "source": source, "split": "train", "eligible": True,
                                   "articles": [first, target]})
                    sample.update(kind="event", blocks=4,
                                  article_ids=[first["id"], target["id"]],
                                  article_text_sha256=[first["text_sha256"], target["text_sha256"]],
                                  article_token_sha256=[first["token_sha256"], target["token_sha256"]])
                samples.append(sample)
        (root / "tokens.uint32").write_bytes(payload)
        for name, rows in (("document_index.jsonl", docs), ("event_index.jsonl", events)):
            (root / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        selection = {"ready": True, "deficits": {}, "split": "train", "seed": 17,
                     "groups_per_source_requested": 4, "source_manifest_sha256": "a" * 64,
                     "tokenizer_fingerprint": "sha256:b", "samples": samples}
        probe.write_json(root / "calibration_selection.json", selection)
        manifest = {"format": "qwen38_native_derived_data_v1",
                    "ready_for_calibration": True, "ready_for_training": False,
                    "derivation_scope": "calibration_subset_only",
                    "model": {"name": probe.MODEL_REPO, "revision": probe.MODEL_REVISION},
                    "source_manifest_sha256": "a" * 64,
                    "tokenizer": {"fingerprint": "sha256:b"},
                    "files": {p.name: probe.file_info(p) for p in root.iterdir()}}
        probe.write_json(root / "data_manifest.json", manifest)
        probe.write_json(root / "COMPLETE.json", {
            "format": "qwen38_native_derived_seal_v1", "ready_for_calibration": True,
            "derivation_scope": "calibration_subset_only",
            "data_manifest": probe.file_info(root / "data_manifest.json"),
            "files": manifest["files"]})

    def test_masks_selection_and_no_query_or_heldout_token_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            _, _, selected = probe.verify_cache(root)
            original, read_ids = probe.read_tokens, []

            def record(path, entry):
                read_ids.append(entry["id"])
                return original(path, entry)

            probe.read_tokens = record
            try:
                frames = probe.observation_frames(root, selected)
            finally:
                probe.read_tokens = original
            self.assertEqual(len(selected), 12)
            self.assertEqual(len(frames), 14)
            self.assertEqual(len(read_ids), 12)
            self.assertTrue(all(":target" not in name and "-2" not in name and "-3" not in name
                                for name in read_ids))
            self.assertTrue(all(f["loss_labels"] is None and f["loss_target_tokens"] == 0
                                and len(f["input_ids"]) % 2 == 0 for f in frames))
            later = [f for f in frames if f["article_token_start"] == 512]
            self.assertEqual(len(later), 2)
            for frame in later:
                self.assertEqual((frame["prefix_tokens"], frame["new_tokens"], frame["padding_tokens"]),
                                 (128, 97, 1))
                self.assertEqual(sum(frame["encoder_valid_mask"]), 225)
                self.assertEqual(sum(frame["write_mask"]), 97)
                self.assertFalse(any(frame["write_mask"][:128]))
                self.assertFalse(frame["write_mask"][-1])

    def test_changed_cache_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            (root / "tokens.uint32").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "cache file hash mismatch"):
                probe.verify_cache(root)

    def test_uint32_hash_checked_at_read(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            entry = probe.read_index(root / "document_index.jsonl")["books-0"]
            entry["token_bytes_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "cached token bytes"):
                probe.read_tokens(root / "tokens.uint32", entry)


if __name__ == "__main__":
    unittest.main()
