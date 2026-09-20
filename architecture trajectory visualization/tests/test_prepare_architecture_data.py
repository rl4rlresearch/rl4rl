"""Preparation tests use tiny local ZIPs and mocked HTTP, never candidate execution."""

import hashlib
import io
import json
import tempfile
import unittest
import urllib.error
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from architecture_trajectory_visualization import prepare as preparation


def archive_bytes():
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("data/c0c3/test/runs/run/manifest.json", '{"run_id":"run"}')
        bundle.writestr(
            "data/c0c3/test/runs/run/candidates/seed/train.py",
            "raise RuntimeError('must never execute')\n",
        )
        bundle.writestr("data/c0c3/test/runs/run/model.pt", b"weights are unnecessary")
    return result.getvalue()


class RangeResponse(io.BytesIO):
    def __init__(self, content, begin, end, *, correct=True):
        super().__init__(content[begin : end + 1])
        self.status = 206
        self.headers = {
            "Content-Range": f"bytes {begin if correct else 0}-{end}/{len(content)}"
        }


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.content = archive_bytes()
        self.spec = {
            "campaign": "tiny",
            "filename": "tiny.zip",
            "oid": hashlib.sha256(self.content).hexdigest(),
            "size": len(self.content),
            "revision": "a" * 40,
        }

    def download(self, *, response_correct=True):
        requested = []

        def open_range(request, timeout):
            begin, end = map(
                int, request.get_header("Range").removeprefix("bytes=").split("-")
            )
            requested.append((begin, end))
            return RangeResponse(self.content, begin, end, correct=response_correct)

        with (
            patch.object(preparation, "CHUNK_BYTES", 32),
            patch.object(
                preparation,
                "download_actions",
                return_value={
                    self.spec["oid"]: {"href": "https://example.invalid/archive"}
                },
            ),
            patch.object(preparation.urllib.request, "urlopen", side_effect=open_range),
            patch.object(preparation.time, "sleep"),
            redirect_stdout(io.StringIO()),
        ):
            result = preparation.download_archives([self.spec], self.root, workers=2)
        return result, requested

    def test_resume_contiguous_prefix_and_verify_final_payload(self):
        (self.root / "tiny.zip.part").write_bytes(self.content[:17])
        result, requested = self.download()
        self.assertEqual(result["tiny"].read_bytes(), self.content)
        self.assertEqual(min(begin for begin, _ in requested), 17)
        self.assertFalse((self.root / "tiny.zip.ranges.json").exists())
        self.assertFalse((self.root / "tiny.zip.lock").exists())

    def test_completed_ranges_are_not_downloaded_again(self):
        partial = bytearray(len(self.content))
        partial[:64] = self.content[:64]
        (self.root / "tiny.zip.part").write_bytes(partial)
        (self.root / "tiny.zip.ranges.json").write_text(
            json.dumps(
                {
                    "prefix_bytes": 32,
                    "done": [32],
                    "oid": self.spec["oid"],
                    "size": len(self.content),
                }
            )
        )
        result, requested = self.download()
        self.assertEqual(min(begin for begin, _ in requested), 64)
        self.assertEqual(result["tiny"].read_bytes(), self.content)

    def test_wrong_full_checksum_never_becomes_a_completed_archive(self):
        self.spec["oid"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.download()
        self.assertFalse((self.root / "tiny.zip").exists())
        self.assertTrue((self.root / "tiny.zip.part").exists())
        self.assertFalse((self.root / "tiny.zip.lock").exists())

    def test_invalid_checkpoint_identity_fails_before_network(self):
        (self.root / "tiny.zip.part").write_bytes(self.content[:32])
        (self.root / "tiny.zip.ranges.json").write_text(
            json.dumps(
                {
                    "prefix_bytes": 32,
                    "done": [],
                    "oid": "0" * 64,
                    "size": len(self.content),
                }
            )
        )
        with patch.object(preparation, "download_actions") as network:
            with self.assertRaisesRegex(ValueError, "identity"):
                preparation.download_archives([self.spec], self.root)
            network.assert_not_called()

    def test_wrong_range_headers_are_rejected(self):
        (self.root / "tiny.zip.part").write_bytes(self.content[:17])
        with self.assertRaisesRegex(ValueError, "byte-range response"):
            self.download(response_correct=False)
        self.assertFalse((self.root / "tiny.zip").exists())

    def test_verified_existing_archive_does_not_need_network(self):
        (self.root / "tiny.zip").write_bytes(self.content)
        with (
            patch.object(preparation.urllib.request, "urlopen") as network,
            redirect_stdout(io.StringIO()),
        ):
            result = preparation.download_archives([self.spec], self.root)
            self.assertEqual(result["tiny"], self.root / "tiny.zip")
            network.assert_not_called()

    def test_expired_public_lfs_action_is_refreshed_without_losing_progress(self):
        expired = {self.spec["oid"]: {"href": "https://example.invalid/expired"}}
        fresh = {self.spec["oid"]: {"href": "https://example.invalid/fresh"}}

        def open_range(request, timeout):
            if request.full_url.endswith("expired"):
                raise urllib.error.HTTPError(request.full_url, 403, "Expired", {}, None)
            return RangeResponse(self.content, 0, len(self.content) - 1)

        with (
            patch.object(preparation, "CHUNK_BYTES", len(self.content) + 1),
            patch.object(
                preparation, "download_actions", side_effect=[expired, fresh]
            ) as actions,
            patch.object(preparation.urllib.request, "urlopen", side_effect=open_range),
            patch.object(preparation.time, "sleep"),
            redirect_stdout(io.StringIO()),
        ):
            result = preparation.download_archives([self.spec], self.root, workers=1)
        self.assertEqual(actions.call_count, 2)
        self.assertEqual(result["tiny"].read_bytes(), self.content)

    def test_preparation_copies_source_as_text_skips_weights_and_is_idempotent(self):
        archive = self.root / "tiny.zip"
        archive.write_bytes(self.content)
        destination = self.root / "prepared"
        with redirect_stdout(io.StringIO()):
            first = preparation.prepare_dataset(
                self.root, self.spec, archive, destination
            )
            second = preparation.prepare_dataset(
                self.root, self.spec, archive, destination
            )
        self.assertTrue(first["checksum_verified"])
        self.assertEqual(second["files_extracted"], first["files_extracted"])
        self.assertEqual(first["files_extracted"], 2)
        self.assertTrue(
            (destination / "data/c0c3/test/runs/run/candidates/seed/train.py").is_file()
        )
        self.assertFalse((destination / "data/c0c3/test/runs/run/model.pt").exists())
        self.assertEqual(
            json.loads((destination / "architecture-replay-source.json").read_text())[
                "source_revision"
            ],
            self.spec["revision"],
        )

    def test_preparation_refuses_to_merge_different_verified_revisions(self):
        archive = self.root / "tiny.zip"
        archive.write_bytes(self.content)
        destination = self.root / "prepared"
        with redirect_stdout(io.StringIO()):
            preparation.prepare_dataset(self.root, self.spec, archive, destination)
        receipt = (destination / "architecture-replay-source.json").read_bytes()
        for changed in ({"revision": "b" * 40}, {"oid": "0" * 64}):
            with self.subTest(changed=changed):
                with patch.object(preparation, "prepare_archive") as extraction:
                    with self.assertRaisesRegex(
                        ValueError, "different archive revision"
                    ):
                        preparation.prepare_dataset(
                            self.root, {**self.spec, **changed}, archive, destination
                        )
                    extraction.assert_not_called()
                self.assertEqual(
                    (destination / "architecture-replay-source.json").read_bytes(),
                    receipt,
                )

    def test_preparation_refuses_unidentified_existing_files(self):
        archive = self.root / "tiny.zip"
        archive.write_bytes(self.content)
        destination = self.root / "prepared"
        destination.mkdir()
        (destination / "unknown.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "without archive provenance"):
            preparation.prepare_dataset(self.root, self.spec, archive, destination)
        self.assertFalse((destination / "architecture-replay-source.json").exists())

    def test_interrupted_extraction_can_resume_only_the_same_archive(self):
        archive = self.root / "tiny.zip"
        archive.write_bytes(self.content)
        destination = self.root / "prepared"
        with (
            patch.object(
                preparation, "prepare_archive", side_effect=OSError("interruption")
            ),
            self.assertRaisesRegex(OSError, "interruption"),
        ):
            preparation.prepare_dataset(self.root, self.spec, archive, destination)
        self.assertTrue((destination / "architecture-replay-preparation.json").exists())
        self.assertFalse((destination / "architecture-replay-source.json").exists())
        with self.assertRaisesRegex(ValueError, "different archive revision"):
            preparation.prepare_dataset(
                self.root, {**self.spec, "revision": "b" * 40}, archive, destination
            )
        with redirect_stdout(io.StringIO()):
            receipt = preparation.prepare_dataset(
                self.root, self.spec, archive, destination
            )
        self.assertTrue(receipt["checksum_verified"])
        self.assertFalse(
            (destination / "architecture-replay-preparation.json").exists()
        )


if __name__ == "__main__":
    unittest.main()
