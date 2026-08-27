#!/usr/bin/env python3
"""Tests for the immutable rank-coherent admission-package manifest."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from paper.validate_rank_coherent_manifest import validate_manifest


class RankCoherentManifestTests(unittest.TestCase):
    def test_repository_manifest_passes(self) -> None:
        self.assertEqual(validate_manifest(), 10)

    def test_changed_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "one.txt").write_text("changed", encoding="utf-8")
            manifest = {
                "schema_version": "rank-coherent-admission-manifest-v1",
                "files": {"one.txt": "0" * 64},
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                validate_manifest(path, root)

    def test_path_traversal_and_extra_schema_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "manifest.json"
            path.write_text(json.dumps({
                "schema_version": "rank-coherent-admission-manifest-v1",
                "files": {"../escape": "0" * 64},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "malformed"):
                validate_manifest(path, root)
            path.write_text(json.dumps({
                "schema_version": "rank-coherent-admission-manifest-v1",
                "files": {},
                "extra": True,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "top-level"):
                validate_manifest(path, root)


if __name__ == "__main__":
    unittest.main()
