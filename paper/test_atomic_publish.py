"""Executable atomic-publication contract for generated paper artifacts."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from paper.atomic_publish import publish_text_artifacts


class AtomicPublishTests(unittest.TestCase):
    def test_success_replaces_artifacts_and_leaves_no_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            first, second = directory / "summary.json", directory / "figure.svg"
            first.write_text("old summary", encoding="utf-8")
            second.write_text("old figure", encoding="utf-8")

            publish_text_artifacts(((first, "new summary\n"), (second, "new figure\n")))

            self.assertEqual(first.read_text(encoding="utf-8"), "new summary\n")
            self.assertEqual(second.read_text(encoding="utf-8"), "new figure\n")
            self.assertEqual(sorted(directory.iterdir()), [second, first])

    def test_second_artifact_write_failure_preserves_both_destinations(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            first, second = directory / "summary.json", directory / "figure.svg"
            first.write_text("old summary", encoding="utf-8")
            second.write_text("old figure", encoding="utf-8")
            real_fdopen = os.fdopen
            calls = 0

            def failing_fdopen(*args: object, **kwargs: object):
                nonlocal calls
                calls += 1
                if calls == 2:
                    os.close(args[0])
                    raise OSError("simulated second-artifact write failure")
                return real_fdopen(*args, **kwargs)

            with mock.patch("paper.atomic_publish.os.fdopen", side_effect=failing_fdopen):
                with self.assertRaisesRegex(OSError, "second-artifact"):
                    publish_text_artifacts(((first, "new summary"), (second, "new figure")))

            self.assertEqual(first.read_text(encoding="utf-8"), "old summary")
            self.assertEqual(second.read_text(encoding="utf-8"), "old figure")
            self.assertEqual(sorted(directory.iterdir()), [second, first])


if __name__ == "__main__":
    unittest.main()
