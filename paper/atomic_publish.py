"""Publish one or more text artifacts via same-directory temporary files."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path


def publish_text_artifacts(artifacts: Iterable[tuple[Path, str]]) -> None:
    """Prepare every artifact before atomically replacing any destination.

    Same-directory temporary files make each final ``os.replace`` atomic.  The
    prepare-first ordering also guarantees that a write failure cannot publish
    only the first member of a multi-artifact result.
    """
    items = list(artifacts)
    destinations = [path.resolve() for path, _ in items]
    if len(set(destinations)) != len(destinations):
        raise ValueError("artifact destinations must be distinct")

    prepared: list[tuple[Path, Path]] = []
    try:
        for destination, content in items:
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                text=True,
            )
            temporary = Path(temporary_name)
            prepared.append((temporary, destination))
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        for temporary, destination in prepared:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in prepared:
            temporary.unlink(missing_ok=True)
