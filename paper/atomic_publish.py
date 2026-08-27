"""Publish text artifacts with prepare-first replacement and error rollback."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path


def publish_text_artifacts(artifacts: Iterable[tuple[Path, str]]) -> None:
    """Prepare every artifact and roll back caught replacement failures.

    Same-directory temporary files make each final ``os.replace`` atomic.  The
    prepare-first ordering also guarantees that a write failure cannot publish
    only the first member of a multi-artifact result. Existing destinations are
    moved to same-directory backups immediately before publication. If a final
    operation raises, every destination is restored to its entry state.

    This is an exception-rollback contract, not a durable transaction journal:
    abrupt process or operating-system termination can still require recovery.
    """
    items = list(artifacts)
    destinations = [path.resolve() for path, _ in items]
    if len(set(destinations)) != len(destinations):
        raise ValueError("artifact destinations must be distinct")

    prepared: list[tuple[Path, Path]] = []
    backups: list[tuple[Path, Path, bool]] = []
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
        try:
            for temporary, destination in prepared:
                existed = destination.exists()
                descriptor, backup_name = tempfile.mkstemp(
                    dir=destination.parent,
                    prefix=f".{destination.name}.",
                    suffix=".bak",
                )
                os.close(descriptor)
                backup = Path(backup_name)
                backup.unlink()
                backups.append((backup, destination, existed))
                if existed:
                    os.replace(destination, backup)
                os.replace(temporary, destination)
        except BaseException:
            rollback_errors: list[OSError] = []
            for backup, destination, existed in reversed(backups):
                try:
                    if existed and backup.exists():
                        os.replace(backup, destination)
                    elif not existed:
                        destination.unlink(missing_ok=True)
                except OSError as error:
                    rollback_errors.append(error)
            if rollback_errors:
                raise RuntimeError(
                    "artifact publication failed and rollback was incomplete"
                ) from rollback_errors[0]
            raise
        for backup, _, _ in backups:
            backup.unlink(missing_ok=True)
    finally:
        for temporary, _ in prepared:
            temporary.unlink(missing_ok=True)
        for backup, _, _ in backups:
            backup.unlink(missing_ok=True)
