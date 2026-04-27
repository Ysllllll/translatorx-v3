"""Schema migrations for video JSON documents.

Extracted from ``store.py`` to keep the orchestration class small and to
make adding new migrations a single-file change. The migration ladder is
keyed by the *source* version: ``_VIDEO_MIGRATIONS[N]`` upgrades a v``N``
document to v``N+1``. Each migration mutates ``data`` in-place and is
expected to bump ``data["schema_version"]``.
"""

from __future__ import annotations

from typing import Any, Callable

SCHEMA_VERSION = 2


class IncompatibleStoreError(RuntimeError):
    """Raised when a stored document cannot be upgraded to the runtime schema."""


def _migrate_video_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a v1 video JSON document to v2 in-place.

    v2 added the ``variants`` / ``prompts`` / ``terms`` top-level dicts
    (see :func:`store.empty_video_data`). Existing v1 files simply
    lacked those keys; we provide empty dicts so downstream code sees a
    uniform shape regardless of whether the file was newly created or
    read from an older deployment.
    """
    data.setdefault("variants", {})
    data.setdefault("prompts", {})
    data.setdefault("terms", {})
    data["schema_version"] = 2
    return data


_VIDEO_MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {
    1: _migrate_video_v1_to_v2,
}


def check_schema(data: dict[str, Any], where: str) -> None:
    """Walk the migration ladder until ``data`` is at ``SCHEMA_VERSION``.

    * Unmarked documents are treated as v1 (the earliest format ever
      persisted).
    * Documents newer than the runtime raise
      :class:`IncompatibleStoreError`.
    * Missing migrations on the ladder raise the same error.
    """
    version = data.get("schema_version")
    if version is None:
        version = 1
        data["schema_version"] = 1
    if version > SCHEMA_VERSION:
        raise IncompatibleStoreError(f"{where}: schema_version={version} is newer than runtime (supports <= {SCHEMA_VERSION})")
    while data.get("schema_version", version) < SCHEMA_VERSION:
        cur = int(data["schema_version"])
        migrate = _VIDEO_MIGRATIONS.get(cur)
        if migrate is None:
            raise IncompatibleStoreError(f"{where}: no migration from schema_version={cur} -> {cur + 1}")
        migrate(data)


__all__ = [
    "SCHEMA_VERSION",
    "IncompatibleStoreError",
    "check_schema",
]
