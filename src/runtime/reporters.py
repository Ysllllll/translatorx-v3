"""Error reporters — real-time surface for :class:`ErrorInfo` (D-038, D-050).

Three built-ins:

* :class:`LoggerReporter` — writes via the stdlib :mod:`logging` module.
* :class:`JsonlErrorReporter` — append-only JSONL file with optional
  rotation (powers the audit log at ``<course>/zzz_logs/errors.jsonl``).
* :class:`ChainReporter` — fans out to N reporters; swallows individual
  failures so one broken reporter can't stop the pipeline.

All reporters honour the D-038 contract:

* Sync API (``def report``, no awaits).
* Must not raise — the framework invokes them via a ``safe_call`` shim
  in the processor base class, but we defend internally anyway.
* Passing ``None`` disables the reporter path entirely; callers must
  handle that upstream.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from .errors import ErrorInfo

if TYPE_CHECKING:
    from model import SentenceRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoggerReporter
# ---------------------------------------------------------------------------


class LoggerReporter:
    """Emit errors through :mod:`logging`.

    Category → default level mapping:

    * ``fatal`` / ``permanent`` → ERROR
    * ``degraded``              → WARNING
    * ``transient``             → INFO

    Override via ``level_map`` if needed. ``logger`` defaults to a
    module-level logger so callers can configure formatting via the
    standard ``logging`` pipeline.
    """

    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        level_map: dict[str, int] | None = None,
    ):
        self._logger = logger if logger is not None else logging.getLogger("runtime.errors")
        self._level_map: dict[str, int] = {
            "fatal": logging.ERROR,
            "permanent": logging.ERROR,
            "degraded": logging.WARNING,
            "transient": logging.INFO,
        }
        if level_map:
            self._level_map.update(level_map)

    def report(
        self,
        err: ErrorInfo,
        record: "SentenceRecord",
        context: dict,
    ) -> None:
        level = self._level_map.get(err.category, logging.WARNING)
        self._logger.log(
            level,
            "[%s/%s] %s — %s (attempts=%d%s)",
            err.processor,
            err.category,
            err.code,
            err.message,
            err.attempts,
            f", cause={err.cause}" if err.cause else "",
        )


# ---------------------------------------------------------------------------
# JsonlErrorReporter
# ---------------------------------------------------------------------------


class JsonlErrorReporter:
    """Append-only JSONL error log (D-050).

    Each report writes one JSON line with fields::

        {"ts", "video", "record_id", "processor", "category", "code",
         "message", "attempts", "fingerprint", "cause"}

    Rotation is handled via :class:`logging.handlers.RotatingFileHandler`
    for proven correctness on multi-process / size-bounded scenarios.
    The underlying logger is dedicated per path (isolated from app logs).

    ``categories`` optionally filters which categories are written;
    ``None`` (default) means all.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        rotate_max_mb: int = 100,
        rotate_keep: int = 10,
        categories: Iterable[str] | None = None,
    ):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._categories: frozenset[str] | None = frozenset(categories) if categories is not None else None

        # Dedicated logger per path; name encodes the path to prevent
        # accidental handler reuse across reporters.
        self._logger = logging.getLogger(f"runtime.errors.jsonl.{self._path}")
        self._logger.propagate = False
        self._logger.setLevel(logging.INFO)

        # Only add our handler once; avoid duplicates on re-construction.
        handler_tag = f"jsonl-{self._path}"
        for existing in self._logger.handlers:
            if getattr(existing, "_trx_tag", None) == handler_tag:
                self._handler = existing
                break
        else:
            handler = RotatingFileHandler(
                filename=str(self._path),
                maxBytes=rotate_max_mb * 1024 * 1024,
                backupCount=rotate_keep,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            handler._trx_tag = handler_tag  # type: ignore[attr-defined]
            self._logger.addHandler(handler)
            self._handler = handler

    def report(
        self,
        err: ErrorInfo,
        record: "SentenceRecord",
        context: dict,
    ) -> None:
        if self._categories is not None and err.category not in self._categories:
            return

        payload = {
            "ts": err.at if err.at else time.time(),
            "video": context.get("video"),
            "course": context.get("course"),
            "record_id": record.extra.get("stream_id") if hasattr(record, "extra") else None,
            "processor": err.processor,
            "category": err.category,
            "code": err.code,
            "message": err.message,
            "attempts": err.attempts,
            "retryable": err.retryable,
            "fingerprint": context.get("fingerprint"),
            "cause": err.cause,
        }
        line = json.dumps(payload, ensure_ascii=False, default=str)
        self._logger.info(line)

    def close(self) -> None:
        """Flush and remove the file handler. Safe to call multiple times."""
        try:
            self._handler.flush()
        except Exception:
            pass
        try:
            self._logger.removeHandler(self._handler)
            self._handler.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ChainReporter
# ---------------------------------------------------------------------------


class ChainReporter:
    """Fan-out to multiple reporters. Any single failure is swallowed."""

    def __init__(self, reporters: Iterable[object]):
        self._reporters = tuple(reporters)

    def report(
        self,
        err: ErrorInfo,
        record: "SentenceRecord",
        context: dict,
    ) -> None:
        for reporter in self._reporters:
            try:
                reporter.report(err, record, context)  # type: ignore[attr-defined]
            except Exception as exc:  # defensive: never let one reporter poison the chain
                logger.warning(
                    "Reporter %r failed: %s",
                    type(reporter).__name__,
                    exc,
                )


__all__ = [
    "ChainReporter",
    "JsonlErrorReporter",
    "LoggerReporter",
]


# Convenience: allow ErrorInfo -> dict for callers that need it.
def error_info_to_dict(err: ErrorInfo) -> dict:
    """Return a plain dict representation of ``err``."""
    return asdict(err)
