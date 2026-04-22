"""Arq worker entrypoint — ``translatorx-worker`` console script.

Usage::

    translatorx-worker path/to/app.yaml

Loads the given :class:`AppConfig`, builds :class:`App`, wires the
Redis resource manager, and hands the arq ``WorkerSettings`` to
:func:`arq.run_worker`.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv:
        print("usage: translatorx-worker <app.yaml>", file=sys.stderr)
        return 2
    cfg_path = Path(argv[0])
    if not cfg_path.exists():
        print(f"config not found: {cfg_path}", file=sys.stderr)
        return 2

    from api.app.app import App
    from api.service.runtime.tasks_arq import build_worker_settings
    from application.config import AppConfig

    cfg = AppConfig.load(cfg_path)
    app = App.from_config(cfg)
    settings = build_worker_settings(app)

    try:
        from arq.worker import run_worker
    except ImportError:
        print("arq is required; install with `pip install arq`", file=sys.stderr)
        return 2

    class _Settings:
        pass

    for k, v in settings.items():
        setattr(_Settings, k, v)

    run_worker(_Settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
