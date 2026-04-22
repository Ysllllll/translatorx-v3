"""``python -m api.service`` / ``translatorx-serve`` — uvicorn launcher.

Reads a config file path from ``--config`` (defaults to ``app.yaml`` in
the current working directory) and starts uvicorn using
``service.host`` / ``service.port`` from :class:`ServiceConfig`.

Examples::

    translatorx-serve --config config/app.yaml
    python -m api.service --config app.yaml --reload
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from api.app import App
from api.service import from_app_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="translatorx-serve")
    p.add_argument("--config", default=os.environ.get("TRX_CONFIG", "app.yaml"))
    p.add_argument("--host", default=None, help="Override service.host")
    p.add_argument("--port", type=int, default=None, help="Override service.port")
    p.add_argument("--reload", action="store_true", help="Enable uvicorn reload")
    p.add_argument("--log-level", default="info")
    return p.parse_args(argv)


def build() -> "object":
    """Entry point for ``uvicorn --factory api.service.__main__:build``."""
    config_path = os.environ.get("TRX_CONFIG", "app.yaml")
    app = App.from_config(config_path)
    return from_app_config(app)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level.upper())

    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn not installed. Install with: pip install 'translatorx[service]'", file=sys.stderr)
        return 2

    trx_app = App.from_config(args.config)
    host = args.host or trx_app.config.service.host
    port = args.port or trx_app.config.service.port

    # Stash the chosen path so the factory path can reload from env.
    os.environ["TRX_CONFIG"] = args.config
    fastapi_app = from_app_config(trx_app)

    uvicorn.run(
        fastapi_app,
        host=host,
        port=port,
        log_level=args.log_level,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
