"""Bootstrap sys.path so demos can be run as plain scripts.

`python demos/<topic>/<demo>.py` — without this shim — would fail with
ImportError because ``src/`` isn't on ``sys.path`` (pyproject's
``pythonpath = ["src"]`` only applies to pytest), nor is ``demos/`` (so
``_print`` / ``_shared`` would not resolve from a subdir).

Every demo does::

    import _bootstrap  # noqa: F401

Subdirectory demos use a 4-line shim ``demos/<subdir>/_bootstrap.py``
that prepends ``demos/`` to ``sys.path`` first, then re-imports this
module to also add ``src/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DEMOS = Path(__file__).resolve().parent
_SRC = _DEMOS.parent / "src"

for _p in (_DEMOS, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
