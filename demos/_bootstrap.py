"""Bootstrap sys.path so demos can be run as plain scripts.

`python demos/demo_xxx.py` — without this shim — would fail with
ImportError because ``src/`` isn't on ``sys.path`` (pyproject's
``pythonpath = ["src"]`` only applies to pytest).

Every demo does::

    import _bootstrap  # noqa: F401
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
