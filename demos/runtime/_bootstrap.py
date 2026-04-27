"""Subdir bootstrap — add ``demos/`` and ``src/`` to sys.path.

Self-contained (does not import top-level ``demos/_bootstrap``) because
Python's module cache would otherwise shadow it.
"""

import sys
from pathlib import Path

_DEMOS = Path(__file__).resolve().parent.parent
_SRC = _DEMOS.parent / "src"

for _p in (_DEMOS, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
