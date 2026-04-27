"""Self-contained reuse of the parent demos/service/_bootstrap.py."""

import sys
from pathlib import Path

_DEMOS_DIR = str(Path(__file__).resolve().parents[2])
_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
for _p in (_DEMOS_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
