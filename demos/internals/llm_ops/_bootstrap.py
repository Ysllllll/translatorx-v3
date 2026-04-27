"""Self-contained bootstrap for ``demos/internals/llm_ops/`` modules."""

import sys
from pathlib import Path

_DEMOS = Path(__file__).resolve().parents[2]
_SRC = _DEMOS.parent / "src"

for _p in (_DEMOS, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
