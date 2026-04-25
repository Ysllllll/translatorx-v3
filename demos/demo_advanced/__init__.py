"""Advanced feature demos (STEP 5/6/7/8) split into one module per topic.

Each module exposes an async ``step_*`` entrypoint and is picked up by
``demos/demo_advanced_features.py`` via the ``--only`` flag. Keeping them
separate avoids the previous 524-line monolith and lets each feature evolve
without merging conflicts.

Run via the public entry::

    python demos/demo_advanced_features.py --only summary
    python demos/demo_advanced_features.py --only chunked,degrade
"""

from __future__ import annotations

from .chunked import step_chunked
from .degrade import step_degrade
from .dynamic import step_dynamic_terms
from .summary import step_summary

__all__ = [
    "step_dynamic_terms",
    "step_degrade",
    "step_chunked",
    "step_summary",
]
