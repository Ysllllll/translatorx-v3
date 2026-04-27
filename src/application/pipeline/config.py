"""Deprecated alias — use :mod:`application.pipeline.loader` instead.

Retained as a re-export shim to avoid breaking existing imports while
Phase 2 lands. New code should import from
:mod:`application.pipeline.loader` (or the package facade).
"""

from __future__ import annotations

from .loader import load_pipeline_dict, load_pipeline_yaml, parse_pipeline_yaml

__all__ = ["load_pipeline_dict", "load_pipeline_yaml", "parse_pipeline_yaml"]
